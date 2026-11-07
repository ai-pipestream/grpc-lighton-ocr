#include "service/lighton_service.h"

#include <cstdint>
#include <unistd.h>
#include <array>

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include "parse/markdown.h"

namespace lighton {
namespace {

grpc::Status invalid(const std::string& message) {
  return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
}

grpc::Status precondition(const std::string& message) {
  return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, message);
}

grpc::Status exhausted(const std::string& message) {
  return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, message);
}

void provenance(locrv1::InferenceProvenance* out) {
  out->set_service_version("0.1.0"); out->set_backend_revision(kBackendRevision);
  out->set_model_revision(kModelRevision); out->set_model_sha256(kModelSha);
  out->set_projector_sha256(kProjectorSha); out->set_device("cpu");
  std::array<char, 256> hostname{};
  if (::gethostname(hostname.data(), hostname.size() - 1) == 0) out->set_server_instance(hostname.data());
}
// The page parser emits texts and pictures only. Rewrite their references
// recursively before merging repeated fields into the folded document.
void offset_refs(google::protobuf::Message* message, int texts, int pictures) {
  const auto* descriptor = message->GetDescriptor();
  const auto* reflection = message->GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const auto* field = descriptor->field(i);
    if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      if (field->is_repeated()) {
        for (int j = 0; j < reflection->FieldSize(*message, field); ++j)
          offset_refs(reflection->MutableRepeatedMessage(message, field, j), texts, pictures);
      } else if (reflection->HasField(*message, field)) {
        offset_refs(reflection->MutableMessage(message, field), texts, pictures);
      }
    } else if (!field->is_repeated() &&
               field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_STRING &&
               (field->name() == "ref" || field->name() == "self_ref")) {
      auto ref = reflection->GetString(*message, field);
      for (const auto& [prefix, offset] : {std::pair{"#/texts/", texts}, {"#/pictures/", pictures}}) {
        if (ref.starts_with(prefix)) {
          ref = std::string(prefix) + std::to_string(std::stoul(ref.substr(std::string(prefix).size())) + offset);
          reflection->SetString(message, field, ref);
          break;
        }
      }
    }
  }
}

}  // namespace

void project_native_result(const NativeResult& result, uint32_t page_no,
                           locrv1::LightonModel model, uint64_t first_step,
                           locrv1::ParseDocumentResponse* event) {
  docv1::CollectorSource source;
  source.set_collector("grpc-lighton-ocr"); source.set_model(model_repo(model));
  source.set_version(kModelRevision);
  docv1::GenerationTrace trace;
  trace.set_page_no(page_no); trace.set_first_step(first_step); trace.set_text(result.raw_text);
  trace.mutable_source()->CopyFrom(source);
  for (const auto& generated : result.steps) {
    auto* step = trace.add_steps(); step->set_token(generated.token);
    step->set_piece(generated.piece); step->set_logprob(generated.logprob);
    step->set_model_token(generated.top_token); step->set_model_logprob(generated.top_logprob);
    step->set_intervention(generated.token == generated.top_token ?
        docv1::GENERATION_INTERVENTION_NONE : docv1::GENERATION_INTERVENTION_UNSPECIFIED);
  }
  docv1::Document document;
  std::string error;
  const bool parsed = parse_lightonocr_markdown(result.text, page_no, result.width, result.height,
                                               &document, &error, &source);
  document.add_generation_traces()->CopyFrom(trace);
  if (parsed && result.finish == NativeFinish::end_token) {
    auto* converted = event->mutable_page_document();
    converted->set_page_no(page_no); provenance(converted->mutable_provenance());
    converted->set_finish_reason(locrv1::FINISH_REASON_END_TOKEN);
    converted->mutable_document()->Swap(&document);
    return;
  }
  auto* failure = event->mutable_page_failure(); failure->set_page_no(page_no);
  failure->mutable_generation_trace()->CopyFrom(trace); provenance(failure->mutable_provenance());
  if (parsed) failure->mutable_partial_document()->Swap(&document);
  switch (result.finish) {
    case NativeFinish::token_limit: failure->set_finish_reason(locrv1::FINISH_REASON_TOKEN_LIMIT); break;
    case NativeFinish::output_limit: failure->set_finish_reason(locrv1::FINISH_REASON_OUTPUT_LIMIT); break;
    case NativeFinish::deadline: failure->set_finish_reason(locrv1::FINISH_REASON_DEADLINE); break;
    case NativeFinish::engine_error: failure->set_finish_reason(locrv1::FINISH_REASON_ENGINE_ERROR); break;
    case NativeFinish::end_token: failure->set_finish_reason(locrv1::FINISH_REASON_PARSE_ERROR); break;
  }
  failure->set_code(result.finish == NativeFinish::token_limit || result.finish == NativeFinish::output_limit ||
                    result.finish == NativeFinish::deadline ? locrv1::PAGE_FAILURE_CODE_RESOURCE_LIMIT :
                    locrv1::PAGE_FAILURE_CODE_INFERENCE);
  failure->set_message(parsed ? "native generation incomplete; available output retained" : error);
}

LightOnOcrServiceImpl::LightOnOcrServiceImpl(Config config) : config_(std::move(config)) {
  if (!config_.diagnostic_no_engine) {
    if (config_.model != locrv1::LIGHTON_MODEL_0_8B)
      throw std::invalid_argument("only CPU LightOnOCR-3-0.8B is qualified; 4B is unavailable");
    engine_ = std::make_unique<NativeEngine>(config_.models_dir);
  }
}

grpc::Status LightOnOcrServiceImpl::GetServiceInfo(grpc::ServerContext* /*context*/,
                                                   const locrv1::GetServiceInfoRequest* /*request*/,
                                                   locrv1::GetServiceInfoResponse* response) {
  response->set_listen_address(config_.listen_address);
  response->set_engine_loaded(engine_loaded());
  if (engine_loaded()) provenance(response->mutable_provenance());
  response->set_model(config_.model);
  response->set_model_repo(model_repo(config_.model));
  return grpc::Status::OK;
}

grpc::Status LightOnOcrServiceImpl::ParseDocument(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<locrv1::ParseDocumentResponse, locrv1::ParseDocumentRequest>* stream) {
  unsigned active = streams_.load();
  do {
    if (active >= 4) return exhausted("four document streams are already active");
  } while (!streams_.compare_exchange_weak(active, active + 1));
  struct Release {
    std::atomic<unsigned>& count;
    ~Release() { --count; }
  } release{streams_};
  locrv1::ParseDocumentRequest request;
  bool saw_options = false;
  locrv1::ParseOptions options;
  std::set<uint32_t> pages;
  uint32_t failed = 0, succeeded = 0;
  uint64_t steps = 0;
  size_t folded_bytes = 0;
  docv1::Document folded;
  while (stream->Read(&request)) {
    if (context->IsCancelled()) return {grpc::StatusCode::CANCELLED, "request cancelled"};
    switch (request.payload_case()) {
      case locrv1::ParseDocumentRequest::kOptions:
        if (saw_options) return invalid("ParseOptions repeated");
        saw_options = true;
        options = request.options();
        if (!locrv1::LightonModel_IsValid(options.model())) return invalid("unknown model");
        if (!std::isfinite(options.temperature()) || options.temperature() < 0) {
          return invalid("temperature must be finite and nonnegative");
        }
        if (options.max_tokens_per_page() > kMaxTokens) return exhausted("max_tokens_per_page exceeds 1024");
        if (options.temperature() > 2 || (options.temperature() > 0 && options.temperature() < 0.01f))
          return invalid("temperature must be zero (greedy) or between 0.01 and 2");
        if (options.model() == locrv1::LIGHTON_MODEL_UNSPECIFIED) {
          return invalid("ParseOptions.model is required");
        }
        if (options.model() != config_.model) {
          return precondition("ParseOptions.model is not the checkpoint this process serves");
        }
        if (options.page_count() == 0) return invalid("page_count is required");
        if (options.page_count() > config_.max_pages) return exhausted("page_count exceeds the cap");
        break;
      case locrv1::ParseDocumentRequest::kPage: {
        if (!saw_options) return invalid("PageImage before ParseOptions");
        const locrv1::PageImage& page = request.page();
        if (page.page_no() == 0 || page.page_no() > options.page_count()) {
          return invalid("page_no is outside 1..page_count");
        }
        if (pages.contains(page.page_no())) return invalid("page_no repeated");
        if (page.png().size() > config_.max_page_bytes) return exhausted("page exceeds the byte cap");
        pages.insert(page.page_no());
        locrv1::ParseDocumentResponse event;
        if (!engine_) {
          auto* failure = event.mutable_page_failure();
          failure->set_page_no(page.page_no());
          failure->set_code(page.png().empty() ? locrv1::PAGE_FAILURE_CODE_BAD_IMAGE
                                              : locrv1::PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE);
          failure->set_message("native engine unavailable in diagnostic mode");
          ++failed;
        } else {
          try {
            auto result = engine_->infer(page.png(), options.max_tokens_per_page(),
                options.temperature(), page.width_px(), page.height_px(),
                [&] { return context->IsCancelled(); });
            project_native_result(result, page.page_no(), config_.model, steps, &event);
            if (event.has_page_failure()) {
              ++failed;
              if (!stream->Write(event)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
              steps += result.steps.size(); request.Clear(); break;
            }
            auto* document = event.mutable_page_document()->mutable_document();
            steps += result.steps.size();
            if (options.emit_document()) {
              folded_bytes += document->ByteSizeLong();
              if (folded_bytes > 16 * 1024 * 1024) return exhausted("folded document exceeds 16 MiB");
              docv1::Document copy = *document;
              offset_refs(&copy, folded.texts_size(), folded.pictures_size());
              folded.MergeFrom(copy);
            }
            ++succeeded;
            if (!options.emit_page_documents()) { request.Clear(); break; }
          } catch (const EngineError& error) {
            if (context->IsCancelled() || error.kind == EngineError::cancelled)
              return {grpc::StatusCode::CANCELLED, error.what()};
            if (error.kind == EngineError::busy) return exhausted(error.what());
            auto* failure = event.mutable_page_failure(); failure->set_page_no(page.page_no());
            failure->set_code(error.kind == EngineError::bad_image ? locrv1::PAGE_FAILURE_CODE_BAD_IMAGE :
                error.kind == EngineError::budget ? locrv1::PAGE_FAILURE_CODE_RESOURCE_LIMIT :
                locrv1::PAGE_FAILURE_CODE_INFERENCE);
            provenance(failure->mutable_provenance());
            failure->set_message(error.what()); ++failed;
          } catch (const std::exception&) {
            if (context->IsCancelled()) return {grpc::StatusCode::CANCELLED, "inference cancelled"};
            auto* failure = event.mutable_page_failure(); failure->set_page_no(page.page_no());
            failure->set_code(locrv1::PAGE_FAILURE_CODE_INFERENCE);
            provenance(failure->mutable_provenance());
            failure->set_finish_reason(locrv1::FINISH_REASON_ENGINE_ERROR);
            failure->set_message("native inference failed"); ++failed;
          }
        }
        if (!stream->Write(event)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
        // Keep page numbers for duplicate detection, never retain raster payloads.
        request.Clear();
        break;
      }
      case locrv1::ParseDocumentRequest::PAYLOAD_NOT_SET:
        return invalid("empty ParseDocumentRequest");
    }
  }
  if (context->IsCancelled()) return {grpc::StatusCode::CANCELLED, "request cancelled"};
  if (!saw_options) return invalid("missing ParseOptions");
  if (pages.size() != options.page_count()) return invalid("fewer pages than page_count");

  if (options.emit_document() && succeeded) {
    locrv1::ParseDocumentResponse event;
    event.mutable_document()->Swap(&folded);
    if (!stream->Write(event)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
  }
  locrv1::ParseDocumentResponse trailer;
  locrv1::ParseComplete* complete = trailer.mutable_complete();
  complete->set_pages(options.page_count());
  complete->set_pages_ok(succeeded);
  complete->set_pages_failed(failed);
  if (!stream->Write(trailer)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
  return grpc::Status::OK;
}

}  // namespace lighton
