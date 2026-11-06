#include "service/lighton_service.h"

#include <cstdint>

#include <cmath>
#include <set>
#include <string>
#include <utility>

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

}  // namespace

grpc::Status LightOnOcrServiceImpl::GetServiceInfo(grpc::ServerContext* /*context*/,
                                                   const locrv1::GetServiceInfoRequest* /*request*/,
                                                   locrv1::GetServiceInfoResponse* response) {
  response->set_listen_address(config_.listen_address);
  response->set_engine_loaded(engine_loaded());
  response->set_model(config_.model);
  response->set_model_repo(model_repo(config_.model));
  return grpc::Status::OK;
}

grpc::Status LightOnOcrServiceImpl::ParseDocument(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<locrv1::ParseDocumentResponse, locrv1::ParseDocumentRequest>* stream) {
  locrv1::ParseDocumentRequest request;
  bool saw_options = false;
  locrv1::ParseOptions options;
  std::set<uint32_t> pages;
  uint32_t failed = 0;
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
        auto* failure = event.mutable_page_failure();
        failure->set_page_no(page.page_no());
        failure->set_code(page.png().empty() ? locrv1::PAGE_FAILURE_CODE_BAD_IMAGE
                                            : locrv1::PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE);
        failure->set_message(page.png().empty() ? "page raster is empty"
                                               : "native LightOnOCR engine is not loaded");
        if (!stream->Write(event)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
        ++failed;
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

  locrv1::ParseDocumentResponse trailer;
  locrv1::ParseComplete* complete = trailer.mutable_complete();
  complete->set_pages(options.page_count());
  complete->set_pages_ok(0);
  complete->set_pages_failed(failed);
  if (!stream->Write(trailer)) return {grpc::StatusCode::CANCELLED, "response stream closed"};
  return grpc::Status::OK;
}

}  // namespace lighton
