#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.h"
#include "config.h"
#include "engine/inference_control.h"
#include "service/lighton_service.h"

namespace locrv1 = ai::pipestream::lightonocr::v1;

namespace {

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

struct Server {
  std::unique_ptr<lighton::LightOnOcrServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string address;
};

Server start(lighton::Config config = {}) {
  Server started;
  config.diagnostic_no_engine = true;
  config.listen_address = "127.0.0.1:0";
  started.service = std::make_unique<lighton::LightOnOcrServiceImpl>(config);
  grpc::ServerBuilder builder;
  int selected = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected);
  builder.RegisterService(started.service.get());
  started.server = builder.BuildAndStart();
  started.address = "127.0.0.1:" + std::to_string(selected);
  return started;
}

}  // namespace

int main() {
  {
    lighton::detail::Control expired{lighton::detail::Clock::time_point::min(), [] { return false; }};
    locrv1::PageFailure failure;
    failure.mutable_provenance()->set_device("stale-device");
    failure.mutable_provenance()->add_stages()->set_runtime_verified(true);
    failure.mutable_generation_trace()->set_text("stale prior page");
    bool threw = false;
    try { expired.check(); }
    catch (const lighton::EngineError& error) {
      threw = true;
      expect(error.kind == lighton::EngineError::deadline, "prefill deadline is not client cancellation");
      lighton::project_engine_error(error, 2, &failure);
    }
    expect(threw, "expired control throws before native work");
    expect(failure.page_no() == 2 && failure.code() == locrv1::PAGE_FAILURE_CODE_RESOURCE_LIMIT,
           "thrown internal deadline is a typed resource-limit page failure");
    expect(failure.finish_reason() == locrv1::FINISH_REASON_DEADLINE, "thrown deadline finish reason");
    expect(failure.provenance().stages_size() == 0 && failure.provenance().device() == "unverified",
           "thrown deadline cannot reuse prior placement");
    expect(!failure.has_generation_trace(), "thrown deadline cannot reuse prior generation");
  }
  for (const auto& backend : {"cpu", "cuda", "sycl", "metal"}) {
    bool rejected = false;
    try { lighton::NativeEngine unavailable("/nonexistent-models", backend, "missing-native-device"); }
    catch (const std::exception& error) {
      rejected = std::string(error.what()).find("requested native backend/device unavailable") != std::string::npos;
    }
    expect(rejected, std::string(backend) + " unavailable device fails before reading model files");
  }

  for (bool four_b : {false, true}) {
    lighton::Config unavailable;
    unavailable.models_dir = "/nonexistent-lighton-models";
    if (four_b) unavailable.model = locrv1::LIGHTON_MODEL_4B;
    bool rejected = false;
    try { lighton::LightOnOcrServiceImpl invalid(unavailable); }
    catch (const std::exception&) { rejected = true; }
    expect(rejected, "production missing weights and unqualified 4B fail startup");
  }
  {
    lighton::NativeResult malformed;
    malformed.width = 1000; malformed.height = 300;
    malformed.text = "![text](99999999999999999,0,1000,1000) café";
    malformed.raw_text = malformed.text + "<|im_end|>";
    malformed.steps.push_back({7, 7, "café", -0.5f, -0.5f});
    locrv1::ParseDocumentResponse response;
    lighton::project_native_result(malformed, 1, locrv1::LIGHTON_MODEL_0_8B, 0, &response);
    expect(response.has_page_failure(), "malformed grounding is typed failure");
    expect(response.page_failure().generation_trace().text() == malformed.raw_text,
           "malformed grounding retains all raw output");
    expect(response.page_failure().generation_trace().steps_size() == 1,
           "malformed grounding retains tokens");
    malformed.text = "![text](0,0,1000,1000) café";
    malformed.raw_text = malformed.text;
    malformed.finish = lighton::NativeFinish::token_limit;
    response.Clear();
    lighton::project_native_result(malformed, 1, locrv1::LIGHTON_MODEL_0_8B, 0, &response);
    expect(response.page_failure().finish_reason() == locrv1::FINISH_REASON_TOKEN_LIMIT,
           "token limit has typed finish reason");
    expect(response.page_failure().partial_document().texts_size() == 1,
           "token limit retains recoverable document");
    expect(response.page_failure().generation_trace().text() == malformed.raw_text,
           "token limit retains raw output");
    malformed.finish = lighton::NativeFinish::deadline;
    response.Clear();
    lighton::project_native_result(malformed, 1, locrv1::LIGHTON_MODEL_0_8B, 0, &response);
    expect(response.page_failure().code() == locrv1::PAGE_FAILURE_CODE_RESOURCE_LIMIT &&
           response.page_failure().finish_reason() == locrv1::FINISH_REASON_DEADLINE,
           "returned deadline has the same typed resource-limit mapping");
    expect(response.page_failure().generation_trace().text() == malformed.raw_text,
           "returned deadline retains current request output");
  }
  Server server = start();
  expect(server.server != nullptr, "server started");
  auto channel = grpc::CreateChannel(server.address, grpc::InsecureChannelCredentials());
  auto stub = locrv1::LightOnOcrService::NewStub(channel);

  {
    grpc::ClientContext context;
    locrv1::GetServiceInfoRequest request;
    locrv1::GetServiceInfoResponse response;
    const grpc::Status status = stub->GetServiceInfo(&context, request, &response);
    expect(status.ok(), status.error_message());
    expect(!response.engine_loaded(), "engine is not loaded");
    expect(response.model_repo() == "lightonai/LightOnOCR-3-0.8B", response.model_repo());
    expect(response.model() == locrv1::LIGHTON_MODEL_0_8B, "model enum");
  }
  {
    grpc::ClientContext context;
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest page;
    page.mutable_page()->set_page_no(1);
    page.mutable_page()->set_png("png");
    expect(stream->Write(page), "write page");
    stream->WritesDone();
    locrv1::ParseDocumentResponse event;
    expect(!stream->Read(&event), "no event before the status");
    const grpc::Status status = stream->Finish();
    expect(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT, status.error_message());
  }
  {
    grpc::ClientContext context;
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest options;
    options.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
    options.mutable_options()->set_page_count(1);
    locrv1::ParseDocumentRequest page;
    page.mutable_page()->set_page_no(1);
    page.mutable_page()->set_png("not-a-real-image");
    expect(stream->Write(options) && stream->Write(page), "write options and page");
    stream->WritesDone();
    locrv1::ParseDocumentResponse failure;
    locrv1::ParseDocumentResponse complete;
    expect(stream->Read(&failure) && stream->Read(&complete), "failure then trailer");
    expect(failure.page_failure().code() == locrv1::PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE,
           "engine unavailable");
    expect(complete.has_complete() && complete.complete().pages_failed() == 1, "one failure");
    expect(stream->Finish().ok(), "stream completes");
  }
  {
    grpc::ClientContext context;
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest options;
    options.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
    options.mutable_options()->set_page_count(1);
    locrv1::ParseDocumentRequest page;
    page.mutable_page()->set_page_no(1);
    expect(stream->Write(options) && stream->Write(page), "write empty raster");
    stream->WritesDone();
    locrv1::ParseDocumentResponse failure;
    expect(stream->Read(&failure), "bad image event");
    expect(failure.page_failure().code() == locrv1::PAGE_FAILURE_CODE_BAD_IMAGE, "empty raster");
    locrv1::ParseDocumentResponse complete;
    expect(stream->Read(&complete) && complete.has_complete(), "trailer");
    expect(stream->Finish().ok(), "empty raster still completes");
  }
  for (const auto& [model, code] :
       {std::pair{locrv1::LIGHTON_MODEL_UNSPECIFIED, grpc::StatusCode::INVALID_ARGUMENT},
        std::pair{locrv1::LIGHTON_MODEL_4B, grpc::StatusCode::FAILED_PRECONDITION}}) {
    grpc::ClientContext context;
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest options;
    options.mutable_options()->set_model(model);
    options.mutable_options()->set_page_count(1);
    stream->Write(options);
    stream->WritesDone();
    locrv1::ParseDocumentResponse event;
    expect(!stream->Read(&event), "no event for a rejected model");
    const grpc::Status status = stream->Finish();
    expect(status.error_code() == code, "model check: " + status.error_message());
  }
  expect(lighton::model_repo(locrv1::LIGHTON_MODEL_4B) == "lightonai/LightOnOCR-3-4B", "4b repo");
  // A page failure must arrive before the client half-closes. Holding all
  // rasters until EOF previously allowed 256 * 32 MiB retained per stream.
  {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(2));
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest options;
    options.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
    options.mutable_options()->set_page_count(2);
    locrv1::ParseDocumentRequest page;
    page.mutable_page()->set_page_no(1);
    page.mutable_page()->set_png("page bytes");
    stream->Write(options);
    stream->Write(page);
    locrv1::ParseDocumentResponse event;
    expect(stream->Read(&event) && event.has_page_failure(), "page event before EOF");
    context.TryCancel();
    while (stream->Read(&event)) {}
    expect(!stream->Finish().ok(), "cancelled stream is not successful");
  }
  for (int invalid_case = 0; invalid_case < 4; ++invalid_case) {
    grpc::ClientContext context;
    auto stream = stub->ParseDocument(&context);
    locrv1::ParseDocumentRequest options;
    options.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
    options.mutable_options()->set_page_count(1);
    if (invalid_case == 0) options.mutable_options()->set_model(static_cast<locrv1::LightonModel>(99));
    if (invalid_case == 1) options.mutable_options()->set_temperature(-1);
    if (invalid_case == 2) options.mutable_options()->set_temperature(std::numeric_limits<float>::quiet_NaN());
    if (invalid_case == 3) options.mutable_options()->set_temperature(1e-38f);
    stream->Write(options);
    locrv1::ParseDocumentRequest page;
    page.mutable_page()->set_page_no(1);
    page.mutable_page()->set_png("page");
    stream->Write(page);
    stream->WritesDone();
    locrv1::ParseDocumentResponse event;
    while (stream->Read(&event)) {}
    expect(stream->Finish().error_code() == grpc::StatusCode::INVALID_ARGUMENT,
           "invalid model or temperature is INVALID_ARGUMENT");
  }
  {
    lighton::Config limits;
    limits.max_pages = 2;
    limits.max_page_bytes = 4;
    Server limited = start(limits);
    auto small = locrv1::LightOnOcrService::NewStub(
        grpc::CreateChannel(limited.address, grpc::InsecureChannelCredentials()));
    for (int scenario = 0; scenario < 3; ++scenario) {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
      auto stream = small->ParseDocument(&context);
      locrv1::ParseDocumentRequest options;
      options.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
      options.mutable_options()->set_page_count(scenario == 0 ? 3 : 1);
      stream->Write(options);
      if (scenario != 0) {
        locrv1::ParseDocumentRequest page;
        page.mutable_page()->set_page_no(1);
        page.mutable_page()->set_png(scenario == 1 ? "12345" : "1234");
        stream->Write(page);
        if (scenario == 2) stream->Write(page);
      }
      stream->WritesDone();
      locrv1::ParseDocumentResponse event;
      while (stream->Read(&event)) expect(!event.has_complete(), "no trailer after invalid stream");
      expect(stream->Finish().error_code() == (scenario == 2 ? grpc::StatusCode::INVALID_ARGUMENT
                                                            : grpc::StatusCode::RESOURCE_EXHAUSTED),
             "page count, byte cap, and duplicate checks");
    }
    limited.server->Shutdown();
  }
  {
    using Stream = grpc::ClientReaderWriter<locrv1::ParseDocumentRequest, locrv1::ParseDocumentResponse>;
    std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < 4; ++i) {
      auto context = std::make_unique<grpc::ClientContext>();
      context->set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
      auto stream = stub->ParseDocument(context.get());
      locrv1::ParseDocumentRequest request;
      request.mutable_options()->set_model(locrv1::LIGHTON_MODEL_0_8B);
      request.mutable_options()->set_page_count(1); stream->Write(request);
      request.Clear(); request.mutable_page()->set_page_no(1); stream->Write(request);
      locrv1::ParseDocumentResponse response;
      expect(stream->Read(&response) && response.has_page_failure(), "active stream admitted");
      contexts.push_back(std::move(context)); streams.push_back(std::move(stream));
    }
    grpc::ClientContext extra;
    extra.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(2));
    auto rejected = stub->ParseDocument(&extra); rejected->WritesDone();
    locrv1::ParseDocumentResponse response;
    while (rejected->Read(&response)) {}
    expect(rejected->Finish().error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
           "fifth active stream rejected");
    for (auto& stream : streams) {
      stream->WritesDone(); while (stream->Read(&response)) {}
      expect(stream->Finish().ok(), "admitted stream completes");
    }
  }
  {
    lighton::Config limits;
    limits.transport_timeout = std::chrono::milliseconds(150);
    Server idle_server = start(limits);
    auto idle_stub = locrv1::LightOnOcrService::NewStub(
        grpc::CreateChannel(idle_server.address, grpc::InsecureChannelCredentials()));
    using Stream = grpc::ClientReaderWriter<locrv1::ParseDocumentRequest, locrv1::ParseDocumentResponse>;
    std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
    std::vector<std::unique_ptr<Stream>> streams;
    const auto began = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; ++i) {
      auto context = std::make_unique<grpc::ClientContext>(); // No client deadline.
      auto stream = idle_stub->ParseDocument(context.get());
      contexts.push_back(std::move(context)); streams.push_back(std::move(stream));
    }
    locrv1::ParseDocumentResponse response;
    for (auto& stream : streams) {
      expect(!stream->Read(&response), "idle input terminates without an event");
      expect(stream->Finish().error_code() == grpc::StatusCode::CANCELLED,
             "server transport timeout is CANCELLED on wire");
    }
    expect(std::chrono::steady_clock::now() - began < std::chrono::seconds(3),
           "idle clients terminate within bound");
    grpc::ClientContext recovered;
    auto stream = idle_stub->ParseDocument(&recovered);
    stream->WritesDone();
    expect(!stream->Read(&response), "recovered slot has no event for empty input");
    expect(stream->Finish().error_code() == grpc::StatusCode::INVALID_ARGUMENT,
           "idle leases released for next client");
    idle_server.server->Shutdown();
  }
  server.server->Shutdown();
  if (failures != 0) return 1;
  std::cout << "service_test ok\n";
  return 0;
}
