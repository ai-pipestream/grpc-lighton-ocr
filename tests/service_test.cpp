#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.h"
#include "config.h"
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

Server start() {
  Server started;
  lighton::Config config;
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
  server.server->Shutdown();
  if (failures != 0) return 1;
  std::cout << "service_test ok\n";
  return 0;
}
