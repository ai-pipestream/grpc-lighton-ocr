#pragma once

#include <utility>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.h"
#include "config.h"

namespace lighton {

namespace locrv1 = ai::pipestream::lightonocr::v1;

// The native weight engine is not linked. GetServiceInfo reports that, and
// a page with bytes fails PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE as soon as
// that page has been checked, without retaining its raster. The markdown parser is covered by its own test
// and runs once a decode produces text.
class LightOnOcrServiceImpl final : public locrv1::LightOnOcrService::Service {
 public:
  explicit LightOnOcrServiceImpl(Config config) : config_(std::move(config)) {}

  grpc::Status ParseDocument(grpc::ServerContext* context,
                             grpc::ServerReaderWriter<locrv1::ParseDocumentResponse,
                                                      locrv1::ParseDocumentRequest>* stream) override;

  grpc::Status GetServiceInfo(grpc::ServerContext* context,
                              const locrv1::GetServiceInfoRequest* request,
                              locrv1::GetServiceInfoResponse* response) override;

 private:
  Config config_;
};

}  // namespace lighton
