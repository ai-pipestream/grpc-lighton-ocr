#pragma once

#include <utility>
#include <atomic>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.h"
#include "config.h"
#include "engine/native_engine.h"

namespace lighton {

namespace locrv1 = ai::pipestream::lightonocr::v1;

// Project native output without dropping raw tokens on parse/length failures.
void project_native_result(const NativeResult& result, uint32_t page_no,
                           locrv1::LightonModel model, uint64_t first_step,
                           locrv1::ParseDocumentResponse* event);

class LightOnOcrServiceImpl final : public locrv1::LightOnOcrService::Service {
 public:
  explicit LightOnOcrServiceImpl(Config config);

  // One readiness source for GetServiceInfo and named gRPC health.
  bool engine_loaded() const noexcept { return engine_ != nullptr; }

  grpc::Status ParseDocument(grpc::ServerContext* context,
                             grpc::ServerReaderWriter<locrv1::ParseDocumentResponse,
                                                      locrv1::ParseDocumentRequest>* stream) override;

  grpc::Status GetServiceInfo(grpc::ServerContext* context,
                              const locrv1::GetServiceInfoRequest* request,
                              locrv1::GetServiceInfoResponse* response) override;

 private:
  std::atomic<unsigned> streams_{0};
  Config config_;
  std::unique_ptr<NativeEngine> engine_;
};

}  // namespace lighton
