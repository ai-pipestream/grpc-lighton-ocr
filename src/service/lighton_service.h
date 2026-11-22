#pragma once

#include <atomic>
#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.h"
#include "config.h"
#include "engine/native_engine.h"

namespace lighton {

namespace locrv1 = ai::pipestream::lightonocr::v1;

// Project native output without dropping raw tokens on parse/length failures.
// engine names the device and artifacts for the provenance; the default is
// the CPU engine.
void project_native_result(const NativeResult& result, uint32_t page_no,
                           locrv1::LightonModel model, uint64_t first_step,
                           locrv1::ParseDocumentResponse* event, const EngineInfo& engine = {});

class LightOnOcrServiceImpl final : public locrv1::LightOnOcrService::Service {
 public:
  explicit LightOnOcrServiceImpl(Config config);

  // One readiness source for GetServiceInfo and named gRPC health.
  bool engine_loaded() const noexcept { return engine_ != nullptr; }

  // One startup log line: the device inference runs on and, for an
  // accelerator, the placement receipt of the startup probe.
  std::string device_report() const;

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
