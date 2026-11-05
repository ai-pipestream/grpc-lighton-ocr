#include "service/lighton_service.h"

#include <map>
#include <string>
#include <utility>

namespace lighton {
namespace {

grpc::Status invalid(const std::string& message) {
  return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
}

grpc::Status exhausted(const std::string& message) {
  return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, message);
}

}  // namespace

grpc::Status LightOnOcrServiceImpl::GetServiceInfo(grpc::ServerContext* /*context*/,
                                                   const locrv1::GetServiceInfoRequest* /*request*/,
                                                   locrv1::GetServiceInfoResponse* response) {
  response->set_listen_address(config_.listen_address);
  response->set_engine_loaded(false);
  response->set_model_repo(config_.model_billion == 4 ? "lightonai/LightOnOCR-3-4B"
                                                      : "lightonai/LightOnOCR-3-0.8B");
  return grpc::Status::OK;
}

grpc::Status LightOnOcrServiceImpl::ParseDocument(
    grpc::ServerContext* /*context*/,
    grpc::ServerReaderWriter<locrv1::ParseDocumentResponse, locrv1::ParseDocumentRequest>* stream) {
  locrv1::ParseDocumentRequest request;
  bool saw_options = false;
  locrv1::ParseOptions options;
  std::map<uint32_t, locrv1::PageImage> pages;
  while (stream->Read(&request)) {
    switch (request.payload_case()) {
      case locrv1::ParseDocumentRequest::kOptions:
        if (saw_options) return invalid("ParseOptions repeated");
        saw_options = true;
        options = request.options();
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
        pages.emplace(page.page_no(), page);
        break;
      }
      case locrv1::ParseDocumentRequest::PAYLOAD_NOT_SET:
        return invalid("empty ParseDocumentRequest");
    }
  }
  if (!saw_options) return invalid("missing ParseOptions");
  if (pages.size() != options.page_count()) return invalid("fewer pages than page_count");

  uint32_t failed = 0;
  for (const auto& [page_no, page] : pages) {
    locrv1::ParseDocumentResponse event;
    locrv1::PageFailure* failure = event.mutable_page_failure();
    failure->set_page_no(page_no);
    if (page.png().empty()) {
      failure->set_code(locrv1::PAGE_FAILURE_CODE_BAD_IMAGE);
      failure->set_message("page raster is empty");
    } else {
      failure->set_code(locrv1::PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE);
      failure->set_message("native LightOnOCR engine is not loaded");
    }
    if (!stream->Write(event)) return grpc::Status::OK;
    failed++;
  }
  locrv1::ParseDocumentResponse trailer;
  locrv1::ParseComplete* complete = trailer.mutable_complete();
  complete->set_pages(options.page_count());
  complete->set_pages_ok(0);
  complete->set_pages_failed(failed);
  stream->Write(trailer);
  return grpc::Status::OK;
}

}  // namespace lighton
