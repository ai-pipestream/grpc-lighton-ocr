#pragma once

#include <cstdint>
#include <chrono>
#include <string>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.pb.h"

namespace lighton {

struct Config {
  std::string listen_address = "0.0.0.0:50074";
  std::string backend = "cpu";
  std::string device; // Exact native device name, empty selects first matching backend.
  std::string models_dir = "/models";
  // GRPC_LIGHTON_MODEL: "0.8b" or "4b". Unset selects 0.8b; any other value
  // fails startup.
  ai::pipestream::lightonocr::v1::LightonModel model =
      ai::pipestream::lightonocr::v1::LIGHTON_MODEL_0_8B;
  bool diagnostic_no_engine = false;
  std::chrono::milliseconds transport_timeout{30000};
  uint32_t max_pages = 256;
  uint64_t max_page_bytes = 32ull * 1024ull * 1024ull;
};

// Throws std::invalid_argument for an unknown GRPC_LIGHTON_MODEL.
Config load_config_from_env();

// Hugging Face repo id for a checkpoint. Throws std::invalid_argument for
// LIGHTON_MODEL_UNSPECIFIED.
std::string model_repo(ai::pipestream::lightonocr::v1::LightonModel model);

}  // namespace lighton
