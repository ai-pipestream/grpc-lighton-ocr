#pragma once

#include <cstdint>
#include <chrono>
#include <string>
#include <string_view>

#include "ai/pipestream/lightonocr/v1/lighton_ocr.pb.h"
#include "engine/device.h"

namespace lighton {

struct Config {
  std::string listen_address = "0.0.0.0:50074";
  std::string models_dir = "/models";
  // GRPC_LIGHTON_MODEL: "0.8b" or "4b". Unset selects 0.8b; any other value
  // fails startup.
  ai::pipestream::lightonocr::v1::LightonModel model =
      ai::pipestream::lightonocr::v1::LIGHTON_MODEL_0_8B;
  // GRPC_LIGHTON_DEVICE: cpu (default), openvino:GPU, openvino:GPU.<n> or
  // openvino:CPU. One explicit device; anything else fails startup.
  DeviceSpec device;
  // GRPC_LIGHTON_THREADS: host threads for the ggml CPU backend, 1..256.
  int threads = 2;
  bool diagnostic_no_engine = false;
  std::chrono::milliseconds transport_timeout{30000};
  uint32_t max_pages = 256;
  uint64_t max_page_bytes = 32ull * 1024ull * 1024ull;
};

// Throws std::invalid_argument for an unknown GRPC_LIGHTON_MODEL, a device
// that parse_device rejects, or a thread count that parse_threads rejects.
Config load_config_from_env();

// A plain decimal integer in 1..256. Throws std::invalid_argument otherwise.
int parse_threads(std::string_view text);

// Hugging Face repo id for a checkpoint. Throws std::invalid_argument for
// LIGHTON_MODEL_UNSPECIFIED.
std::string model_repo(ai::pipestream::lightonocr::v1::LightonModel model);

}  // namespace lighton
