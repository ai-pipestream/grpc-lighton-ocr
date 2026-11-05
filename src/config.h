#pragma once

#include <cstdint>
#include <string>

namespace lighton {

struct Config {
  std::string listen_address = "0.0.0.0:50074";
  std::string models_dir = "/models";
  // 0 selects the 0.8B checkpoint. 4 selects the 4B checkpoint.
  int model_billion = 0;
  uint32_t max_pages = 256;
  uint64_t max_page_bytes = 32ull * 1024ull * 1024ull;
};

Config load_config_from_env();

}  // namespace lighton
