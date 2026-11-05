#include "config.h"

#include <cstdlib>
#include <string>

namespace lighton {
namespace {

std::string env_or(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return value;
}

}  // namespace

Config load_config_from_env() {
  Config config;
  config.listen_address = env_or("GRPC_LIGHTON_LISTEN_ADDRESS", config.listen_address);
  config.models_dir = env_or("GRPC_LIGHTON_MODELS_DIR", config.models_dir);
  const std::string model = env_or("GRPC_LIGHTON_MODEL", "0.8b");
  if (model == "4b" || model == "4B") config.model_billion = 4;
  return config;
}

}  // namespace lighton
