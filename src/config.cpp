#include "config.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace lighton {
namespace {

namespace locrv1 = ai::pipestream::lightonocr::v1;

std::string env_or(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  return value;
}

locrv1::LightonModel parse_model(const std::string& value) {
  if (value == "0.8b") return locrv1::LIGHTON_MODEL_0_8B;
  if (value == "4b") return locrv1::LIGHTON_MODEL_4B;
  throw std::invalid_argument("GRPC_LIGHTON_MODEL must be 0.8b or 4b, got '" + value + "'");
}

}  // namespace

Config load_config_from_env() {
  Config config;
  config.listen_address = env_or("GRPC_LIGHTON_LISTEN_ADDRESS", config.listen_address);
  config.models_dir = env_or("GRPC_LIGHTON_MODELS_DIR", config.models_dir);
  config.model = parse_model(env_or("GRPC_LIGHTON_MODEL", "0.8b"));
  return config;
}

std::string model_repo(locrv1::LightonModel model) {
  switch (model) {
    case locrv1::LIGHTON_MODEL_0_8B:
      return "lightonai/LightOnOCR-3-0.8B";
    case locrv1::LIGHTON_MODEL_4B:
      return "lightonai/LightOnOCR-3-4B";
    case locrv1::LIGHTON_MODEL_UNSPECIFIED:
    default:
      break;
  }
  throw std::invalid_argument("no checkpoint for LightonModel " + std::to_string(model));
}

}  // namespace lighton
