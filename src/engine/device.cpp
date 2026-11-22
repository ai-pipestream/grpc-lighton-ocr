#include "engine/device.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "ggml-backend.h"

#ifdef LIGHTON_OPENVINO
#include <openvino/runtime/core.hpp>
#endif

namespace lighton {
namespace {

constexpr std::string_view kAccepted =
    "cpu, openvino:GPU, openvino:GPU.<n> or openvino:CPU";
constexpr char kOpenvinoEnv[] = "GGML_OPENVINO_DEVICE";
constexpr std::string_view kOpenvinoRegistry = "OPENVINO";

std::invalid_argument rejected(std::string_view text, std::string_view reason) {
  return std::invalid_argument("GRPC_LIGHTON_DEVICE='" + std::string(text) + "' " +
                               std::string(reason) + "; use " + std::string(kAccepted));
}

bool routing_device(std::string_view id) {
  for (std::string_view name : {"AUTO", "HETERO", "MULTI", "BATCH"})
    if (id.starts_with(name)) return true;
  return false;
}

// GPU, or GPU.<n> with a plain decimal index and no leading zero.
bool gpu_id(std::string_view id) {
  if (id == "GPU") return true;
  if (!id.starts_with("GPU.")) return false;
  const std::string_view index = id.substr(4);
  if (index.empty() || index.size() > 3) return false;
  if (index.size() > 1 && index.front() == '0') return false;
  return std::ranges::all_of(index, [](char c) { return c >= '0' && c <= '9'; });
}

std::string selected_tag(const DeviceSpec& spec) {
  return std::string(kOpenvinoEnv) + "=" + spec.openvino_device + " (selected)";
}

std::string join(const std::vector<std::string>& values) {
  std::string out;
  for (const auto& value : values) out += (out.empty() ? "" : ", ") + value;
  return out.empty() ? "none" : out;
}

}  // namespace

std::string DeviceSpec::to_string() const {
  return backend == Backend::cpu ? "cpu" : "openvino:" + openvino_device;
}

DeviceSpec parse_device(std::string_view text) {
  if (text == "cpu") return {};
  constexpr std::string_view prefix = "openvino:";
  if (!text.starts_with(prefix)) throw rejected(text, "is not a known device");
  const std::string_view id = text.substr(prefix.size());
  if (id.find(',') != std::string_view::npos || routing_device(id))
    throw rejected(text, "names a device list or a routing device, which may run on the CPU");
  if (id != "CPU" && !gpu_id(id)) throw rejected(text, "is not a known OpenVINO device");
  return {DeviceSpec::Backend::openvino, std::string(id)};
}

bool openvino_built() {
#ifdef LIGHTON_OPENVINO
  return true;
#else
  return false;
#endif
}

std::vector<std::string> openvino_available_devices() {
#ifdef LIGHTON_OPENVINO
  std::vector<std::string> devices = ov::Core().get_available_devices();
  std::ranges::sort(devices);
  return devices;
#else
  throw std::runtime_error("this build has no OpenVINO backend (LIGHTON_OPENVINO is OFF)");
#endif
}

ggml_backend_device* select_device(const DeviceSpec& spec) {
  if (!spec.accelerated()) return nullptr;
  const std::string label = "GRPC_LIGHTON_DEVICE=" + spec.to_string();
  if (!openvino_built())
    throw std::runtime_error(label + " needs the OpenVINO build of this server (Dockerfile.openvino)");

  // ggml takes the device from the environment once, when its registry is
  // built. A different value already there would be the one in force.
  if (const char* preset = std::getenv(kOpenvinoEnv);
      preset != nullptr && preset != spec.openvino_device)
    throw std::runtime_error(label + " conflicts with " + kOpenvinoEnv + "=" + preset);

  // Ask the runtime before ggml does: ggml ends the process with abort() on
  // a missing device, which gives the operator no usable status.
  const auto available = openvino_available_devices();
  if (std::ranges::find(available, spec.openvino_device) == available.end())
    throw std::runtime_error(label + ": OpenVINO device " + spec.openvino_device +
                             " is not available; available: " + join(available));
  if (::setenv(kOpenvinoEnv, spec.openvino_device.c_str(), 1) != 0)
    throw std::runtime_error("cannot export the OpenVINO device selection");

  // The backend registers every OpenVINO device and marks the selected one
  // as the only GPU-type device. Exactly one must match, by type and by id.
  ggml_backend_device* found = nullptr;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
    if (kOpenvinoRegistry != ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate))) continue;
    if (ggml_backend_dev_type(candidate) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
    if (found != nullptr) throw std::runtime_error(label + ": ggml offers more than one selected OpenVINO device");
    found = candidate;
  }
  if (found == nullptr) throw std::runtime_error(label + ": ggml registered no selected OpenVINO device");
  if (!std::string_view(ggml_backend_dev_description(found)).starts_with(selected_tag(spec)))
    throw std::runtime_error(label + ": ggml selected a different device: " +
                             ggml_backend_dev_description(found));
  return found;
}

std::string describe_device(const DeviceSpec& spec, ggml_backend_device* device) {
  if (!spec.accelerated() || device == nullptr) return spec.to_string();
  // ggml describes the device as "<selected tag> - <runtime device name>".
  std::string_view name = ggml_backend_dev_description(device);
  const std::string tag = selected_tag(spec) + " - ";
  if (name.starts_with(tag)) name.remove_prefix(tag.size());
  while (!name.empty() && name.back() == ' ') name.remove_suffix(1);  // CPU brand strings are padded
  return spec.to_string() + " - " + std::string(name);
}

}  // namespace lighton
