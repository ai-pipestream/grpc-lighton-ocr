#pragma once

#include <string>
#include <string_view>
#include <vector>

struct ggml_backend_device;

namespace lighton {

// Typed form of GRPC_LIGHTON_DEVICE. One device, named exactly; never a list.
struct DeviceSpec {
  enum class Backend { cpu, openvino };
  Backend backend = Backend::cpu;
  // OpenVINO device id as its runtime spells it: GPU, GPU.1 or CPU. Empty for
  // the ggml CPU backend.
  std::string openvino_device;

  bool accelerated() const { return backend == Backend::openvino; }
  // The configuration spelling: "cpu" or "openvino:<id>".
  std::string to_string() const;
  bool operator==(const DeviceSpec&) const = default;
};

// Accepts cpu, openvino:GPU, openvino:GPU.<n> and openvino:CPU. Everything
// else throws std::invalid_argument, including the routing devices AUTO,
// HETERO, MULTI and BATCH and any comma list: those let the runtime move work
// to another device without telling the caller.
DeviceSpec parse_device(std::string_view text);

// True when this binary carries the ggml OpenVINO backend.
bool openvino_built();

// OpenVINO device ids the runtime offers in this process, sorted. Throws
// std::runtime_error in a build without the OpenVINO backend.
std::vector<std::string> openvino_available_devices();

// Selects the device in ggml and returns its handle; nullptr for cpu. For an
// OpenVINO device this must run before anything else touches the ggml backend
// registry, because the backend reads its device once, at registration.
// Throws std::runtime_error when the build has no OpenVINO backend, when the
// device is not present, or when ggml selected a different one.
ggml_backend_device* select_device(const DeviceSpec& spec);

// The reported device: "cpu", or "openvino:<id> - <name>" where the name is
// the one ggml reports for the handle select_device returned.
std::string describe_device(const DeviceSpec& spec, ggml_backend_device* device);

}  // namespace lighton
