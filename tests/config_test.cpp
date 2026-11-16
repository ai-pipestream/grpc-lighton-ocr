// Model-free checks of GRPC_LIGHTON_DEVICE and GRPC_LIGHTON_THREADS parsing.
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "config.h"
#include "engine/device.h"
#include "engine/native_engine.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

template <class Call> std::string rejection(Call call) {
  try { call(); } catch (const std::invalid_argument& error) { return error.what(); }
  return {};
}

void with_env(const char* name, const char* value, auto call) {
  if (value == nullptr) ::unsetenv(name); else ::setenv(name, value, 1);
  call();
  ::unsetenv(name);
}

}  // namespace

int main() {
  using lighton::DeviceSpec;
  using lighton::parse_device;

  expect(parse_device("cpu") == DeviceSpec{}, "cpu is the ggml CPU backend");
  expect(!parse_device("cpu").accelerated(), "cpu is not an accelerator");
  for (const char* id : {"GPU", "GPU.0", "GPU.1", "GPU.12", "CPU"}) {
    const DeviceSpec spec = parse_device(std::string("openvino:") + id);
    expect(spec.backend == DeviceSpec::Backend::openvino && spec.openvino_device == id,
           std::string("openvino:") + id + " keeps the device id exactly");
    expect(spec.accelerated() && spec.to_string() == std::string("openvino:") + id,
           std::string("openvino:") + id + " round-trips");
  }

  // One named device only. A list or a routing device may run on the CPU
  // without saying so, and a near miss must not be guessed at.
  for (const char* text : {
           "", "CPU", "gpu", "GPU", "openvino", "openvino:", "openvino:gpu", "openvino:cpu",
           "openvino:AUTO", "openvino:AUTO:GPU,CPU", "openvino:HETERO:GPU,CPU", "openvino:MULTI:GPU,CPU",
           "openvino:BATCH:GPU", "openvino:GPU,CPU", "openvino:GPU.0,GPU.1", "openvino:GPU.", "openvino:GPU.x",
           "openvino:GPU.01", "openvino:GPU.-1", "openvino:GPU.1234", "openvino:GPU.1 ", " openvino:GPU",
           "openvino:NPU", "openvino:GPU:0", "OPENVINO:GPU", "cuda", "sycl:0", "auto"}) {
    const std::string what = rejection([&] { parse_device(text); });
    expect(!what.empty(), std::string("device '") + text + "' is rejected");
    expect(what.find("openvino:GPU.<n>") != std::string::npos,
           std::string("rejection of '") + text + "' names the accepted forms");
  }
  expect(rejection([] { parse_device("openvino:HETERO:GPU,CPU"); }).find("routing device") != std::string::npos,
         "a routing device is rejected for what it is");

  for (const auto& [text, value] : {std::pair{"1", 1}, {"2", 2}, {"16", 16}, {"256", 256}})
    expect(lighton::parse_threads(text) == value, std::string("threads ") + text);
  for (const char* text : {"", "0", "-1", "257", "2x", " 2", "2 ", "0x10", "1e2", "two", "99999999999"})
    expect(!rejection([&] { lighton::parse_threads(text); }).empty(),
           std::string("threads '") + text + "' is rejected");

  with_env("GRPC_LIGHTON_DEVICE", nullptr, [] {
    ::unsetenv("GRPC_LIGHTON_THREADS");
    const lighton::Config config = lighton::load_config_from_env();
    expect(config.device == DeviceSpec{} && config.threads == 2, "defaults stay cpu with two threads");
  });
  with_env("GRPC_LIGHTON_DEVICE", "openvino:GPU.1", [] {
    ::setenv("GRPC_LIGHTON_THREADS", "8", 1);
    const lighton::Config config = lighton::load_config_from_env();
    expect(config.device.openvino_device == "GPU.1" && config.threads == 8, "device and threads come from the environment");
    ::unsetenv("GRPC_LIGHTON_THREADS");
  });
  with_env("GRPC_LIGHTON_DEVICE", "openvino:AUTO", [] {
    expect(!rejection([] { lighton::load_config_from_env(); }).empty(), "a routing device fails configuration");
  });
  with_env("GRPC_LIGHTON_THREADS", "0", [] {
    expect(!rejection([] { lighton::load_config_from_env(); }).empty(), "zero threads fail configuration");
  });

  // The CPU backend needs no device handle and reports itself as cpu.
  expect(lighton::select_device(DeviceSpec{}) == nullptr, "cpu selects no accelerator device");
  expect(lighton::describe_device(DeviceSpec{}, nullptr) == "cpu", "cpu is reported as cpu");
  expect(std::string(lighton::artifacts_for(DeviceSpec{}).model) == "model-bf16.gguf",
         "cpu loads the BF16 artifacts");
  expect(std::string(lighton::artifacts_for(parse_device("openvino:GPU")).model) == "model-f16.gguf" &&
             std::string(lighton::artifacts_for(parse_device("openvino:GPU")).projector) == "vision-f16.gguf",
         "OpenVINO devices load the F16 artifacts");

  if (!lighton::openvino_built()) {
    // A CPU-only binary must refuse an accelerator, never serve it from CPU.
    bool refused = false;
    try { lighton::select_device(parse_device("openvino:GPU")); }
    catch (const std::runtime_error& error) {
      refused = std::string(error.what()).find("OpenVINO build") != std::string::npos;
    }
    expect(refused, "a build without OpenVINO refuses an OpenVINO device");
  }

  if (failures != 0) return 1;
  std::cout << "config_test: OK\n";
  return 0;
}
