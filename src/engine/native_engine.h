#pragma once
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "engine/device.h"
namespace lighton {
struct EngineError : std::runtime_error {
  enum Kind { bad_image, budget, cancelled, busy, inference } kind;
  EngineError(Kind k, const char* message) : std::runtime_error(message), kind(k) {}
};
struct NativeStep { uint32_t token, top_token; std::string piece; float logprob, top_logprob; };
enum class NativeFinish { end_token, token_limit, output_limit, engine_error, deadline };
struct NativeResult { std::string text, raw_text; uint32_t width = 0, height = 0, tokens = 0; std::vector<NativeStep> steps; NativeFinish finish = NativeFinish::end_token; };
inline constexpr char kBackendRevision[] = "f3586e66bc916d672773d4a6505513ba91a3595c";
inline constexpr char kModelRevision[] = "36d8636925327c671bd8e479c517a004f7566bd9";
inline constexpr char kModelSha[] = "a73ab10d787c68cbfc518f51bfe044abcfb9dc27ce9c457f2068bf7d3ee0241d";
inline constexpr char kProjectorSha[] = "a0af0fe69443cfe35edd1294466d0f1d9780682177a101944d0fe8a9f5b72bd4";
// One pinned pair of converted artifacts, see models/SHA256SUMS.
struct ModelArtifacts { const char *model, *model_sha256, *projector, *projector_sha256; };
// BF16 serves the ggml CPU backend. The OpenVINO GPU plugin does not take
// BF16 weights, so OpenVINO devices load the F16 conversion of the same
// checkpoint revision.
inline constexpr ModelArtifacts kBf16Artifacts{"model-bf16.gguf", kModelSha, "vision-bf16.gguf", kProjectorSha};
inline constexpr ModelArtifacts kF16Artifacts{
    "model-f16.gguf", "57989c0efbebf4506470bd60eb53c6641e8a30ea1901a65c3db48181b6436176",
    "vision-f16.gguf", "984817f5c138c8400a171770175ae016e77a904252d3d456dcca65f217874f6b"};
inline constexpr const ModelArtifacts& artifacts_for(const DeviceSpec& device) {
  return device.accelerated() ? kF16Artifacts : kBf16Artifacts;
}
// The vision encoder an OpenVINO device runs: an OpenVINO IR exported from
// vision-f16.gguf by tools/export_vision_encoder, see models/SHA256SUMS.
struct EncoderArtifacts { const char *xml, *xml_sha256, *bin, *bin_sha256; };
inline constexpr EncoderArtifacts kF16Encoder{
    "vision-encoder-f16.xml", "069c37e4f2fc4ccbb611333f8cc089a148e5419bc169de913bae86313267a71d",
    "vision-encoder-f16.bin", "39fe3a5080be77c576263344ffd66bc9d7ead94ba31195cd6c4f0a4d816cd14c"};
struct EngineOptions {
  std::string directory;
  DeviceSpec device;
  int threads = 2;  // host threads for the ggml CPU backend and host-side work
};
// What a loaded engine runs on. The defaults describe the CPU engine.
struct EngineInfo {
  // "cpu", or "openvino:<id> - <device name from the runtime>" once every
  // graph node of a real inference was seen on that device.
  std::string device = "cpu";
  std::string model_sha256 = kModelSha;
  std::string projector_sha256 = kProjectorSha;
  // SHA256 of the OpenVINO vision encoder weights an accelerator runs. Empty
  // on cpu, where the projector file itself is executed.
  std::string vision_encoder_sha256;
  // Startup placement receipt for the log. Empty on cpu.
  std::string placement;
};
class NativeEngine {
 public:
  // Loads the artifacts. On an accelerator the vision encoder is the exported
  // OpenVINO model, compiled for that device alone, and the text model runs
  // on ggml's OpenVINO backend. The constructor then runs the compiled-in
  // reference page and throws unless both stages ran on the device only and
  // the page was read correctly.
  explicit NativeEngine(const EngineOptions& options);
  ~NativeEngine();
  const EngineInfo& info() const;
  // On an accelerator, a page whose graph left the device fails with
  // EngineError::inference naming the ops; it is never served from the CPU.
  NativeResult infer(std::string_view bytes, uint32_t max_tokens, float temperature,
                     uint32_t width, uint32_t height, std::function<bool()> cancelled);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
inline constexpr uint32_t kMaxTokens = 1024;
inline constexpr uint32_t kContext = 4096;
inline constexpr uint64_t kMaxPixels = 4000000;
inline constexpr size_t kMaxOutputBytes = 1024 * 1024;
}
