#pragma once
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace lighton {
struct EngineError : std::runtime_error {
  enum Kind { bad_image, budget, cancelled, busy, inference } kind;
  EngineError(Kind k, const char* message) : std::runtime_error(message), kind(k) {}
};
struct NativeStep { uint32_t token, top_token; std::string piece; float logprob, top_logprob; };
enum class NativeFinish { end_token, token_limit, output_limit, engine_error, deadline };
struct NativePlacement {
  std::string backend, device_name, device_id;
  bool runtime_verified = false;
  bool cpu_compute = false;
};
struct NativePlacementPair { NativePlacement vision, decoder; };
struct NativeResult { std::string text, raw_text; uint32_t width = 0, height = 0, tokens = 0; std::vector<NativeStep> steps; NativeFinish finish = NativeFinish::end_token; NativePlacementPair placement{}; };

class NativeEngine {
 public:
  explicit NativeEngine(const std::string& directory, const std::string& backend = "cpu",
                        const std::string& device = "");
  NativePlacementPair placement() const;
  NativePlacement vision_placement() const;
  NativePlacement decoder_placement() const;
  ~NativeEngine();
  NativeResult infer(std::string_view bytes, uint32_t max_tokens, float temperature,
                     uint32_t width, uint32_t height, std::function<bool()> cancelled);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
inline constexpr char kBackendRevision[] = "f3586e66bc916d672773d4a6505513ba91a3595c";
inline constexpr char kModelRevision[] = "36d8636925327c671bd8e479c517a004f7566bd9";
inline constexpr char kModelSha[] = "a73ab10d787c68cbfc518f51bfe044abcfb9dc27ce9c457f2068bf7d3ee0241d";
inline constexpr char kProjectorSha[] = "a0af0fe69443cfe35edd1294466d0f1d9780682177a101944d0fe8a9f5b72bd4";
inline constexpr uint32_t kMaxTokens = 1024;
inline constexpr uint32_t kContext = 4096;
inline constexpr uint64_t kMaxPixels = 4000000;
inline constexpr size_t kMaxOutputBytes = 1024 * 1024;
}
