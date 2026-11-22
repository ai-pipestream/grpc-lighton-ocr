#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lighton::vision {

// Arithmetic the device computes in. The weights on disk are the same
// either way.
enum class EncoderPrecision { f32, f16 };

struct EncoderOptions {
  // The exported encoder: an OpenVINO IR pair (tools/export_vision_encoder).
  std::string xml_path, bin_path;
  // One OpenVINO device, spelled as its runtime does: GPU, GPU.<n> or CPU.
  std::string device;
  // Compiled-kernel cache directory, or empty for none.
  std::string cache_dir;
  // Page sizes kept compiled at once; the least recently used one goes first.
  size_t max_compiled_sizes = 6;
  // Always stated to the runtime: left alone, a plugin picks its own (the GPU
  // plugin half precision, the CPU plugin bfloat16 on processors that have
  // it). f32 reproduces the CPU engine's encoder most closely; f16 is about
  // ten times faster on an Arc GPU and moves a few tokens per large page
  // noticeably (docs/vision-encoder.md).
  EncoderPrecision precision = EncoderPrecision::f32;
};

// What one compiled page size runs on, read back from the runtime.
struct EncoderPlacement {
  uint32_t width = 0, height = 0;
  // Devices the runtime says execute the model; exactly the configured one.
  std::vector<std::string> execution_devices;
  uint64_t device_nodes = 0;
  // Executed nodes whose implementation the plugin reports as host code,
  // "<layer type> <implementation>" each.
  std::vector<std::string> host_nodes;

  bool clean() const { return host_nodes.empty(); }
  std::string summary() const;
};

// Runs the exported vision encoder on one OpenVINO device.
//
// The IR has a dynamic page size. Each distinct size is reshaped to a fixed
// shape and compiled once for the device, then kept; a page of a size seen
// before runs without compiling.
//
// The model is compiled for the named device and nothing else: no AUTO,
// HETERO or MULTI, no second device. compile throws when the runtime reports
// another execution device or a node implemented in host code, so a page is
// either encoded on the device or fails.
//
// Not thread safe: one encoder serves one inference at a time.
class OpenvinoEncoder {
 public:
  // Reads the IR. Throws std::runtime_error when it cannot be read, is not
  // the encoder this code expects, or the device is not present.
  explicit OpenvinoEncoder(const EncoderOptions& options);
  ~OpenvinoEncoder();

  // Pixels per output token along one axis, and the embedding width.
  uint32_t token_stride() const;
  uint32_t out_hidden() const;

  // pixels: one preprocessed page, interleaved RGB float, width * height * 3
  // values; both sides multiples of token_stride(). Returns the embeddings
  // row by row: (width / stride) * (height / stride) rows of out_hidden().
  std::vector<float> encode(std::span<const float> pixels, uint32_t width, uint32_t height);

  // Placement of the size the last encode ran.
  const EncoderPlacement& placement() const;
  // Sizes compiled since construction; a repeated size does not count.
  uint64_t compilations() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lighton::vision
