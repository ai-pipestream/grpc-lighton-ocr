#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov {
class Model;
}

namespace lighton::vision {

// One checkpoint tensor as float values, dimensions outermost first: a
// linear weight is [out, in], the patch kernel is [out, in, height, width].
struct Tensor {
  std::vector<size_t> shape;
  std::vector<float> values;
  // The checkpoint stores it as half-precision floats; the model keeps it so.
  bool half = false;
};

// Geometry of the vision tower. The defaults are the pinned checkpoint's;
// read_encoder_weights fills them from the projector file's metadata.
struct EncoderSpec {
  uint32_t patch_size = 16;
  uint32_t merge_size = 2;      // tokens are merge_size x merge_size patches
  uint32_t hidden = 768;
  uint32_t heads = 12;
  uint32_t layers = 12;
  uint32_t out_hidden = 1024;   // width of the text model's input embedding
  uint32_t position_side = 48;  // the learned position table is side x side
  float norm_eps = 1e-6f;

  uint32_t head_dim() const { return hidden / heads; }
  // Pixels per output token along one axis.
  uint32_t token_stride() const { return patch_size * merge_size; }
  bool operator==(const EncoderSpec&) const = default;
};

// Weights under the projector file's tensor names (v.blk.0.ln1.weight, ...).
struct EncoderWeights {
  EncoderSpec spec;
  std::map<std::string, Tensor> tensors;
  // Throws std::runtime_error naming the tensor when it is absent.
  const Tensor& at(const std::string& name) const;
};

// Names of the model's only input and output.
inline constexpr char kPixelInput[] = "pixel_values";
inline constexpr char kEmbeddingOutput[] = "image_embeds";

// The vision tower and its projector as one OpenVINO model.
//
//   pixel_values  f32 [1, 3, H, W]   RGB, normalized as the checkpoint's
//                                    preprocessor does; H and W are dynamic
//                                    and must be multiples of token_stride()
//   image_embeds  f32 [H/stride * W/stride, out_hidden]
//
// Everything that depends on the page size (the resized position table and
// the rotary tables) is computed inside the graph from the input's shape, so
// one model serves every page size: reshape it to a fixed [1, 3, H, W] and
// those subgraphs fold to constants at compile time.
//
// The arithmetic follows the clip graph llama.cpp builds for this projector
// type (tools/mtmd/models/qwen3vl.cpp at the pinned revision), which is the
// reference the CPU engine runs.
//
// Weights keep the checkpoint's precision: a half-precision tensor becomes a
// half-precision constant followed by a conversion, a float tensor a float
// constant. Saved without further compression the model holds the
// checkpoint's values exactly.
std::shared_ptr<ov::Model> build_encoder_model(const EncoderWeights& weights);

}  // namespace lighton::vision
