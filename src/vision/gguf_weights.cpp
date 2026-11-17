#include "vision/gguf_weights.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace lighton::vision {
namespace {

constexpr char kProjectorType[] = "qwen3vl_merger";

using GgufPtr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

int64_t required_key(const gguf_context* gguf, const char* key) {
  const int64_t id = gguf_find_key(gguf, key);
  if (id < 0) throw std::runtime_error(std::string("projector file lacks metadata key ") + key);
  return id;
}

uint32_t u32(const gguf_context* gguf, const char* key) { return gguf_get_val_u32(gguf, required_key(gguf, key)); }

// One tensor's bytes as float, with the dimension order reversed: ggml lists
// the innermost dimension first.
Tensor read_tensor(const gguf_context* gguf, std::ifstream& file, int64_t id) {
  const ggml_type type = gguf_get_tensor_type(gguf, id);
  const int64_t* ne = gguf_get_tensor_ne(gguf, id);
  int dims = GGML_MAX_DIMS;
  while (dims > 1 && ne[dims - 1] == 1) --dims;
  Tensor tensor;
  size_t count = 1;
  for (int d = dims - 1; d >= 0; --d) {
    tensor.shape.push_back(static_cast<size_t>(ne[d]));
    count *= static_cast<size_t>(ne[d]);
  }
  std::vector<char> bytes(gguf_get_tensor_size(gguf, id));
  file.seekg(static_cast<std::streamoff>(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id)));
  if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error(std::string("projector tensor read failed: ") + gguf_get_tensor_name(gguf, id));
  tensor.values.resize(count);
  const size_t width = type == GGML_TYPE_F32 ? sizeof(float) : sizeof(uint16_t);
  if (bytes.size() != count * width || (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16))
    throw std::runtime_error(std::string("projector tensor has an unsupported type: ") +
                             gguf_get_tensor_name(gguf, id));
  const auto n = static_cast<int64_t>(count);
  if (type == GGML_TYPE_F32) std::copy_n(reinterpret_cast<const float*>(bytes.data()), count, tensor.values.data());
  if (type == GGML_TYPE_F16) {
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), tensor.values.data(), n);
    tensor.half = true;
  }
  if (type == GGML_TYPE_BF16)
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), tensor.values.data(), n);
  return tensor;
}

EncoderSpec read_spec(const gguf_context* gguf) {
  const std::string type = gguf_get_val_str(gguf, required_key(gguf, "clip.projector_type"));
  if (type != kProjectorType)
    throw std::runtime_error("projector type '" + type + "' is not " + kProjectorType);
  if (!gguf_get_val_bool(gguf, required_key(gguf, "clip.use_gelu")))
    throw std::runtime_error("projector does not use the GELU activation this encoder implements");
  EncoderSpec spec;
  spec.patch_size = u32(gguf, "clip.vision.patch_size");
  spec.merge_size = u32(gguf, "clip.vision.spatial_merge_size");
  spec.hidden = u32(gguf, "clip.vision.embedding_length");
  spec.heads = u32(gguf, "clip.vision.attention.head_count");
  spec.layers = u32(gguf, "clip.vision.block_count");
  spec.out_hidden = u32(gguf, "clip.vision.projection_dim");
  spec.norm_eps = gguf_get_val_f32(gguf, required_key(gguf, "clip.vision.attention.layer_norm_epsilon"));
  // The exported model documents its input as (value / 255 - 0.5) / 0.5.
  for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const int64_t id = required_key(gguf, key);
    const auto* values = static_cast<const float*>(gguf_get_arr_data(gguf, id));
    for (size_t i = 0; i < gguf_get_arr_n(gguf, id); ++i)
      if (values[i] != 0.5f) throw std::runtime_error(std::string("projector normalization is not 0.5: ") + key);
  }
  // This encoder has no deepstack branch; a checkpoint that uses one needs it.
  if (const int64_t id = gguf_find_key(gguf, "clip.vision.is_deepstack_layers"); id >= 0) {
    const auto* flags = static_cast<const int8_t*>(gguf_get_arr_data(gguf, id));
    for (size_t i = 0; i < gguf_get_arr_n(gguf, id); ++i)
      if (flags[i] != 0) throw std::runtime_error("projector uses deepstack layers, which this encoder lacks");
  }
  return spec;
}

}  // namespace

EncoderWeights read_encoder_weights(const std::string& gguf_path) {
  GgufPtr gguf(gguf_init_from_file(gguf_path.c_str(), {true, nullptr}), gguf_free);
  if (!gguf) throw std::runtime_error("cannot read projector file: " + gguf_path);
  std::ifstream file(gguf_path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open projector file: " + gguf_path);
  EncoderWeights weights;
  weights.spec = read_spec(gguf.get());
  for (int64_t id = 0; id < gguf_get_n_tensors(gguf.get()); ++id)
    weights.tensors.emplace(gguf_get_tensor_name(gguf.get(), id), read_tensor(gguf.get(), file, id));
  const Tensor& positions = weights.at("v.position_embd.weight");
  const auto side = static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(positions.shape.at(0)))));
  if (positions.shape.size() != 2 || side * side != positions.shape[0] || positions.shape[1] != weights.spec.hidden)
    throw std::runtime_error("projector position table is not a square grid of hidden-size rows");
  weights.spec.position_side = side;
  return weights;
}

}  // namespace lighton::vision
