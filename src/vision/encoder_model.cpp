#include "vision/encoder_model.h"

#include <cmath>
#include <stdexcept>

#include <openvino/core/model.hpp>
#include <openvino/op/ops.hpp>

namespace lighton::vision {
namespace {

using Out = ov::Output<ov::Node>;
using ov::element::f32;
using ov::element::i64;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

Out ints(std::vector<int64_t> values) {
  return v0::Constant::create(i64, ov::Shape{values.size()}, values);
}
Out scalar(int64_t value) { return v0::Constant::create(i64, ov::Shape{}, {value}); }

Out constant(const Tensor& tensor, const ov::Shape& shape) {
  if (ov::shape_size(shape) != tensor.values.size())
    throw std::runtime_error("vision encoder weight has an unexpected size");
  if (!tensor.half) return v0::Constant::create(f32, shape, tensor.values);
  // Values that came from half precision go back into it unchanged.
  return std::make_shared<v0::Convert>(v0::Constant::create(ov::element::f16, shape, tensor.values), f32);
}
Out constant(const Tensor& tensor) { return constant(tensor, ov::Shape(tensor.shape)); }

Out named(Out node, const std::string& name) {
  node.get_node()->set_friendly_name(name);
  return node;
}

// Builds the graph for one set of weights. Hidden states are [1, tokens, C].
class Builder {
 public:
  explicit Builder(const EncoderWeights& weights) : w_(weights), spec_(weights.spec) {}
  std::shared_ptr<ov::Model> build();

 private:
  Out linear(const Out& x, const std::string& prefix) const;
  Out layer_norm(const Out& x, const std::string& prefix) const;
  Out patch_embedding(const Out& pixels) const;
  Out position_table(const Out& feature_map) const;
  Out merge_order(const Out& map, int64_t channels) const;
  void rotary_tables(const Out& feature_map);
  Out rotate(const Out& x) const;
  Out attention(const Out& x, const std::string& prefix) const;
  Out block(const Out& x, uint32_t index) const;

  const EncoderWeights& w_;
  const EncoderSpec& spec_;
  Out rows_, cols_;  // patch rows and columns, i64 scalars
  Out cos_, sin_;    // [1, tokens, 1, head_dim / 2]
};

Out Builder::linear(const Out& x, const std::string& prefix) const {
  auto product = std::make_shared<v0::MatMul>(x, constant(w_.at(prefix + ".weight")), false, true);
  return std::make_shared<v1::Add>(product, constant(w_.at(prefix + ".bias")));
}

// ggml_norm, then weight and bias: variance over the last axis, eps inside
// the square root.
Out Builder::layer_norm(const Out& x, const std::string& prefix) const {
  auto normalized = std::make_shared<ov::op::v6::MVN>(x, ints({-1}), true, spec_.norm_eps,
                                                      ov::op::MVNEpsMode::INSIDE_SQRT);
  auto scaled = std::make_shared<v1::Multiply>(normalized, constant(w_.at(prefix + ".weight")));
  return std::make_shared<v1::Add>(scaled, constant(w_.at(prefix + ".bias")));
}

// The checkpoint's patch layer is a 3-D convolution over two frames. A still
// image is both frames, so the two frame kernels add up to one 2-D kernel,
// kept as float: a sum of two half-precision values need not be one.
Out Builder::patch_embedding(const Out& pixels) const {
  Tensor kernel = w_.at("v.patch_embd.weight");
  const Tensor& second = w_.at("v.patch_embd.weight.1");
  if (second.values.size() != kernel.values.size())
    throw std::runtime_error("vision encoder patch kernels disagree in size");
  for (size_t i = 0; i < kernel.values.size(); ++i) kernel.values[i] += second.values[i];
  kernel.half = false;
  const ov::Strides stride{spec_.patch_size, spec_.patch_size};
  auto conv = std::make_shared<v1::Convolution>(pixels, constant(kernel), stride, ov::CoordinateDiff{0, 0},
                                                ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
  return std::make_shared<v1::Add>(conv, constant(w_.at("v.patch_embd.bias"), ov::Shape{1, spec_.hidden, 1, 1}));
}

// The learned side x side position table, resized to the page's patch grid:
// bilinear with aligned corners, [1, C, rows, cols].
Out Builder::position_table(const Out& feature_map) const {
  const Tensor& table = w_.at("v.position_embd.weight");  // [side * side, C], row-major in (y, x)
  const size_t side = spec_.position_side, channels = spec_.hidden;
  Tensor planar{{1, channels, side, side}, std::vector<float>(table.values.size()), table.half};
  for (size_t p = 0; p < side * side; ++p)
    for (size_t c = 0; c < channels; ++c) planar.values[c * side * side + p] = table.values[p * channels + c];
  using Interpolate = ov::op::v11::Interpolate;
  Interpolate::InterpolateAttrs attrs;
  attrs.mode = Interpolate::InterpolateMode::LINEAR;
  attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
  attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
  attrs.pads_begin = {0, 0, 0, 0};
  attrs.pads_end = {0, 0, 0, 0};
  auto shape = std::make_shared<ov::op::v3::ShapeOf>(feature_map, i64);
  auto size = std::make_shared<ov::op::v8::Gather>(shape, ints({2, 3}), scalar(0));
  return std::make_shared<Interpolate>(constant(planar), size, ints({2, 3}), attrs);
}

// [1, C, rows, cols] to [1, tokens * merge * merge, C], patches ordered as
// the projector merges them: token row, token column, then the patches of
// that token row by row.
Out Builder::merge_order(const Out& map, int64_t channels) const {
  const int64_t m = spec_.merge_size;
  auto part = [&](const Out& dim) -> Out {
    return std::make_shared<v0::Unsqueeze>(std::make_shared<v1::Divide>(dim, scalar(m)), scalar(0));
  };
  auto split_shape =
      std::make_shared<v0::Concat>(ov::OutputVector{ints({channels}), part(rows_), ints({m}), part(cols_), ints({m})}, 0);
  auto split = std::make_shared<v1::Reshape>(map, split_shape, false);
  auto ordered = std::make_shared<v1::Transpose>(split, ints({1, 3, 2, 4, 0}));
  return std::make_shared<v1::Reshape>(ordered, ints({1, -1, channels}), false);
}

// Rotary angles per patch: the first quarter of a head's dimensions turns
// with the patch row, the second with the patch column, and each of those
// pairs with the dimension half a head further on.
void Builder::rotary_tables(const Out& feature_map) {
  auto shape = std::make_shared<ov::op::v3::ShapeOf>(feature_map, i64);
  rows_ = std::make_shared<ov::op::v8::Gather>(shape, scalar(2), scalar(0));
  cols_ = std::make_shared<ov::op::v8::Gather>(shape, scalar(3), scalar(0));
  auto grid_shape = std::make_shared<v0::Concat>(
      ov::OutputVector{ints({1, 1}), std::make_shared<v0::Unsqueeze>(rows_, scalar(0)),
                       std::make_shared<v0::Unsqueeze>(cols_, scalar(0))},
      0);
  auto axis_positions = [&](const Out& count, std::vector<int64_t> shape_of_axis) -> Out {
    auto range = std::make_shared<ov::op::v4::Range>(scalar(0), count, scalar(1), i64);
    auto as_float = std::make_shared<v0::Convert>(range, f32);
    auto shaped = std::make_shared<v1::Reshape>(as_float, ints(std::move(shape_of_axis)), false);
    auto grid = std::make_shared<ov::op::v3::Broadcast>(shaped, grid_shape);
    return merge_order(grid, 1);  // [1, patches, 1]
  };
  const size_t quarter = spec_.head_dim() / 4;
  // ggml multiplies the angle by this factor once per frequency, in float.
  const float step = std::pow(10000.0f, -2.0f / static_cast<float>(spec_.head_dim() / 2));
  std::vector<float> frequencies(quarter);
  float frequency = 1.0f;
  for (auto& value : frequencies) {
    value = frequency;
    frequency *= step;
  }
  Out frequency_row = v0::Constant::create(f32, ov::Shape{1, 1, quarter}, frequencies);
  auto row_angles = std::make_shared<v1::Multiply>(axis_positions(rows_, {1, 1, -1, 1}), frequency_row);
  auto col_angles = std::make_shared<v1::Multiply>(axis_positions(cols_, {1, 1, 1, -1}), frequency_row);
  auto angles = std::make_shared<v0::Concat>(ov::OutputVector{row_angles, col_angles}, 2);
  auto per_head = std::make_shared<v0::Unsqueeze>(angles, scalar(2));  // [1, patches, 1, head_dim / 2]
  cos_ = std::make_shared<v0::Cos>(per_head);
  sin_ = std::make_shared<v0::Sin>(per_head);
}

// x is [1, patches, heads, head_dim]; rotates dimension d with d + head_dim / 2.
Out Builder::rotate(const Out& x) const {
  auto halves = std::make_shared<v1::Split>(x, scalar(3), 2);
  const Out a = halves->output(0), b = halves->output(1);
  auto first = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(a, cos_),
                                              std::make_shared<v1::Multiply>(b, sin_));
  auto second = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(a, sin_),
                                          std::make_shared<v1::Multiply>(b, cos_));
  return std::make_shared<v0::Concat>(ov::OutputVector{first, second}, 3);
}

Out Builder::attention(const Out& x, const std::string& prefix) const {
  const int64_t heads = spec_.heads, head_dim = spec_.head_dim();
  // The fused projection lays out query, key and value one after another.
  auto fused = std::make_shared<v1::Reshape>(linear(x, prefix + ".attn_qkv"), ints({0, 0, 3, heads, head_dim}), true);
  auto parts = std::make_shared<v1::Split>(fused, scalar(2), 3);
  auto head_major = [&](const Out& part, bool rotary) -> Out {
    Out flat = std::make_shared<v0::Squeeze>(part, ints({2}));  // [1, patches, heads, head_dim]
    if (rotary) flat = rotate(flat);
    return std::make_shared<v1::Transpose>(flat, ints({0, 2, 1, 3}));
  };
  auto mixed = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
      head_major(parts->output(0), true), head_major(parts->output(1), true), head_major(parts->output(2), false),
      false);
  auto token_major = std::make_shared<v1::Transpose>(mixed, ints({0, 2, 1, 3}));
  auto merged = std::make_shared<v1::Reshape>(token_major, ints({0, 0, heads * head_dim}), true);
  return linear(merged, prefix + ".attn_out");
}

Out Builder::block(const Out& x, uint32_t index) const {
  const std::string prefix = "v.blk." + std::to_string(index);
  Out attended = std::make_shared<v1::Add>(x, attention(layer_norm(x, prefix + ".ln1"), prefix));
  Out hidden = linear(layer_norm(attended, prefix + ".ln2"), prefix + ".ffn_up");
  hidden = std::make_shared<ov::op::v7::Gelu>(hidden, ov::op::GeluApproximationMode::TANH);
  hidden = linear(hidden, prefix + ".ffn_down");
  return named(std::make_shared<v1::Add>(attended, hidden), prefix + ".out");
}

std::shared_ptr<ov::Model> Builder::build() {
  auto pixels = std::make_shared<v0::Parameter>(f32, ov::PartialShape{1, 3, -1, -1});
  pixels->set_friendly_name(kPixelInput);
  pixels->output(0).get_tensor().set_names({kPixelInput});

  const Out patches = patch_embedding(pixels);
  rotary_tables(patches);
  Out x = std::make_shared<v1::Add>(patches, position_table(patches));
  x = named(merge_order(x, spec_.hidden), "patch_sequence");
  for (uint32_t i = 0; i < spec_.layers; ++i) x = block(x, i);
  x = layer_norm(x, "v.post_ln");

  // Projector: the patches of one token side by side, then a two-layer MLP.
  const int64_t group = static_cast<int64_t>(spec_.hidden) * spec_.merge_size * spec_.merge_size;
  x = std::make_shared<v1::Reshape>(x, ints({-1, group}), false);
  x = std::make_shared<ov::op::v7::Gelu>(linear(x, "mm.0"), ov::op::GeluApproximationMode::TANH);
  x = named(linear(x, "mm.2"), kEmbeddingOutput);
  x.get_tensor().set_names({kEmbeddingOutput});

  auto result = std::make_shared<v0::Result>(x);
  auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{pixels}, "lighton_vision_encoder");
  return model;
}

}  // namespace

const Tensor& EncoderWeights::at(const std::string& name) const {
  const auto found = tensors.find(name);
  if (found == tensors.end()) throw std::runtime_error("vision encoder weight is missing: " + name);
  return found->second;
}

std::shared_ptr<ov::Model> build_encoder_model(const EncoderWeights& weights) {
  const EncoderSpec& spec = weights.spec;
  if (spec.heads == 0 || spec.hidden % spec.heads != 0 || spec.head_dim() % 4 != 0 || spec.merge_size == 0)
    throw std::runtime_error("vision encoder geometry is not supported");
  return Builder(weights).build();
}

}  // namespace lighton::vision
