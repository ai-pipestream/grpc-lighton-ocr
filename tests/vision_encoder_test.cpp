// The exported vision encoder on the OpenVINO CPU device, with a small
// encoder and made-up weights: the graph must compute what a plain C++
// evaluation of the same arithmetic computes, one model must serve several
// page sizes, and each size must compile once. Needs no model file and no GPU.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "vision/encoder_model.h"
#include "vision/openvino_encoder.h"

namespace {

using lighton::vision::EncoderSpec;
using lighton::vision::EncoderWeights;
using lighton::vision::Tensor;
using Rows = std::vector<std::vector<double>>;  // one row per patch or token

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

// Repeatable values in [-0.5, 0.5).
struct Generator {
  uint32_t state = 12345;
  float next() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
  }
};

Tensor filled(Generator& generator, std::vector<size_t> shape, float offset = 0) {
  size_t count = 1;
  for (size_t d : shape) count *= d;
  Tensor tensor{std::move(shape), std::vector<float>(count)};
  for (float& value : tensor.values) value = generator.next() + offset;
  return tensor;
}

EncoderWeights small_encoder() {
  EncoderWeights weights;
  weights.spec = EncoderSpec{2, 2, 16, 2, 2, 6, 3, 1e-6f};
  const size_t c = weights.spec.hidden, p = weights.spec.patch_size, side = weights.spec.position_side;
  Generator generator;
  auto& t = weights.tensors;
  auto linear = [&](const std::string& name, size_t out, size_t in) {
    t[name + ".weight"] = filled(generator, {out, in});
    t[name + ".bias"] = filled(generator, {out});
  };
  auto norm = [&](const std::string& name) {
    t[name + ".weight"] = filled(generator, {c}, 1.0f);
    t[name + ".bias"] = filled(generator, {c});
  };
  t["v.patch_embd.weight"] = filled(generator, {c, 3, p, p});
  t["v.patch_embd.weight.1"] = filled(generator, {c, 3, p, p});
  t["v.patch_embd.bias"] = filled(generator, {c});
  t["v.position_embd.weight"] = filled(generator, {side * side, c});
  for (uint32_t i = 0; i < weights.spec.layers; ++i) {
    const std::string block = "v.blk." + std::to_string(i);
    norm(block + ".ln1");
    linear(block + ".attn_qkv", 3 * c, c);
    linear(block + ".attn_out", c, c);
    norm(block + ".ln2");
    linear(block + ".ffn_up", 2 * c, c);
    linear(block + ".ffn_down", c, 2 * c);
  }
  norm("v.post_ln");
  const size_t group = c * weights.spec.merge_size * weights.spec.merge_size;
  linear("mm.0", group, group);
  linear("mm.2", weights.spec.out_hidden, group);
  return weights;
}

// The plain evaluation. It shares no code with the graph builder.
struct Reference {
  const EncoderWeights& w;
  uint32_t rows, cols;  // patch grid

  std::vector<double> linear(const std::vector<double>& x, const std::string& name) const {
    const Tensor& weight = w.at(name + ".weight");
    const Tensor& bias = w.at(name + ".bias");
    std::vector<double> out(weight.shape[0]);
    for (size_t o = 0; o < out.size(); ++o) {
      out[o] = bias.values[o];
      for (size_t i = 0; i < x.size(); ++i) out[o] += weight.values[o * x.size() + i] * x[i];
    }
    return out;
  }
  std::vector<double> norm(const std::vector<double>& x, const std::string& name) const {
    double mean = 0, variance = 0;
    for (double v : x) mean += v / static_cast<double>(x.size());
    for (double v : x) variance += (v - mean) * (v - mean) / static_cast<double>(x.size());
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      out[i] = (x[i] - mean) / std::sqrt(variance + w.spec.norm_eps) * w.at(name + ".weight").values[i] +
               w.at(name + ".bias").values[i];
    return out;
  }
  static double gelu(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
  }
  Rows patches(const std::vector<float>& pixels, uint32_t width) const;
  Rows attention(const Rows& x, const std::string& block) const;
  std::vector<double> run(const std::vector<float>& pixels, uint32_t width) const;
};

// Patch embedding plus the resized position table, in the merge order.
Rows Reference::patches(const std::vector<float>& pixels, uint32_t width) const {
  const size_t c = w.spec.hidden, p = w.spec.patch_size, m = w.spec.merge_size, side = w.spec.position_side;
  const Tensor& k0 = w.at("v.patch_embd.weight");
  const Tensor& k1 = w.at("v.patch_embd.weight.1");
  const Tensor& table = w.at("v.position_embd.weight");
  auto patch = [&](size_t r, size_t col) {
    std::vector<double> out(c);
    const double fy = static_cast<double>(r) * (side - 1) / (rows - 1), fx = static_cast<double>(col) * (side - 1) / (cols - 1);
    const size_t y0 = static_cast<size_t>(fy), x0 = static_cast<size_t>(fx);
    const size_t y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
    const double wy = fy - y0, wx = fx - x0;
    for (size_t ch = 0; ch < c; ++ch) {
      double sum = w.at("v.patch_embd.bias").values[ch];
      for (size_t in = 0; in < 3; ++in)
        for (size_t ky = 0; ky < p; ++ky)
          for (size_t kx = 0; kx < p; ++kx) {
            const size_t k = ((ch * 3 + in) * p + ky) * p + kx;
            sum += (k0.values[k] + k1.values[k]) * pixels[((r * p + ky) * width + col * p + kx) * 3 + in];
          }
      auto at = [&](size_t y, size_t x) { return table.values[(y * side + x) * c + ch]; };
      sum += (1 - wy) * ((1 - wx) * at(y0, x0) + wx * at(y0, x1)) + wy * ((1 - wx) * at(y1, x0) + wx * at(y1, x1));
      out[ch] = sum;
    }
    return out;
  };
  Rows sequence;
  for (size_t tr = 0; tr < rows / m; ++tr)
    for (size_t tc = 0; tc < cols / m; ++tc)
      for (size_t dy = 0; dy < m; ++dy)
        for (size_t dx = 0; dx < m; ++dx) sequence.push_back(patch(tr * m + dy, tc * m + dx));
  return sequence;
}

Rows Reference::attention(const Rows& x, const std::string& block) const {
  const size_t c = w.spec.hidden, heads = w.spec.heads, d = w.spec.head_dim(), m = w.spec.merge_size;
  const size_t n = x.size(), quarter = d / 4;
  std::vector<std::vector<double>> q(n), k(n), v(n);
  for (size_t i = 0; i < n; ++i) {
    const auto fused = linear(norm(x[i], block + ".ln1"), block + ".attn_qkv");
    q[i].assign(fused.begin(), fused.begin() + c);
    k[i].assign(fused.begin() + c, fused.begin() + 2 * c);
    v[i].assign(fused.begin() + 2 * c, fused.end());
    // Position of sequence entry i in the patch grid.
    const size_t token = i / (m * m), inside = i % (m * m);
    const double position[2] = {static_cast<double>(token / (cols / m) * m + inside / m),
                                static_cast<double>(token % (cols / m) * m + inside % m)};
    for (auto* vector : {&q[i], &k[i]})
      for (size_t h = 0; h < heads; ++h)
        for (size_t j = 0; j < 2 * quarter; ++j) {
          const double angle = position[j / quarter] * std::pow(10000.0, -2.0 * (j % quarter) / (d / 2));
          double& a = (*vector)[h * d + j];
          double& b = (*vector)[h * d + j + d / 2];
          const double a0 = a, b0 = b;
          a = a0 * std::cos(angle) - b0 * std::sin(angle);
          b = a0 * std::sin(angle) + b0 * std::cos(angle);
        }
  }
  Rows out(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<double> mixed(c, 0.0);
    for (size_t h = 0; h < heads; ++h) {
      std::vector<double> score(n);
      double total = 0;
      for (size_t j = 0; j < n; ++j) {
        double dot = 0;
        for (size_t e = 0; e < d; ++e) dot += q[i][h * d + e] * k[j][h * d + e];
        total += score[j] = std::exp(dot / std::sqrt(static_cast<double>(d)));
      }
      for (size_t j = 0; j < n; ++j)
        for (size_t e = 0; e < d; ++e) mixed[h * d + e] += score[j] / total * v[j][h * d + e];
    }
    out[i] = linear(mixed, block + ".attn_out");
  }
  return out;
}

std::vector<double> Reference::run(const std::vector<float>& pixels, uint32_t width) const {
  Rows x = patches(pixels, width);
  for (uint32_t layer = 0; layer < w.spec.layers; ++layer) {
    const std::string block = "v.blk." + std::to_string(layer);
    const Rows attended = attention(x, block);
    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t e = 0; e < x[i].size(); ++e) x[i][e] += attended[i][e];
      auto hidden = linear(norm(x[i], block + ".ln2"), block + ".ffn_up");
      for (double& value : hidden) value = gelu(value);
      const auto down = linear(hidden, block + ".ffn_down");
      for (size_t e = 0; e < x[i].size(); ++e) x[i][e] += down[e];
    }
  }
  const size_t group = w.spec.merge_size * w.spec.merge_size;
  std::vector<double> out;
  for (size_t token = 0; token < x.size() / group; ++token) {
    std::vector<double> joined;
    for (size_t i = 0; i < group; ++i) {
      const auto normed = norm(x[token * group + i], "v.post_ln");
      joined.insert(joined.end(), normed.begin(), normed.end());
    }
    auto hidden = linear(joined, "mm.0");
    for (double& value : hidden) value = gelu(value);
    const auto projected = linear(hidden, "mm.2");
    out.insert(out.end(), projected.begin(), projected.end());
  }
  return out;
}

std::vector<float> page(uint32_t width, uint32_t height) {
  Generator generator{777u + width * 31u + height};
  std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
  for (float& value : pixels) value = 2.0f * generator.next();
  return pixels;
}

void check_size(lighton::vision::OpenvinoEncoder& encoder, const EncoderWeights& weights, uint32_t width,
                uint32_t height) {
  const std::string label = std::to_string(width) + "x" + std::to_string(height);
  const auto pixels = page(width, height);
  const std::vector<float> got = encoder.encode(pixels, width, height);
  const Reference reference{weights, height / weights.spec.patch_size, width / weights.spec.patch_size};
  const std::vector<double> want = reference.run(pixels, width);
  expect(got.size() == want.size(), label + ": " + std::to_string(got.size()) + " values, want " +
                                        std::to_string(want.size()));
  if (got.size() != want.size()) return;
  double worst = 0;
  for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
  expect(worst < 2e-4, label + ": largest difference to the plain evaluation is " + std::to_string(worst));
  expect(encoder.placement().width == width && encoder.placement().height == height && encoder.placement().clean() &&
             encoder.placement().device_nodes > 0,
         label + ": placement " + encoder.placement().summary());
}

template <typename Call>
bool throws(Call&& call) {
  try {
    call();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const EncoderWeights weights = small_encoder();
  const auto directory = std::filesystem::temp_directory_path() / "lighton-vision-encoder-test";
  std::filesystem::create_directories(directory);
  const std::string xml = (directory / "encoder.xml").string(), bin = (directory / "encoder.bin").string();
  auto model = lighton::vision::build_encoder_model(weights);
  model->set_rt_info(weights.spec.token_stride(), "lighton", "token_stride");
  ov::save_model(model, xml, false);

  lighton::vision::OpenvinoEncoder encoder({xml, bin, "CPU", {}, 2, lighton::vision::EncoderPrecision::f32});
  expect(encoder.token_stride() == 4 && encoder.out_hidden() == 6, "the IR carries its token stride and width");
  check_size(encoder, weights, 12, 8);
  check_size(encoder, weights, 8, 16);
  expect(encoder.compilations() == 2, "two page sizes, two compilations");
  check_size(encoder, weights, 12, 8);
  expect(encoder.compilations() == 2, "a page size seen before does not compile again");
  // Room for two sizes: a third replaces the least recently used, 8x16.
  check_size(encoder, weights, 16, 8);
  check_size(encoder, weights, 12, 8);
  expect(encoder.compilations() == 3, "the most recently used size stays compiled");
  check_size(encoder, weights, 8, 16);
  expect(encoder.compilations() == 4, "the least recently used size was dropped");

  expect(throws([&] { encoder.encode(page(10, 8), 10, 8); }), "a width that is no whole number of tokens is refused");
  expect(throws([&] { encoder.encode(page(12, 8), 8, 8); }), "a pixel count that disagrees with the size is refused");
  expect(throws([&] { lighton::vision::OpenvinoEncoder absent({xml, bin, "GPU.9", {}, 2, {}}); }),
         "a device the runtime does not offer is refused");
  expect(throws([&] { lighton::vision::OpenvinoEncoder missing({xml + ".none", bin, "CPU", {}, 2, {}}); }),
         "a missing IR is refused");

  std::filesystem::remove_all(directory);
  if (failures != 0) return 1;
  std::cout << "vision_encoder_test: PASS\n";
  return 0;
}
