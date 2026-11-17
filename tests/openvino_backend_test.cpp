// The two carried fixes to ggml's OpenVINO backend (patches/llama.cpp), on
// the OpenVINO CPU device, against the ggml CPU backend as the reference.
// The graph is the attention block a vision encoder builds without flash
// attention: query, key and value are VIEWs of one projection, query and key
// get rotary positions, and the value VIEW is permuted with (1, 2, 0, 3).
// Needs no model file and no GPU.
//
//   1. That permutation used to be translated as if it were (0, 2, 1, 3),
//      which scrambled the values.
//   2. The backend used to serve a second input size from the graph it had
//      compiled for the first one.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace {

constexpr int kHeads = 2, kHeadDim = 4, kHidden = kHeads * kHeadDim, kLayers = 3;

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

// Repeatable values in [-0.5, 0.5).
float next_value(uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
}

void fill(ggml_tensor* tensor, uint32_t seed) {
  std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
  for (float& value : values) value = next_value(seed);
  ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}

struct Weights {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  std::vector<ggml_tensor*> projection, bias;  // one fused query, key, value projection per layer
};

Weights make_weights(ggml_backend_t backend) {
  Weights weights;
  weights.ctx = ggml_init({ggml_tensor_overhead() * 4 * kLayers, nullptr, true});
  for (int layer = 0; layer < kLayers; ++layer) {
    weights.projection.push_back(ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, kHidden, 3 * kHidden));
    weights.bias.push_back(ggml_new_tensor_1d(weights.ctx, GGML_TYPE_F32, 3 * kHidden));
    ggml_format_name(weights.projection.back(), "qkv.%d.weight", layer);
    ggml_format_name(weights.bias.back(), "qkv.%d.bias", layer);
  }
  weights.buffer = ggml_backend_alloc_ctx_tensors(weights.ctx, backend);
  ggml_backend_buffer_set_usage(weights.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for (int layer = 0; layer < kLayers; ++layer) {
    fill(weights.projection[layer], 100u + layer);
    fill(weights.bias[layer], 200u + layer);
  }
  return weights;
}

// One attention block as tools/mtmd/clip.cpp builds it with flash attention
// off: x is [hidden, tokens], positions one int per token.
ggml_tensor* attention(ggml_context* ctx, ggml_tensor* x, ggml_tensor* positions, const Weights& weights,
                       int layer) {
  const int64_t tokens = x->ne[1];
  ggml_tensor* fused = ggml_add(ctx, ggml_mul_mat(ctx, weights.projection[layer], x), weights.bias[layer]);
  auto part = [&](int index) {
    return ggml_view_3d(ctx, fused, kHeadDim, kHeads, tokens, ggml_row_size(fused->type, kHeadDim), fused->nb[1],
                        ggml_row_size(fused->type, index * kHidden));
  };
  auto rotary = [&](ggml_tensor* part_view) {
    return ggml_rope_ext(ctx, part_view, positions, nullptr, kHeadDim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  };
  ggml_tensor* q = ggml_permute(ctx, rotary(part(0)), 0, 2, 1, 3);
  ggml_tensor* k = ggml_permute(ctx, rotary(part(1)), 0, 2, 1, 3);
  ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, part(2), 1, 2, 0, 3));  // the value VIEW, permuted directly
  ggml_tensor* weights_kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), nullptr, 1.0f / std::sqrt(float(kHeadDim)), 0.0f);
  ggml_tensor* mixed = ggml_permute(ctx, ggml_mul_mat(ctx, v, weights_kq), 0, 2, 1, 3);
  return ggml_add(ctx, x, ggml_cont_2d(ctx, mixed, kHidden, tokens));
}

// Runs the block stack for one token count and returns the output values.
std::vector<float> run(ggml_backend_sched_t sched, const Weights& weights, int tokens) {
  ggml_context* ctx = ggml_init({ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true});
  ggml_tensor* x = ggml_set_name(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, tokens), "page");
  ggml_set_input(x);
  ggml_tensor* positions = ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens), "positions");
  ggml_set_input(positions);
  ggml_tensor* out = x;
  for (int layer = 0; layer < kLayers; ++layer) out = attention(ctx, out, positions, weights, layer);
  ggml_set_name(out, "out");
  ggml_set_output(out);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, out);
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("test graph allocation failed");
  fill(x, 7u + tokens);
  std::vector<int32_t> order(tokens);
  for (int i = 0; i < tokens; ++i) order[i] = i;
  ggml_backend_tensor_set(positions, order.data(), 0, ggml_nbytes(positions));
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("test graph compute failed for " + std::to_string(tokens) + " tokens");
  std::vector<float> values(static_cast<size_t>(ggml_nelements(out)));
  ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
  ggml_free(ctx);
  return values;
}

struct Runner {
  std::vector<ggml_backend_t> backends;
  ggml_backend_sched_t sched;
  Weights weights;
  explicit Runner(std::vector<ggml_backend_t> list) : backends(std::move(list)) {
    sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()),
                                   GGML_DEFAULT_GRAPH_SIZE, false, true);
    weights = make_weights(backends.front());
  }
};

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0;
  double dot = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return dot / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
}

ggml_backend_dev_t openvino_device() {
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    if (std::string_view(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) == "OPENVINO" &&
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU)
      return device;
  }
  throw std::runtime_error("ggml registered no selected OpenVINO device");
}

}  // namespace

int main() {
  // The backend reads its device once, when ggml builds its registry.
  ::setenv("GGML_OPENVINO_DEVICE", "CPU", 1);
  try {
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    ggml_backend_t reference_cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    ggml_backend_t openvino = ggml_backend_dev_init(openvino_device(), nullptr);
    if (cpu == nullptr || reference_cpu == nullptr || openvino == nullptr)
      throw std::runtime_error("backend initialization failed");
    Runner reference({reference_cpu}), device({openvino, cpu});
    // The OpenVINO CPU plugin may compute in bfloat16; a scrambled tensor is
    // far outside this floor, rounding is far inside it.
    constexpr double floor = 0.999;
    // Several sizes through one scheduler, a repeat included: each must be
    // computed for its own size, not served from the first one compiled.
    for (int tokens : {6, 10, 6, 4}) {
      const std::vector<float> want = run(reference.sched, reference.weights, tokens);
      const std::vector<float> got = run(device.sched, device.weights, tokens);
      const double agreement = cosine(want, got);
      expect(got.size() == want.size() && agreement >= floor,
             std::to_string(tokens) + " tokens: cosine to the ggml CPU backend is " + std::to_string(agreement));
      expect(ggml_backend_sched_get_n_splits(device.sched) == 1,
             std::to_string(tokens) + " tokens: the graph ran as one OpenVINO split");
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) return 1;
  std::cout << "openvino_backend_test: PASS\n";
  return 0;
}
