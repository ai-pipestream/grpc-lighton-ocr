// Qualification tool for the exported vision encoder: encodes pages with the
// ggml CPU encoder (the reference the CPU engine serves) and with the
// OpenVINO IR on one device, from the same preprocessed pixels, and compares
// the embeddings token by token.
//
//   vision_encoder_parity MODEL_DIRECTORY DEVICE f32|f16 IMAGE [IMAGE...]
//
// MODEL_DIRECTORY holds model-f16.gguf, vision-f16.gguf and the exported
// vision-encoder-f16.xml/.bin. DEVICE is an OpenVINO device id (CPU, GPU,
// GPU.<n>); f32 or f16 is the arithmetic the device computes in. Exits 1
// when a page misses the floors below.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "vision/openvino_encoder.h"

namespace {

// What a page must reach (measurements in docs/vision-encoder.md). In f32 the
// encoder agrees with the reference to a page cosine of 0.99997 or better,
// with no token below 0.99. In f16 a few tokens with very large activations
// turn further, as they do between the reference's own BF16 and F16 runs
// (page cosine 0.9997, up to 1% of the tokens below 0.99).
constexpr double kPageCosineFloorF32 = 0.9999;
constexpr double kPageCosineFloorF16 = 0.9995;
constexpr double kLowTokenShare = 0.01;

struct Agreement {
  double cosine = 0, worst_token_cosine = 1, max_abs = 0, mean_abs = 0, reference_max = 0;
  size_t low_tokens = 0;  // tokens whose cosine is below 0.99
};

double cosine(std::span<const float> a, std::span<const float> b) {
  double dot = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return dot / (std::sqrt(aa) * std::sqrt(bb) + 1e-30);
}

Agreement compare(std::span<const float> reference, std::span<const float> candidate, size_t width) {
  Agreement out;
  out.cosine = cosine(reference, candidate);
  double total = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double difference = std::fabs(double(reference[i]) - candidate[i]);
    out.max_abs = std::max(out.max_abs, difference);
    out.reference_max = std::max(out.reference_max, std::fabs(double(reference[i])));
    total += difference;
  }
  out.mean_abs = total / static_cast<double>(reference.size());
  for (size_t row = 0; row < reference.size() / width; ++row) {
    const double token = cosine(reference.subspan(row * width, width), candidate.subspan(row * width, width));
    out.worst_token_cosine = std::min(out.worst_token_cosine, token);
    if (token < 0.99) ++out.low_tokens;
  }
  return out;
}

std::vector<unsigned char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return {std::istreambuf_iterator<char>(in), {}};
}

// Encodes one page both ways. Returns false when it misses the floors.
bool check_page(mtmd_context* vision, lighton::vision::OpenvinoEncoder& encoder, const std::string& path,
                double cosine_floor) {
  const auto bytes = read_file(path);
  auto decoded = mtmd_helper_bitmap_init_from_buf(vision, bytes.data(), bytes.size(), false,
                                                  mtmd_helper_init_opt_default());
  std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(decoded.bitmap, mtmd_bitmap_free);
  if (!bitmap) throw std::runtime_error("cannot decode " + path);
  const std::string prompt = mtmd_default_marker();
  mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
  std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(),
                                                                                mtmd_input_chunks_free);
  const mtmd_bitmap* images[] = {bitmap.get()};
  if (mtmd_tokenize(vision, chunks.get(), &text, images, 1)) throw std::runtime_error("tokenization failed: " + path);
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), i);
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
    uint32_t width = 0, height = 0;
    const float* pixels = mtmd_image_tokens_get_pixels(mtmd_input_chunk_get_tokens_image(chunk), &width, &height);
    if (pixels == nullptr) throw std::runtime_error("no preprocessed page: " + path);
    const size_t tokens = mtmd_input_chunk_get_n_tokens(chunk);
    // Twice: the first call may compile this page size, the second does not.
    const auto started = std::chrono::steady_clock::now();
    encoder.encode({pixels, size_t(width) * height * 3}, width, height);
    const auto compiled = std::chrono::steady_clock::now();
    const std::vector<float> candidate = encoder.encode({pixels, size_t(width) * height * 3}, width, height);
    const auto encoded = std::chrono::steady_clock::now();
    using Ms = std::chrono::duration<double, std::milli>;
    if (mtmd_encode_chunk(vision, chunk)) throw std::runtime_error("reference encoder failed: " + path);
    if (candidate.size() != tokens * encoder.out_hidden()) throw std::runtime_error("token count differs: " + path);
    const Agreement result = compare({mtmd_get_output_embd(vision), candidate.size()}, candidate, encoder.out_hidden());
    std::printf("%s %ux%u tokens=%zu cosine=%.6f worst_token_cosine=%.6f tokens_below_0.99=%zu max_abs=%.4f "
                "mean_abs=%.5f reference_max=%.2f first_ms=%.0f encode_ms=%.1f\n",
                path.c_str(), width, height, tokens, result.cosine, result.worst_token_cosine, result.low_tokens,
                result.max_abs, result.mean_abs, result.reference_max, Ms(compiled - started).count(),
                Ms(encoded - compiled).count());
    return result.cosine >= cosine_floor &&
           static_cast<double>(result.low_tokens) <= kLowTokenShare * static_cast<double>(tokens);
  }
  throw std::runtime_error("no image chunk: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string precision = argc > 3 ? argv[3] : "";
  if (argc < 5 || (precision != "f32" && precision != "f16")) {
    std::cerr << "usage: vision_encoder_parity MODEL_DIRECTORY DEVICE f32|f16 IMAGE [IMAGE...]\n";
    return 64;
  }
  try {
    const std::string directory = argv[1];
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ggml_backend_dev_t no_devices[] = {nullptr};
    model_params.devices = no_devices;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file((directory + "/model-f16.gguf").c_str(), model_params), llama_model_free);
    if (!model) throw std::runtime_error("text model load failed");
    auto vision_params = mtmd_context_params_default();
    vision_params.use_gpu = false;
    vision_params.warmup = false;
    vision_params.n_threads = 4;
    vision_params.image_max_tokens = 2048;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision(
        mtmd_init_from_file((directory + "/vision-f16.gguf").c_str(), model.get(), vision_params), mtmd_free);
    if (!vision) throw std::runtime_error("projector load failed");
    lighton::vision::OpenvinoEncoder encoder(
        {directory + "/vision-encoder-f16.xml", directory + "/vision-encoder-f16.bin", argv[2], {}, 6,
         precision == "f16" ? lighton::vision::EncoderPrecision::f16 : lighton::vision::EncoderPrecision::f32});
    bool ok = true;
    const double cosine_floor = precision == "f16" ? kPageCosineFloorF16 : kPageCosineFloorF32;
    for (int i = 4; i < argc; ++i) ok = check_page(vision.get(), encoder, argv[i], cosine_floor) && ok;
    std::printf("placement of the last page: %s; sizes compiled: %llu\n", encoder.placement().summary().c_str(),
                static_cast<unsigned long long>(encoder.compilations()));
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "vision_encoder_parity: " << error.what() << '\n';
    return 1;
  }
}
