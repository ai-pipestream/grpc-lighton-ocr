#include "engine/native_engine.h"
#include <chrono>
#include <fstream>
#include <mutex>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <openssl/sha.h>
#include "engine/placement.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "stb_image.h"
namespace lighton {
namespace {
using Clock = std::chrono::steady_clock;
struct Control {
  Clock::time_point deadline;
  std::function<bool()> cancelled;
  bool stop() const { return cancelled() || Clock::now() >= deadline; }
  void check() const {
    if (stop()) throw EngineError(EngineError::cancelled, "inference cancelled or deadline exceeded");
  }
};
// The ggml CPU backend calls its abort callback once after every node it
// computes, so the call count is the number of decoder nodes the CPU ran.
struct DecodeWatch {
  Control* control;
  uint64_t cpu_nodes = 0;
};
bool abort_decode(void* p) {
  auto* watch = static_cast<DecodeWatch*>(p);
  ++watch->cpu_nodes;
  return watch->control->stop();
}
void verify(const std::string& file, const char* expected) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("missing configured model artifact: " + file);
  SHA256_CTX context; SHA256_Init(&context);
  std::array<char, 65536> bytes;
  while (in.read(bytes.data(), bytes.size()) || in.gcount())
    SHA256_Update(&context, bytes.data(), in.gcount());
  if (!in.eof()) throw std::runtime_error("model artifact read failed: " + file);
  unsigned char digest[SHA256_DIGEST_LENGTH]; SHA256_Final(digest, &context);
  std::string hex; const char* digits = "0123456789abcdef";
  for (auto c : digest) { hex += digits[c >> 4]; hex += digits[c & 15]; }
  if (hex != expected) throw std::runtime_error("model artifact SHA256 mismatch: " + file);
}
// Reference page for the startup check on an accelerator: the repository's
// invoice fixture, compiled in. Placement says where a graph ran, not that
// the device computed it correctly, so the device must also read this page.
constexpr unsigned char kReferencePage[] = {
#include "reference_page.inc"
};
constexpr std::string_view kReferenceWords[] = {"INVOICE", "1042", "125.00", "Paid"};
constexpr uint32_t kProbeTokens = 96;
}
struct NativeEngine::Impl {
  std::mutex mutex;
  std::string projector;
  EngineInfo info;
  int threads = 2;
  ggml_backend_dev_t device = nullptr;  // nullptr: the ggml CPU backend
  Control* active = nullptr;
  // llama keeps these pointers in the model's parameters, so they live here.
  std::array<ggml_backend_dev_t, 2> devices{};  // explicit list, null-terminated
  std::array<llama_model_tensor_buft_override, 2> on_device{};
  uint64_t decoder_cpu_nodes = 0;  // nodes the ggml CPU backend computed for the decoder
  // Set only on an accelerator. They outlive the contexts that call them.
  std::unique_ptr<PlacementRecorder> vision_placement, decoder_placement;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
  std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision{nullptr, mtmd_free};

  void generate(const mtmd_bitmap* bitmap, uint32_t max_tokens, float temperature,
                Control& control, NativeResult& result);
  void reset_placement();
  std::string require_device_placement(bool require_both_stages);
  void probe_placement();
};
void NativeEngine::Impl::reset_placement() {
  if (vision_placement) vision_placement->take();
  if (decoder_placement) decoder_placement->take();
  decoder_cpu_nodes = 0;
}
// Throws unless everything recorded since the last reset was on the device.
// Returns the receipt. A no-op on the CPU backend.
std::string NativeEngine::Impl::require_device_placement(bool require_both_stages) {
  if (!vision_placement) return {};
  const PlacementReport vision_report = vision_placement->take();
  const PlacementReport decoder_report = decoder_placement->take();
  const std::string receipt = "vision: " + vision_report.summary() + " | decoder: " + decoder_report.summary() +
      ", " + std::to_string(decoder_cpu_nodes) + " nodes computed by the ggml CPU backend";
  const bool unobserved = vision_report.device_nodes == 0 || decoder_report.device_nodes == 0;
  if (vision_report.clean() && decoder_report.clean() && decoder_cpu_nodes == 0 &&
      !(require_both_stages && unobserved))
    return receipt;
  const std::string message = "inference left device " + info.device + ": " + receipt;
  std::cerr << "grpc-lighton-ocr placement violation: " << message << '\n';
  throw EngineError(EngineError::inference, message.c_str());
}
// The reference page through the vision encoder, the prefill and the decoder,
// so that a graph the device cannot run, or runs wrongly, stops startup and
// not a request.
void NativeEngine::Impl::probe_placement() {
  auto decoded = mtmd_helper_bitmap_init_from_buf(vision.get(), kReferencePage, sizeof(kReferencePage), false,
                                                  mtmd_helper_init_opt_default());
  std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(decoded.bitmap, mtmd_bitmap_free);
  if (!bitmap) throw std::runtime_error("startup reference page decode failed");
  // The first inference on a GPU compiles every graph; allow for that.
  Control control{Clock::now() + std::chrono::minutes(15), [] { return false; }};
  NativeResult result;
  reset_placement();
  generate(bitmap.get(), kProbeTokens, 0, control, result);
  info.placement = require_device_placement(true);
  const bool read = result.finish == NativeFinish::end_token &&
      std::ranges::all_of(kReferenceWords, [&](std::string_view word) {
        return result.text.find(word) != std::string::npos;
      });
  if (!read)
    throw std::runtime_error("device " + info.device + " does not reproduce the startup reference page; it read: '" +
                             result.text.substr(0, 160) + "'");
}
NativeEngine::NativeEngine(const EngineOptions& options) : impl_(std::make_unique<Impl>()) {
  // Device first: a missing device must stop startup before any weight is
  // read, and ggml must learn the selection before its registry exists.
  impl_->device = select_device(options.device);
  impl_->threads = options.threads;
  const ModelArtifacts& artifacts = artifacts_for(options.device);
  const std::string text = options.directory + "/" + artifacts.model;
  impl_->projector = options.directory + "/" + artifacts.projector;
  verify(text, artifacts.model_sha256); verify(impl_->projector, artifacts.projector_sha256);
  impl_->info = {describe_device(options.device, impl_->device), artifacts.model_sha256,
                 artifacts.projector_sha256, {}};
  static std::once_flag initialized;
  std::call_once(initialized, [] { llama_backend_init(); });
  const bool accelerated = impl_->device != nullptr;
  // An explicit list in both modes: llama never picks a device by itself.
  impl_->devices = {impl_->device, nullptr};
  auto params = llama_model_default_params();
  params.devices = impl_->devices.data();
  params.n_gpu_layers = accelerated ? -1 : 0;  // negative: every layer
  if (accelerated) {
    // llama keeps the input layer in host memory unless told otherwise.
    impl_->on_device = {{{"^token_embd\\.weight$", ggml_backend_dev_buffer_type(impl_->device)},
                         {nullptr, nullptr}}};
    params.tensor_buft_overrides = impl_->on_device.data();
  }
  impl_->model.reset(llama_model_load_from_file(text.c_str(), params));
  if (!impl_->model) throw std::runtime_error("native model load failed");
  auto vp = mtmd_context_params_default(); vp.use_gpu = accelerated; vp.device = impl_->device;
  vp.warmup = false; vp.n_threads = impl_->threads; vp.image_max_tokens = 2048;
  if (accelerated) {
    impl_->vision_placement = std::make_unique<PlacementRecorder>(impl_->device);
    impl_->decoder_placement = std::make_unique<PlacementRecorder>(impl_->device);
    // The recorder never asks for per-node data, so the device receives each
    // graph whole. Cancellation is checked between stages and tokens.
    vp.cb_eval = PlacementRecorder::on_eval;
    vp.cb_eval_user_data = impl_->vision_placement.get();
  } else {
    vp.cb_eval_user_data = impl_.get();
    vp.cb_eval = [](ggml_tensor*, bool ask, void* opaque) {
      auto* state = static_cast<Impl*>(opaque);
      return ask || state->active == nullptr || !state->active->stop();
    };
  }
  impl_->vision.reset(mtmd_init_from_file(impl_->projector.c_str(), impl_->model.get(), vp));
  if (!impl_->vision) throw std::runtime_error("native projector load failed");
  if (accelerated) impl_->probe_placement();
}
NativeEngine::~NativeEngine() = default;
const EngineInfo& NativeEngine::info() const { return impl_->info; }
NativeResult NativeEngine::infer(std::string_view bytes, uint32_t max_tokens, float temperature,
                                uint32_t width, uint32_t height, std::function<bool()> cancelled) {
  std::unique_lock lock(impl_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) throw EngineError(EngineError::busy, "native engine busy");
  if (!std::isfinite(temperature) || temperature < 0 || temperature > 2 ||
      (temperature > 0 && temperature < 0.01f))
    throw EngineError(EngineError::budget, "invalid sampling temperature");
  if (max_tokens == 0) max_tokens = 512;
  if (max_tokens > kMaxTokens) throw EngineError(EngineError::budget, "token cap exceeds 1024");
  if (bytes.empty() || bytes.size() > 32 * 1024 * 1024)
    throw EngineError(EngineError::bad_image, "invalid image byte size");
  const bool png = bytes.size() >= 8 && bytes.substr(0, 8) == std::string_view("\x89PNG\r\n\x1a\n", 8);
  const bool jpeg = bytes.size() >= 3 && static_cast<unsigned char>(bytes[0]) == 0xff &&
      static_cast<unsigned char>(bytes[1]) == 0xd8 && static_cast<unsigned char>(bytes[2]) == 0xff;
  if (!png && !jpeg) throw EngineError(EngineError::bad_image, "only PNG and JPEG are accepted");
  int w = 0, h = 0, channels = 0;
  if (!stbi_info_from_memory(reinterpret_cast<const unsigned char*>(bytes.data()),
                             static_cast<int>(bytes.size()), &w, &h, &channels) || w <= 0 || h <= 0)
    throw EngineError(EngineError::bad_image, "invalid image header");
  if (static_cast<uint64_t>(w) * h > kMaxPixels || w > 8192 || h > 8192)
    throw EngineError(EngineError::budget, "raster exceeds pixel limit");
  if ((width && width != static_cast<uint32_t>(w)) || (height && height != static_cast<uint32_t>(h)))
    throw EngineError(EngineError::bad_image, "declared raster dimensions disagree");
  NativeResult result{{}, {}, static_cast<uint32_t>(w), static_cast<uint32_t>(h), 0, {}};
  Control control{Clock::now() + std::chrono::seconds(60), std::move(cancelled)};
  control.check();
  auto decoded = mtmd_helper_bitmap_init_from_buf(impl_->vision.get(),
      reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), false, mtmd_helper_init_opt_default());
  std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(decoded.bitmap, mtmd_bitmap_free);
  if (!bitmap) throw EngineError(EngineError::bad_image, "native image-buffer decode failed");
  impl_->reset_placement();
  impl_->generate(bitmap.get(), max_tokens, temperature, control, result);
  impl_->require_device_placement(false);
  return result;
}
// Vision encode, prefill and sampling for one decoded page. Fills result and
// sets its finish reason; throws for cancellation and for setup failures.
void NativeEngine::Impl::generate(const mtmd_bitmap* bitmap, uint32_t max_tokens, float temperature,
                                  Control& control, NativeResult& result) {
    active = &control;
    struct ClearActive { Impl* state; ~ClearActive() { state->active = nullptr; } } clear{this};
    auto cp = llama_context_default_params();
    cp.n_ctx = kContext; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    DecodeWatch watch{&control};
    struct KeepCount { Impl* state; DecodeWatch* watch; ~KeepCount() { state->decoder_cpu_nodes += watch->cpu_nodes; } }
        keep{this, &watch};
    cp.abort_callback = abort_decode; cp.abort_callback_data = &watch;
    if (decoder_placement) {
      cp.cb_eval = PlacementRecorder::on_eval;
      cp.cb_eval_user_data = decoder_placement.get();
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("native decoder context failed");
    auto* vision = this->vision.get();
    control.check();
    // Exact single-user, no-tools, thinking-disabled Qwen3.5 chat-template
    // form. libmtmd expands its media marker into the model's vision tokens.
    std::string prompt = std::string("<|im_start|>user\n") + mtmd_default_marker() +
        "grounding<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const mtmd_bitmap* images[] = {bitmap};
    if (mtmd_tokenize(vision, chunks.get(), &text, images, 1))
      throw std::runtime_error("native multimodal tokenization failed");
    if (mtmd_helper_get_n_tokens(chunks.get()) + max_tokens > cp.n_ctx)
      throw EngineError(EngineError::budget, "prompt exceeds context budget");
    llama_pos past = 0;
    if (mtmd_helper_eval_chunks(vision, ctx.get(), chunks.get(), 0, 0, 512, true, &past)) {
      control.check();
      throw std::runtime_error("native multimodal prefill failed");
    }
    try {
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (temperature == 0) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(0));
    }
    const auto* vocab = llama_model_get_vocab(model.get());
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    bool complete = false;
    size_t trace_bytes = 0;
    for (uint32_t i = 0; i < max_tokens; ++i) {
      if (control.cancelled()) throw EngineError(EngineError::cancelled, "client cancelled inference");
      if (Clock::now() >= control.deadline) { result.finish = NativeFinish::deadline; return; }
      llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      const int count = llama_vocab_n_tokens(vocab);
      const int top = static_cast<int>(std::max_element(logits, logits + count) - logits);
      double sum = 0;
      for (int j = 0; j < count; ++j) sum += std::exp(double(logits[j]) - logits[top]);
      const double normalizer = logits[top] + std::log(sum);
      const std::string_view raw_piece(llama_vocab_get_text(vocab, token));
      if (raw_piece.size() > kMaxOutputBytes - trace_bytes)
        { result.finish = NativeFinish::output_limit; return; }
      trace_bytes += raw_piece.size();
      result.steps.push_back({static_cast<uint32_t>(token), static_cast<uint32_t>(top),
          llama_vocab_get_text(vocab, token), static_cast<float>(logits[token] - normalizer),
          static_cast<float>(logits[top] - normalizer)});
      if (llama_vocab_is_eog(vocab, token)) {
        if (raw_piece.size() > kMaxOutputBytes - result.raw_text.size())
          { result.finish = NativeFinish::output_limit; return; }
        result.raw_text += raw_piece; complete = true; break;
      }
      std::vector<char> piece(256);
      int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
      if (n < 0) { if (n < -65536) throw EngineError(EngineError::budget, "token piece too large"); piece.resize(-n); n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
      if (n < 0) throw std::runtime_error("token decode failed");
      if (result.text.size() + static_cast<size_t>(n) > kMaxOutputBytes)
        { result.finish = NativeFinish::output_limit; return; }
      result.text.append(piece.data(), n); result.raw_text.append(piece.data(), n); ++result.tokens;
      llama_batch_ext_clear(batch.get());
      int index = llama_batch_ext_add_token(batch.get(), 0, token);
      if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &past) ||
          !llama_batch_ext_set_output_logits(batch.get(), index, true))
        throw std::runtime_error("native batch setup failed");
      ++past;
      if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        if (control.cancelled()) throw EngineError(EngineError::cancelled, "client cancelled inference");
        result.finish = Clock::now() >= control.deadline ? NativeFinish::deadline : NativeFinish::engine_error;
        return;
      }
    }
    if (!complete) result.finish = NativeFinish::token_limit;
    return;
    } catch (const EngineError& error) {
      if (error.kind == EngineError::cancelled) throw;
      result.finish = error.kind == EngineError::budget ? NativeFinish::output_limit : NativeFinish::engine_error;
      return;
    } catch (const std::exception&) {
      result.finish = NativeFinish::engine_error;
      return;
    }
}
}
