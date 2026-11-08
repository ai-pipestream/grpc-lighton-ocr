#include "engine/native_engine.h"
#include <chrono>
#include <atomic>
#include <iostream>
#include "ggml-backend.h"
#include <fstream>
#include <mutex>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <openssl/sha.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "stb_image.h"
extern "C" void lighton_ggml_set_observer(
    void (*observer)(ggml_backend_t, ggml_tensor*, void*), void* data);
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
bool abort_decode(void* p) { return static_cast<Control*>(p)->stop(); }
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
}
struct NativeEngine::Impl {
  std::mutex mutex;
  std::string projector;
  Control* active = nullptr;
  ggml_backend_dev_t device = nullptr;
  std::string backend;
  struct Observation {
    uint64_t device_arithmetic = 0, cpu_arithmetic = 0, cpu_ops = 0;
    ggml_backend_dev_t cpu_device = nullptr;
    void observe(ggml_backend_t backend, ggml_tensor* tensor, ggml_backend_dev_t requested) {
      // Copies/views are not model arithmetic. Device accessibility says
      // nothing about execution, especially for Metal/unified memory.
      if (tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_VIEW ||
          tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_TRANSPOSE ||
          tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_CPY ||
          tensor->op == GGML_OP_CONT || tensor->op == GGML_OP_DUP) return;
      const auto dev = ggml_backend_get_device(backend);
      const std::string_view registry = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
      // BLAS/Accelerate advertises ACCEL, but performs model math on CPU.
      const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || registry == "BLAS";
      if (cpu) ++cpu_ops;
      if (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) {
        if (cpu) { ++cpu_arithmetic; cpu_device = dev; }
        if (dev == requested) ++device_arithmetic;
      }
    }
  } vision_observation, decoder_observation;
  mutable std::mutex placement_mutex;
  NativePlacement vision_last, decoder_last;
  static void observe(ggml_backend_t backend, ggml_tensor* tensor, void* opaque) {
    auto& state = *static_cast<Impl*>(opaque);
    // mtmd and llama use distinct graphs. The caller selects the stage at
    // the encode/eval boundary below through stage-specific callbacks.
    auto& target = state.vision_active ? state.vision_observation : state.decoder_observation;
    target.observe(backend, tensor, state.device);
  }
  bool vision_active = false;
  NativePlacement snapshot(const Observation& observation) const {
    const auto actual = backend == "cpu" && observation.cpu_device ? observation.cpu_device : device;
    return {backend, ggml_backend_dev_description(actual), ggml_backend_dev_name(actual),
            backend == "cpu" ? observation.cpu_arithmetic > 0 : observation.device_arithmetic > 0,
            observation.cpu_ops > 0};
  }
  void publish() {
    std::lock_guard lock(placement_mutex);
    vision_last = snapshot(vision_observation);
    decoder_last = snapshot(decoder_observation);
  }
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
  std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision{nullptr, mtmd_free};
};
NativeEngine::NativeEngine(const std::string& directory, const std::string& backend,
                           const std::string& device) : impl_(std::make_unique<Impl>()) {
  const std::string text = directory + "/model-bf16.gguf";
  impl_->projector = directory + "/vision-bf16.gguf";

  static std::once_flag initialized;
  std::call_once(initialized, [] { llama_backend_init(); });
  if (backend != "cpu" && backend != "cuda" && backend != "sycl" && backend != "metal")
    throw std::invalid_argument("unsupported native backend: " + backend);
  const std::string registry = backend == "cpu" ? "CPU" : backend == "cuda" ? "CUDA" :
                               backend == "sycl" ? "SYCL" : "Metal";
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    auto candidate = ggml_backend_dev_get(i);
    if (registry != ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate))) continue;
    if (!device.empty() && device != ggml_backend_dev_name(candidate)) continue;
    if (backend != "cpu" && ggml_backend_dev_type(candidate) != GGML_BACKEND_DEVICE_TYPE_GPU &&
        ggml_backend_dev_type(candidate) != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
    impl_->device = candidate; break;
  }
  if (!impl_->device) throw std::runtime_error("requested native backend/device unavailable: " + backend + "/" + device);
  impl_->backend = backend;
  // Initialize the selected device before reading weights; never fall back.
  auto validation = ggml_backend_dev_init(impl_->device, nullptr);
  if (!validation) throw std::runtime_error("requested device initialization failed");
  ggml_backend_free(validation);
  verify(text, kModelSha); verify(impl_->projector, kProjectorSha);
  ggml_backend_dev_t devices[] = {backend == "cpu" ? nullptr : impl_->device, nullptr};
  auto params = llama_model_default_params();
  params.devices = devices;
  params.split_mode = LLAMA_SPLIT_MODE_NONE;
  params.n_gpu_layers = backend == "cpu" ? 0 : 999;
  impl_->model.reset(llama_model_load_from_file(text.c_str(), params));
  if (!impl_->model) throw std::runtime_error("native model load failed");
  auto vp = mtmd_context_params_default(); vp.use_gpu = backend != "cpu"; vp.device = impl_->device; vp.warmup = false;
  vp.n_threads = 2; vp.image_max_tokens = 2048;
  vp.cb_eval_user_data = impl_.get();
  vp.cb_eval = [](ggml_tensor*, bool ask, void* opaque) {
    auto& state = *static_cast<Impl*>(opaque);
    state.vision_active = true;
    return ask ? (state.active && state.active->stop()) : false;
  };
  impl_->vision.reset(mtmd_init_from_file(impl_->projector.c_str(), impl_->model.get(), vp));
  if (!impl_->vision) throw std::runtime_error("native projector load failed");
  impl_->publish(); // Identities are known; runtime verification is still false.

}
NativeEngine::~NativeEngine() = default;
NativePlacement NativeEngine::vision_placement() const {
  std::lock_guard lock(impl_->placement_mutex); return impl_->vision_last;
}
NativePlacement NativeEngine::decoder_placement() const {
  std::lock_guard lock(impl_->placement_mutex); return impl_->decoder_last;
}
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
    impl_->active = &control;
    impl_->vision_observation = {}; impl_->decoder_observation = {};
    struct PublishPlacement { Impl* state; ~PublishPlacement() { state->publish(); } } publish{impl_.get()};
    lighton_ggml_set_observer(Impl::observe, impl_.get());
    struct ClearObserver { ~ClearObserver() { lighton_ggml_set_observer(nullptr, nullptr); } } clear_observer;
    struct ClearActive { Impl* state; ~ClearActive() { state->active = nullptr; } } clear{impl_.get()};
    auto cp = llama_context_default_params();
    cp.n_ctx = kContext; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.cb_eval_user_data = impl_.get();
    cp.cb_eval = [](ggml_tensor*, bool ask, void* opaque) {
      auto& state = *static_cast<Impl*>(opaque);
      state.vision_active = false;
      return ask ? (state.active && state.active->stop()) : false;
    };
    cp.offload_kqv = impl_->backend != "cpu";
    cp.op_offload = impl_->backend != "cpu";
    cp.abort_callback = abort_decode; cp.abort_callback_data = &control;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(impl_->model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("native decoder context failed");
    auto* vision = impl_->vision.get();
    control.check();
    auto decoded = mtmd_helper_bitmap_init_from_buf(vision, reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), false,
                                                   mtmd_helper_init_opt_default());
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(decoded.bitmap, mtmd_bitmap_free);
    if (!bitmap) throw EngineError(EngineError::bad_image, "native image-buffer decode failed");
    // Exact single-user, no-tools, thinking-disabled Qwen3.5 chat-template
    // form. libmtmd expands its media marker into the model's vision tokens.
    std::string prompt = std::string("<|im_start|>user\n") + mtmd_default_marker() +
        "grounding<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const mtmd_bitmap* images[] = {bitmap.get()};
    if (mtmd_tokenize(vision, chunks.get(), &text, images, 1))
      throw std::runtime_error("native multimodal tokenization failed");
    if (mtmd_helper_get_n_tokens(chunks.get()) + max_tokens > cp.n_ctx)
      throw EngineError(EngineError::budget, "prompt exceeds context budget");
    llama_pos past = 0;
    if (mtmd_helper_eval_chunks(vision, ctx.get(), chunks.get(), 0, 0, 512, true, &past)) {
      control.check();
      throw std::runtime_error("native multimodal prefill failed");
    }
    if (!impl_->snapshot(impl_->vision_observation).runtime_verified ||
        !impl_->snapshot(impl_->decoder_observation).runtime_verified)
      throw std::runtime_error("model stage did not execute arithmetic on the requested device");
    try {
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (temperature == 0) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(0));
    }
    const auto* vocab = llama_model_get_vocab(impl_->model.get());
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    bool complete = false;
    size_t trace_bytes = 0;
    for (uint32_t i = 0; i < max_tokens; ++i) {
      if (control.cancelled()) throw EngineError(EngineError::cancelled, "client cancelled inference");
      if (Clock::now() >= control.deadline) { result.finish = NativeFinish::deadline; return result; }
      llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      const int count = llama_vocab_n_tokens(vocab);
      const int top = static_cast<int>(std::max_element(logits, logits + count) - logits);
      double sum = 0;
      for (int j = 0; j < count; ++j) sum += std::exp(double(logits[j]) - logits[top]);
      const double normalizer = logits[top] + std::log(sum);
      const std::string_view raw_piece(llama_vocab_get_text(vocab, token));
      if (raw_piece.size() > kMaxOutputBytes - trace_bytes)
        { result.finish = NativeFinish::output_limit; return result; }
      trace_bytes += raw_piece.size();
      result.steps.push_back({static_cast<uint32_t>(token), static_cast<uint32_t>(top),
          llama_vocab_get_text(vocab, token), static_cast<float>(logits[token] - normalizer),
          static_cast<float>(logits[top] - normalizer)});
      if (llama_vocab_is_eog(vocab, token)) {
        if (raw_piece.size() > kMaxOutputBytes - result.raw_text.size())
          { result.finish = NativeFinish::output_limit; return result; }
        result.raw_text += raw_piece; complete = true; break;
      }
      std::vector<char> piece(256);
      int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
      if (n < 0) { if (n < -65536) throw EngineError(EngineError::budget, "token piece too large"); piece.resize(-n); n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
      if (n < 0) throw std::runtime_error("token decode failed");
      if (result.text.size() + static_cast<size_t>(n) > kMaxOutputBytes)
        { result.finish = NativeFinish::output_limit; return result; }
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
        return result;
      }
    }
    if (!complete) result.finish = NativeFinish::token_limit;
    impl_->publish();
    std::cerr << "placement vision=" << vision_placement().device_id
              << " matmuls=" << impl_->vision_observation.device_arithmetic
              << " cpu_ops=" << impl_->vision_observation.cpu_ops
              << " decoder=" << decoder_placement().device_id
              << " matmuls=" << impl_->decoder_observation.device_arithmetic
              << " cpu_ops=" << impl_->decoder_observation.cpu_ops << '\n';
    return result;
    } catch (const EngineError& error) {
      if (error.kind == EngineError::cancelled) throw;
      result.finish = error.kind == EngineError::budget ? NativeFinish::output_limit : NativeFinish::engine_error;
      return result;
    } catch (const std::exception&) {
      result.finish = NativeFinish::engine_error;
      return result;
    }
}
}
