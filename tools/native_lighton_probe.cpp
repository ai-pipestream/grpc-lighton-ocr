// CPU/Metal feasibility probe. File I/O belongs to this test harness; the
// native multimodal API receives the image as a byte buffer.
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"

using Clock = std::chrono::steady_clock;
static bool expired(void* data) { return Clock::now() > *static_cast<Clock::time_point*>(data); }
struct Placement {
  ggml_backend_dev_t selected;
  Clock::time_point deadline;
  unsigned selected_ops = 0, cpu_ops = 0, other_ops = 0;
  static bool observe(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& p = *static_cast<Placement*>(opaque);
    if (Clock::now() > p.deadline) return false;
    if (ask) return true;
    if (tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
        tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) return true;
    auto device = tensor->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) : nullptr;
    if (device == p.selected) ++p.selected_ops;
    else if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) ++p.cpu_ops;
    else ++p.other_ops;
    return true;
  }
};
int main(int argc, char** argv) {
  if (argc != 5) { std::cerr << "usage: native_lighton_probe MODEL PROJECTOR PNG cpu|metal\n"; return 64; }
  try {
    llama_backend_init();
    const std::string backend = argv[4];
    if (backend != "cpu" && backend != "metal") throw std::invalid_argument("expected cpu or metal");
    ggml_backend_dev_t selected = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
      auto dev = ggml_backend_dev_get(i);
      if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) ==
          (backend == "metal" ? "Metal" : "CPU")) { selected = dev; break; }
    }
    if (!selected) throw std::runtime_error("requested backend unavailable");
    auto deadline = Clock::now() + std::chrono::minutes(5);
    Placement encoder{selected, deadline}, decoder{selected, deadline};
    ggml_backend_dev_t devices[] = {backend == "metal" ? selected : nullptr, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices; mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.n_gpu_layers = backend == "metal" ? 999 : 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("native text-model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.abort_callback = expired; cp.abort_callback_data = &deadline;
    cp.offload_kqv = backend == "metal"; cp.op_offload = backend == "metal";
    cp.cb_eval = Placement::observe; cp.cb_eval_user_data = &decoder;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("native decoder context failed");
    auto vp = mtmd_context_params_default();
    vp.use_gpu = backend == "metal"; vp.device = selected; vp.n_threads = 2; vp.warmup = false;
    vp.cb_eval = Placement::observe; vp.cb_eval_user_data = &encoder;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision(
        mtmd_init_from_file(argv[2], model.get(), vp), mtmd_free);
    if (!vision) throw std::runtime_error("native projector load failed");
    std::ifstream input(argv[3], std::ios::binary);
    if (!input) throw std::runtime_error("fixture open failed");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.empty() || bytes.size() > 32*1024*1024) throw std::runtime_error("fixture size invalid");
    auto decoded = mtmd_helper_bitmap_init_from_buf(vision.get(), bytes.data(), bytes.size(), false,
                                                   mtmd_helper_init_opt_default());
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(decoded.bitmap, mtmd_bitmap_free);
    if (!bitmap) throw std::runtime_error("native image-buffer decode failed");
    // Exact single-user, no-tools, thinking-disabled Qwen3.5 chat-template
    // form. libmtmd expands its media marker into the model's vision tokens.
    std::string prompt = std::string("<|im_start|>user\n") + mtmd_default_marker() +
        "grounding<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const mtmd_bitmap* images[] = {bitmap.get()};
    if (mtmd_tokenize(vision.get(), chunks.get(), &text, images, 1))
      throw std::runtime_error("native multimodal tokenization failed");
    if (mtmd_helper_get_n_tokens(chunks.get()) + 256 > cp.n_ctx)
      throw std::runtime_error("prompt exceeds context budget");
    llama_pos past = 0;
    if (mtmd_helper_eval_chunks(vision.get(), ctx.get(), chunks.get(), 0, 0, 512, true, &past))
      throw std::runtime_error("native multimodal prefill failed");
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_init_greedy(), llama_sampler_free);
    const auto* vocab = llama_model_get_vocab(model.get());
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    bool complete = false;
    for (int i = 0; i < 256; ++i) {
      if (expired(&deadline)) throw std::runtime_error("decode deadline exceeded");
      llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
      if (llama_vocab_is_eog(vocab, token)) { complete = true; break; }
      std::vector<char> piece(256);
      int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
      if (n < 0) { piece.resize(-n); n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false); }
      if (n < 0) throw std::runtime_error("token decode failed");
      std::cout.write(piece.data(), n); std::cout.flush();
      llama_batch_ext_clear(batch.get());
      int index = llama_batch_ext_add_token(batch.get(), 0, token);
      if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &past) ||
          !llama_batch_ext_set_output_logits(batch.get(), index, true))
        throw std::runtime_error("native batch setup failed");
      ++past;
      if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()))
        throw std::runtime_error("native token decode failed");
    }
    if (!complete) throw std::runtime_error("decode token budget exhausted");
    std::cerr << "device=" << ggml_backend_dev_description(selected)
      << " vision_selected=" << encoder.selected_ops << " vision_cpu=" << encoder.cpu_ops
      << " vision_other=" << encoder.other_ops << " decoder_selected=" << decoder.selected_ops
      << " decoder_cpu=" << decoder.cpu_ops << " decoder_other=" << decoder.other_ops << '\n';
    if (!encoder.selected_ops || !decoder.selected_ops || encoder.other_ops || decoder.other_ops)
      throw std::runtime_error("both stages must execute on the selected device with known placement");
    if (backend == "metal" && (encoder.cpu_ops || decoder.cpu_ops))
      throw std::runtime_error("partial Metal offload: CPU model compute observed");
    std::cout << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << "probe: " << e.what() << '\n'; return 1; }
}
