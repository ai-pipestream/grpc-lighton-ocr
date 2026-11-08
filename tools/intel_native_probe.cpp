// Native engine smoke helper, without RPC. RPC acceptance is separate.
#include "engine/native_engine.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: intel_native_probe MODELS PNG cpu|sycl\n";
    return 64;
  }
  try {
    const std::string backend = argv[3];
    if (backend != "cpu" && backend != "sycl") throw std::runtime_error("explicit cpu or sycl required");
    lighton::NativeEngine engine(argv[1], backend);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("fixture missing");
    std::string bytes((std::istreambuf_iterator<char>(input)), {});
    for (int repeat = 0; repeat < 6; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      auto result = engine.infer(bytes, 256, 0, 0, 0, [] { return false; });
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
      std::cout << "repeat=" << repeat << " elapsed_us=" << us << " text=" << result.text << '\n';
      if (result.finish != lighton::NativeFinish::end_token ||
          result.text.find("INVOICE 1042") == std::string::npos ||
          result.text.find("125.00") == std::string::npos || result.text.find("Paid") == std::string::npos)
        throw std::runtime_error("invoice text/completion assertion failed");
      std::smatch box;
      if (!std::regex_search(result.text, box, std::regex(R"(\((\d+),(\d+),(\d+),(\d+)\))")) ||
          std::stoi(box[1]) >= std::stoi(box[3]) || std::stoi(box[2]) >= std::stoi(box[4]))
        throw std::runtime_error("positive generated box assertion failed");
      for (const auto& placement : {engine.vision_placement(), engine.decoder_placement()}) {
        std::cout << "placement backend=" << placement.backend << " device=" << placement.device_name
                  << " id=" << placement.device_id << " verified=" << placement.runtime_verified
                  << " cpu_compute=" << placement.cpu_compute << '\n';
        if (!placement.runtime_verified || placement.backend != backend || placement.cpu_compute != (backend == "cpu"))
          throw std::runtime_error("full stage placement assertion failed");
      }
    }
    // Pre-cancel smoke only. The real RPC gate must cancel active inference.
    bool cancelled = false;
    try { engine.infer(bytes, 256, 0, 0, 0, [] { return true; }); }
    catch (const lighton::EngineError& error) { cancelled = error.kind == lighton::EngineError::cancelled; }
    if (!cancelled) throw std::runtime_error("pre-cancellation not observed");
    auto recovered = engine.infer(bytes, 256, 0, 0, 0, [] { return false; });
    if (recovered.finish != lighton::NativeFinish::end_token || recovered.text.find("INVOICE 1042") == std::string::npos)
      throw std::runtime_error("recovery failed");
    std::cout << "PASS native engine text/basic-box/repeats/pre-cancel/recovery; RPC is separate\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
