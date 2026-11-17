// Build-time model preparation, like tools/prepare_model.py: converts the
// vision tower of a projector file into an OpenVINO IR pair. The server never
// runs this; it reads the IR.
//
//   export_vision_encoder VISION.gguf OUT.xml
//
// Writes OUT.xml and the weights beside it as OUT.bin, each tensor in the
// precision the projector file stores it in. The page size stays dynamic in
// the IR.
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <openvino/openvino.hpp>

#include "vision/encoder_model.h"
#include "vision/gguf_weights.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: export_vision_encoder VISION.gguf OUT.xml\n";
    return 64;
  }
  try {
    const lighton::vision::EncoderWeights weights = lighton::vision::read_encoder_weights(argv[1]);
    const std::shared_ptr<ov::Model> model = lighton::vision::build_encoder_model(weights);
    const auto& spec = weights.spec;
    // What a reader of the IR needs to feed it, kept inside the IR.
    model->set_rt_info(spec.token_stride(), "lighton", "token_stride");
    model->set_rt_info(spec.patch_size, "lighton", "patch_size");
    model->set_rt_info(spec.merge_size, "lighton", "merge_size");
    model->set_rt_info(std::string("RGB, (value / 255 - 0.5) / 0.5, NCHW"), "lighton", "pixel_format");
    model->set_rt_info(std::filesystem::path(argv[1]).filename().string(), "lighton", "source_file");
    // No compression on top: the constants already have the file's precision.
    ov::save_model(model, argv[2], false);
    std::cout << "wrote " << argv[2] << ": " << spec.layers << " layers, hidden " << spec.hidden << ", token stride "
              << spec.token_stride() << ", embedding width " << spec.out_hidden << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "export_vision_encoder: " << error.what() << '\n';
    return 1;
  }
}
