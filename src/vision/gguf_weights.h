#pragma once

#include <string>

#include "vision/encoder_model.h"

namespace lighton::vision {

// Reads the vision tower from a projector file written by llama.cpp's
// converter (vision-f16.gguf or vision-bf16.gguf). Values come out as float,
// dimensions outermost first. Throws std::runtime_error when the file is not
// a projector of the one type this encoder implements, or lacks a tensor.
EncoderWeights read_encoder_weights(const std::string& gguf_path);

}  // namespace lighton::vision
