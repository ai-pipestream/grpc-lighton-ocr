#include "vision/openvino_encoder.h"

#include <algorithm>
#include <list>
#include <stdexcept>

#include <openvino/openvino.hpp>

#include "vision/encoder_model.h"

namespace lighton::vision {
namespace {

// Runtime-model node types that carry no computation.
bool bookkeeping(const std::string& layer_type) {
  return layer_type == "Parameter" || layer_type == "Result" || layer_type == "Const" ||
         layer_type == "Constant" || layer_type == "Input" || layer_type == "Output";
}

std::string info(const std::shared_ptr<ov::Node>& node, const char* key) {
  const auto& rt = node->get_rt_info();
  const auto found = rt.find(key);
  return found == rt.end() ? std::string() : found->second.as<std::string>();
}

// The GPU plugin names a primitive it runs in host code "<name>_cpu" or
// gives it the implementation type "cpu".
bool host_implementation(std::string implementation) {
  std::ranges::transform(implementation, implementation.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return implementation == "cpu" || implementation.starts_with("cpu:") || implementation.ends_with("_cpu") ||
         implementation.find("cpu_impl") != std::string::npos;
}

std::string join(const std::vector<std::string>& values) {
  std::string out;
  for (const auto& value : values) out += (out.empty() ? "" : ", ") + value;
  return out.empty() ? "none" : out;
}

struct Compiled {
  uint32_t width, height;
  ov::InferRequest request;
  EncoderPlacement placement;
};

}  // namespace

std::string EncoderPlacement::summary() const {
  std::string out = std::to_string(width) + "x" + std::to_string(height) + ": " + std::to_string(device_nodes) +
                    " nodes on " + join(execution_devices);
  if (!host_nodes.empty()) out += "; NOT on the device: " + join(host_nodes);
  return out;
}

struct OpenvinoEncoder::Impl {
  EncoderOptions options;
  ov::Core core;
  std::shared_ptr<ov::Model> model;
  uint32_t stride = 0, out_hidden = 0;
  std::list<Compiled> compiled;  // most recently used first
  uint64_t compilations = 0;

  Compiled& compiled_for(uint32_t width, uint32_t height);
  EncoderPlacement inspect(const ov::CompiledModel& compiled, uint32_t width, uint32_t height) const;
};

// Reads back where the runtime put the model and throws unless that is the
// configured device alone.
EncoderPlacement OpenvinoEncoder::Impl::inspect(const ov::CompiledModel& compiled, uint32_t width,
                                                uint32_t height) const {
  EncoderPlacement placement{width, height, compiled.get_property(ov::execution_devices), 0, {}};
  const bool only_device = placement.execution_devices.size() == 1 &&
      (placement.execution_devices[0] == options.device || placement.execution_devices[0] == options.device + ".0");
  if (!only_device)
    throw std::runtime_error("vision encoder was placed on " + join(placement.execution_devices) +
                             ", not on OpenVINO device " + options.device + " alone");
  if (options.device.starts_with("GPU")) {
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
      const std::string type = info(node, "layerType");
      if (bookkeeping(type)) continue;
      const std::string implementation = info(node, "primitiveType");
      if (host_implementation(implementation)) {
        placement.host_nodes.push_back(type + " " + implementation);
      } else {
        ++placement.device_nodes;
      }
    }
  } else {
    placement.device_nodes = compiled.get_runtime_model()->get_ordered_ops().size();
  }
  if (!placement.clean())
    throw std::runtime_error("vision encoder left OpenVINO device " + options.device + ": " + placement.summary());
  return placement;
}

Compiled& OpenvinoEncoder::Impl::compiled_for(uint32_t width, uint32_t height) {
  const auto hit = std::ranges::find_if(
      compiled, [&](const Compiled& entry) { return entry.width == width && entry.height == height; });
  if (hit != compiled.end()) {
    compiled.splice(compiled.begin(), compiled, hit);
    return compiled.front();
  }
  // A private copy: reshape fixes the page size in place.
  std::shared_ptr<ov::Model> fixed = model->clone();
  fixed->reshape(ov::PartialShape{1, 3, height, width});
  const ov::element::Type precision =
      options.precision == EncoderPrecision::f16 ? ov::element::f16 : ov::element::f32;
  ov::CompiledModel compiled_model =
      core.compile_model(fixed, options.device, ov::hint::inference_precision(precision));
  EncoderPlacement placement = inspect(compiled_model, width, height);
  while (compiled.size() >= std::max<size_t>(options.max_compiled_sizes, 1)) compiled.pop_back();
  compiled.push_front({width, height, compiled_model.create_infer_request(), std::move(placement)});
  ++compilations;
  return compiled.front();
}

OpenvinoEncoder::OpenvinoEncoder(const EncoderOptions& options) : impl_(std::make_unique<Impl>()) {
  impl_->options = options;
  const auto devices = impl_->core.get_available_devices();
  if (std::ranges::find(devices, options.device) == devices.end())
    throw std::runtime_error("vision encoder: OpenVINO device " + options.device +
                             " is not available; available: " + join(devices));
  if (!options.cache_dir.empty()) impl_->core.set_property(ov::cache_dir(options.cache_dir));
  impl_->model = impl_->core.read_model(options.xml_path, options.bin_path);
  const auto& inputs = impl_->model->inputs();
  const auto& outputs = impl_->model->outputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !inputs[0].get_names().contains(kPixelInput) ||
      !outputs[0].get_names().contains(kEmbeddingOutput))
    throw std::runtime_error("vision encoder IR does not have the expected input and output: " + options.xml_path);
  const ov::PartialShape out_shape = outputs[0].get_partial_shape();
  if (out_shape.rank().is_dynamic() || out_shape.size() != 2 || out_shape[1].is_dynamic())
    throw std::runtime_error("vision encoder IR has an unexpected output shape: " + options.xml_path);
  impl_->out_hidden = static_cast<uint32_t>(out_shape[1].get_length());
  if (impl_->model->has_rt_info("lighton", "token_stride"))
    impl_->stride = impl_->model->get_rt_info<uint32_t>("lighton", "token_stride");
  if (impl_->stride == 0) throw std::runtime_error("vision encoder IR lacks its token stride: " + options.xml_path);
}

OpenvinoEncoder::~OpenvinoEncoder() = default;
uint32_t OpenvinoEncoder::token_stride() const { return impl_->stride; }
uint32_t OpenvinoEncoder::out_hidden() const { return impl_->out_hidden; }
uint64_t OpenvinoEncoder::compilations() const { return impl_->compilations; }

const EncoderPlacement& OpenvinoEncoder::placement() const {
  if (impl_->compiled.empty()) throw std::logic_error("vision encoder has not run yet");
  return impl_->compiled.front().placement;
}

std::vector<float> OpenvinoEncoder::encode(std::span<const float> pixels, uint32_t width, uint32_t height) {
  const uint32_t stride = impl_->stride;
  if (width == 0 || height == 0 || width % stride != 0 || height % stride != 0 ||
      pixels.size() != static_cast<size_t>(width) * height * 3)
    throw std::invalid_argument("vision encoder page size is not a whole number of tokens");
  Compiled& entry = impl_->compiled_for(width, height);
  ov::Tensor input = entry.request.get_input_tensor();
  // Interleaved RGB to one plane per channel.
  float* planes = input.data<float>();
  const size_t area = static_cast<size_t>(width) * height;
  for (size_t p = 0; p < area; ++p)
    for (size_t c = 0; c < 3; ++c) planes[c * area + p] = pixels[p * 3 + c];
  entry.request.infer();
  const ov::Tensor output = entry.request.get_output_tensor();
  const size_t tokens = static_cast<size_t>(width / stride) * (height / stride);
  if (output.get_element_type() != ov::element::f32 || output.get_size() != tokens * impl_->out_hidden)
    throw std::runtime_error("vision encoder returned an unexpected embedding shape");
  const float* values = output.data<const float>();
  return std::vector<float>(values, values + output.get_size());
}

}  // namespace lighton::vision
