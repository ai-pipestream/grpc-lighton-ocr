#include "engine/placement.h"

#include <stdexcept>

#include "ggml-backend.h"
#include "ggml.h"

namespace lighton {
namespace {

// Ops that only rename memory. No backend computes anything for them.
bool layout_only(ggml_op op) {
  return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
         op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

const ggml_tensor* storage(const ggml_tensor* tensor) {
  return tensor->view_src != nullptr ? tensor->view_src : tensor;
}

void add(std::map<std::string, PlacementReport::Stray>& strays, std::string key, const ggml_tensor* tensor) {
  auto& stray = strays[std::move(key)];
  if (stray.count++ == 0) stray.first_tensor = tensor->name;
}

std::string buffer_type_name(ggml_backend_buffer_t buffer) {
  return buffer == nullptr ? "no buffer" : ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void append(std::string& out, const std::map<std::string, PlacementReport::Stray>& strays) {
  for (const auto& [key, stray] : strays) {
    out += "; " + key + " x" + std::to_string(stray.count);
    if (!stray.first_tensor.empty()) out += " (first: " + stray.first_tensor + ")";
  }
}

}  // namespace

std::string PlacementReport::summary() const {
  std::string out = std::to_string(device_nodes) + " compute nodes and " +
                    std::to_string(device_weights) + " weight tensors on the device";
  if (clean()) return out;
  out += "; NOT on the device";
  append(out, stray_nodes);
  append(out, stray_weights);
  return out;
}

PlacementRecorder::PlacementRecorder(ggml_backend_device* device)
    : device_(device), device_buft_(device == nullptr ? nullptr : ggml_backend_dev_buffer_type(device)) {
  if (device_buft_ == nullptr) throw std::invalid_argument("placement recorder needs a device with a buffer type");
}

bool PlacementRecorder::on_eval(ggml_tensor* node, bool ask, void* recorder) {
  if (ask) static_cast<PlacementRecorder*>(recorder)->observe(node);
  // false on the question: this node's data is not needed. The call after a
  // compute step, which that answer rules out, would mean "keep going".
  return !ask;
}

void PlacementRecorder::observe(const ggml_tensor* node) {
  if (layout_only(node->op)) return;
  for (const ggml_tensor* source : node->src) {
    if (source == nullptr) continue;
    const ggml_tensor* base = storage(source);
    if (base->op == GGML_OP_NONE && base->buffer != nullptr &&
        ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
      observe_weight(base);
  }
  if (ggml_backend_dev_supports_op(device_, node)) {
    ++report_.device_nodes;
  } else {
    add(report_.stray_nodes, std::string(ggml_op_desc(node)) + " (the device cannot run it)", node);
  }
}

void PlacementRecorder::observe_weight(const ggml_tensor* weight) {
  if (!weights_seen_.insert(weight).second) return;
  if (ggml_backend_buffer_get_type(weight->buffer) == device_buft_) {
    ++report_.device_weights;
    return;
  }
  add(report_.stray_weights, "weights in " + buffer_type_name(weight->buffer), weight);
}

PlacementReport PlacementRecorder::take() {
  PlacementReport out = std::move(report_);
  report_ = {};
  weights_seen_.clear();
  return out;
}

}  // namespace lighton
