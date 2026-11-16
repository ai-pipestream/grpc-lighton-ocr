#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>

struct ggml_tensor;
struct ggml_backend_device;
struct ggml_backend_buffer_type;

namespace lighton {

// Where the scheduler put one stage of inference.
struct PlacementReport {
  struct Stray {
    uint64_t count = 0;
    std::string first_tensor;  // name of the first tensor seen for this key
  };
  // Compute nodes the device runs.
  uint64_t device_nodes = 0;
  // Distinct weight tensors read from the device buffer type.
  uint64_t device_weights = 0;
  // Compute nodes the device does not run, keyed "<op> (<reason>)".
  std::map<std::string, Stray> stray_nodes;
  // Weight tensors elsewhere, keyed "weights in <buffer type>".
  std::map<std::string, Stray> stray_weights;

  bool clean() const { return stray_nodes.empty() && stray_weights.empty(); }
  // One line for a log or an error: counts, then every stray key.
  std::string summary() const;
};

// Watches the graphs the ggml scheduler runs for one selected device.
//
// For a compute node it asks the device what the scheduler asks when it
// places the node: can you run this op. The scheduler gives a node to its
// first backend that says yes, the device is that first backend, and the CPU
// backend is the last, so a node the device turns down is a node the CPU
// computes. (The scheduler pins graph inputs to the CPU backend, but ggml
// only lets leaf tensors be inputs, and a leaf computes nothing.)
//
// A node's own buffer type cannot answer this. The allocator lets a node
// reuse its parent's memory, so a node the device computes can live in the
// host buffer of a graph input.
//
// Weights are checked by buffer type, which is exact for them: every weight
// tensor a node reads must be in the device's buffer type.
// Not thread safe: one recorder serves one inference at a time.
class PlacementRecorder {
 public:
  explicit PlacementRecorder(ggml_backend_device* device);

  // ggml_backend_sched_eval_callback with a PlacementRecorder as user data.
  // It records on the scheduler's question and always answers "not needed",
  // so every split still reaches its backend as one graph.
  static bool on_eval(ggml_tensor* node, bool ask, void* recorder);

  void observe(const ggml_tensor* node);
  // Returns what was recorded since the last call and starts over.
  PlacementReport take();

 private:
  void observe_weight(const ggml_tensor* weight);

  ggml_backend_device* device_;
  ggml_backend_buffer_type* device_buft_;
  PlacementReport report_;
  std::unordered_set<const ggml_tensor*> weights_seen_;
};

}  // namespace lighton
