// Placement accounting against real ggml buffers and the real ggml scheduler.
// The CPU build checks the weight rule and the scheduler wiring. The OpenVINO build also runs
// a graph with one op the OpenVINO backend cannot translate and requires the
// recorder to name it, on the OpenVINO CPU device (no GPU needed).
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/device.h"
#include "engine/placement.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace {

using lighton::PlacementRecorder;
using lighton::PlacementReport;

int failures = 0;

void expect(bool ok, const std::string& message) {
  if (ok) return;
  std::cerr << "FAIL " << message << '\n';
  failures++;
}

ggml_context* metadata_context() {
  return ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true});
}

ggml_tensor* vector4(ggml_context* ctx, const char* name) {
  return ggml_set_name(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4), name);
}

void place(ggml_backend_buffer_t buffer, ggml_tensor* tensor, size_t slot) {
  auto* base = static_cast<char*>(ggml_backend_buffer_get_base(buffer));
  if (ggml_backend_tensor_alloc(buffer, tensor, base + slot * 64) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("test tensor placement failed");
}

bool near(const std::array<float, 4>& got, const std::array<float, 4>& want) {
  for (size_t i = 0; i < 4; ++i)
    if (std::fabs(got[i] - want[i]) > 1e-4f) return false;
  return true;
}

// Hand-placed tensors: the weight rule and the layout-only ops, without a
// scheduler. The CPU device stands in for the accelerator; it runs every op.
void classification_rules() {
  ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  ggml_backend_buffer_type_t device_buft = ggml_backend_dev_buffer_type(device);
  ggml_backend_buffer_t on_device = ggml_backend_buft_alloc_buffer(device_buft, 1024);
  ggml_backend_buffer_t device_weights = ggml_backend_buft_alloc_buffer(device_buft, 1024);
  ggml_backend_buffer_set_usage(device_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  // A second buffer type stands in for "not the device".
  alignas(64) static char elsewhere_memory[1024], stray_weight_memory[1024];
  ggml_backend_buffer_t elsewhere = ggml_backend_cpu_buffer_from_ptr(elsewhere_memory, sizeof(elsewhere_memory));
  ggml_backend_buffer_t stray_weights =
      ggml_backend_cpu_buffer_from_ptr(stray_weight_memory, sizeof(stray_weight_memory));
  ggml_backend_buffer_set_usage(stray_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  const std::string other = ggml_backend_buft_name(ggml_backend_buffer_get_type(elsewhere));
  expect(ggml_backend_buffer_get_type(elsewhere) != device_buft, "the stand-in buffer type differs from the device's");

  ggml_context* ctx = metadata_context();
  ggml_tensor* input = vector4(ctx, "input");
  ggml_tensor* w_device = vector4(ctx, "w_device");
  ggml_tensor* w_host = vector4(ctx, "w_host");
  place(on_device, input, 0);
  place(device_weights, w_device, 0);
  place(stray_weights, w_host, 0);

  ggml_tensor* good = ggml_set_name(ggml_add(ctx, input, w_device), "good");
  place(on_device, good, 1);
  // A node the device computes may reuse host memory; that is no stray.
  ggml_tensor* reused = ggml_set_name(ggml_sqr(ctx, good), "reused");
  place(elsewhere, reused, 1);
  ggml_tensor* reads_host = ggml_set_name(ggml_mul(ctx, reused, w_host), "reads_host");
  place(on_device, reads_host, 2);
  ggml_tensor* view = ggml_set_name(ggml_reshape_2d(ctx, reads_host, 2, 2), "view");
  ggml_backend_view_init(view);

  PlacementRecorder recorder(device);
  for (ggml_tensor* node : {good, reused, reads_host, view})
    expect(!PlacementRecorder::on_eval(node, true, &recorder), "the recorder does not ask for per-node data");
  expect(PlacementRecorder::on_eval(good, false, &recorder), "an after-compute call continues");
  recorder.observe(reads_host);  // nodes count per visit, weights once per tensor

  const PlacementReport report = recorder.take();
  expect(report.device_nodes == 4, "good, reused and reads_host twice run on the device, the view does not count, got " +
                                       std::to_string(report.device_nodes));
  expect(report.device_weights == 1, "one weight tensor on the device");
  expect(report.stray_nodes.empty(), "a device that runs every op has no stray node: " + report.summary());
  expect(!report.clean(), "a stray weight makes the report unclean");
  const auto weight = report.stray_weights.find("weights in " + other);
  expect(report.stray_weights.size() == 1 && weight != report.stray_weights.end() &&
             weight->second.count == 1 && weight->second.first_tensor == "w_host",
         "a weight off the device is reported once, by name");
  const std::string summary = report.summary();
  for (const char* part : {"NOT on the device", "weights in ", "x1 (first: w_host)", "4 compute nodes"})
    expect(summary.find(part) != std::string::npos, std::string("summary has '") + part + "': " + summary);

  const PlacementReport empty = recorder.take();
  expect(empty.clean() && empty.device_nodes == 0 && empty.device_weights == 0, "take starts a new report");
  expect(empty.summary().find("NOT") == std::string::npos, "a clean summary reports no stray");

  bool rejected = false;
  try { PlacementRecorder none(nullptr); } catch (const std::invalid_argument&) { rejected = true; }
  expect(rejected, "a recorder without a device is an error");

  ggml_free(ctx);
  for (ggml_backend_buffer_t buffer : {on_device, device_weights, elsewhere, stray_weights})
    ggml_backend_buffer_free(buffer);
}

struct ScheduledRun {
  PlacementReport report;
  std::array<float, 4> output{};
  int splits = 0;
};

// out = scale(act(input + weight), 2), with the weight in the first backend's
// memory and the recorder installed the way the engine installs it.
ScheduledRun run_scheduled(std::vector<ggml_backend_t> backends, ggml_backend_dev_t device, bool leaky) {
  ggml_context* weights_ctx = metadata_context();
  ggml_tensor* weight = vector4(weights_ctx, "weight");
  ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(weights_ctx, backends.front());
  ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  const std::array<float, 4> ones{1, 1, 1, 1};
  ggml_backend_tensor_set(weight, ones.data(), 0, sizeof(ones));

  ggml_context* ctx = metadata_context();
  ggml_tensor* input = vector4(ctx, "input");
  ggml_set_input(input);
  ggml_tensor* sum = ggml_set_name(ggml_add(ctx, input, weight), "sum");
  ggml_tensor* act = ggml_set_name(leaky ? ggml_leaky_relu(ctx, sum, 0.1f, false) : ggml_sqr(ctx, sum), "act");
  ggml_tensor* out = ggml_set_name(ggml_scale(ctx, ggml_reshape_2d(ctx, act, 2, 2), 2.0f), "out");
  ggml_set_output(out);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, out);

  ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()),
                                                      GGML_DEFAULT_GRAPH_SIZE, false, true);
  PlacementRecorder recorder(device);
  ggml_backend_sched_set_eval_callback(sched, PlacementRecorder::on_eval, &recorder);
  ScheduledRun run;
  if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("test graph allocation failed");
  const std::array<float, 4> values{1, -2, 3, -4};
  ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
  if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("test graph compute failed");
  ggml_backend_tensor_get(out, run.output.data(), 0, sizeof(run.output));
  run.splits = ggml_backend_sched_get_n_splits(sched);
  run.report = recorder.take();

  ggml_backend_sched_free(sched);
  ggml_free(ctx);
  ggml_backend_buffer_free(weights);
  ggml_free(weights_ctx);
  return run;
}

// The scheduler drives the recorder; the device is the only backend.
void scheduler_on_one_backend() {
  ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
  const ScheduledRun run = run_scheduled({backend}, device, true);
  expect(near(run.output, {4.0f, -0.2f, 8.0f, -0.6f}), "the graph still computes whole and correctly");
  expect(run.report.clean(), "one backend keeps every node on the device: " + run.report.summary());
  expect(run.report.device_nodes == 3, "ADD, LEAKY_RELU and SCALE are the compute nodes, got " +
                                           std::to_string(run.report.device_nodes));
  expect(run.report.device_weights == 1, "the weight is read from the device");
  ggml_backend_free(backend);
}

#ifdef LIGHTON_OPENVINO
// The real failure this service must not hide: ggml's scheduler moves an op
// the accelerator cannot run to the CPU backend and reports success.
void openvino_fallback_is_named() {
  bool absent = false;
  try { lighton::select_device(lighton::parse_device("openvino:GPU.9")); }
  catch (const std::runtime_error& error) {
    const std::string what = error.what();
    absent = what.find("OpenVINO device GPU.9 is not available") != std::string::npos &&
             what.find("available: ") != std::string::npos;
  }
  expect(absent, "an absent OpenVINO device is a clean error that lists the available ones");

  const lighton::DeviceSpec spec = lighton::parse_device("openvino:CPU");
  ggml_backend_dev_t device = lighton::select_device(spec);
  expect(device != nullptr, "the OpenVINO CPU device is selected");
  expect(lighton::describe_device(spec, device).starts_with("openvino:CPU - "),
         "the reported device carries the runtime's name: " + lighton::describe_device(spec, device));
  std::string conflict;
  try { lighton::select_device(lighton::parse_device("openvino:GPU")); }
  catch (const std::runtime_error& error) { conflict = error.what(); }
  expect(conflict.find("conflicts with GGML_OPENVINO_DEVICE=CPU") != std::string::npos,
         "a different device after one is selected is a conflict: " + conflict);

  ggml_backend_t accelerator = ggml_backend_dev_init(device, nullptr);
  ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
  expect(accelerator != nullptr && cpu != nullptr, "both backends initialize");

  // Every op has an OpenVINO translator: nothing may leave the device.
  const ScheduledRun supported = run_scheduled({accelerator, cpu}, device, false);
  expect(near(supported.output, {8.0f, 2.0f, 32.0f, 18.0f}), "the OpenVINO graph computes correctly");
  expect(supported.report.clean() && supported.report.device_nodes == 3 && supported.report.device_weights == 1,
         "a supported graph stays on the OpenVINO device: " + supported.report.summary());

  // LEAKY_RELU has no OpenVINO translator. The scheduler gives it to the CPU
  // backend and the result is still right, which is why it must be reported.
  const ScheduledRun fallback = run_scheduled({accelerator, cpu}, device, true);
  expect(near(fallback.output, {4.0f, -0.2f, 8.0f, -0.6f}), "the fallback graph still computes correctly");
  expect(fallback.splits > 1, "the scheduler split the graph across backends");
  expect(!fallback.report.clean(), "a CPU fallback makes the report unclean");
  bool named = false;
  for (const auto& [key, stray] : fallback.report.stray_nodes)
    named = named || (key == "LEAKY_RELU (the device cannot run it)" && stray.first_tensor == "act");
  expect(named, "the fallback op is named: " + fallback.report.summary());
  expect(fallback.report.device_nodes == 2, "ADD and SCALE stayed on the OpenVINO device");

  ggml_backend_free(accelerator);
  ggml_backend_free(cpu);
}
#endif

}  // namespace

int main() {
#ifdef LIGHTON_OPENVINO
  // First: the device selection must precede every other use of ggml.
  openvino_fallback_is_named();
#endif
  classification_rules();
  scheduler_on_one_backend();
  if (failures != 0) return 1;
  std::cout << "placement_test: OK\n";
  return 0;
}
