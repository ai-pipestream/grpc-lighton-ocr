#!/usr/bin/env bash
# Intel GPU qualification for the Dockerfile.openvino image: proves the
# service runs on the GPU it was told to use and that a missing or vague
# device stops startup instead of running on the CPU.
#
#   scripts/test-openvino-gpu.sh IMAGE MODEL_DIRECTORY
#
# Needs a host with an Intel GPU render node (/dev/dri/renderD*), docker,
# grpcurl and python3. MODEL_DIRECTORY holds model-f16.gguf and
# vision-f16.gguf (tools/prepare_model.py ... f16). Without a render node or
# without the F16 artifacts the test FAILS; it never skips, so a host that
# lost its GPU cannot pass. LIGHTON_RENDER_GID is the host gid of the render
# node (default: read with stat).
#
# Checks, in order:
#   1. no /dev/dri in the container: startup must fail and name the missing
#      OpenVINO device (no CPU engine is built);
#   2. a routing device or a device list (AUTO, HETERO): startup must fail;
#   3. a GPU index the host does not have: startup must fail;
#   4. with the GPU: the startup log must carry the device line and the
#      placement receipt of the startup probe, GetServiceInfo must report the
#      device, the real-model RPC suite (tests/native_rpc_test.py) must pass,
#      a page must advance the server's own xe compute-engine cycles
#      (drm-cycles-ccs in /proc/1/fdinfo), and the log must hold no
#      placement violation afterwards. The GPU memory the server holds is
#      printed after startup and after the suite.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 IMAGE MODEL_DIRECTORY" >&2; exit 64; }
image=$1
models=$(realpath "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
server=lighton-openvino-gpu-$$
cleanup() {
  if docker inspect "$server" >/dev/null 2>&1; then
    echo "== server log (tail)"
    docker logs "$server" 2>&1 | tail -40 || true
    docker rm -f "$server" >/dev/null 2>&1 || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT
fail() { echo "openvino-gpu: FAIL: $*" >&2; exit 1; }

render=$(ls /dev/dri/renderD* 2>/dev/null | head -1 || true)
[[ -n "$render" ]] || fail "no /dev/dri/renderD* on this host; this test needs the Intel GPU"
for artifact in model-f16.gguf vision-f16.gguf; do
  [[ -f "$models/$artifact" ]] || fail "$models/$artifact is missing (tools/prepare_model.py ... f16)"
done
gid=${LIGHTON_RENDER_GID:-$(stat -c %g "$render")}
gpu_args=(--device /dev/dri:/dev/dri --group-add "$gid")
run_args=(--read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges -v "$models:/models:ro")

# A startup that must fail: exits non-zero within the timeout, never reaches
# its listener, and its log carries the expected text.
expect_startup_failure() {
  local label=$1 text=$2; shift 2
  echo "== $label: startup must fail"
  set +e
  timeout 300 docker run --rm "${run_args[@]}" "$@" "$image" > "$work/fail.log" 2>&1
  local code=$?
  set -e
  grep -E "GRPC_LIGHTON_DEVICE|not available" "$work/fail.log" | head -3 || true
  [[ "$code" != 0 && "$code" != 124 ]] || fail "$label: startup exit $code, wanted a failure"
  ! grep -q "listening on" "$work/fail.log" || fail "$label: the server reached its listener"
  grep -qF -- "$text" "$work/fail.log" || fail "$label: log lacks '$text'"
  echo "failed as required (exit $code)"
}

expect_startup_failure "no GPU device" "OpenVINO device GPU is not available"
expect_startup_failure "AUTO device list" "routing device" \
  "${gpu_args[@]}" -e GRPC_LIGHTON_DEVICE=openvino:AUTO:GPU,CPU
expect_startup_failure "HETERO device list" "routing device" \
  "${gpu_args[@]}" -e GRPC_LIGHTON_DEVICE=openvino:HETERO:GPU,CPU
expect_startup_failure "absent GPU index" "OpenVINO device GPU.9 is not available" \
  "${gpu_args[@]}" -e GRPC_LIGHTON_DEVICE=openvino:GPU.9

ccs_cycles() {
  docker exec "$server" sh -c \
    'total=0; for f in /proc/1/fdinfo/*; do
       v=$(grep -s "^drm-cycles-ccs:" "$f" | awk "{print \$2}");
       [ -n "$v" ] && total=$((total + v));
     done; echo $total'
}

# GPU memory the server process holds, from the xe driver's per-client
# accounting. One DRM client can be open on several descriptors; each client
# is counted once.
gpu_memory() {
  docker exec "$server" sh -c 'cat /proc/1/fdinfo/* 2>/dev/null' | awk '
    /^drm-client-id:/ { client = $2; fresh = !(client in clients); clients[client] = 1 }
    /^drm-(total|resident)-/ && fresh { sub(":", "", $1); kib[$1] += $2 }
    END { for (key in kib) printf "%s=%dMiB ", key, kib[key] / 1024; print "" }'
}

echo "== GPU boot: device line, placement receipt, real-model RPC"
docker run -d --name "$server" "${run_args[@]}" "${gpu_args[@]}" \
  -e GRPC_LIGHTON_DEVICE=openvino:GPU -p 127.0.0.1::50074 "$image" >/dev/null
# The first boot compiles every graph for the GPU before it listens.
for _ in $(seq 1 900); do
  docker logs "$server" 2>&1 | grep -q "listening on" && break
  [[ "$(docker inspect -f '{{.State.Running}}' "$server")" == true ]] || fail "server exited during startup"
  sleep 1
done
docker logs "$server" > "$work/boot.log" 2>&1
grep -q "listening on" "$work/boot.log" || fail "server did not reach its listener in 900 s"
grep -E "^grpc-lighton-ocr device: " "$work/boot.log" || fail "no device line in the startup log"
grep -qE "^grpc-lighton-ocr device: openvino:GPU - .+ startup placement probe: vision: [1-9][0-9]* compute nodes .+ decoder: [1-9][0-9]* compute nodes .+, 0 nodes computed by the ggml CPU backend" \
  "$work/boot.log" || fail "the startup log has no clean placement receipt for the GPU"
! grep -q "NOT on the device" "$work/boot.log" || fail "the startup probe saw nodes off the GPU"

echo "GPU memory after startup (text model, vision projector, compiled probe graphs): $(gpu_memory)"

address=$(docker port "$server" 50074/tcp)
before=$(ccs_cycles)
LIGHTON_RPC_ADDRESS="$address" LIGHTON_EXPECT_DEVICE=openvino:GPU LIGHTON_BUSY_PAGES=8 \
  python3 "$root/tests/native_rpc_test.py" unused "$models"
after=$(ccs_cycles)
echo "xe compute-engine cycles: $before -> $after"
echo "GPU memory after the RPC suite: $(gpu_memory)"
[[ "$after" -gt "$before" ]] || fail "the GPU compute engine did no work during inference"
! docker logs "$server" 2>&1 | grep -q "placement violation" || fail "inference left the GPU"

docker stop -t 7 "$server" >/dev/null
[[ "$(docker wait "$server")" == 0 ]] || fail "server did not exit cleanly on SIGTERM"
docker rm -f "$server" >/dev/null 2>&1 || true
echo "openvino-gpu: PASS ($image)"
