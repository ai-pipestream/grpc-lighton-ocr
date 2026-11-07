#!/usr/bin/env bash
# Real CPU OCR qualification against the runtime image, not diagnostic boot.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 IMAGE MODEL_DIRECTORY" >&2; exit 64; }
image=$1
model_dir=$(realpath "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
container="lighton-native-test-$$"
cleanup() { docker logs "$container" > /tmp/lighton-native-container.log 2>&1 || true; docker stop -t 7 "$container" >/dev/null 2>&1 || true; docker container remove "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$container" --read-only --cap-drop ALL --security-opt no-new-privileges \
  -v "$model_dir:/models:ro" -p 127.0.0.1::50074 "$image" >/dev/null
address=$(docker port "$container" 50074/tcp)
LIGHTON_RPC_ADDRESS="$address" python3 "$root/tests/native_rpc_test.py" unused "$model_dir"
python3 "$root/tests/shutdown_inference_test.py" "$address" "$container"
docker logs "$container" > /tmp/lighton-native-container.log 2>&1
[[ $(docker wait "$container") == 0 ]]
[[ $(docker inspect -f '{{.HostConfig.ReadonlyRootfs}}' "$container") == true ]]
echo "read-only runtime image inference PASS: $(docker inspect -f '{{.Image}}' "$container")"
