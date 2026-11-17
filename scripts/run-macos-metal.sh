#!/bin/bash
set -euo pipefail
test "$(uname -s)" = Darwin && test "$(uname -m)" = arm64 || exit 1
test "$#" = 1 || { echo "usage: $0 READ_ONLY_MODEL_DIRECTORY" >&2; exit 64; }
task_models=$(cd "$1" && pwd)
for artifact in model-bf16.gguf vision-bf16.gguf; do
  test -r "$task_models/$artifact" && test ! -w "$task_models/$artifact" || {
    echo "Model must exist and be read-only: $task_models/$artifact" >&2; exit 1;
  }
done
task_root=$(cd "$(dirname "$0")/.." && pwd)
export GRPC_LIGHTON_BACKEND=metal
export GRPC_LIGHTON_MODELS_DIR="$task_models"
export GRPC_LIGHTON_LISTEN_ADDRESS=127.0.0.1:50074
exec nice -n 10 "$task_root/build-metal/grpc-lighton-ocr-server"
