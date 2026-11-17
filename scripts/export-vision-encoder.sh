#!/usr/bin/env bash
# Model preparation for OpenVINO devices: exports the vision encoder of
# vision-f16.gguf to the OpenVINO IR pair the server loads, and checks both
# files against models/SHA256SUMS. Runs the export tool from the build stage
# of Dockerfile.openvino; the runtime image does not contain it.
#
#   docker build -f Dockerfile.openvino --target build -t grpc-lighton-ocr:openvino-build .
#   scripts/export-vision-encoder.sh grpc-lighton-ocr:openvino-build MODEL_DIRECTORY
#
# MODEL_DIRECTORY must hold vision-f16.gguf (tools/prepare_model.py ... f16)
# and be writable; vision-encoder-f16.xml and .bin are written into it.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 BUILD_STAGE_IMAGE MODEL_DIRECTORY" >&2; exit 64; }
image=$1
models=$(realpath "$2")
root=$(cd "$(dirname "$0")/.." && pwd)
[[ -f "$models/vision-f16.gguf" ]] || { echo "$models/vision-f16.gguf is missing" >&2; exit 1; }
grep -E ' vision-f16\.gguf$' "$root/models/SHA256SUMS" | (cd "$models" && sha256sum -c -)
docker run --rm --user "$(id -u):$(id -g)" -v "$models:/models" "$image" \
  /out/tools/export_vision_encoder /models/vision-f16.gguf /models/vision-encoder-f16.xml
grep -E ' vision-encoder-f16\.(xml|bin)$' "$root/models/SHA256SUMS" | (cd "$models" && sha256sum -c -)
echo "exported and verified: $models/vision-encoder-f16.xml, $models/vision-encoder-f16.bin"
