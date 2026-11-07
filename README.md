# grpc-lighton-ocr

A diskless streaming gRPC OCR service using native libllama/libmtmd and the
LightOnOCR-3-0.8B checkpoint on CPU. Request images stay in RAM; inference uses
no Python, subprocess, HTTP model server, or writable request directory.

Only **CPU 0.8B** is enabled. The 4B enum remains on the wire but production
startup rejects it. Table markup is retained as text; typed table extraction
and continuation-block joining are unfinished. gRParse does not dial this
service yet. Setting `GRPARSE_LIGHTON_OCR_TARGET` does not enable a route.

## Build and models

Requires CMake 3.25+, a C++23 compiler, and Git. gRPC is pinned to v1.83.0;
llama.cpp is pinned to `f3586e66bc916d672773d4a6505513ba91a3595c`.
The CPU backend is static, without native CPU tuning, dynamic backend plugins,
video subprocess helpers, or HTTP inference support. The x86 build uses the
upstream fixed AVX2 baseline; ARM uses its own architecture-specific kernels.
ARM inference qualification is not claimed by the x86 fixture receipt.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Weights remain external. `models/source-manifest.json` records the upstream
Apache-2.0 checkpoint revision and every source file hash. `models/SHA256SUMS`
records the converted BF16 text and vision artifacts. Startup verifies both
hashes, loads both artifacts, then opens the listener. Missing, mismatched, or
unloadable production artifacts fail startup.

For offline preparation, use a Python environment with CPU Torch 2.11.0 and
Transformers 5.5.4 and the pinned llama.cpp checkout:

```sh
python tools/prepare_model.py /path/to/pinned/llama.cpp /path/to/models
```

Python is only used for offline conversion and test clients. The service never
runs the preparation script. Mount the resulting directory read-only:

```sh
docker build -t grpc-lighton-ocr:local .
docker run --rm --read-only --cap-drop ALL -p 50074:50074 \
  -v /absolute/path/to/models:/models:ro grpc-lighton-ocr:local
```

## Readiness and limits

The service name is `ai.pipestream.lightonocr.v1.LightOnOcrService`.
Check that name with gRPC Health/Check for inference readiness. The empty
service name is transport liveness. `GetServiceInfo.engine_loaded` agrees
with named readiness. `GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE=1` explicitly boots
without models for transport diagnostics; named health stays NOT_SERVING and
pages fail ENGINE_UNAVAILABLE. This is not an inference mode.

Limits are four active document streams and one active inference, with no
inference queue; 256 pages; 32 MiB compressed raster; 4 million decoded pixels
and at most 8192 per dimension; 4096 context tokens; 2048 vision tokens;
1024 generated tokens maximum (512 default); 1 MiB generated text; 16 MiB
folded document. PNG/JPEG dimensions are checked before raster allocation.
Zero temperature (including omitted) selects greedy decoding; positive
temperatures must be finite and within 0.01..2.
Sampling uses seed zero. A decode has a 60-second deadline and checks client
cancellation in vision graph evaluation and decoder operations. Shutdown
marks health non-serving and gives active RPCs five seconds before cancellation.

Partial or malformed output remains in the typed failure generation trace, with a
finish reason and recoverable partial document where available.

Successful pages carry typed artifact hashes, revisions, device and service
version, plus page boxes and a token generation trace. Failures remain typed;
clients must inspect page failures and completion counts even with OK status.

## Qualification

Model-free unit tests use diagnostic mode. They do not prove OCR.
The real-model regression requires grpcurl, Python, and the pinned artifacts:

```sh
python3 tests/native_rpc_test.py build/grpc-lighton-ocr-server /path/to/models
scripts/test-native-image.sh grpc-lighton-ocr:local /path/to/models
```

It checks expected invoice text, valid positive page boxes (not exact generated
coordinates), provenance, generation traces, two-page folded references,
resource limits, concurrent admission, loaded health, cancellation and recovery.
The image test uses a read-only root and read-only model mount. Missing weights
are failures, not skipped tests. `scripts/smoke-test.sh` checks only diagnostic
transport boot, library closure and shutdown. See `docs/native-proof.md` for
the original standalone checkpoint proof, which is separate from RPC/image proof.

## Contract and licence

The contract is `proto/ai/pipestream/lightonocr/v1/lighton_ocr.proto`.
The shared document and OpenNLP protos remain byte-identical to the fleet copy.
`buf lint` and `buf format` run in CI. Apache-2.0.
