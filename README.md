# grpc-lighton-ocr

A diskless streaming gRPC OCR service using native libllama/libmtmd and the
LightOnOCR-3-0.8B checkpoint. Request images stay in RAM; inference uses
no Python, subprocess, HTTP model server, or writable request directory.

Only **0.8B** is enabled: on CPU by default, or on one Intel GPU with the
OpenVINO image (see [Intel GPU](#intel-gpu-openvino)). The 4B enum remains on
the wire but production startup rejects it. Table markup is retained as text; typed table extraction
and continuation-block joining are unfinished. gRParse does not dial this
service yet. Setting `GRPARSE_LIGHTON_OCR_TARGET` does not enable a route.

## Build and models

Requires CMake 3.25+, a C++23 compiler, and Git. gRPC is pinned to v1.83.0;
llama.cpp is pinned to `f3586e66bc916d672773d4a6505513ba91a3595c`.
The CPU backend is static, without native CPU tuning, dynamic backend plugins,
video subprocess helpers, or HTTP inference support. `-DLIGHTON_OPENVINO=ON`
adds the OpenVINO backend and needs the OpenVINO runtime and OpenCL
development files; it is OFF by default. The x86 build uses the
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
python tools/prepare_model.py /path/to/pinned/llama.cpp /path/to/models f16
```

The first line writes the BF16 pair the CPU backend loads. The second writes
`model-f16.gguf` and `vision-f16.gguf` from the same source files, for OpenVINO
devices only. Both pairs are pinned in `models/SHA256SUMS`.

Python is only used for offline conversion and test clients. The service never
runs the preparation script. Mount the resulting directory read-only:

```sh
docker build -t grpc-lighton-ocr:local .
docker run --rm --read-only --cap-drop ALL -p 50074:50074 \
  -v /absolute/path/to/models:/models:ro grpc-lighton-ocr:local
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GRPC_LIGHTON_LISTEN_ADDRESS` | `0.0.0.0:50074` | gRPC listen address |
| `GRPC_LIGHTON_MODELS_DIR` | `/models` | directory with the converted artifacts |
| `GRPC_LIGHTON_MODEL` | `0.8b` | checkpoint; `4b` is rejected |
| `GRPC_LIGHTON_DEVICE` | `cpu` | `cpu`, `openvino:GPU`, `openvino:GPU.<n>` or `openvino:CPU` |
| `GRPC_LIGHTON_THREADS` | `2` | host threads for the ggml CPU backend, 1..256 |
| `GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE` | `0` | `1` boots without an engine |

An unknown value in any of them fails startup. `GRPC_LIGHTON_DEVICE` names one
device. Routing devices and lists (`AUTO`, `HETERO`, `MULTI`, `BATCH`,
`GPU,CPU`) are rejected, because they let the runtime move work to the CPU
without reporting it.

## Intel GPU (OpenVINO)

`Dockerfile.openvino` builds the server with `-DLIGHTON_OPENVINO=ON`, which
adds llama.cpp's ggml OpenVINO backend (OpenVINO 2026.4.1 and the Intel GPU
compute runtime, linux/amd64). The pinned backend is written against the 2026
API and does not compile with OpenVINO 2025.4.1. The default build and `Dockerfile` stay CPU
only. The image defaults to `GRPC_LIGHTON_DEVICE=openvino:GPU` and loads the
F16 artifacts; the OpenVINO GPU plugin does not take BF16 weights.

```sh
docker build -f Dockerfile.openvino -t grpc-lighton-ocr:openvino .
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" -p 50074:50074 \
  -v /absolute/path/to/models:/models:ro grpc-lighton-ocr:openvino
```

The image runs as the numeric user 65532, so the render node is opened through
the host's render group (`--group-add`). On a host with several GPUs, name one:
`openvino:GPU.1`.

The device is a requirement, not a preference:

- All layers of the text model, its token embedding and the vision projector
  are loaded into the device's buffer type. llama.cpp is given exactly one
  device (`params.devices`) and offloads every layer.
- Startup fails when the named device is absent, and the error lists the
  devices OpenVINO does offer. A CPU-only binary given an OpenVINO device also
  fails; nothing is served from the CPU instead.
- ggml's scheduler gives an op the accelerator cannot run to its CPU backend
  and still reports success. The server watches both graphs (vision and
  decoder) through the scheduler's callback. For each compute node it asks
  the device what the scheduler asks when it places the node
  (`ggml_backend_dev_supports_op`), it requires each weight tensor to be in
  the device's buffer type, and for the decoder it counts the nodes the ggml
  CPU backend computed, which must be zero. After loading it runs a
  compiled-in reference page (the invoice fixture) through the vision
  encoder, the prefill and the decoder, and fails startup with the op names
  on any miss. The receipt is in the startup log: `grpc-lighton-ocr device: openvino:GPU - <device name>,
  ...; startup placement probe: vision: N compute nodes ... | decoder: N
  compute nodes ..., 0 nodes computed by the ggml CPU backend`.
- Placement is not correctness, so the same startup run must also read the
  reference page: the output has to end normally and contain the fixture's
  words. A device that computes a wrong answer stops startup with the text it
  read.
- The placement check runs again on every page. A page whose graph leaves the device
  fails as `PAGE_FAILURE_CODE_INFERENCE` with the op names, and the log gets a
  `placement violation` line.
- `GetServiceInfo` and every page report the device in
  `InferenceProvenance.device`, with the name the runtime gives it, and the
  hashes of the F16 artifacts that were loaded.

**State of the pinned backend (llama.cpp f3586e66).** Measured on the
OpenVINO CPU device, which uses the same graph translation as the GPU; not
yet measured on a GPU:

- The text model runs entirely on OpenVINO (25 of 25 layers, 320 weight
  tensors, no node on the ggml CPU backend) and gives the same output as the
  CPU build for the reference page.
- The vision projector is also placed entirely on OpenVINO (311 compute
  nodes, 154 weight tensors), but the backend computes it wrongly: the
  reference page comes back as unrelated digits. The backend also keeps the
  vision graph it compiled for the first raster size and fails a page of any
  other size. llama.cpp's own OpenVINO notes list image input as work in
  progress.

So an OpenVINO device currently stops at the reference page check and the
image does not serve OCR yet. The build flavour, device selection, placement
check and tests are in place for a backend revision that fixes the vision
path; moving the pin is the remaining step.

The check covers what ggml schedules. How the OpenVINO GPU plugin implements
a primitive inside a graph it accepted is not visible at this layer.
`scripts/test-openvino-gpu.sh` adds an independent signal: the GPU's
compute-engine cycle counter must advance during inference.

On an OpenVINO device the vision encoder runs as one graph, so cancellation
and the deadline are checked before it, after it and between decode tokens,
not inside it. `GRPC_LIGHTON_THREADS` still sizes host-side work.

`openvino:CPU` runs the same backend on OpenVINO's CPU plugin under the same
placement rules. It exists to qualify the OpenVINO path on a host without a
render node; it is not a faster CPU mode. The shipped image carries the GPU
plugin only, with no CPU plugin and no routing plugin, so `openvino:CPU` fails
there as an absent device. The build stage has the CPU plugin:

```sh
docker build -f Dockerfile.openvino --target build -t grpc-lighton-ocr:openvino-build .
docker run --rm -e GRPC_LIGHTON_DEVICE=openvino:CPU -e GRPC_LIGHTON_MODELS_DIR=/models \
  -v /absolute/path/to/models:/models:ro -p 50074:50074 \
  --entrypoint /out/grpc-lighton-ocr-server grpc-lighton-ocr:openvino-build
```

The backend reads its own `GGML_OPENVINO_*` variables from the environment
(see llama.cpp `docs/backend/OPENVINO.md` at the pinned commit), for example
`GGML_OPENVINO_CACHE_DIR` for a compiled-kernel cache. The image creates
`/var/cache/openvino` owned by 65532 for a volume mounted there.
`GGML_OPENVINO_DEVICE` is set by the server from `GRPC_LIGHTON_DEVICE`; a
different value already in the environment fails startup.

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

Blocking reads and writes each have a 30-second transport timeout. Inference
does not use that timer. Timeout cancels the RPC (CANCELLED on the wire) and
releases admission capacity; there is no total document-duration cap.

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
transport boot, library closure and shutdown; it also runs against the
OpenVINO image, without a GPU.

The Intel GPU qualification needs a host with a render node and the F16
artifacts, and fails without either:

```sh
scripts/test-openvino-gpu.sh grpc-lighton-ocr:openvino /path/to/models
```

It requires startup to fail without `/dev/dri`, with `AUTO` or `HETERO`, and
with a GPU index the host does not have. With the GPU it requires the device
line and placement receipt in the startup log, the device in
`GetServiceInfo`, a pass of `tests/native_rpc_test.py`, an advancing xe
compute-engine cycle counter, and no placement violation.

`config_test` covers device and thread parsing. `placement_test` checks the
placement accounting against the real ggml scheduler; in the OpenVINO build it
also runs a graph with an op OpenVINO cannot translate on the OpenVINO CPU
device and requires the fallback to be named. Both run in CTest and in both
image builds. See `docs/native-proof.md` for
the original standalone checkpoint proof, which is separate from RPC/image proof.

## Contract and licence

The contract is `proto/ai/pipestream/lightonocr/v1/lighton_ocr.proto`.
The shared document and OpenNLP protos remain byte-identical to the fleet copy.
`buf lint` and `buf format` run in CI. Apache-2.0. The OpenVINO image adds
distribution packages under Apache-2.0, MIT and BSD licences, listed in
`NOTICE`.
