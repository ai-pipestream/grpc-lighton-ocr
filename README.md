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
llama.cpp is pinned to `f3586e66bc916d672773d4a6505513ba91a3595c`, with the
three patches under `patches/llama.cpp` applied at configure time
(`patches/llama.cpp/README.md`).
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

An OpenVINO device needs one more artifact, the vision encoder as an OpenVINO
model, exported from `vision-f16.gguf` by a C++ tool that is built in the
build stage of `Dockerfile.openvino` and is not in the runtime image:

```sh
docker build -f Dockerfile.openvino --target build -t grpc-lighton-ocr:openvino-build .
scripts/export-vision-encoder.sh grpc-lighton-ocr:openvino-build /path/to/models
```

It writes `vision-encoder-f16.xml` and `vision-encoder-f16.bin`, which are
pinned in `models/SHA256SUMS` too; the export is reproducible.

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

`Dockerfile.openvino` builds the server with `-DLIGHTON_OPENVINO=ON`
(OpenVINO 2026.4.1 and the Intel GPU compute runtime, linux/amd64). The
default build and `Dockerfile` stay CPU only. The image defaults to
`GRPC_LIGHTON_DEVICE=openvino:GPU` and loads the F16 artifacts; the OpenVINO
GPU plugin does not take BF16 weights.

```sh
docker build -f Dockerfile.openvino -t grpc-lighton-ocr:openvino .
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" -p 50074:50074 \
  -v /absolute/path/to/models:/models:ro grpc-lighton-ocr:openvino
```

The image runs as the numeric user 65532, so the render node is opened through
the host's render group (`--group-add`). On a host with several GPUs, name one:
`openvino:GPU.1`.

Both stages of a page run on the device, each through the path that is
correct there:

- **Vision encoder**: an OpenVINO model exported once from `vision-f16.gguf`
  (`vision-encoder-f16.xml` and `.bin`), run through the OpenVINO runtime's
  C++ API. mtmd still decodes, resizes and normalizes the page and lays out
  the prompt, so preprocessing is the CPU engine's; the server takes mtmd's
  preprocessed pixels, encodes them on the device and hands the embeddings
  back to mtmd for the decoder. Each page size is compiled once and kept (six
  sizes at a time). Design and measurements: `docs/vision-encoder.md`.
- **Text model**: llama.cpp's ggml OpenVINO backend, all 25 layers and the
  token embedding in the device's buffer type, with exactly one device given
  to llama.cpp (`params.devices`). Two things differ from the CPU engine,
  both forced by the backend on the GPU (`docs/vision-encoder.md`, "End to
  end"): the prompt of a page goes in as one batch, because a page's
  embeddings in several batches gave an incomplete reading; and every page
  compiles the text model's graphs again, because reusing them across pages
  of different sizes crashed the backend. The second costs several seconds a
  page.

The device is a requirement, not a preference:

- Startup fails when the named device is absent, and the error lists the
  devices OpenVINO does offer. A CPU-only binary given an OpenVINO device also
  fails; nothing is served from the CPU instead.
- The encoder is compiled for the named device alone. After compiling a page
  size the server reads back the devices the runtime executes it on, which
  must be exactly that device, and on a GPU it walks the compiled model and
  fails on any node the plugin implements in host code. mtmd's own encoder is
  host code; its scheduler callback counts the nodes it computes, and that
  count must stay zero.
- ggml's scheduler gives an op the accelerator cannot run to its CPU backend
  and still reports success. The server watches the decoder graph through the
  scheduler's callback. For each compute node it asks the device what the
  scheduler asks when it places the node (`ggml_backend_dev_supports_op`), it
  requires each weight tensor to be in the device's buffer type, and it counts
  the nodes the ggml CPU backend computed, which must be zero.
- After loading, a compiled-in reference page (the invoice fixture) runs
  through the encoder, the prefill and the decoder. Any placement miss fails
  startup. Placement is not correctness, so the page must also be read: the
  output has to end normally and contain the fixture's words. A device that
  computes a wrong answer stops startup with the text it read. The receipt is
  in the startup log: `grpc-lighton-ocr device: openvino:GPU - <device name>,
  ...; startup placement probe: vision: 992x288: N nodes on GPU.0, 0 nodes
  computed by the host projector | decoder: N compute nodes ..., 0 nodes
  computed by the ggml CPU backend`.
- The same checks run on every page. A page that leaves the device fails as
  `PAGE_FAILURE_CODE_INFERENCE` with the reason, and the log gets a
  `placement violation` or `vision encoder failure` line.
- `GetServiceInfo` and every page report the device in
  `InferenceProvenance.device`, with the name the runtime gives it, the hashes
  of the F16 artifacts, and the hash of the encoder's weights file in
  `InferenceProvenance.vision_encoder_sha256`.

**Why the encoder is not the ggml graph.** At the pinned revision the ggml
OpenVINO backend computed this encoder wrongly (a `PERMUTE` of a `VIEW` was
translated for one axis order only) and served every page from the graph
compiled for the first page size. Both faults are fixed by carried patches
with a regression test (`patches/llama.cpp`, `tests/openvino_backend_test.cpp`,
write-up for the backend's maintainers in `docs/upstream/ggml-openvino-vision.md`),
and with them the ggml encoder agrees with the CPU backend again. The server
still uses the exported model: the runtime compiles it directly with its own
graph optimizations and kernels, and its placement can be read back.

The decoder check covers what ggml schedules. How the OpenVINO GPU plugin
implements a primitive inside a decoder graph it accepted is not visible at
that layer. `scripts/test-openvino-gpu.sh` adds an independent signal: the
GPU's compute-engine cycle counter must advance during inference.

The encoder runs as one request and each prefill batch of the text model as
one graph, so cancellation and the deadline are checked between those steps
and between decode tokens, not inside them. On the GPU a step includes
compiling its graph, so a cancelled page can hold the engine for several
seconds; `scripts/test-openvino-gpu.sh` allows 15. A page takes 6 to 14 s on
an Arc Pro B70, most of it compilation. A page of a size the encoder has not
seen before also compiles the encoder inside its own deadline (5 to 21 s,
largest pages longest); `GGML_OPENVINO_CACHE_DIR` keeps compiled kernels
across restarts. `GRPC_LIGHTON_THREADS` still sizes host-side work.

`openvino:CPU` runs both stages on OpenVINO's CPU plugin under the same rules.
It exists to qualify the OpenVINO path on a host without a render node; it is
not a faster CPU mode. The shipped image carries the GPU plugin only, with no
CPU plugin and no routing plugin, so `openvino:CPU` fails there as an absent
device. The build stage has the CPU plugin:

```sh
docker build -f Dockerfile.openvino --target build -t grpc-lighton-ocr:openvino-build .
docker run --rm -e GRPC_LIGHTON_DEVICE=openvino:CPU -e GRPC_LIGHTON_MODELS_DIR=/models \
  -v /absolute/path/to/models:/models:ro -p 50074:50074 \
  --entrypoint /out/grpc-lighton-ocr-server grpc-lighton-ocr:openvino-build
```

The ggml backend reads its own `GGML_OPENVINO_*` variables from the
environment (see llama.cpp `docs/backend/OPENVINO.md` at the pinned commit),
for example `GGML_OPENVINO_CACHE_DIR` for a compiled-kernel cache, which the
vision encoder shares. The image creates `/var/cache/openvino` owned by 65532
for a volume mounted there. `GGML_OPENVINO_DEVICE` is set by the server from
`GRPC_LIGHTON_DEVICE`; a different value already in the environment fails
startup.

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
coordinates), provenance, generation traces, two-page folded references, a
page of a second raster size followed by the first size again,
resource limits, concurrent admission, loaded health, cancellation and recovery.
The image test uses a read-only root and read-only model mount. Missing weights
are failures, not skipped tests. `scripts/smoke-test.sh` checks only diagnostic
transport boot, library closure and shutdown; it also runs against the
OpenVINO image, without a GPU.

The Intel GPU qualification needs a host with a render node and the F16
artifacts including the exported encoder, and fails without either:

```sh
scripts/test-openvino-gpu.sh grpc-lighton-ocr:openvino /path/to/models
```

It requires startup to fail without `/dev/dri`, with `AUTO` or `HETERO`, and
with a GPU index the host does not have. With the GPU it requires the device
line and placement receipt in the startup log, the device in
`GetServiceInfo`, a pass of `tests/native_rpc_test.py`, an advancing xe
compute-engine cycle counter, and no placement violation.

The exported encoder is qualified against the ggml CPU encoder, from the same
preprocessed pixels, with the parity tool of the build stage (any OpenVINO
device; pages are PNG or JPEG files under the mounted directory):

```sh
docker run --rm -v /path/to/models:/models:ro -v /path/to/pages:/pages:ro \
  grpc-lighton-ocr:openvino-build /out/tools/vision_encoder_parity /models CPU f32 /pages/page.png
```

`config_test` covers device and thread parsing. `placement_test` checks the
placement accounting against the real ggml scheduler; in the OpenVINO build it
also runs a graph with an op OpenVINO cannot translate on the OpenVINO CPU
device and requires the fallback to be named. Both run in CTest and in both
image builds. The OpenVINO build adds two more, on the OpenVINO CPU device
and without a model file: `vision_encoder_test` exports a small encoder with
made-up weights and compares it with a plain C++ evaluation of the same
arithmetic, over several page sizes, and `openvino_backend_test` holds the two
carried backend fixes against the ggml CPU backend. See `docs/native-proof.md` for
the original standalone checkpoint proof, which is separate from RPC/image proof.

## Contract and licence

The contract is `proto/ai/pipestream/lightonocr/v1/lighton_ocr.proto`.
The shared document and OpenNLP protos remain byte-identical to the fleet copy.
`buf lint` and `buf format` run in CI. Apache-2.0. The OpenVINO image adds
distribution packages under Apache-2.0, MIT and BSD licences, listed in
`NOTICE`.
