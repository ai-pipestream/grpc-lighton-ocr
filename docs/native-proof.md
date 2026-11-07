# Native CPU checkpoint proof

Native CPU 0.8B inference now runs through the streaming RPC. Loaded health is
SERVING; diagnostic mode reports NOT_SERVING. No gRParse route is enabled.
The 4B checkpoint and aarch64 inference remain untested.

Backend: llama.cpp f3586e66bc916d672773d4a6505513ba91a3595c, CPU only,
libllama and libmtmd. Model: lightonai/LightOnOCR-3-0.8B revision
36d8636925327c671bd8e479c517a004f7566bd9, Apache-2.0.

Original model.safetensors SHA256:
`ae46aa6fd7136d57297ac006ed9ec345b7a2083feb2e99501bedcb014ca3efdf`

Conversion uses the pinned backend's convert_hf_to_gguf.py, Transformers 5.5.4,
CPU Torch 2.11.0, BF16. Text conversion requires `--no-mtp` because this
checkpoint has zero MTP layers. Projector conversion uses `--mmproj`.
Python is only an offline conversion dependency, not an inference dependency.

Converted text SHA256:
`a73ab10d787c68cbfc518f51bfe044abcfb9dc27ce9c457f2068bf7d3ee0241d`

Converted projector SHA256:
`a0af0fe69443cfe35edd1294466d0f1d9780682177a101944d0fe8a9f5b72bd4`

Fixture tests/fixtures/native-invoice.png SHA256:
`a3d9057022912f90aa6de461dd60e87dd1ba64557c5c9638e086c8d5795b45f7`

The tools/native_lighton_probe.cpp harness reads the fixture into a buffer and
passes it to mtmd_helper_bitmap_init_from_buf. The prompt is `grounding`,
with the upstream single-user, no-tools, thinking-disabled chat template.
The prompt does not contain the expected invoice text. Two CPU threads,
4096-token context, 256-token generation cap, five-minute abort deadline.

Observed output, exit 0:

```text
![text](39,105,457,692) INVOICE 1042 Total USD 125.00 Paid
```

One local run took 3.94 seconds with peak RSS 1928308 KiB. This is neither a
throughput benchmark nor proof of general OCR accuracy. Receipt files on krick:
/tmp/lighton-native-output.txt, /tmp/lighton-native-inference.log and
/work/models/lightonocr3-0.8b-native/source-receipt.json.

## RPC receipt

Initial qualification used a manually assembled portable static binary and
library closure, with the repository runtime-stage instructions. It did not
use the full Dockerfile build. Image ID:
`sha256:e04ea89f9865927a205b91c275515786142671c3abfbb4303b039d98f6618fdc`.
Source was an uncommitted tree based on `508dcc8`, without an exact commit ID.
Receipt: `/tmp/lighton-native-container-test-final.log` on krick.

Passed: two-page invoice text, positive boxes, source metadata, Unicode spans,
raw token traces, document references, token-limit partial output, greedy
sampling, admission limits, cancellation, recovery, and active-call shutdown.
Root and model mounts were read-only; UID was 65532. This is one fixture,
not a general accuracy or speed measurement.

Reproduction with the repository Dockerfile:

```sh
docker build -t grpc-lighton-ocr:native-review .
./scripts/smoke-test.sh grpc-lighton-ocr:native-review
./scripts/test-native-image.sh grpc-lighton-ocr:native-review /work/models/lightonocr3-0.8b-native
```

The Dockerfile runs markdown_test and service_test. The service test checks
that malformed grounding retains raw text and token traces. CI smoke checks
transport boot only. Hosted checks and publication are separate gates.
The runtime includes project and pinned llama.cpp licenses.

## Committed-source Dockerfile qualification (2026-11-06)

Source commit: `41041d99097c02e0c4ba0024a96498005766148b`.
A clean worktree at this commit produced a new repository Dockerfile image.
The dependency cache was reused; first-party files were recompiled, including
`src/engine/native_engine.cpp` (build log step 10/22). The image revision label
matches this commit. This receipt adds documentation only after that run.

Image ID:
`sha256:38d20ae2fef298a19a33e1933645e3d6d6ec8b2a7fe39a90a942ca911a2175fb`

Server binary SHA256:
`f7862c4d4b035975249c7c0eaa752055dfaaa8f079f4b337df7c725354fbb2df`

Commands, from the source worktree:

```sh
docker build --label org.opencontainers.image.revision=41041d99097c02e0c4ba0024a96498005766148b -t grpc-lighton-ocr:41041d9 .
./scripts/smoke-test.sh grpc-lighton-ocr:41041d9
./scripts/test-native-image.sh grpc-lighton-ocr:41041d9 /work/models/lightonocr3-0.8b-native
docker run --rm --read-only --entrypoint /usr/bin/sha256sum grpc-lighton-ocr:41041d9 /usr/local/bin/grpc-lighton-ocr-server
```

Results: Dockerfile CTest 2/2 passed; diagnostic image smoke passed; read-only
real-model RPC qualification passed. Checks covered invoice text, positive
page boxes, source metadata, Unicode spans, raw traces, two-page references,
partial output, greedy decoding, admission limits, cancellation and recovery.
Eight-page active-call shutdown exited 0 in 5.370 seconds; the client received
Unavailable with "Cancelling all calls". Root and model mounts were read-only.
The runtime license files match the project and pinned backend licenses.

The service unit test also passed four idle clients without client deadlines:
server timeout returned CANCELLED and released capacity for another request.
That test uses 150 milliseconds; production uses 30 seconds per blocked Read
or Write, with no transport timer during inference. Write backpressure does
not have a separate client-visible regression receipt.

Logs on krick:

- `/tmp/lighton-41041d9-docker-build.log`
- `/tmp/lighton-41041d9-smoke.log`
- `/tmp/lighton-41041d9-rpc.log`

Hosted CI and publication are separate from this local qualification. No 4B,
aarch64, GPU, general document accuracy, or published-image result is claimed.
