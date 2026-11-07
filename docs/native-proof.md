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
