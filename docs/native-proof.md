# Native CPU checkpoint proof

This is a standalone inference proof, not yet an enabled RPC engine. The service
still reports engine_loaded=false and named health NOT_SERVING. No gRParse route
is enabled. The 4B checkpoint has not been tested.

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

Before RPC exposure: verify artifact hashes during loading; bound pixels before
image decoding, context and generated tokens before allocation, and output bytes
while generating; test cancellation, admission limits, actual RPC image-to-text,
and loaded-model health. Current runtime readiness tests cover the absent engine
only. Shutdown marks health non-serving and uses a five-second gRPC deadline.
