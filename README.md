# grpc-lighton-ocr

A gRPC collector for LightOnOCR-3. It speaks a typed streaming contract and
ships as its own container so it can scale apart from gRParse and from
`grpc-unlimited-ocr`.

**Status:** the wire contract, the grounding-markdown parser, the hardened
image, and CI are in place. The native weight engine is not linked yet.
`GetServiceInfo` reports `engine_loaded = false`, and a page raster returns
`PAGE_FAILURE_CODE_ENGINE_UNAVAILABLE`. The parser is what will turn a
decode into `Document` items. Current tests exercise the parser and failure
paths, not image-to-text inference. Table markup is still retained as text;
typed table extraction and continuation-block joining are unfinished.

The service is `ai.pipestream.lightonocr.v1.LightOnOcrService` on port
**50074**. The gRParse coordinator does not dial this service yet; setting
`GRPARSE_LIGHTON_OCR_TARGET` does not enable a parse route. The proposed
compose integration is an opt-in service shell, not an OCR implementation.

The standard gRPC health RPC indicates transport availability only. Use
`GetServiceInfo.engine_loaded` to check inference readiness, and inspect
page failures and completion counts even when the stream status is OK.
The process holds request bytes in memory and needs no writable request
directory. Model files are separate from request data.

## Build and test

Requires CMake 3.25 or newer and a C++23 compiler.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIGHTON_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

gRPC is fetched at configure time (v1.83.0).

## Image

The runtime is numeric user 65532, one binary, gRPC only. Run it read-only.
Weights, when the engine loads them, mount at `/models` and are never baked
into the image.

```sh
docker build -t grpc-lighton-ocr:local .
docker run --rm --read-only -p 50074:50074 grpc-lighton-ocr:local
```

`scripts/smoke-test.sh` checks library closure, the user, the exposed port,
the listen line, and a clean SIGTERM.

## Contract

[`proto/ai/pipestream/lightonocr/v1/lighton_ocr.proto`](proto/ai/pipestream/lightonocr/v1/lighton_ocr.proto).
`document.proto` and the OpenNLP protos it imports are vendored
byte-identical from gRParse. `buf lint` and `buf format` run in CI.

## Licence

Apache-2.0.

## Native engine work remaining

The declared 0.8B and 4B checkpoints use Qwen3.5, per the
[upstream model card](https://huggingface.co/lightonai/LightOnOCR-3-0.8B).
A candidate C++ integration is libllama plus libmtmd, with an image decoded
from its input buffer, a bounded decode loop, and cancellation checked
between decode steps. This is a proposed path, not implemented support.
No checkpoint or projector checksums are pinned in `models/MANIFEST` yet.

Before advertising either checkpoint as working, pin the native library and
converted model/projector artifacts, verify preprocessing and the grounding
prompt against upstream, and run an actual image-to-text fixture through the
read-only service container. Record the model revision and output, test
resource limits and cancellation, then add the gRParse client route. The
existing parser and stub tests do not satisfy that gate.
