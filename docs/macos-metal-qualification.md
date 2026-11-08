# Native macOS Metal qualification

Build on Apple Silicon macOS with `scripts/build-macos-metal.sh`. Run
`scripts/run-macos-metal.sh /absolute/read-only/models` in the foreground.
The model directory must contain non-writable `model-bf16.gguf` and
`vision-bf16.gguf`; their pinned hashes are checked before loading. The server
binds to loopback `127.0.0.1:50074`. No daemon or machine configuration change
is needed. An arm64 Linux container does not provide Apple's Metal runtime.

## Checkpoint on 2026-11-08

Host: Apple M2, 24 GiB, native arm64 (Rosetta translation disabled), macOS
27.2 build 26B123; Apple Clang 21, Homebrew gRPC 1.85.1 and Protobuf 36.2.
The production runtime remains pinned llama.cpp
`f3586e66bc916d672773d4a6505513ba91a3595c`; models remain the pinned BF16
LightOnOCR-3-0.8B text and vision artifacts.

Actual RPC qualification at `c95bcfb` **failed**. The CPU baseline emitted
the correct invoice and geometry for page 1, but page 2 reached the existing
60-second native deadline after 27 tokens. Vision encoding took 42.787 and
44.806 seconds; decoder prefill took 0.887 and 0.925 seconds. No deadline or
acceptance threshold was raised.

The Metal request executed both stages on MTL0, but the scheduler observed
52 vision matmuls with zero CPU operations and 7106 decoder matmuls with
76 CPU operations. The service accurately disclosed decoder CPU computation;
the strict full-offload gate rejected it. No sustained latency speedup or
cancellation/recovery qualification is claimed from this attempt.

The subsequent common embedding-buffer override addresses that decoder
placement. The CPU investigation found no ARM vector branch in the pinned
BF16 dot routine. An Apple-only, hash-guarded generated-source repair adds
NEON products with FP64 accumulation; it does not edit the fetched checkout,
change weights, or extend the deadline. At `5ad28b4`, the full native build
and all four unit tests passed (1.08 seconds), including 840 numerical BF16
reference cases covering tails and unaligned inputs. Actual RPC reruns remain
required before claiming either fix qualifies the service.

## Required hardware rerun

Use the parent's standalone `tests/hardware_rpc_test.py` with Metal and
explicit CPU endpoints on the same host. Its paired endpoint design needs
two model instances during the bounded comparison; inspect memory pressure
before starting, serialize requests, and stop both owned instances afterward.
Preserve raw outputs in a fresh output directory. Run
`tests/metal_lifecycle_test.py` against the Metal endpoint for repeated
cancellation/recovery. Both scripts enforce actual placement and output
assertions; neither should be weakened to make this checkpoint pass.
