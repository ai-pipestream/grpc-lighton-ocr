# Actual B70 LightOn RPC qualification — PASS

Host krick-1, Intel Arc Pro B70, SYCL0, Level Zero1.15.39122,
oneAPI compiler2025.3.3, Ubuntu26.10. Both stages' exact device name and ID
were additionally asserted from comparison.json. Instrumentation remained
on (`GRPC_LIGHTON_OBSERVE_PLACEMENT=1`) throughout. Common f550a4e is imported
as c34a56b, with EVP preserved. Runtime sources correspond to56c2f51;
98ee9a7 records the qualification build helper used to build them.

Actual same-binary explicit SYCL/CPU servers ran at127.0.0.1:51074/51075.
Parent hardware_rpc_test.py passed warmup plus five alternating measured
requests per backend, two pages per request. Assertions cover known invoice
text, hand-read raster anchor locations, bounded boxes, page ordering,
unique folded references, completion, same model/runtime identities,
verified vision+decoder placement, and explicit cpu_compute=false for SYCL
and true for CPU. All24 paired-gate pages passed. This is a narrow invoice
fixture, not a general OCR accuracy or memory-soak claim.

Median two-page RPC: SYCL1.938829721s, CPU9.049890531s, speedup4.668x.
SYCL measured range1.852655252..1.999360410s; CPU8.208988044..9.972541402s.
Cold warmups6.822sSYCL/11.155sCPU are excluded. Own compilation was paused;
other CI/training activity remained, so these are shared-host fixture
measurements, not stable machine-wide throughput guarantees. Earlier native
and failed placement receipts remain separate and unchanged.

The common693860f cancellation harness (import0524e14) passed three cycles:
valid first page completes, seven uploaded pages remain, client waits0.2s
for subsequent inference, cancels the active stream, requires nonzero exit
and no completion, then validates a real two-page content/geometry/placement
recovery. Recovery RPCs took3.375s,3.247s,2.763s. This is actual stream
cancellation/recovery, unlike the older native pre-cancel smoke.

Raw RPC outputs, comparison.json, cancellations, exits, server phase timings
and actual placement logs are retained in the three subdirectories. Large
server stderr and RPC JSONL logs are losslessly compressed as `.gz` with
mtime=0; `compressed-logs.json` records every original path, byte count and
SHA256, plus the compressed hash. Summaries, info JSON and source manifests
remain readable. Use `gzip -dc FILE.gz` to inspect an original log. All
compressed files were decompressed and checked byte-for-byte before the
original text copies were replaced. Earlier failed placement proof in
`../intel-native` is unchanged. Server
binary hash is in rpc-f550-servers/binary.sha256. Model/projector identities
and SHA256 are in both info JSONs. Native runtime libraries:

- libllama.so: 1d1eb01de0b5b901dfed8c582d50c9d528eaef7bd9c2239af557500921f83a97
- libmtmd.so: 1a7da7e7203a97a17a99e3cafcf37902d44d04babdbb20043cdf47097452bc2b
- libggml-base.so: a0a9105ce4b98689b34635db7190f29a7168a8592b2381874f5a09abf57e8cc6

All runtime source files were SHA256-compared local versus remote. The
private remote Ninja dependency log had been damaged by earlier concurrent
build drivers, and retained stale timestamps after recovery. The completed
pinned gRPC1.83.0 libraries were reused; the current CMake-generated proto,
first-party service, reflection and link commands were executed explicitly.
The resulting server uses the actual native llama/mtmd SYCL forward pass.
No engine substitution, metadata fabrication, or CPU fallback was used.
This qualifies the native server binary, not a container image.
