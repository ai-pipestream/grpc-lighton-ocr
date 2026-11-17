# Scheduler stop fix: native M2 qualification

Code/test commit: `bda85f4dd28ac6bad822e11d7673175c0eb504ed`.
Native server SHA256:
`07a47d205577dce75c8452cf5bd85e11a563ac0c351edf4d65847e792c6248a8`.

Apple M2, 24 GiB, native Apple Silicon, Rosetta disabled, macOS 27.2
(26B123). Exact architecture, compiler, gRPC/Protobuf versions, 39 source
hashes, generated scheduler/BF16 runtime hashes, four binary hashes and BF16
model hashes are in source-manifest.json. Source hashes match Git. Runtime:
llama.cpp f3586e66bc916d672773d4a6505513ba91a3595c; model:
LightOnOCR-3-0.8B revision 36d8636925327c671bd8e479c517a004f7566bd9.

## Results

- Native incremental build passed, bounded to 2 jobs at nice 10.
- All 6 configured native tests passed in 1.34s: BF16 reference,
  scheduler stop, control, markdown, service, hardware output checker.
- Actual two-page RPC passed invoice text, anchored boxes, references,
  completion, and verified vision/decoder placement on Apple M2 / MTL0.
  Both stages explicitly report cpuCompute=false. Elapsed: 2.505s.
- 3 valid eight-page streams cancelled after first-page output, before
  completion. Each following two-page TCP recovery passed all content and
  placement assertions: 2.456, 2.371, 2.409s. Raw outputs rechecked locally 4/4.
- Observation enabled, single loaded model, loopback endpoint 50074.
  No CPU server or CPU comparison. Owned server stopped in finally;
  independent process check found none remaining (63% memory free).

The fake async backend test uses the actual generated pinned scheduler and
production callback setup: 16 combinations of observation on/off,
vision/decoder, cancellation/deadline, and first/middle split stops. Each
case checks a different-size successful next request on the same scheduler.
A queued cross-backend copy writes only on synchronization; all queues must
be empty before caller cleanup, and later splits must not execute. Fake
native success with no matmul and final-token boundaries test stop priority.
The test passes locally with ASan/UBSan/leak checking and natively on the M2.
Removing the full drain, restoring the inner-loop break, or removing the
post-call check each fails the test. Full native engine/service compilation
with WERROR and the local service suite also pass.

This bounded smoke does not measure general OCR quality, long-duration
memory use, or a hardware 60-second expiration. Deterministic tests cover
those stop boundaries; no deadline or threshold changed. Observation-off
stop behavior is covered by fake-backend tests, not a separate hardware run.
Native placement logs are retained, not an Xcode profiler capture.

Previous CPU/Metal performance (8.52225x) belongs only to source
1bfc1df0cdee5f17e211be07001996155facfad5. No new speedup claim.
The 77dcad smoke stays separate as an intermediate receipt in ../metal-m2-20261108-postdeadline.
Files ending .gz retain raw outputs with deterministic gzip headers.
