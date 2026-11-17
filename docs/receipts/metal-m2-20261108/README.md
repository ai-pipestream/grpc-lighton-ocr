# Apple M2 actual LightOn qualification

Tested source: `1bfc1df0cdee5f17e211be07001996155facfad5`.
Native arm64 server SHA256:
`b4516118d9c3aada7bd0157db64325aaf04e0b2a7be08151185648d87f499695`.

Host: Apple M2, 24 GiB, native arm64, no Rosetta, macOS 27.2 build 26B123,
Apple Clang 21, system gRPC 1.85.1 and Protobuf 36.2. Runtime:
llama.cpp `f3586e66bc916d672773d4a6505513ba91a3595c`; LightOnOCR-3-0.8B
revision `36d8636925327c671bd8e479c517a004f7566bd9`, original BF16 weights.
Exact model/projector/fixture hashes are in `paired/comparison.json`.

`source-manifest.json` records SHA256 for all 36 tracked build/source/proto/test
inputs at the tested commit and maps the relevant changes:

- `ec4e8f4` → `6517a76`: accelerator embeddings and request-owned placement.
- `f550a4e` → `c377a4f`: device-classification cache, preserved whole graph UID,
  bounded cancellation polling and phase timing. Observation was explicitly
  enabled; no unverified diagnostic mode was used.
- `cfac05a`, `3de285d`, `5ad28b4`: Apple-only BF16 NEON dot, test include and
  C linkage fixes. Private generated source preserves FP32 products and FP64
  accumulation; the pinned source checkout is not mutated. All 840 primitive
  reference cases passed, then the combined build passed actual model gates.
- `693860f` → `1bfc1df`: shared valid-stream cancellation/TCP-recovery runner.

The paired harness ran one warm pair followed by five measured pairs,
alternating endpoint order. Both processes used this exact binary and the same
read-only models; CPU was explicit. No concurrent compile/model work ran.
Two loaded endpoints were necessary for the harness; reported memory free
was 74% before requests and 70% during them. Both were stopped by the runner
finally block. An independent process check found no remaining ASR/LightOn
servers, with 78% memory free afterward.

All twelve paired RPCs passed text, anchored geometry, reference and
completion checks. Each contained two pages with the invoice text
`INVOICE 1042 Total USD 125.00 Paid`. Both Metal stages reported MTL0,
runtime verification true and CPU computation false. Both CPU stages reported
CPU, runtime verification true and CPU computation true. Median two-page
latency: CPU 21.942903583 s, Metal 2.574776792 s, speedup 8.522254687x.

Three eight-page valid streams were interrupted after receiving their
first page. Each cancellation exited nonzero before document completion;
each subsequent two-page TCP RPC passed the same output/placement checks.
Recovery RPC times were 2.538, 2.628 and 2.579 s. Their outputs and exit codes
are under `cancellation/`.

Files ending `.gz` contain unedited raw outputs compressed with deterministic
gzip headers. `paired/comparison.json` is the unchanged harness receipt;
cold and warmed info responses are retained separately. Complete native
server logs retain scheduler operation counts and phase times. This is native
placement/log evidence, not an Xcode GPU profiler capture.

Scope: one repeated invoice fixture, bounded cancellation/recovery and this
0.8B model. No general OCR quality, 4B model, long-duration memory soak or
isolated NEON speedup claim. Earlier failures remain documented in
`../../macos-metal-qualification.md`; no deadline, tolerance, metadata
requirement or geometry assertion was weakened.
