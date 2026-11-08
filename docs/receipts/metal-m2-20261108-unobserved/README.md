# Observation-off regression on Apple M2

Runtime source: `bda85f4dd28ac6bad822e11d7673175c0eb504ed`.
Test-only commit: `809f8a7ef3ab1516836cd1dcfb96955a214c12f0`.
Same server binary as the observed run; no rebuild:
`07a47d205577dce75c8452cf5bd85e11a563ac0c351edf4d65847e792c6248a8`.

Apple M2, 24 GiB, native Apple Silicon, Rosetta disabled. Explicit startup:
GRPC_LIGHTON_BACKEND=metal, GRPC_LIGHTON_OBSERVE_PLACEMENT=0, loopback 50074,
read-only pinned BF16 models. This is **not observed GPU-offload evidence**.
Both stages report configured MTL0 identities and unverified execution.
False runtimeVerified values are omitted in protobuf JSON; cpuCompute is
also omitted, so no CPU-computation claim is made.

Two-page invoice text, hand-located boxes, references and completion passed in
2.905171s. 3 valid eight-page streams cancelled after first-page output,
all exiting -2 before completion. Following two-page recovery RPCs passed
the same content checks in 2.695132/2.714225/2.754396s. All completed outputs
and each cancellation's first page required unverified stage metadata.
Single server PID 53380 remained alive through all cycles, demonstrating
lease reuse without restart. The runner then stopped the owned server;
independent process check found none remaining, with 62% memory free.

The new --unobserved-regression flag is explicit and isolated. Default
cancellation checks still require verified placement. Mode-validator tests
pass 3/3 locally and on Mac; existing content/geometry checker tests pass
3/3 locally. Local receipt recheck passed 4 full outputs, and the default
strict gate correctly rejects this unverified evidence.

source-manifest.json records runtime/model/binary hashes plus the separate
test hashes. All 39 runtime-source hashes and both test hashes match their
commits; binary and generated runtime hashes equal the observed-run receipt.
runner.py records startup and cleanup. Use the test commit's cancellation
script with --unobserved-regression, --backend metal, --address 127.0.0.1:50074,
and a new --output directory. The runner loads those test tools separately
from the unchanged bda85f4 server checkout. Raw outputs/logs are gzip files.

No CPU benchmark or speedup claim. Earlier paired performance remains tied
to source 1bfc1df0cdee5f17e211be07001996155facfad5. No timeout or threshold changed.
