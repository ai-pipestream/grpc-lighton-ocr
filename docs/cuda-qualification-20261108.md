# CUDA qualification, 2026-11-08

Current native source tested on krick:
`c18cdcf529dd7be05167be3306602910212e6e7b`, including the shared deadline
fix and `bda85f4` scheduler drain/stop-callback fix. This is runtime evidence,
not a merge verdict. Orin and the final container remain pending below.

## RTX results

Five configured unit/checker gates passed, including the scheduler stop
cases. Three valid-stream cancellation/recovery cycles then passed actual
CUDA RPC inference. Every recovery validated two pages of expected invoice
text, hand-read anchor boxes, references, generation output and completion,
plus verified CUDA arithmetic in both vision and decoder stages and explicit
`cpu_compute=false`. Recovery times were 0.758, 0.733 and 0.714 seconds.

Cancellation is sent after the first page while an eight-page valid stream
remains active; all clients exit with SIGINT status -2. This proves valid
stream cancellation/recovery, not a precisely timed mid-kernel abort.

With `GRPC_LIGHTON_OBSERVE_PLACEMENT=0`, three additional diagnostic cycles
passed the same content and geometry checks. They explicitly require
unverified stages and absent CPU-compute disclosure. Stop callbacks remain
installed. This diagnostic mode does not pass the normal hardware placement
gate, and the shared checker was not weakened.

The old optional `native_rpc_test.py` was also attempted under CUDA. It
rejects startup because it explicitly requires `device == "cpu"`. That RED
log is retained; no six-of-six CUDA claim is made. Actual CUDA qualification
uses the separate hardware and cancellation harnesses. An initial client
started before model readiness also failed and is retained; the reported
cycles followed a successful readiness RPC.

## Earlier paired timing and identities

The earlier EVP source `64555ed` passed the paired hardware gate with five
warm two-page RPCs per backend. Median CPU latency was 9.6003 seconds and
CUDA 0.4713 seconds, a 20.37x ratio. Both stages, content and anchor geometry
were asserted. This was one invoice on a shared host with background CI
bursts, not an exclusive benchmark or general quality measurement. The
latest stop fix received incremental runtime tests, not a new benchmark.

Host: RTX 4080 SUPER 16 GiB, driver 615.71.09, CUDA 12.4, Release architecture
89, GCC 15 C++ and GCC 13 nvcc host compiler. The native decoder and mtmd
vision use llama pin `f3586e66bc916d672773d4a6505513ba91a3595c` and gRPC
`c876f4da50f7da2f331888b88b2a7243514139fe` (v1.83.0, static BoringSSL).
Host preprocessing and sampling are excluded from graph CPU disclosure.

Model revision: `36d8636925327c671bd8e479c517a004f7566bd9`.
Text SHA256: `a73ab10d787c68cbfc518f51bfe044abcfb9dc27ce9c457f2068bf7d3ee0241d`.
Vision SHA256: `a0af0fe69443cfe35edd1294466d0f1d9780682177a101944d0fe8a9f5b72bd4`.

## Image and Orin limits

`Dockerfile.cuda` includes `libssl-dev` in the builder and `libssl3t64`
(libssl and libcrypto) in the runtime, and builds every configured test gate
with two jobs. The toolkit driver stub is used only to load build-time unit
tests without a GPU driver. Five stub-only test gates and Docker's recipe
check passed. The stub is not copied into the runtime image or used for
actual hardware qualification.

Nano1 is reachable and its original two-job dependency build is continuing.
Its first-party source must be updated before the actual Orin GPU/RPC tests.
The existing TEI service remains untouched. No working-Orin or final CUDA
container qualification is claimed by this receipt batch.

## Receipts

The [compressed archive](qualification/cuda-20261108-receipts.tar.gz) and
[SHA256](qualification/cuda-20261108-receipts.sha256) contain source snapshots,
exact binary hashes, raw RPC output, build/test logs and historical failures.
Extract with `tar -xzf`. Models and executables are not embedded. Frozen
current tested binaries are in `/tmp/lighton-qualified-c18cdcf/`; the local
expanded handoff is `/work/main/grpc-services/hardware-receipts/cuda-20261108/lighton/`.
Earlier deadline-only smoke evidence is labeled intermediate separately.
