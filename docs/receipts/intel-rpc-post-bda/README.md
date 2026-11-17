# B70 scheduler-stop follow-up: observed and unobserved RPC PASS

Native runtime includes bda85f4 (local e074b68), following deadline77dcad5
(local b66f93e). Validators include 809f8a7/e09a909 (local bb723e4/619fbc4).
Host krick-1, Intel Arc Pro B70 Graphics, SYCL0, oneAPI2025.3.3,
Level Zero1.15.39122, the same pinned BF16 model/projector as ../intel-rpc-f550.

Each mode passed a real two-page RPC with known text, authored invoice
geometry, page order, references and completion. Observed mode verified
both stages on the exact B70/SYCL0 device with cpu_compute=false. Observation
off required unverified stage metadata, retained exact device identity and
passed content/geometry; it is a control regression, not offload evidence.

Each mode then passed three valid-stream cancellation/recovery cycles:
first valid page received, seven uploaded pages pending, 0.2-second delay,
SIGINT client cancellation, nonzero exit/no completion, followed by correct
two-page output and mode-appropriate placement metadata. Six cycles total.
Server instances were sequential, with a shared GPU flock held throughout.
Both servers stopped; exitcode is zero and the lock was released to Tesla.
Timing fields are diagnostic only: Tesla's CPU reference oracle was active.
No CPU comparison or performance qualification was performed here.

The native llama build rebuilt its generated scheduler/engine through normal
Ninja. The RPC binary used six unchanged CMake-emitted compile/link commands
from the earlier recovered native build; that earlier build limitation still
applies. This receipt does not qualify a clean build or a container image.
The independent clean Docker build has separate logs and status.

`identities.sha256` records the binary, generated scheduler and ggml-base
library. `native-source.sha256` was checked against the remote source bytes.
`run.sh` records the complete run. Its initial pre-run wrapper failed before
locking because oneAPI setvars references unset OCL_ICD_FILENAMES under
`set -u`; the successful script enables nounset after setvars. No test result
from that failed launch is claimed.

Large raw logs are gzip mtime=0, verified byte-for-byte on decompression.
`compressed-logs.json` retains original paths, sizes and SHA256 plus gzip
hashes. JSON assertions, summaries and identities remain readable.
