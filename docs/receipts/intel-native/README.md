# Native placement smoke, RPC qualification pending

Host krick-1; Intel Arc Pro B70; Level Zero 1.15.39122; oneAPI 2025.3.3.
Pinned llama f3586e66bc916d672773d4a6505513ba91a3595c. Common engine ec4e8f4
reconciled as1294627, preserving EVP verification. Native executable SHA256
15edb01f993d4c537d46f30790abd6ac8002685c51e0ee24faae759c7574a1c3.
Models match the SHA256 constants in native_engine.h, verified at load.

The first failed receipt native-sycl.txt/log is retained: correct text but
76 CPU decoder operations, therefore partial placement and failed qualification.
The current sycl.stdout/stderr applies the common input embedding override:
vision52 GPU matmuls/0 CPU ops, decoder7106 GPU matmuls/0 CPU ops. Six repeats
return the expected invoice text, with both stages verified on SYCL0. CPU
comparison reports CPU execution explicitly. Pre-cancel and subsequent
inference pass; this is not actual in-flight cancellation proof. The box
regex checks only positive extents, not authored raster locations.

Warm runs1..5 median SYCL1048.921 ms, CPU13746.908 ms. Cold run0 SYCL7342.539ms,
CPU9098.485ms is excluded. Both modes use the same executable/model/fixture.
Shared host load varied substantially during these sequential runs; the
alternating two-page RPC comparison and authored geometry gate remain required.
