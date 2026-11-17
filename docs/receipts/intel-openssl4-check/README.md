# Pinned gRPC / OpenSSL 4 targeted compile check

The system OpenSSL provider avoids IntelLLVM's bundled BoringSSL bcm.cc
compilation exceeding the isolated builder's 24 GiB cap. That clean image
attempt exited102 with icpx error10106 and cgroup oom_kill=1; it did not pass.

Before waiting for the replacement image's full dependency build, three
pristine gRPC1.83.0 TLS sources were copied from the private checkout into
/tmp/lighton-openssl-check on krick-1. The existing CMake-generated compile
commands were converted to syntax-only checks (one at a time, nice10).
All three failed on OpenSSL4 const getters/opaque ASN1 structures. After
the guarded ASR compatibility patch, all three passed. `red-*.log` and
`green-*.log` retain the results; empty green logs mean no diagnostics.
`check.py` records the procedure and `sources.sha256` the patched outputs.
Shared fetched sources were not modified by this targeted check.

This proves the three compatibility edits compile against the actual Intel
toolchain and OpenSSL4 headers; it is not a complete image/link/runtime test.
LightOn includes this patch only for fetched gRPC with package crypto.
