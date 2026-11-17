# Intermediate typed-deadline Metal smoke

Tested source: `77dcad584b823ae38e1bb4fe3a87382db9592392`.
Host: Apple M2, 24 GiB, Rosetta disabled. Server SHA256:
`6b4f52497871d6d94f9697ff2705343168d496dba8cb0427e9bd0afc4e2dc5d4`.
Exact source, generated runtime, binaries, toolchain and BF16 model hashes:
`source-manifest.json`. All 38 source hashes match Git; pinned llama input
hashes match the preceding qualification.

Native incremental build passed; control/service tests passed 2/2 in 0.77s.
Actual two-page RPC passed invoice text, anchored geometry, references,
completion, and both-stage verified MTL0 placement with cpuCompute=false
(3.135s, first request, not a performance comparison). 3 valid eight-page
streams cancelled after the first page; subsequent two-page recovery RPCs
passed all checks in 2.423/2.429/2.443s. Raw outputs revalidated locally 4/4.
Observation was enabled. No CPU model was loaded. The runner stopped the
owned server in finally; independent process check found none remaining.

This is intermediate evidence: independent review later found scheduler
abort propagation and observation-disabled cancellation gaps. These passing
results do not cover those boundary cases. A fix requires fresh qualification.
No 60-second hardware wait was performed; deterministic control tests cover
direct classification.

Previous paired CPU/Metal performance (8.52225x) applies only to
`1bfc1df0cdee5f17e211be07001996155facfad5`; it was not repeated here.
Files ending .gz are unedited raw outputs with deterministic gzip headers.
Use runner.py for this bounded smoke after building the pinned source.
