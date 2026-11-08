# Intel SYCL build and qualification

The production configuration is `GRPC_LIGHTON_BACKEND=sycl` and
`GRPC_LIGHTON_DEVICE=SYCL0`. CPU remains explicit with `cpu`. Both vision and
decoder use the native pinned llama/mtmd implementation, with common actual
execution placement observation. Input embeddings are explicitly assigned
to the selected accelerator; configuration alone never verifies placement.

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLIGHTON_SYCL=ON
cmake --build build-sycl -j2
```

Run two instances of the same binary on the same host, with identical model
artifacts and server-instance configuration, one explicitly SYCL and one
explicitly CPU. Give them separate unused loopback ports. Then run the
parent-owned actual RPC gate:

```bash
python3 tests/hardware_rpc_test.py --candidate 127.0.0.1:51074 \
  --cpu 127.0.0.1:51075 --backend sycl --output receipts/new-sycl-rpc
```

The gate checks authored invoice locations, text, page order, references,
completion, both observed stages, and five alternating warm RPC repetitions
against CPU. Use a reserved GPU window. In-flight cancellation/recovery is
also required, using the shared lifecycle gate. Native helper results and
`docs/receipts/intel-native` are smoke evidence only; their positive-box
regex and pre-cancel callback cannot replace those RPC gates.

## Container recipe

`Dockerfile.sycl` pins the Ubuntu and Intel oneAPI image digests, carries
SYCL/oneMKL plus Level Zero driver libraries, and bounds compilation to two
jobs by default. This is a recipe, not a claim that an image was published
or qualified on hardware. Requalify the resulting container via real RPC.

```bash
docker build --platform linux/amd64 -f Dockerfile.sycl \
  --build-arg BUILD_JOBS=2 -t grpc-lighton-ocr:sycl-local .
docker run --rm --read-only --tmpfs /tmp --device=/dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v /work/models/lighton-intel-qualification:/models:ro \
  -p 127.0.0.1:51074:50074 grpc-lighton-ocr:sycl-local
```

The host must have a working B70 kernel driver/firmware and accessible DRM
render node. Containers supply user-space drivers, not the host kernel.
The numeric render group permits access for image user65532; choose the
correct render node on multi-GPU hosts. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
is set in the image. Runtime placement must still identify the intended
B70. Driver packages are pinned to `libze1=1.32.0-1` and
`libze-intel-gpu1=26.27.39122.14-1`; the package repository must retain them.

The native krick-1 runs use oneAPI compiler2025.3.3 and Level Zero1.15.39122.
The image's pinned base is a separately versioned toolchain and needs its
own recorded compiler/runtime identity and hardware results. Shared host
contention and instrumentation changes must be documented with timings;
do not promote provisional native timings into general throughput claims.
