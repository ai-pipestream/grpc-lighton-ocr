# syntax=docker/dockerfile:1.28
# CPU-only native LightOnOCR 0.8B. Weights remain external, read-only.
# No-engine smoke checks explicitly select diagnostic mode.
# The runtime base is ubuntu:26.04, the same default the other C++ collectors
# use. A Docker Hardened Image debian-base (Debian 13, glibc 2.41) cannot load
# a binary built on ubuntu 26.04 (glibc 2.43). Swap the runtime with
# --build-arg LIGHTON_RUNTIME_IMAGE when a base with glibc 2.43 or newer exists.
#
#   docker build -t grpc-lighton-ocr:local .
#   docker run --rm --read-only -p 50074:50074 grpc-lighton-ocr:local

ARG LIGHTON_RUNTIME_IMAGE=ubuntu:26.04

FROM ubuntu:26.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake g++ git ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ARG TARGETARCH
RUN --mount=type=cache,id=grpc-lighton-ocr-ubuntu26-cxx23-grpc1.83.0-${TARGETARCH},target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIGHTON_WERROR=ON \
    && cmake --build /build --target grpc-lighton-ocr-server markdown_test service_test --parallel 2 \
    && ctest --test-dir /build --output-on-failure \
    && mkdir -p /out/lib /out/licenses && cp LICENSE /out/licenses/grpc-lighton-ocr-LICENSE \
    && cp /build/_deps/llama-src/LICENSE /out/licenses/llama.cpp-LICENSE \
    && cp /build/grpc-lighton-ocr-server /out/ \
    && ldd /out/grpc-lighton-ocr-server | awk '/=> \// {print $3}' | xargs -r -I '{}' cp -L '{}' /out/lib/ \
    && ldd /out/grpc-lighton-ocr-server | awk '/ld-linux/ {print $1}' | xargs -r -I '{}' cp -L '{}' /out/lib/ld.so

FROM ${LIGHTON_RUNTIME_IMAGE}
COPY --from=build /out/grpc-lighton-ocr-server /usr/local/bin/grpc-lighton-ocr-server
COPY --from=build /out/lib/ /opt/lighton/lib/
COPY --from=build /out/licenses/ /usr/share/licenses/grpc-lighton-ocr/
ENV LD_LIBRARY_PATH=/opt/lighton/lib
ENV GRPC_LIGHTON_LISTEN_ADDRESS=0.0.0.0:50074 \
    GRPC_LIGHTON_MODELS_DIR=/models
USER 65532:65532
EXPOSE 50074
ENTRYPOINT ["/opt/lighton/lib/ld.so", "/usr/local/bin/grpc-lighton-ocr-server"]
