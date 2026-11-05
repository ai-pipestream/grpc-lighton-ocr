# syntax=docker/dockerfile:1.28
# CPU image. The native LightOnOCR engine is not linked yet, so this image
# needs no GPU and bakes no weights. It still builds the server, runs the
# test suite in the build stage, and boots as the numeric user 65532.
#
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
    && cmake --build /build --target grpc-lighton-ocr-server markdown_test service_test --parallel \
    && ctest --test-dir /build --output-on-failure \
    && mkdir -p /out && cp /build/grpc-lighton-ocr-server /out/

FROM ${LIGHTON_RUNTIME_IMAGE}
COPY --from=build /out/grpc-lighton-ocr-server /usr/local/bin/grpc-lighton-ocr-server
ENV GRPC_LIGHTON_LISTEN_ADDRESS=0.0.0.0:50074 \
    GRPC_LIGHTON_MODELS_DIR=/models
USER 65532:65532
EXPOSE 50074
ENTRYPOINT ["/usr/local/bin/grpc-lighton-ocr-server"]
