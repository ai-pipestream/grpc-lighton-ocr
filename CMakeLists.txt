cmake_minimum_required(VERSION 3.25)
project(grpc-lighton-ocr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIGHTON_WERROR "Treat warnings as errors" ON)
# OFF keeps the CPU-only build. ON adds the ggml OpenVINO backend for Intel
# GPU inference and needs the OpenVINO runtime and OpenCL development files.
option(LIGHTON_OPENVINO "Build the ggml OpenVINO backend (Intel GPU)" OFF)

# Warnings stay on our targets. add_compile_options would also instrument
# the fetched gRPC build, and that tree does not compile with -Werror.
function(lighton_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra)
  if(LIGHTON_WERROR)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()

include(CTest)
include(FetchContent)

set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
foreach(plugin CSHARP NODE OBJECTIVE_C PHP PYTHON RUBY)
  set(gRPC_BUILD_GRPC_${plugin}_PLUGIN OFF CACHE BOOL "" FORCE)
endforeach()
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.83.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
  SYSTEM)
FetchContent_MakeAvailable(grpc)

# In-process backend. No downloads or subprocesses during inference. The
# default build is CPU only; LIGHTON_OPENVINO adds one accelerator backend.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENVINO ${LIGHTON_OPENVINO} CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Keep upstream architecture-specific SIMD defaults. GGML_NATIVE stays OFF:
# x86 uses the fixed AVX2 baseline, while ARM selects its own kernels and flags.
# No host-specific AVX512/AMX flags or dynamic backend plugins are enabled.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(LIGHTON_X86_SIMD ON)
else()
  set(LIGHTON_X86_SIMD OFF)
endif()
foreach(instruction AVX AVX2 FMA F16C SSE42)
  set(GGML_${instruction} ${LIGHTON_X86_SIMD} CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
# The pinned revision with the patches under patches/llama.cpp on top; see
# cmake/PatchLlama.cmake for how they are applied. Their combined hash is
# part of the patch command, so a changed patch runs the step again.
file(GLOB LIGHTON_LLAMA_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama.cpp/*.patch")
list(SORT LIGHTON_LLAMA_PATCHES)
set(LIGHTON_LLAMA_PATCH_HASH "")
foreach(patch IN LISTS LIGHTON_LLAMA_PATCHES)
  file(SHA256 "${patch}" patch_hash)
  string(SHA256 LIGHTON_LLAMA_PATCH_HASH "${LIGHTON_LLAMA_PATCH_HASH}${patch_hash}")
endforeach()
FetchContent_Declare(llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG f3586e66bc916d672773d4a6505513ba91a3595c
  PATCH_COMMAND ${CMAKE_COMMAND} -DLLAMA_SOURCE_DIR=<SOURCE_DIR>
                -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches/llama.cpp
                -DPATCH_HASH=${LIGHTON_LLAMA_PATCH_HASH}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PatchLlama.cmake
  SYSTEM)
FetchContent_MakeAvailable(llama)

set(LIGHTON_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIGHTON_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(LIGHTON_PROTOS
  "${LIGHTON_PROTO_DIR}/ai/pipestream/document/v1/document.proto"
  "${LIGHTON_PROTO_DIR}/org/apache/opennlp/grpc/v1/opennlp_document.proto"
  "${LIGHTON_PROTO_DIR}/org/apache/opennlp/grpc/v1/opennlp_annotations.proto"
  "${LIGHTON_PROTO_DIR}/ai/pipestream/lightonocr/v1/lighton_ocr.proto")
set(LIGHTON_PROTO_SOURCES
  "${LIGHTON_GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
  "${LIGHTON_GENERATED_DIR}/org/apache/opennlp/grpc/v1/opennlp_document.pb.cc"
  "${LIGHTON_GENERATED_DIR}/org/apache/opennlp/grpc/v1/opennlp_annotations.pb.cc"
  "${LIGHTON_GENERATED_DIR}/ai/pipestream/lightonocr/v1/lighton_ocr.pb.cc"
  "${LIGHTON_GENERATED_DIR}/ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.cc")
set(LIGHTON_PROTO_HEADERS ${LIGHTON_PROTO_SOURCES})
list(TRANSFORM LIGHTON_PROTO_HEADERS REPLACE "\\.cc$" ".h")

add_custom_command(
  OUTPUT ${LIGHTON_PROTO_SOURCES} ${LIGHTON_PROTO_HEADERS}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${LIGHTON_GENERATED_DIR}"
  COMMAND $<TARGET_FILE:protoc>
  ARGS --proto_path=${LIGHTON_PROTO_DIR}
       --proto_path=${protobuf_SOURCE_DIR}/src
       --cpp_out=${LIGHTON_GENERATED_DIR}
       --grpc_out=${LIGHTON_GENERATED_DIR}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
       ${LIGHTON_PROTOS}
  DEPENDS ${LIGHTON_PROTOS} protoc grpc_cpp_plugin
  VERBATIM)

add_library(lighton_proto STATIC ${LIGHTON_PROTO_SOURCES})
target_include_directories(lighton_proto SYSTEM PUBLIC "${LIGHTON_GENERATED_DIR}")
target_compile_options(lighton_proto PRIVATE -w)
target_link_libraries(lighton_proto PUBLIC grpc++)

add_library(lighton_parse STATIC src/parse/markdown.cpp)
target_include_directories(lighton_parse PUBLIC src)
target_link_libraries(lighton_parse PUBLIC lighton_proto re2::re2)
lighton_warnings(lighton_parse)

# Device selection and placement accounting: ggml only, no model code.
add_library(lighton_device STATIC
  src/engine/device.cpp
  src/engine/placement.cpp)
target_include_directories(lighton_device PUBLIC src)
target_link_libraries(lighton_device PUBLIC ggml)
if(LIGHTON_OPENVINO)
  # The same runtime ggml links. Used once at startup to list devices, so a
  # missing device is a clean startup error and not an abort inside ggml.
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  target_link_libraries(lighton_device PRIVATE openvino::runtime)
  target_compile_definitions(lighton_device PUBLIC LIGHTON_OPENVINO=1)
endif()
lighton_warnings(lighton_device)

# The page an accelerator must read correctly before the server listens. It
# is compiled in, so the check needs no file beside the weights.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/native-invoice.png" LIGHTON_REFERENCE_PAGE_HEX HEX)
string(REGEX REPLACE "(..)" "0x\\1," LIGHTON_REFERENCE_PAGE_BYTES "${LIGHTON_REFERENCE_PAGE_HEX}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/reference/reference_page.inc"
  CONTENT "${LIGHTON_REFERENCE_PAGE_BYTES}\n" @ONLY)

# The vision stage of an OpenVINO device: the exported encoder, run through
# the OpenVINO runtime. A build without OpenVINO gets a stand-in that cannot
# be constructed.
if(LIGHTON_OPENVINO)
  add_library(lighton_vision STATIC src/vision/openvino_encoder.cpp)
  target_link_libraries(lighton_vision PRIVATE openvino::runtime)
else()
  add_library(lighton_vision STATIC src/vision/no_encoder.cpp)
endif()
target_include_directories(lighton_vision PUBLIC src)
lighton_warnings(lighton_vision)

add_library(lighton_service STATIC
  src/engine/native_engine.cpp
  src/config.cpp
  src/service/lighton_service.cpp)
target_include_directories(lighton_service PUBLIC src)
target_include_directories(lighton_service SYSTEM PRIVATE "${llama_SOURCE_DIR}/vendor/stb")
target_include_directories(lighton_service PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/reference")
target_link_libraries(lighton_service PUBLIC lighton_parse lighton_device lighton_vision llama mtmd)
lighton_warnings(lighton_service)

if(LIGHTON_OPENVINO)
  # Model preparation, not part of the server: builds the encoder graph from
  # the projector file's weights and writes the IR the server reads.
  add_library(lighton_vision_export STATIC
    src/vision/encoder_model.cpp
    src/vision/gguf_weights.cpp)
  target_include_directories(lighton_vision_export PUBLIC src)
  target_link_libraries(lighton_vision_export PUBLIC ggml openvino::runtime)
  lighton_warnings(lighton_vision_export)

  add_executable(export_vision_encoder tools/export_vision_encoder.cpp)
  target_link_libraries(export_vision_encoder PRIVATE lighton_vision_export)
  lighton_warnings(export_vision_encoder)

  # Qualification: the exported encoder against the ggml CPU encoder.
  add_executable(vision_encoder_parity tools/vision_encoder_parity.cpp)
  target_link_libraries(vision_encoder_parity PRIVATE lighton_vision llama mtmd)
  lighton_warnings(vision_encoder_parity)
endif()

add_executable(grpc-lighton-ocr-server src/main.cpp)
target_link_libraries(grpc-lighton-ocr-server PRIVATE lighton_service grpc++_reflection)
lighton_warnings(grpc-lighton-ocr-server)

if(BUILD_TESTING)
  add_executable(markdown_test tests/markdown_test.cpp)
  target_link_libraries(markdown_test PRIVATE lighton_parse)
  lighton_warnings(markdown_test)
  add_test(NAME markdown_test COMMAND markdown_test)

  add_executable(service_test tests/service_test.cpp)
  target_link_libraries(service_test PRIVATE lighton_service)
  lighton_warnings(service_test)
  add_test(NAME service_test COMMAND service_test)

  add_executable(config_test tests/config_test.cpp)
  target_link_libraries(config_test PRIVATE lighton_service)
  lighton_warnings(config_test)
  add_test(NAME config_test COMMAND config_test)

  # With LIGHTON_OPENVINO this also runs the OpenVINO backend on its CPU
  # device and needs the OpenVINO CPU plugin; it never needs a GPU.
  add_executable(placement_test tests/placement_test.cpp)
  target_link_libraries(placement_test PRIVATE lighton_device)
  lighton_warnings(placement_test)
  add_test(NAME placement_test COMMAND placement_test)

  if(LIGHTON_OPENVINO)
    # A small encoder with made-up weights, exported and run on the OpenVINO
    # CPU device, against a plain C++ evaluation of the same arithmetic.
    add_executable(vision_encoder_test tests/vision_encoder_test.cpp)
    target_link_libraries(vision_encoder_test PRIVATE lighton_vision lighton_vision_export)
    lighton_warnings(vision_encoder_test)
    add_test(NAME vision_encoder_test COMMAND vision_encoder_test)

    # The carried ggml OpenVINO backend fixes, on the OpenVINO CPU device.
    add_executable(openvino_backend_test tests/openvino_backend_test.cpp)
    target_link_libraries(openvino_backend_test PRIVATE ggml)
    lighton_warnings(openvino_backend_test)
    add_test(NAME openvino_backend_test COMMAND openvino_backend_test)
  endif()
endif()

# Explicit qualification gate: supplied models are required, never skipped.
set(LIGHTON_QUALIFICATION_MODELS "" CACHE PATH "Pinned artifacts for real native RPC tests")
if(BUILD_TESTING AND LIGHTON_QUALIFICATION_MODELS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_program(LIGHTON_GRPCURL grpcurl REQUIRED)
  add_test(NAME native_rpc_test COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/native_rpc_test.py
    $<TARGET_FILE:grpc-lighton-ocr-server> ${LIGHTON_QUALIFICATION_MODELS})
  set_tests_properties(native_rpc_test PROPERTIES TIMEOUT 600)
endif()
