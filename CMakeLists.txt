cmake_minimum_required(VERSION 3.25)
project(grpc-lighton-ocr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIGHTON_WERROR "Treat warnings as errors" ON)

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

option(LIGHTON_SYSTEM_GRPC "Use installed native protobuf and gRPC" OFF)
if(LIGHTON_SYSTEM_GRPC)
  find_package(Protobuf CONFIG REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  add_executable(protoc ALIAS protobuf::protoc)
  add_executable(grpc_cpp_plugin ALIAS gRPC::grpc_cpp_plugin)
  add_library(grpc++ ALIAS gRPC::grpc++)
  add_library(grpc++_reflection ALIAS gRPC::grpc++_reflection)
  get_target_property(LIGHTON_PROTOBUF_INCLUDE protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
  list(GET LIGHTON_PROTOBUF_INCLUDE 0 LIGHTON_PROTOBUF_INCLUDE)
  set(LIGHTON_PROTOBUF_IMPORT_DIR "${LIGHTON_PROTOBUF_INCLUDE}")
else()
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
set(LIGHTON_PROTOBUF_IMPORT_DIR "${protobuf_SOURCE_DIR}/src")
endif()

# Model identity hashing uses EVP directly. Match gRPC's crypto provider so
# bundled BoringSSL headers cannot accidentally be paired with system OpenSSL.
if(LIGHTON_SYSTEM_GRPC OR gRPC_SSL_PROVIDER STREQUAL "package")
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  set(LIGHTON_CRYPTO_TARGET OpenSSL::Crypto)
elseif(TARGET crypto)
  set(LIGHTON_CRYPTO_TARGET crypto)
  set(LIGHTON_CRYPTO_INCLUDE "${grpc_SOURCE_DIR}/third_party/boringssl-with-bazel/include")
else()
  message(FATAL_ERROR "LightOn requires gRPC's OpenSSL or bundled BoringSSL crypto provider")
endif()

# In-process backends. CPU remains available in every build.
option(LIGHTON_CUDA "Build the native CUDA backend" OFF)
option(LIGHTON_SYCL "Build the native SYCL backend" OFF)
option(LIGHTON_METAL "Build the native Metal backend" OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${LIGHTON_CUDA} CACHE BOOL "" FORCE)
set(GGML_SYCL ${LIGHTON_SYCL} CACHE BOOL "" FORCE)
set(GGML_METAL ${LIGHTON_METAL} CACHE BOOL "" FORCE)
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
FetchContent_Declare(llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG f3586e66bc916d672773d4a6505513ba91a3595c
  SYSTEM)
FetchContent_MakeAvailable(llama)
include(cmake/AppleLlamaCompleteType.cmake)
include(cmake/ObserveGgmlPlacement.cmake)

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
       --proto_path=${LIGHTON_PROTOBUF_IMPORT_DIR}
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

add_library(lighton_service STATIC
  src/engine/native_engine.cpp
  src/config.cpp
  src/service/lighton_service.cpp)
target_include_directories(lighton_service PUBLIC src)
target_include_directories(lighton_service SYSTEM PRIVATE "${llama_SOURCE_DIR}/vendor/stb")
target_link_libraries(lighton_service PUBLIC lighton_parse llama mtmd)
target_link_libraries(lighton_service PRIVATE ${LIGHTON_CRYPTO_TARGET})
if(LIGHTON_CRYPTO_INCLUDE)
  target_include_directories(lighton_service SYSTEM PRIVATE "${LIGHTON_CRYPTO_INCLUDE}")
endif()
lighton_warnings(lighton_service)

add_executable(grpc-lighton-ocr-server src/main.cpp)
target_link_libraries(grpc-lighton-ocr-server PRIVATE lighton_service grpc++_reflection)
lighton_warnings(grpc-lighton-ocr-server)

if(BUILD_TESTING)
  add_executable(scheduler_stop_test tests/scheduler_stop_test.cpp)
  target_include_directories(scheduler_stop_test PRIVATE src "${llama_SOURCE_DIR}/ggml/src")
  target_link_libraries(scheduler_stop_test PRIVATE ggml ggml-cpu)
  lighton_warnings(scheduler_stop_test)
  add_test(NAME scheduler_stop_test COMMAND scheduler_stop_test)
  add_executable(inference_control_test tests/inference_control_test.cpp)
  target_include_directories(inference_control_test PRIVATE src)
  lighton_warnings(inference_control_test)
  add_test(NAME inference_control_test COMMAND inference_control_test)
  add_executable(markdown_test tests/markdown_test.cpp)
  target_link_libraries(markdown_test PRIVATE lighton_parse)
  lighton_warnings(markdown_test)
  add_test(NAME markdown_test COMMAND markdown_test)

  add_executable(service_test tests/service_test.cpp)
  target_link_libraries(service_test PRIVATE lighton_service)
  lighton_warnings(service_test)
  add_test(NAME service_test COMMAND service_test)
endif()

if(BUILD_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    foreach(check_name hardware_rpc_checks_test hardware_cancellation_checks_test)
      add_test(NAME ${check_name} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/${check_name}.py)
    endforeach()
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
