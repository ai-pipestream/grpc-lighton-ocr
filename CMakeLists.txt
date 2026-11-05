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
target_link_libraries(lighton_parse PUBLIC lighton_proto)
lighton_warnings(lighton_parse)

add_library(lighton_service STATIC
  src/config.cpp
  src/service/lighton_service.cpp)
target_include_directories(lighton_service PUBLIC src)
target_link_libraries(lighton_service PUBLIC lighton_parse)
lighton_warnings(lighton_service)

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
endif()
