# Qualification-only extension to a grpc-asr build on the same host/toolchain.
# Reuse its pinned gRPC targets and this checkout's independently built llama
# shared libraries. No ASR engine code is linked into the LightOn server.
# Configure the ASR build with:
# -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=/path/to/lighton/tools/IntelRpcReuse.cmake
# -DLIGHTON_NATIVE_BUILD=/path/to/lighton/build-llama-intel
get_filename_component(LIGHTON_RPC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
function(lighton_add_rpc_reuse)
  if(NOT TARGET grpc++ OR NOT TARGET grpc++_reflection)
    message(FATAL_ERROR "RPC reuse requires the existing pinned gRPC build")
  endif()
  set(native "${LIGHTON_NATIVE_BUILD}")
  set(llama "${LIGHTON_RPC_ROOT}/build-deps/llama-src")
  foreach(lib llama mtmd ggml ggml-base ggml-cpu ggml-sycl)
    if(NOT EXISTS "${native}/bin/lib${lib}.so")
      message(FATAL_ERROR "Build the native SYCL libraries first: ${lib}")
    endif()
    list(APPEND native_libs "${native}/bin/lib${lib}.so")
  endforeach()
  set(gen "${CMAKE_BINARY_DIR}/lighton-generated")
  set(proto "${LIGHTON_RPC_ROOT}/proto")
  set(names ai/pipestream/document/v1/document
    org/apache/opennlp/grpc/v1/opennlp_document
    org/apache/opennlp/grpc/v1/opennlp_annotations
    ai/pipestream/lightonocr/v1/lighton_ocr)
  foreach(name IN LISTS names)
    list(APPEND inputs "${proto}/${name}.proto")
    list(APPEND outputs "${gen}/${name}.pb.cc")
  endforeach()
  list(APPEND outputs "${gen}/ai/pipestream/lightonocr/v1/lighton_ocr.grpc.pb.cc")
  add_custom_command(OUTPUT ${outputs}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${gen}"
    COMMAND $<TARGET_FILE:protoc> --proto_path=${proto}
      --proto_path=${ASR_PROTOBUF_INCLUDE} --cpp_out=${gen} --grpc_out=${gen}
      --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin> ${inputs}
    DEPENDS ${inputs} protoc grpc_cpp_plugin VERBATIM)
  add_library(intel_lighton_proto STATIC ${outputs})
  target_include_directories(intel_lighton_proto SYSTEM PUBLIC "${gen}")
  target_compile_options(intel_lighton_proto PRIVATE -w)
  target_link_libraries(intel_lighton_proto PUBLIC grpc++)
  add_executable(intel-lighton-rpc
    "${LIGHTON_RPC_ROOT}/src/main.cpp"
    "${LIGHTON_RPC_ROOT}/src/config.cpp"
    "${LIGHTON_RPC_ROOT}/src/parse/markdown.cpp"
    "${LIGHTON_RPC_ROOT}/src/engine/native_engine.cpp"
    "${LIGHTON_RPC_ROOT}/src/service/lighton_service.cpp")
  target_include_directories(intel-lighton-rpc PRIVATE "${LIGHTON_RPC_ROOT}/src")
  target_include_directories(intel-lighton-rpc SYSTEM PRIVATE
    "${llama}/include" "${llama}/ggml/include" "${llama}/tools/mtmd" "${llama}/vendor/stb"
    "${grpc_SOURCE_DIR}/include" "${grpc_SOURCE_DIR}/third_party/abseil-cpp"
    "${grpc_SOURCE_DIR}/third_party/re2" "${ASR_PROTOBUF_INCLUDE}")
  target_compile_options(intel-lighton-rpc PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(intel-lighton-rpc PRIVATE intel_lighton_proto
    grpc++_reflection re2::re2 OpenSSL::Crypto ${native_libs})
  set_target_properties(intel-lighton-rpc PROPERTIES BUILD_RPATH "${native}/bin")
endfunction()
cmake_language(DEFER CALL lighton_add_rpc_reuse)
