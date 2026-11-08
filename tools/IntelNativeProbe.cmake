# Add the first-party native runner to the pinned llama build, reusing its
# compiled SYCL objects. Invoke via CMAKE_PROJECT_TOP_LEVEL_INCLUDES.
get_filename_component(LIGHTON_PROBE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
function(lighton_add_native_probe)
  set(llama_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  find_package(OpenSSL REQUIRED)
  include("${LIGHTON_PROBE_ROOT}/cmake/ObserveGgmlPlacement.cmake")
  add_executable(intel_native_probe
    "${LIGHTON_PROBE_ROOT}/tools/intel_native_probe.cpp"
    "${LIGHTON_PROBE_ROOT}/src/engine/native_engine.cpp")
  target_compile_features(intel_native_probe PRIVATE cxx_std_23)
  target_include_directories(intel_native_probe PRIVATE
    "${LIGHTON_PROBE_ROOT}/src" "${llama_SOURCE_DIR}/vendor/stb")
  target_link_libraries(intel_native_probe PRIVATE llama mtmd OpenSSL::Crypto)
endfunction()
cmake_language(DEFER CALL lighton_add_native_probe)
