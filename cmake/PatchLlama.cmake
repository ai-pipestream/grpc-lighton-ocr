# Applies the carried llama.cpp patches to the fetched source tree. Run by
# FetchContent as the PATCH_COMMAND of the llama dependency:
#
#   cmake -DLLAMA_SOURCE_DIR=<dir> -DPATCH_DIR=<repo>/patches/llama.cpp -P PatchLlama.cmake
#
# The tree is first put back to the pinned revision, then every *.patch in
# PATCH_DIR is applied in name order with `git apply`. Starting from the
# pinned revision each time makes a rebuild over a kept tree (a Docker cache
# mount) correct after a patch changed. A patch that does not apply stops the
# configure: after a pin move it needs a person, never a silent build without
# it. patches/llama.cpp/README.md says what each patch is for.
if(NOT LLAMA_SOURCE_DIR OR NOT PATCH_DIR)
  message(FATAL_ERROR "PatchLlama.cmake needs LLAMA_SOURCE_DIR and PATCH_DIR")
endif()
find_package(Git REQUIRED)

function(llama_git what)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
    RESULT_VARIABLE failed
    OUTPUT_QUIET
    ERROR_VARIABLE reason)
  if(NOT failed EQUAL 0)
    message(FATAL_ERROR "llama.cpp: ${what} failed\n${reason}")
  endif()
endfunction()

llama_git("restoring the pinned revision" reset --hard --quiet HEAD)
llama_git("removing files of earlier patches" clean -fdq)
file(GLOB patches "${PATCH_DIR}/*.patch")
list(SORT patches)
foreach(patch IN LISTS patches)
  get_filename_component(name "${patch}" NAME)
  llama_git("applying ${name} to the pinned revision" apply --whitespace=nowarn "${patch}")
  message(STATUS "llama.cpp patch applied: ${name}")
endforeach()
