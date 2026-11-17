# libc++ instantiates unique_ptr cleanup while compiling the constructor.
# Complete the pinned nested type before that constructor, without changing
# any code or mutating the fetched source directory.
if(APPLE)
  set(_source "${llama_SOURCE_DIR}/src/llama-memory-hybrid-idx.cpp")
  file(SHA256 "${_source}" _hash)
  if(NOT _hash STREQUAL "08cbd47bcac5e481ef66c586c27510b210f34d76042dbafa6d51d9dbc489831a")
    message(FATAL_ERROR "Apple complete-type repair needs review for changed llama source")
  endif()
  file(READ "${_source}" _text)
  string(FIND "${_text}" "// Which cells of a sequence make up which pool" _begin)
  string(FIND "${_text}" "// Which pools of the layout" _end)
  math(EXPR _length "${_end} - ${_begin}")
  string(SUBSTRING "${_text}" ${_begin} ${_length} _definition)
  string(REPLACE "${_definition}" "" _text "${_text}")
  string(REPLACE "//\n// llama_memory_hybrid_idx\n//"
    "${_definition}//\n// llama_memory_hybrid_idx\n//" _text "${_text}")
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/apple-llama-memory-hybrid-idx.cpp")
  file(WRITE "${_generated}" "${_text}")
  get_target_property(_sources llama SOURCES)
  list(FILTER _sources EXCLUDE REGEX "(^|/)llama-memory-hybrid-idx\\.cpp$")
  set_property(TARGET llama PROPERTY SOURCES "${_sources};${_generated}")
  set_source_files_properties("${_generated}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()
