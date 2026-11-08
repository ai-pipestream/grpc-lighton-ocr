# Observe actual scheduler assignments after successful graph submission. Keep
# the pinned upstream checkout untouched; compile one generated source instead.
set(_source "${llama_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
file(SHA256 "${_source}" _hash)
if(NOT _hash STREQUAL "7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e")
  message(FATAL_ERROR "GGML placement observer needs review for changed scheduler source")
endif()
file(READ "${_source}" _text)
string(REPLACE "// backend buffer type" [=[
using lighton_observer = void (*)(ggml_backend_t, ggml_tensor *, void *);
static thread_local lighton_observer lighton_observe = nullptr;
static thread_local void * lighton_observe_data = nullptr;
extern "C" GGML_API void lighton_ggml_set_observer(lighton_observer observer, void * data) {
    lighton_observe = observer;
    lighton_observe_data = data;
}
// backend buffer type]=] _text "${_text}")
# Observe only the submitted range, including when cancellation stops a split.
string(REPLACE [=[            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {]=] [=[            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (lighton_observe) {
                for (int n = 0; n < split->graph.n_nodes; ++n)
                    lighton_observe(split_backend, split->graph.nodes[n], lighton_observe_data);
            }
        } else {]=] _text "${_text}")
string(REPLACE "                // TODO: pass backend to the callback, then the user can decide if they want to synchronize" [=[
                if (lighton_observe) {
                    for (int n = j0; n <= j1; ++n)
                        lighton_observe(split_backend, split->graph.nodes[n], lighton_observe_data);
                }
                // TODO: pass backend to the callback, then the user can decide if they want to synchronize]=] _text "${_text}")
# No tensor data is requested, so no synchronization is needed. Preserve
# upstream cancellation/data callbacks when they do request a result.
string(REPLACE "                ggml_backend_synchronize(split_backend);"
  "                if (need) ggml_backend_synchronize(split_backend);" _text "${_text}")
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/observed-ggml-backend.cpp")
file(WRITE "${_generated}" "${_text}")
get_target_property(_sources ggml-base SOURCES)
list(FILTER _sources EXCLUDE REGEX "(^|/)ggml-backend\\.cpp$")
set_property(TARGET ggml-base PROPERTY SOURCES "${_sources};${_generated}")
