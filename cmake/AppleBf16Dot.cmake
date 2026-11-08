# The pinned CPU BF16 dot has no ARM vector branch. Preserve its FP32
# products and FP64 accumulation while vectorizing on native Apple arm64.
# Compile a private generated file; never modify a supplied source checkout.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_source "${llama_SOURCE_DIR}/ggml/src/ggml-cpu/vec.cpp")
  file(SHA256 "${_source}" _hash)
  if(NOT _hash STREQUAL "ed6690d3246f420c37d1441469a3631e365c24425f9722ca2f46651c88c1e292")
    message(FATAL_ERROR "Apple BF16 dot repair needs review for changed vector source")
  endif()
  file(READ "${_source}" _text)
  string(REPLACE "#if defined(__AVX512BF16__)" [=[
#if defined(__aarch64__) && defined(__ARM_NEON)
    float64x2_t acc0 = vdupq_n_f64(0), acc1 = vdupq_n_f64(0);
    float64x2_t acc2 = vdupq_n_f64(0), acc3 = vdupq_n_f64(0);
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t ax = vld1q_u16(reinterpret_cast<const uint16_t *>(x + i));
        const uint16x8_t ay = vld1q_u16(reinterpret_cast<const uint16_t *>(y + i));
        const float32x4_t xl = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(ax), 16));
        const float32x4_t xh = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(ax), 16));
        const float32x4_t yl = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(ay), 16));
        const float32x4_t yh = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(ay), 16));
        const float32x4_t lo = vmulq_f32(xl, yl), hi = vmulq_f32(xh, yh);
        acc0 = vaddq_f64(acc0, vcvt_f64_f32(vget_low_f32(lo)));
        acc1 = vaddq_f64(acc1, vcvt_f64_f32(vget_high_f32(lo)));
        acc2 = vaddq_f64(acc2, vcvt_f64_f32(vget_low_f32(hi)));
        acc3 = vaddq_f64(acc3, vcvt_f64_f32(vget_high_f32(hi)));
    }
    sumf += vaddvq_f64(vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3)));
#elif defined(__AVX512BF16__)]=] _text "${_text}")
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/apple-bf16-vec.cpp")
  file(WRITE "${_generated}" "${_text}")
  get_target_property(_sources ggml-cpu SOURCES)
  list(FILTER _sources EXCLUDE REGEX "(^|/)vec\\.cpp$")
  set_property(TARGET ggml-cpu PROPERTY SOURCES "${_sources};${_generated}")
  if(BUILD_TESTING)
    add_executable(apple_bf16_dot_test tests/apple_bf16_dot_test.cpp)
    target_link_libraries(apple_bf16_dot_test PRIVATE ggml-cpu)
    add_test(NAME apple_bf16_dot_test COMMAND apple_bf16_dot_test)
  endif()
endif()
