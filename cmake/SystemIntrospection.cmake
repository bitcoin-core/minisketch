include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)

# Check for carryless-multiply support: first x86 CLMUL, then ARMv8 PMULL.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_REQUIRED_FLAGS "-mpclmul")
endif()
check_cxx_source_compiles("
  #include <immintrin.h>
  #include <stdint.h>

  int main()
  {
    __m128i a = _mm_cvtsi64_si128((uint64_t)7);
    __m128i b = _mm_clmulepi64_si128(a, a, 37);
    __m128i c = _mm_srli_epi64(b, 41);
    __m128i d = _mm_xor_si128(b, c);
    uint64_t e = _mm_cvtsi128_si64(d);
    return e == 0;
  }
  " HAVE_CLMUL_X86
)
if(HAVE_CLMUL_X86)
  set(CLMUL_CXXFLAGS ${CMAKE_REQUIRED_FLAGS})
  set(HAVE_CLMUL TRUE)
endif()

# Try aarch64 PMULL.
if(NOT HAVE_CLMUL AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+crypto")
  check_cxx_source_compiles("
    #include <arm_neon.h>
    #include <stdint.h>

    int main()
    {
      uint64x2_t a = vcombine_u64(vcreate_u64((uint64_t)7), vdup_n_u64(0));
      poly64_t al = (poly64_t)vgetq_lane_u64(a, 0);
      poly128_t p = vmull_p64(al, al);
      uint64x2_t b = vreinterpretq_u64_p128(p);
      uint64x2_t c = vshrq_n_u64(b, 41);
      uint64x2_t d = veorq_u64(b, c);
      uint64_t e = vgetq_lane_u64(d, 0);
      return e == 0;
    }
    " HAVE_PMULL_AARCH64
  )
  if(HAVE_PMULL_AARCH64)
    set(CLMUL_CXXFLAGS ${CMAKE_REQUIRED_FLAGS})
    set(HAVE_CLMUL TRUE)
  endif()
endif()

if(CMAKE_CXX_STANDARD LESS 20)
  # Check for working clz builtins.
  check_cxx_source_compiles("
    int main()
    {
      unsigned a = __builtin_clz(1);
      unsigned long b = __builtin_clzl(1);
      unsigned long long c = __builtin_clzll(1);
    }
    " HAVE_CLZ
  )
endif()

cmake_pop_check_state()
