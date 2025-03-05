# Check for clmul instructions support.
if(MSVC)
  set(CLMUL_CXXFLAGS "")
else()
  set(CLMUL_CXXFLAGS "-mpclmul")
endif()
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS ${CLMUL_CXXFLAGS})
include(CheckCXXSourceCompiles)
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
  " HAVE_CLMUL
)
cmake_pop_check_state()

add_subdirectory(fields)

add_library(minisketch minisketch.cpp)
target_include_directories(minisketch
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(minisketch
  PRIVATE
   minisketch_field_generic_sources
   $<$<BOOL:${HAVE_CLMUL}>:minisketch_field_clmul_sources>
)

add_library(minisketch_verify EXCLUDE_FROM_ALL minisketch.cpp)
target_compile_definitions(minisketch_verify
  PUBLIC
    MINISKETCH_VERIFY
)
target_link_libraries(minisketch_verify
  PRIVATE
   minisketch_field_generic_sources
   $<$<BOOL:${HAVE_CLMUL}>:minisketch_field_clmul_sources>
)

if(MINISKETCH_TESTS)
  add_executable(test test.cpp)
  target_link_libraries(test PRIVATE minisketch)

  add_executable(test-verify test.cpp)
  target_link_libraries(test-verify PRIVATE minisketch_verify)
endif()

if(MINISKETCH_BENCHMARK)
  add_executable(bench bench.cpp)
  target_link_libraries(bench PRIVATE minisketch)
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
  target_compile_definitions(minisketch
    PRIVATE
      $<$<BOOL:${HAVE_CLZ}>:HAVE_CLZ>
  )
  target_compile_definitions(minisketch_verify
    PRIVATE
      $<$<BOOL:${HAVE_CLZ}>:HAVE_CLZ>
  )
endif()

