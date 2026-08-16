cmake_minimum_required(VERSION 3.16)
project(vector_simd_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Baseline optimization for everyone. NOTE: we deliberately do NOT pass a
# blanket -march=native / -mavx2 here — the AVX2/AVX-512 kernels opt in to
# their own instruction sets per-function via
# __attribute__((target("avx2,fma"))) / __attribute__((target("avx512f"))),
# so the library stays safe to run (with runtime CPUID dispatch) on CPUs
# that lack those extensions, while still emitting real AVX2/AVX-512 code
# for the kernels that need it.
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenMP)

add_library(vector_ops STATIC src/vector_ops.cpp)
target_include_directories(vector_ops PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vector_ops PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark benchmark/benchmark_main.cpp)
target_link_libraries(benchmark PRIVATE vector_ops)

enable_testing()
add_executable(unit_tests tests/test_correctness.cpp)
target_link_libraries(unit_tests PRIVATE vector_ops)
add_test(NAME unit_tests COMMAND unit_tests)
