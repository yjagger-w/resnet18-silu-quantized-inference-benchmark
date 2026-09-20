cmake_minimum_required(VERSION 3.22)

project(silu_cuda_kernels VERSION 1.7.0 LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(
        CMAKE_CUDA_ARCHITECTURES
        75
        CACHE STRING
        "CUDA architectures to compile for; 75 targets NVIDIA T4"
    )
endif()

add_library(silu_cuda_kernels STATIC
    kernels/bias_silu.cu
    kernels/reduction.cu
    kernels/vector_add.cu
)
target_include_directories(silu_cuda_kernels PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(silu_cuda_kernels PUBLIC CUDA::cudart)
target_compile_features(silu_cuda_kernels PUBLIC
    cxx_std_17
    cuda_std_17
)
target_compile_options(silu_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(silu_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(cuda_reduction_benchmark
    tools/reduction_benchmark.cu
)
target_link_libraries(cuda_reduction_benchmark PRIVATE
    silu_cuda_kernels
    CUDA::cudart
)
target_compile_features(cuda_reduction_benchmark PRIVATE
    cxx_std_17
    cuda_std_17
)
target_compile_definitions(cuda_reduction_benchmark PRIVATE
    SILU_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
    SILU_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    SILU_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)

if(BUILD_TESTING)
    add_executable(test_cuda_vector_add
        tests/test_vector_add.cu
    )
    target_link_libraries(test_cuda_vector_add PRIVATE
        silu_cuda_kernels
        CUDA::cudart
    )
    target_compile_features(test_cuda_vector_add PRIVATE
        cxx_std_17
        cuda_std_17
    )

    add_executable(test_cuda_reduction
        tests/test_reduction.cu
    )
    target_link_libraries(test_cuda_reduction PRIVATE
        silu_cuda_kernels
        CUDA::cudart
    )
    target_compile_features(test_cuda_reduction PRIVATE
        cxx_std_17
        cuda_std_17
    )

    add_executable(test_cuda_bias_silu
        tests/test_bias_silu.cu
    )
    target_link_libraries(test_cuda_bias_silu PRIVATE
        silu_cuda_kernels
        CUDA::cudart
    )
    target_compile_features(test_cuda_bias_silu PRIVATE
        cxx_std_17
        cuda_std_17
    )

    add_test(NAME cuda_vector_add COMMAND test_cuda_vector_add)
    add_test(NAME cuda_reduction COMMAND test_cuda_reduction)
    add_test(NAME cuda_bias_silu COMMAND test_cuda_bias_silu)
    set_tests_properties(
        cuda_vector_add
        cuda_reduction
        cuda_bias_silu
        PROPERTIES
        SKIP_RETURN_CODE 77
    )

    add_test(
        NAME cuda_reduction_benchmark_help
        COMMAND cuda_reduction_benchmark --help
    )
    add_test(
        NAME cuda_reduction_benchmark_rejects_zero_iterations
        COMMAND cuda_reduction_benchmark --iterations 0
    )
    set_tests_properties(
        cuda_reduction_benchmark_rejects_zero_iterations
        PROPERTIES
        WILL_FAIL TRUE
    )
endif()
