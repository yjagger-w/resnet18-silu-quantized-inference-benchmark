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

    add_test(NAME cuda_vector_add COMMAND test_cuda_vector_add)
    set_tests_properties(cuda_vector_add PROPERTIES
        SKIP_RETURN_CODE 77
    )
endif()
