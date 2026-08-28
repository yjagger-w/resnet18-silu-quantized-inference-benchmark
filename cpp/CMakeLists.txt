cmake_minimum_required(VERSION 3.20)

project(silu_quantized_kernel VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

add_library(silu_quantized_kernel STATIC
    src/quantized_silu_kernel.cpp
)
target_include_directories(silu_quantized_kernel PUBLIC include)
target_compile_features(silu_quantized_kernel PUBLIC cxx_std_17)
set_target_properties(silu_quantized_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(silu_quantized_kernel PRIVATE /W4 /permissive- /fp:precise)
    set(SILU_KERNEL_LIBRARY_FLAGS "/W4 /permissive- /fp:precise, default ISA")
else()
    target_compile_options(silu_quantized_kernel PRIVATE -Wall -Wextra -Wpedantic)
    set(SILU_KERNEL_LIBRARY_FLAGS "-Wall -Wextra -Wpedantic, default ISA")
endif()

if(DEFINED ORT_ROOT AND NOT "${ORT_ROOT}" STREQUAL "")
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "The v1.2 ORT custom operator requires Windows x64")
    endif()
    set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
    set(ORT_IMPORT_LIBRARY "${ORT_ROOT}/lib/onnxruntime.lib")
    foreach(required_file
            "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h"
            "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h"
            "${ORT_IMPORT_LIBRARY}")
        if(NOT EXISTS "${required_file}")
            message(FATAL_ERROR "Missing official ONNX Runtime SDK file: ${required_file}")
        endif()
    endforeach()

    add_library(silu_ort_custom_op SHARED
        src/ort_quantized_silu_custom_op.cpp
    )
    target_include_directories(silu_ort_custom_op PRIVATE
        include
        "${ORT_INCLUDE_DIR}"
    )
    target_link_libraries(silu_ort_custom_op PRIVATE
        silu_quantized_kernel
        "${ORT_IMPORT_LIBRARY}"
    )
    target_compile_features(silu_ort_custom_op PRIVATE cxx_std_17)
    target_compile_definitions(silu_ort_custom_op PRIVATE
        SILU_ORT_SDK_VERSION="1.19.2"
    )
    set_target_properties(silu_ort_custom_op PROPERTIES
        OUTPUT_NAME "silu_ort_custom_op"
    )
    if(MSVC)
        target_compile_options(silu_ort_custom_op PRIVATE /W4 /permissive- /fp:precise)
    endif()
endif()

add_executable(silu_kernel_benchmark
    tools/silu_kernel_benchmark.cpp
)
target_link_libraries(silu_kernel_benchmark PRIVATE silu_quantized_kernel)
target_compile_features(silu_kernel_benchmark PRIVATE cxx_std_17)
target_compile_definitions(silu_kernel_benchmark PRIVATE
    SILU_KERNEL_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    SILU_KERNEL_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    SILU_KERNEL_BUILD_TYPE="$<CONFIG>"
    SILU_KERNEL_LIBRARY_FLAGS="${SILU_KERNEL_LIBRARY_FLAGS}"
)

if(BUILD_TESTING)
    add_executable(test_quantized_silu_kernel
        tests/test_quantized_silu_kernel.cpp
    )
    target_link_libraries(test_quantized_silu_kernel PRIVATE silu_quantized_kernel)
    target_include_directories(test_quantized_silu_kernel PRIVATE tests)
    target_compile_features(test_quantized_silu_kernel PRIVATE cxx_std_17)

    add_test(NAME quantized_silu_kernel COMMAND test_quantized_silu_kernel)
    add_test(
        NAME benchmark_cli_rejects_outside_output
        COMMAND silu_kernel_benchmark
            --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/act_call_0_golden.json
            --size 16 --warmup 0 --iterations 1 --seed 1
            --implementation scalar --output-dir outside-v1-results
    )
    set_tests_properties(benchmark_cli_rejects_outside_output PROPERTIES
        WILL_FAIL TRUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    add_test(
        NAME benchmark_cli_rejects_avx2
        COMMAND silu_kernel_benchmark
            --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/act_call_0_golden.json
            --size 16 --warmup 0 --iterations 1 --seed 1
            --implementation avx2
            --output-dir results/benchmarks/v1.0_silu_kernel_ctest
    )
    set_tests_properties(benchmark_cli_rejects_avx2 PROPERTIES
        WILL_FAIL TRUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endif()
