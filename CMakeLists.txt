# microtorch — autograd tape + nn API over transformer_core's primitives.
#
# The seam DESIGN.md section 1 proposed: consume transformer_cpp via
# add_subdirectory and link the existing transformer_core STATIC target.
# EXCLUDE_FROM_ALL so only transformer_core is built (as a dependency),
# not that repo's executables. CUDA stays off here: phase 1a is CPU-first
# per PHASE0_KERNEL_AUDIT.md section 6.
cmake_minimum_required(VERSION 3.18)
project(microtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../transformer_cpp
                 ${CMAKE_BINARY_DIR}/transformer_cpp EXCLUDE_FROM_ALL)

add_library(microtorch STATIC
    src/autograd.cpp
    src/ops.cpp
    src/nn.cpp
    src/safetensors.cpp
    src/kimi_linear.cpp
    src/cerebellum.cpp
    src/mamba.cpp
)
target_include_directories(microtorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(microtorch PUBLIC transformer_core
                                 PRIVATE nlohmann_json::nlohmann_json)

enable_testing()
add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE microtorch)
add_test(NAME gradcheck COMMAND test_gradcheck)

add_executable(test_nn tests/test_nn.cpp)
target_link_libraries(test_nn PRIVATE microtorch)
add_test(NAME nn COMMAND test_nn)

add_executable(test_kimi_linear tests/test_kimi_linear.cpp)
target_link_libraries(test_kimi_linear PRIVATE microtorch)
add_test(NAME kimi_linear COMMAND test_kimi_linear)

add_executable(test_cerebellum tests/test_cerebellum.cpp)
target_link_libraries(test_cerebellum PRIVATE microtorch)
add_test(NAME cerebellum COMMAND test_cerebellum)

add_executable(test_mamba tests/test_mamba.cpp)
target_link_libraries(test_mamba PRIVATE microtorch)
add_test(NAME mamba COMMAND test_mamba)

# Phase-1c success test; needs a checkpoint + reference on disk, so it is a
# tool rather than a registered test (tools/make_gpt2_reference.py builds
# the reference).
add_executable(gpt2_parity tools/gpt2_parity.cpp)
target_link_libraries(gpt2_parity PRIVATE microtorch nlohmann_json::nlohmann_json)

# Phase-2b Llama-family test: load Qwen 2B safetensors and verify RMSNorm + RoPE
# work end-to-end. Needs a checkpoint on disk (downloaded separately).
add_executable(qwen_parity tools/qwen_parity.cpp)
target_link_libraries(qwen_parity PRIVATE microtorch nlohmann_json::nlohmann_json)

# Phase 3a: Benchmark Kimi Linear vs standard attention
add_executable(benchmark_attention tools/benchmark_attention.cpp)
target_link_libraries(benchmark_attention PRIVATE microtorch)

# Phase 3c: Benchmark Mamba vs Kimi vs standard attention
add_executable(benchmark_mamba tools/benchmark_mamba.cpp)
target_link_libraries(benchmark_mamba PRIVATE microtorch)

# Phase 3 Integration: Unified test of all three mechanisms
add_executable(phase3_integration tools/phase3_integration_test.cpp)
target_link_libraries(phase3_integration PRIVATE microtorch)
