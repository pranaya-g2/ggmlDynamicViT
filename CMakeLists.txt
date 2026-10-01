cmake_minimum_required(VERSION 3.16)
project(ggmlDViT LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(ggml)

add_library(dynvit-core STATIC
    src/model_load.cpp
    src/layers.cpp
    src/attention.cpp
    src/block.cpp
    src/predictor.cpp
    src/prune.cpp
    src/graph.cpp
    src/inference.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)

target_compile_features(dynvit-core PUBLIC cxx_std_17)
target_include_directories(dynvit-core PUBLIC src PRIVATE third_party)
target_link_libraries(dynvit-core PUBLIC ggml ggml-cpu)

add_executable(dynvit-eval tools/eval.cpp)
target_link_libraries(dynvit-eval PRIVATE dynvit-core)
