cmake_minimum_required(VERSION 3.10)

project(mouse)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)



add_executable(mouse main.cpp main.h
        ghub_mouse.h)
add_library(mousedll SHARED main.cpp main.h)