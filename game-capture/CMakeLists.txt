cmake_minimum_required(VERSION 3.26)
project(screen_record)
enable_language(CUDA)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
add_definitions(-DUNICODE -D_UNICODE)


if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif ()


set(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")  #3
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
set(OpenCV_DIR "G:\\kaifa_environment\\opencv\\opencv\\build")  #2

find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")


#G:\kaifa_environment\code\clion\screen-record\third_party\g2log\g2log.h
file(GLOB_RECURSE capture_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/third-part/g2log/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/third-part/g2log/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu
)

#include_directories(G:\\kaifa_environment\\code\\clion\\tensorrtx\\yolov5\\)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../yolov5/src/)

#cuda
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARIES})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-part/lib/x64/)

add_executable(bebo-capture Main.cpp ${capture_SOURCES})
target_precompile_headers(bebo-capture PUBLIC <windows.h> <d3d11.h> <dxgi.h>)
target_link_libraries(bebo-capture libyuv mincore.lib
        d3d11.lib
        dxgi.lib
        gdiplus.lib
        libyuv.lib
        util.lib
        ipc-util.lib
        psapi.lib
        winmm.lib
        msvcrtd.lib
        legacy_stdio_definitions.lib

)# opencv_world452d.lib


target_link_libraries(bebo-capture ${yolov5_BINARY_DIR}\\bin\\yolov5_det_dll.lib)




include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-part/libyuv/include/)
include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-part/g2log/)
include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-part/util/)
include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third-part/ipc-util/)
include_directories( PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda/)


add_library(bebo-capturedll SHARED ${capture_SOURCES})
target_link_libraries(bebo-capturedll libyuv mincore.lib
        d3d11.lib
        dxgi.lib
        gdiplus.lib
        libyuv.lib
        util.lib
        ipc-util.lib
        psapi.lib
        winmm.lib
        msvcrtd.lib
        legacy_stdio_definitions.lib
)


#add_executable(simialr-image cuda/similarImage.cu
#        cuda/fewgweg.cpp)
