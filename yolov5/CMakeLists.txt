cmake_minimum_required(VERSION 3.10)

project(yolov5)
enable_language(CUDA)


add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)
set(OpenCV_DIR "G:\\kaifa_environment\\opencv\\opencv\\build")
set(TRT_DIR "G:\\kaifa_environment\\tensorrt\\TensorRT-8.2.2.1")
#set(TRT_DIR "G:\\kaifa_environment\\tensorrt\\TensorRT-8.6.1.6")

set(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")


# TODO(Call for PR): make cmake compatible with Windows
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")


set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# tensorrt
include_directories(${TRT_DIR}\\include)
link_directories(${TRT_DIR}\\lib)

# opencv
include_directories(${OpenCV_DIR}\\include)
link_directories(${OpenCV_DIR}\\x64\\vc15\\lib\\)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu ${PROJECT_SOURCE_DIR}/plugin/*.h)

#add_library(myplugins SHARED ${PLUGIN_SRCS})


#target_link_libraries(nvinfer cudart)


#设置编译后输出目录
set(CMAKE_RESULT_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(yolov5_det yolov5_det.cpp ${SRCS} ${PLUGIN_SRCS})

target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
#target_link_libraries(yolov5_det myplugins)
#target_link_libraries(yolov5_det opencv_world452d)





set_target_properties(yolov5_det PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY}) #exe
set_target_properties(yolov5_det PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #不清楚
set_target_properties(yolov5_det PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #lib


#set_target_properties(myplugins PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY}) #exe
#set_target_properties(myplugins PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #不清楚
#set_target_properties(myplugins PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #lib




add_library(yolov5_det_dll SHARED yolov5_det.cpp ${SRCS} ${PLUGIN_SRCS})
target_link_libraries(yolov5_det_dll nvinfer)
target_link_libraries(yolov5_det_dll cudart)
#target_link_libraries(yolov5_det_dll myplugins)
#target_link_libraries(yolov5_det_dll opencv_world452d)


#判断 因为有写dll可能是区分debug和release版本的，这里主要做这个判断
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    # Debug 配置
    target_link_libraries(yolov5_det opencv_world452d)
    target_link_libraries(yolov5_det_dll opencv_world452d)

elseif (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    # Release 配置
    target_link_libraries(yolov5_det opencv_world452)
    target_link_libraries(yolov5_det_dll opencv_world452)
else ()
    # 其他配置
endif ()


set_target_properties(yolov5_det_dll PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY}) #exe
set_target_properties(yolov5_det_dll PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #不清楚
set_target_properties(yolov5_det_dll PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RESULT_OUTPUT_DIRECTORY})  #lib



#install(TARGETS yolov5_det_dll DESTINATION  ${CMAKE_INSTALL_PREFIX}/output)


#add_executable(yolov5_cls yolov5_cls.cpp ${SRCS})
#target_link_libraries(yolov5_cls nvinfer)
#target_link_libraries(yolov5_cls cudart)
#target_link_libraries(yolov5_cls myplugins)
#target_link_libraries(yolov5_cls ${OpenCV_LIBS})

#add_executable(yolov5_seg yolov5_seg.cpp ${SRCS})
#target_link_libraries(yolov5_seg nvinfer)
#target_link_libraries(yolov5_seg cudart)
#target_link_libraries(yolov5_seg myplugins)
#target_link_libraries(yolov5_seg ${OpenCV_LIBS})

