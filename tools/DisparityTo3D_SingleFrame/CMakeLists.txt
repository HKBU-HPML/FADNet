cmake_minimum_required(VERSION 2.8)
project(DisparityTo3D)

set(CMAKE_CXX_STANDARD 11)
include_directories(
        /opt/opencv-3.1.0/include
)
link_directories(
        /opt/opencv-3.1.0/lib
)
set(SOURCE_FILES main.cpp)
add_executable(DisparityTo3D ${SOURCE_FILES})
target_link_libraries(DisparityTo3D
        opencv_core
		opencv_imgcodecs
)
