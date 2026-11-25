cmake_minimum_required(VERSION 3.16)
project(kitti_vo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)

add_executable(kitti_vo main.cpp)
target_link_libraries(kitti_vo ${OpenCV_LIBS} Threads::Threads)