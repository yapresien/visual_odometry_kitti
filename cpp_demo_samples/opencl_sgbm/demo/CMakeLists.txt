cmake_minimum_required(VERSION 3.16)
project(opencl_sgbm_kitti CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    /usr/include
    )

set(OPENCL_TARGET_VERSION 120 CACHE STRING "Target OpenCL version")

add_compile_definitions(
    CL_TARGET_OPENCL_VERSION=${OPENCL_TARGET_VERSION}
)

add_executable(opencl_sgbm_kitti kitti_dataset_demo.cpp)
target_link_libraries(opencl_sgbm_kitti PRIVATE opencl_sgbm::opencl_sgbm OpenCL::OpenCL ${OpenCV_LIBS} Threads::Threads)

