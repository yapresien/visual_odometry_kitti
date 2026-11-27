cmake_minimum_required(VERSION 3.16)
project(kitti_vo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
    /home/master/myspace/projects/visual_odometry_kitti/cpp_demo_samples/cl_sgbm
)

find_package(Threads REQUIRED)

add_executable(kitti_vo main.cpp 
    /home/master/myspace/projects/visual_odometry_kitti/cpp_demo_samples/cl_sgbm/sgbm_opencl.cpp
)
target_link_libraries(kitti_vo ${OpenCV_LIBS} ${OpenCL_LIBRARIES} OpenCL::OpenCL Threads::Threads)