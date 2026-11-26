cmake_minimum_required(VERSION 3.16)
project(sgbm_cpu_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
/usr/include)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

find_package(Threads REQUIRED)

#target_compile_options(helper_sgbm PRIVATE -O3 -march=native -DNDEBUG)


add_executable(sgbm_opencl sgbm_opencl.hpp main.cpp)
target_link_libraries(sgbm_opencl PRIVATE OpenCL::OpenCL ${OpenCV_LIBS} Threads::Threads)
target_compile_options(sgbm_opencl PRIVATE -O0 -g)

# target_compile_definitions(sgbm_opencl PRIVATE UNIT_TESTS)

add_executable(sgbm_opencl_gtest sgbm_opencl.hpp test_opencl_kitti_100frames.cpp)
target_link_libraries(sgbm_opencl_gtest PRIVATE OpenCL::OpenCL GTest::gtest_main ${OpenCV_LIBS} Threads::Threads)
target_compile_definitions(sgbm_opencl_gtest PRIVATE UNIT_TESTS)

include(GoogleTest)
gtest_discover_tests(sgbm_opencl_gtest)