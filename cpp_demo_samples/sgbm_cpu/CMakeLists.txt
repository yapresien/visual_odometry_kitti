cmake_minimum_required(VERSION 3.16)
project(sgbm_cpu_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)


add_library(helper_sgbm helper_sgbm.hpp sgbm_cpu.cpp)
target_link_libraries(helper_sgbm ${OpenCV_LIBS} Threads::Threads)

target_compile_options(helper_sgbm PRIVATE -O3 -march=native -DNDEBUG)
# or:
# target_compile_options(helper_sgbm PRIVATE -O0 -g)
add_executable(sgbm_gtest sgbm_gtest.cpp helper_sgbm.hpp sgbm_gtest_kitti_100frames.cpp sgbm_cpu.cpp)
target_link_libraries(sgbm_gtest helper_sgbm GTest::gtest_main ${OpenCV_LIBS} Threads::Threads)
target_compile_definitions(sgbm_gtest PRIVATE UNIT_TESTS)



include(GoogleTest)
gtest_discover_tests(sgbm_gtest)