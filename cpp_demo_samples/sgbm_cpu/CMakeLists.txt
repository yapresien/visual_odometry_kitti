cmake_minimum_required(VERSION 3.16)
project(sgbm_cpu_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

add_executable(test_sgbm test_sgbm.cpp sgbm_cpu.cpp)
#target_compile_options(test_sgbm PRIVATE -O3 -march=native)
# or:
target_compile_options(test_sgbm PRIVATE -O0 -g)

