# =============================================================================
# Project     : opencl_sgbm
# Author      : <Anshul Yadav> <<er.anshulyadav@gmail.com>>
# Created     : 2026-11-28
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(sgbm_opencl_project VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    /usr/include
                    )

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()


set(OPENCL_TARGET_VERSION 120 CACHE STRING "Target OpenCL version" )
add_compile_definitions(
    CL_TARGET_OPENCL_VERSION=${OPENCL_TARGET_VERSION}
)

# ─── 1. Hardware-Agnostic OpenCL SGBM Library (libopencl_sgbm) ─────────────────
add_library(opencl_sgbm SHARED
  src/opencl_sgbm.cpp
)
add_library(opencl_sgbm::opencl_sgbm ALIAS opencl_sgbm)
target_include_directories(opencl_sgbm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(opencl_sgbm
  PUBLIC
    OpenCL::OpenCL
    Threads::Threads
)
target_compile_options(opencl_sgbm PRIVATE
  $<$<CONFIG:Release>:-O3 -DNDEBUG -Wall -Wextra>
)
target_compile_definitions(opencl_sgbm PUBLIC
  OPENCL_SGBM_DEFAULT_KERNEL_PATH="${CMAKE_INSTALL_PREFIX}/share/opencl_sgbm/opencl_sgbm.cl"
)


# ─── 2. Optional Demo & GoogleTest Benchmarks ─────────────────────────────────
option(OPENCL_SGBM_BUILD_TESTS "Build demo and GoogleTest benchmarks" ON)
if(OPENCL_SGBM_BUILD_TESTS)
  find_package(OpenCV REQUIRED)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(opencl_sgbm_gtest test_opencl_kitti_100frames.cpp)
  target_link_libraries(opencl_sgbm_gtest PRIVATE opencl_sgbm::opencl_sgbm GTest::gtest_main ${OpenCV_LIBS})
  target_compile_definitions(opencl_sgbm_gtest PRIVATE UNIT_TESTS OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION})
  target_compile_options(opencl_sgbm_gtest PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)

  include(GoogleTest)
  gtest_discover_tests(opencl_sgbm_gtest)
endif()

# add_executable(opencl_sgbm_gtest test_opencl_kitti_100frames.cpp)
# target_link_libraries(opencl_sgbm_gtest PRIVATE opencl_sgbm OpenCL::OpenCL GTest::gtest_main ${OpenCV_LIBS} Threads::Threads)

# target_compile_definitions(
#     opencl_sgbm_gtest
#     PRIVATE
#     UNIT_TESTS
#     OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
# )

# target_compile_options(opencl_sgbm PRIVATE -O3 -march=native -DNDEBUG)
# target_compile_options(opencl_sgbm_gtest PRIVATE -O3 -march=native -DNDEBUG)

# include(GoogleTest)
# gtest_discover_tests(opencl_sgbm_gtest)

# ─── 3. Install & CMake Package Export ────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS opencl_sgbm
  EXPORT opencl_sgbmTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/opencl_sgbm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES kernels/opencl_sgbm.cl
  DESTINATION ${CMAKE_INSTALL_DATADIR}/opencl_sgbm
)

install(EXPORT opencl_sgbmTargets
  FILE opencl_sgbmTargets.cmake
  NAMESPACE opencl_sgbm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opencl_sgbm
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/opencl_sgbmConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/opencl_sgbmConfig.cmake"
  "include(CMakeFindDependencyMacro)\n"
  "find_dependency(OpenCL REQUIRED)\n"
  "find_dependency(Threads REQUIRED)\n"
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/opencl_sgbmTargets.cmake\")\n"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/opencl_sgbmConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/opencl_sgbmConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opencl_sgbm
)

set(CPACK_PACKAGE_NAME "libopencl-sgbm")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "<Anshul Yadav>")
set(CPACK_PACKAGE_CONTACT "<Anshul Yadav> <<er.anshulyadav@gmail.com>>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "<Anshul Yadav> <<er.anshulyadav@gmail.com>>")
set(CPACK_GENERATOR "TGZ;DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "ocl-icd-libopencl1 | libopencl1")
include(CPack)
