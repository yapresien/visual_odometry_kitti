# =============================================================================
# Project     : opencl_sgbm
# Author      : Anshul Yadav <er.anshulyadav@gmail.com>
# Created     : 2026-11-28
# License     : MIT (SPDX-License-Identifier: MIT)
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(opencl_sgbm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()


#target_compile_options(helper_sgbm PRIVATE -O3 -march=native -DNDEBUG)
set(OPENCL_TARGET_VERSION 120 CACHE STRING
    "Target OpenCL version")
add_compile_definitions(
    CL_TARGET_OPENCL_VERSION=${OPENCL_TARGET_VERSION}
)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

# ─── 1. Shared Library: libopencl_sgbm ────────────────────────────────────────
add_library(opencl_sgbm SHARED
  src/opencl_sgbm.cpp
  src/binary_cache.cpp
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
  OPENCL_SGBM_SOURCE_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/opencl_sgbm.cl"
  OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
)

# ─── 2. Standalone CLI Target Profiler (Zero OpenCV Dependency) ────────────────
add_executable(opencl_sgbm_profile
  apps/opencl_sgbm_profile.cpp
)
target_link_libraries(opencl_sgbm_profile PRIVATE opencl_sgbm::opencl_sgbm)
target_compile_options(opencl_sgbm_profile PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
target_compile_definitions(
    opencl_sgbm_profile
    PRIVATE
    OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
)

# ─── 3.  KITTI DATASET DEMO ─────
option(OPENCL_KITTI_DEMO_RUN "Build kitti data set demo run" ON)
if(OPENCL_KITTI_DEMO_RUN)
find_package(OpenCV REQUIRED)
add_executable(opencl_sgbm_kitti
  demo/kitti_dataset_demo.cpp
)
target_link_libraries(opencl_sgbm_kitti PRIVATE  opencl_sgbm::opencl_sgbm OpenCL::OpenCL ${OpenCV_LIBS} Threads::Threads)
target_compile_options(opencl_sgbm_kitti PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
target_compile_definitions(
    opencl_sgbm_kitti
    PRIVATE
    OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
)
endif()

# ─── 3. GoogleTest Suite (Synthetic Unit Tests + Optional KITTI Benchmark) ─────
option(OPENCL_SGBM_BUILD_TESTS "Build GoogleTest unit and benchmark suites" ON)
if(OPENCL_SGBM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
  )

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  # Hardware-agnostic synthetic unit tests (zero external dataset required)
  add_executable(opencl_sgbm_unit_tests
    tests/test_unit_stages.cpp
  )
  target_link_libraries(opencl_sgbm_unit_tests PRIVATE opencl_sgbm::opencl_sgbm GTest::gtest_main)
  target_compile_options(opencl_sgbm_unit_tests PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
  target_compile_definitions(
      opencl_sgbm_unit_tests
      PRIVATE
      UNIT_TESTS
      OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
  )
  include(GoogleTest)
  gtest_discover_tests(opencl_sgbm_unit_tests)

  # Optional 100-frame KITTI benchmark if OpenCV is installed
  find_package(OpenCV QUIET)
  if(OpenCV_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kitti_benchmark.cpp")
    add_executable(opencl_sgbm_kitti_gtest
      tests/test_kitti_benchmark.cpp
    )
    target_link_libraries(opencl_sgbm_kitti_gtest PRIVATE opencl_sgbm::opencl_sgbm GTest::gtest_main ${OpenCV_LIBS})
    target_compile_options(opencl_sgbm_kitti_gtest PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    target_compile_definitions(
      opencl_sgbm_kitti_gtest
      PRIVATE
      UNIT_TESTS
      OPENCL_TARGET_VERSION=${OPENCL_TARGET_VERSION}
    )
    gtest_discover_tests(opencl_sgbm_kitti_gtest)
  endif()
endif()

# ─── 4. Install Rules & CMake Config Export ───────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS opencl_sgbm opencl_sgbm_profile
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

# ─── 5. CPack Release Package (.tar.gz & .deb) ────────────────────────────────
set(CPACK_PACKAGE_NAME "libopencl-sgbm")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "Anshul Yadav")
set(CPACK_PACKAGE_CONTACT "Anshul Yadav <er.anshulyadav@gmail.com>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Anshul Yadav <er.anshulyadav@gmail.com>")
set(CPACK_GENERATOR "TGZ;DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "ocl-icd-libopencl1 | libopencl1")
include(CPack)
