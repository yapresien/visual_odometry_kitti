// =============================================================================
// Project     : opencl_sgbm — Hardware-Agnostic OpenCL Semi-Global Block Matching
// File        : binary_cache.h
// Author      : <Your Name> <<your.email@example.com>>
// Created     : 2026-11-28
// License     : MIT (SPDX-License-Identifier: MIT)
// Description : OS/CPU architecture detection and persistent on-disk OpenCL
//               ISA binary cache keyed by device, driver, flags, and source hash.
// =============================================================================

#pragma once

#ifndef OPENCL_TARGET_VERSION
#define OPENCL_TARGET_VERSION 120
#endif

#include <CL/cl.h>

#include <cstdint>
#include <string>

namespace opencl_sgbm {

struct HostPlatformInfo {
  std::string osName;
  std::string cpuArch;
};

HostPlatformInfo detectHostPlatform();

uint64_t fnv1aHash64(const std::string& data);

std::string resolveCacheDirectory();

std::string buildCacheKeyPath(
    const HostPlatformInfo& host,
    const std::string& deviceName,
    const std::string& driverVersion,
    const std::string& buildOptions,
    const std::string& kernelSource);

bool loadProgramBinary(
    cl_context context,
    cl_device_id device,
    const std::string& cacheFilePath,
    const std::string& buildOptions,
    cl_program& outProgram);

void saveProgramBinary(cl_program program, const std::string& cacheFilePath);

}  // namespace opencl_sgbm
