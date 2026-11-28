// =============================================================================
// Project     : opencl_sgbm — Hardware-Agnostic OpenCL Semi-Global Block Matching
// File        : binary_cache.cpp
// Author      : <Your Name> <<your.email@example.com>>
// Created     : 2026-11-28
// License     : MIT (SPDX-License-Identifier: MIT)
// =============================================================================

#include "opencl_sgbm/binary_cache.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace opencl_sgbm {

HostPlatformInfo detectHostPlatform() {
  HostPlatformInfo info;
#if defined(_WIN32)
  info.osName = "Windows";
#elif defined(__APPLE__)
  info.osName = "macOS";
#elif defined(__linux__)
  info.osName = "Linux";
#else
  info.osName = "UnknownOS";
#endif

#if defined(__x86_64__) || defined(_M_X64)
  info.cpuArch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  info.cpuArch = "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  info.cpuArch = "armv7";
#else
  info.cpuArch = "unknown_arch";
#endif
  return info;
}

uint64_t fnv1aHash64(const std::string& data) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : data) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string resolveCacheDirectory() {
  if (const char* envDir = std::getenv("OPENCL_SGBM_CACHE_DIR")) {
    if (*envDir != '\0') return std::string(envDir);
  }
#if defined(_WIN32)
  if (const char* localApp = std::getenv("LOCALAPPDATA")) {
    return (fs::path(localApp) / "opencl_sgbm" / "cache").string();
  }
#else
  if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
    if (*xdg != '\0') return (fs::path(xdg) / "opencl_sgbm").string();
  }
  if (const char* home = std::getenv("HOME")) {
    return (fs::path(home) / ".cache" / "opencl_sgbm").string();
  }
#endif
  return (fs::temp_directory_path() / "opencl_sgbm_cache").string();
}

std::string buildCacheKeyPath(
    const HostPlatformInfo& host,
    const std::string& deviceName,
    const std::string& driverVersion,
    const std::string& buildOptions,
    const std::string& kernelSource) {
  auto sanitize = [](std::string s) {
    for (char& ch : s) {
      if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
    }
    return s;
  };

  const uint64_t hash = fnv1aHash64(buildOptions + "\n" + kernelSource);
  std::ostringstream oss;
  oss << host.osName << "_" << host.cpuArch << "_"
      << sanitize(deviceName) << "_" << sanitize(driverVersion)
      << "_" << std::hex << hash << ".bin";

  const fs::path dir(resolveCacheDirectory());
  std::error_code ec;
  fs::create_directories(dir, ec);
  return (dir / oss.str()).string();
}

bool loadProgramBinary(
    cl_context context,
    cl_device_id device,
    const std::string& cacheFilePath,
    const std::string& buildOptions,
    cl_program& outProgram) {
  std::ifstream ifs(cacheFilePath, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) return false;

  const std::streamsize size = ifs.tellg();
  if (size <= 0) return false;
  ifs.seekg(0, std::ios::beg);

  std::vector<u_char> binary(static_cast<size_t>(size));
  if (!ifs.read(reinterpret_cast<char*>(binary.data()), size)) return false;

  const size_t binLen = binary.size();
  const u_char* binPtr = binary.data();
  cl_int binStatus = CL_SUCCESS;
  cl_int err = CL_SUCCESS;

  outProgram = clCreateProgramWithBinary(context, 1, &device, &binLen, &binPtr, &binStatus, &err);
  if (err != CL_SUCCESS || binStatus != CL_SUCCESS || !outProgram) {
    if (outProgram) clReleaseProgram(outProgram);
    outProgram = nullptr;
    return false;
  }

  err = clBuildProgram(outProgram, 1, &device, buildOptions.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseProgram(outProgram);
    outProgram = nullptr;
    return false;
  }
  return true;
}

void saveProgramBinary(cl_program program, const std::string& cacheFilePath) {
  size_t binSize = 0;
  if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr) != CL_SUCCESS || binSize == 0) {
    return;
  }
  std::vector<unsigned char> binary(binSize);
  unsigned char* ptrs[1] = {binary.data()};
  if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(ptrs), ptrs, nullptr) != CL_SUCCESS) {
    return;
  }
  std::ofstream ofs(cacheFilePath, std::ios::binary | std::ios::trunc);
  if (ofs.is_open()) {
    ofs.write(reinterpret_cast<const char*>(binary.data()), static_cast<std::streamsize>(binSize));
  }
}

}  // namespace opencl_sgbm
