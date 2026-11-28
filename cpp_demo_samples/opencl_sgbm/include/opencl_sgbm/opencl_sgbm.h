// =============================================================================
// Project     : opencl_sgbm — Hardware-Agnostic OpenCL Semi-Global Block Matching
// File        : opencl_sgbm.h
// Author      : <Anshul Yadav> <<er.anshulyadav@gmail.com>>
// Created     : 2026-11-28
// License     : MIT / Apache-2.0 (SPDX-License-Identifier: MIT)
// Description : Self-tuning OpenCL 1.2 SGBM stereo matching library portable
//               across AMD APUs/dGPUs, NVIDIA CUDA/Jetson, and Intel GPUs.
// =============================================================================

#pragma once

#ifndef OPENCL_TARGET_VERSION
#define OPENCL_TARGET_VERSION 120
#endif

#ifndef OPENCL_SGBM_DEFAULT_KERNEL_PATH
#define OPENCL_SGBM_DEFAULT_KERNEL_PATH "kernels/opencl_sgbm.cl"
#endif

#include <CL/cl.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct DeviceSpecs {
  std::string deviceName;
  cl_uint maxComputeUnits = 1;
  size_t maxWorkGroupSize = 256;
  cl_ulong localMemSizeBytes = 32768;
  cl_bool unifiedMemory = CL_FALSE;
  size_t warpWidth = 32;         // Queried via CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
  size_t preferredWgSize = 256;  // Tunable override (clamped to device maxWorkGroupSize)

  // Derived work-group shapes (always multiples of warpWidth)
  size_t pfWgW = 32;
  size_t pfWgH = 8;
  size_t costWgW = 32;
  size_t costWgH = 8;
  size_t sgmWgSize = 32;
};

struct TuningConfig {
  size_t overrideWarpWidth = 0;      // 0 = auto-detect via CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
  size_t overrideWgSize = 0;         // 0 = auto-select (e.g., 256)
  size_t overrideSgmScanWgSize = 0;  // 0 = auto-match warpWidth (32 or 64)
  int maxBlockSize = 9;              // Sets LDS apron allocation size
  std::string preferredVendor = "";  // "" = auto-select, or "AMD", "NVIDIA", "Intel"
};

struct ClSgbmTimings {
  double prefilterMs = 0.0;
  double costVolumeMs = 0.0;
  double aggregationMs = 0.0;
  double wtaLrMs = 0.0;
  double speckleMs = 0.0;
  double totalGpuKernelMs = 0.0;
  double endToEndMs = 0.0;
};

struct ClSgbmParams {
  int minDisparity = 0;
  int blockSize = 5;
  int P1 = 200;
  int P2 = 800;
  int preFilterCap = 63;
  int uniquenessRatio = 10;
  int speckleWindowSize = 100;
  int speckleRange = 2;
  int disp12MaxDiff = 1;
  float invalidDisparity = -1.0f;
  int sgmPaths = 5;  // 4 = 4-path, 5 = MODE_SGBM (default), 8 = MODE_HH
  int downsampleFactor = 1;
};


class OpenClSgbm {
 public:
  // Zero-path constructor: automatically loads OPENCL_SGBM_DEFAULT_KERNEL_PATH
  // (with automatic fallback to "./kernels/opencl_sgbm.cl" for uninstalled local runs)
  OpenClSgbm(
      int width,
      int height,
      int numDisparities,
      const TuningConfig& tuning = TuningConfig{});

  // Explicit single-file .cl path constructor
  OpenClSgbm(
      const std::vector<std::string>& clKernelPath,
      int width,
      int height,
      int numDisparities,
      const TuningConfig& tuning = TuningConfig{});


  ~OpenClSgbm();

  OpenClSgbm(const OpenClSgbm&) = delete;
  OpenClSgbm& operator=(const OpenClSgbm&) = delete;

  const DeviceSpecs& deviceSpecs() const noexcept { return specs_; }

  void compute(
      const std::vector<uint8_t>& rawLeft,
      const std::vector<uint8_t>& rawRight,
      const ClSgbmParams& p,
      std::vector<float>& dispLOut,
      ClSgbmTimings& timings);

  // Stage-level benchmarking / unit-test entry points
  double runPrefilter(
      const std::vector<uint8_t>& rawLeft,
      int preFilterCap,
      std::vector<int16_t>& sobelOut,
      std::vector<cl_short2>& sobelBoundsOut,
      std::vector<cl_short2>& rawBoundsOut);

  double runBoxCost(
      const std::vector<uint8_t>& rawLeft,
      const std::vector<uint8_t>& rawRight,
      int preFilterCap,
      int minDisparity,
      int blockSize,
      std::vector<short>& costOut);

  double runAggregate(int P1, int P2, std::vector<uint16_t>& aggOut);

  double runDisparitySelection(const ClSgbmParams& p, std::vector<float>& dispLOut);

 private:
  std::vector<cl_device_id> selectDevice(const std::string& preferredVendor = "");
  void initContextAndQueue();
  DeviceSpecs queryAndTuneDevice(cl_device_id device, const TuningConfig& cfg);
  std::string detectOpenClCStandard(cl_device_id device);
  std::string buildCompilerOptions(const DeviceSpecs& specs, const TuningConfig& cfg, int numDisparities);
  void buildProgram(const std::vector<std::string>& clFilePaths);
  void refineWarpWidthFromKernel();
  void allocateBuffers();

  void speckleFilterHost(
      std::vector<float>& disp,
      int width,
      int height,
      int speckleWindowSize,
      int speckleRange,
      float invalidDisparity);

 private:
  std::vector<std::string> clFilePaths_;
  int width_ = 0;
  int height_ = 0;
  int numDisparities_ = 0;
  size_t numPixels_ = 0;

  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;

  // Active OpenCL kernels
  cl_kernel kernelPrefilter_ = nullptr;
  cl_kernel kernelBtPixel_ = nullptr;
  cl_kernel kernelAggDir_ = nullptr;
  cl_kernel kernelAggBidiAxis_ = nullptr;
  cl_kernel kernelClearUint_ = nullptr;
  cl_kernel kernelWtaScatter_ = nullptr;
  cl_kernel kernelLrCheck_ = nullptr;
  cl_kernel kernelBtBoxFused_ = nullptr;

  // Active OpenCL device buffers
  cl_mem bufRawL_ = nullptr;
  cl_mem bufRawR_ = nullptr;
  cl_mem bufSobelL_ = nullptr;
  cl_mem bufSobelR_ = nullptr;
  cl_mem bufSobelBoundsL_ = nullptr;
  cl_mem bufSobelBoundsR_ = nullptr;
  cl_mem bufRawBoundsL_ = nullptr;
  cl_mem bufRawBoundsR_ = nullptr;
  cl_mem bufPixelCost_ = nullptr;
  cl_mem bufCostL_ = nullptr;
  cl_mem bufAggL_ = nullptr;
  cl_mem bufRightMinPacked_ = nullptr;
  cl_mem bufDispL_ = nullptr;

  DeviceSpecs specs_;
  TuningConfig tuningConfig_;

  // Reusable scratch memory for host connected-component speckle filtering
  std::vector<uint8_t> speckleVisited_;
  std::vector<std::pair<int, int>> speckleStack_;
  std::vector<std::pair<int, int>> speckleComponent_;

  cl_kernel kernelDownsample2x_ = nullptr;
  cl_kernel kernelUpsample2x_ = nullptr;
  cl_mem bufFullRawL_ = nullptr;
  cl_mem bufFullRawR_ = nullptr;
  cl_mem bufFullDispL_ = nullptr;
};
