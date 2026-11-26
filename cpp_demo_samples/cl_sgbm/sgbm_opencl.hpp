// #define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CL_CHECK(err)                                                                                                        \
  do {                                                                                                                       \
    cl_int _err = (err);                                                                                                     \
    if (_err != CL_SUCCESS) {                                                                                                \
      throw std::runtime_error("OpenCL error " + std::to_string(_err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                                                                        \
  } while (0)

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
  size_t overrideWarpWidth = 0;              // 0 = auto-detect via CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
  size_t overrideWgSize = 0;                 // 0 = auto-select (e.g., 256)
  size_t overrideSgmScanWgSize = 0;          // 0 = auto-match warpWidth (32 or 64)
  int maxBlockSize = 9;                      // Sets LDS apron allocation size
  std::string preferredVendor = "";          // "" = auto-select, or "NVIDIA", "Intel", "AMD", etc.
  bool enableSobelBounds = true;             // Enable/disable Sobel bounds optimization
  bool enableRawBounds = true;               // Enable/disable raw bounds optimization
  bool enableBoxCostBounds = true;           // Enable/disable box cost bounds optimization
  bool enableSgmBounds = true;               // Enable/disable SGM bounds optimization
  bool enableSgmScanBounds = true;           // Enable/disable SGM scan bounds
  bool enableSgmScanWarpSync = true;         // Enable/disable warp-synchronous SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffle = true;      // Enable/disable warp-shuffle SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffleXor = true;   // Enable/disable warp-shuffle-xor SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffleXor2 = true;  // Enable/disable warp-shuffle-xor2 SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffleXor3 = true;  // Enable/disable warp-shuffle-xor3 SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffleXor4 = true;  // Enable/disable warp-shuffle-xor4 SGM scan (requires warp-synchronous scan support)
  bool enableSgmScanWarpShuffleXor5 = true;  // Enable/disable warp-shuffle-xor5 SGM scan (requires warp-synchronous scan support)
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
  float invalidDisparity = -1.0f;  // Match sgbm::kInvalidDisparity
};

class SgbmOpenClRunner {
 public:
  SgbmOpenClRunner(const std::vector<std::string>& clFilePaths, int width, int height, int numDisparities, const TuningConfig& tuning = TuningConfig{}) : clFilePaths_(clFilePaths), width_(width), height_(height), numDisparities_(numDisparities), numPixels_(static_cast<size_t>(width) * height), tuningConfig_(tuning) {
    if (width_ <= 0 || height_ <= 0 || numDisparities_ <= 0) {
      throw std::invalid_argument("Invalid image dimensions or numDisparities");
    }

    auto devices = selectDevice(tuningConfig_.preferredVendor);
    device_ = devices.empty() ? nullptr : devices[0];
    assert(device_ != nullptr);
    specs_ = queryAndTuneDevice(device_, tuningConfig_);
    initContextAndQueue();
    buildProgram(clFilePaths);
    refineWarpWidthFromKernel();  // Queries CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE & rebuilds if needed
    allocateBuffers();
  }

  ~SgbmOpenClRunner() {
    if (bufRawL_) clReleaseMemObject(bufRawL_);
    if (bufSobelL_) clReleaseMemObject(bufSobelL_);
    if (bufSobelBoundsL_) clReleaseMemObject(bufSobelBoundsL_);
    if (bufRawBoundsL_) clReleaseMemObject(bufRawBoundsL_);
    if (kernelPrefilter_) clReleaseKernel(kernelPrefilter_);
    if (kernelBtPixel_) clReleaseKernel(kernelBtPixel_);
    if (kernelBoxFilter_) clReleaseKernel(kernelBoxFilter_);
    if (kernelAggDir_) clReleaseKernel(kernelAggDir_);
    if (kernelWtaSubpixel_) clReleaseKernel(kernelWtaSubpixel_);
    if (kernelRightWta_) clReleaseKernel(kernelRightWta_);
    if (kernelLrCheck_) clReleaseKernel(kernelLrCheck_);

    if (bufDispL_) clReleaseMemObject(bufDispL_);
    if (bufDispR_) clReleaseMemObject(bufDispR_);

    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
  }


  std::vector<cl_device_id> selectDevice(const std::string& preferredVendor = "") {
    cl_uint numPlatforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &numPlatforms));
    if (numPlatforms == 0) throw std::runtime_error("No OpenCL platforms found");

    std::vector<cl_platform_id> platforms(numPlatforms);
    CL_CHECK(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

    cl_device_id fallbackGpu = nullptr;

    for (auto plat : platforms) {
      cl_uint numDevices = 0;
      if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) != CL_SUCCESS || numDevices == 0) {
        continue;
      }

      std::vector<cl_device_id> devices(numDevices);
      CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr));

      std::array<char, 256> vendor{};
      clGetPlatformInfo(plat, CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr);
      const std::string vendorStr(vendor.data());

      if (!preferredVendor.empty() && vendorStr.find(preferredVendor) != std::string::npos) {
        platform_ = plat;
        return devices;
      }

      if (!fallbackGpu) {
        platform_ = plat;
        fallbackGpu = devices[0];
      }
    }

    if (!fallbackGpu) throw std::runtime_error("No OpenCL GPU device found");
    return {fallbackGpu};
  }



  void refineWarpWidthFromKernel() {
    if (tuningConfig_.overrideWarpWidth > 0) return;  // User explicitly forced a warp width

    size_t hwWarpWidth = 0;
    CL_CHECK(clGetKernelWorkGroupInfo(kernelPrefilter_, device_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(hwWarpWidth), &hwWarpWidth, nullptr));

    if (hwWarpWidth > 0 && hwWarpWidth != specs_.warpWidth) {
      tuningConfig_.overrideWarpWidth = hwWarpWidth;
      specs_ = queryAndTuneDevice(device_, tuningConfig_);
      // Release old program/kernels and rebuild with exact hardware warp width
      clReleaseKernel(kernelBoxFilter_);
      clReleaseKernel(kernelBtPixel_);
      clReleaseKernel(kernelPrefilter_);
      clReleaseKernel(kernelAggDir_);
      clReleaseProgram(program_);
      buildProgram(clFilePaths_);
    }
  }

  DeviceSpecs queryAndTuneDevice(cl_device_id device, const TuningConfig& cfg) {
    DeviceSpecs specs;
    char nameBuf[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nameBuf), nameBuf, nullptr);
    specs.deviceName = nameBuf;

    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(specs.maxComputeUnits), &specs.maxComputeUnits, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(specs.maxWorkGroupSize), &specs.maxWorkGroupSize, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(specs.localMemSizeBytes), &specs.localMemSizeBytes, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(specs.unifiedMemory), &specs.unifiedMemory, nullptr);

    // Use user override or safe default before kernel inspection
    specs.warpWidth = (cfg.overrideWarpWidth > 0) ? cfg.overrideWarpWidth : 32;
    size_t targetWg = (cfg.overrideWgSize > 0) ? cfg.overrideWgSize : 256;
    specs.preferredWgSize = std::min(targetWg, specs.maxWorkGroupSize);

    // Derive 2D tile dimensions: pfWgW = warpWidth (coalesced row), pfWgH = preferredWgSize / warpWidth
    specs.pfWgW = std::min(specs.warpWidth, specs.preferredWgSize);
    specs.pfWgH = std::max<size_t>(1, specs.preferredWgSize / specs.pfWgW);

    specs.costWgW = specs.pfWgW;
    specs.costWgH = specs.pfWgH;
    specs.sgmWgSize = (cfg.overrideSgmScanWgSize > 0) ? cfg.overrideSgmScanWgSize : specs.warpWidth;

    return specs;
  }

  // Runs k_prefilter_and_bt_bounds on gfx1102 and returns GPU kernel execution time (ms)
  double runPrefilter(const std::vector<uint8_t>& rawLeft, int preFilterCap, std::vector<int16_t>& sobelOut, std::vector<cl_short2>& sobelBoundsOut, std::vector<cl_short2>& rawBoundsOut) {
    cl_int err = CL_SUCCESS;

    // 1. Zero-copy map input buffer into host address space (Radeon 890M unified memory)
    void* mappedIn = clEnqueueMapBuffer(queue_, bufRawL_, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, numPixels_ * sizeof(uint8_t), 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    std::memcpy(mappedIn, rawLeft.data(), numPixels_ * sizeof(uint8_t));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawL_, mappedIn, 0, nullptr, nullptr));

    // 2. Bind kernel arguments
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &preFilterCap));

    // 3. Dispatch 32x8 Wave32 work-groups (256 threads/WG)
    // Lines 68-71 in runPrefilter():
    // Stage 1 dispatch:
    const std::array<size_t, 2> pfLocal = {specs_.pfWgW, specs_.pfWgH};
    const std::array<size_t, 2> pfGlobal = {((static_cast<size_t>(width_) + specs_.pfWgW - 1) / specs_.pfWgW) * specs_.pfWgW, ((static_cast<size_t>(height_) + specs_.pfWgH - 1) / specs_.pfWgH) * specs_.pfWgH};

    cl_event profEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &profEvent));

    CL_CHECK(clFinish(queue_));

    // Query hardware GPU profiling timestamps (1ns resolution per clinfo)
    cl_ulong tStart = 0, tEnd = 0;
    CL_CHECK(clGetEventProfilingInfo(profEvent, CL_PROFILING_COMMAND_START, sizeof(tStart), &tStart, nullptr));
    CL_CHECK(clGetEventProfilingInfo(profEvent, CL_PROFILING_COMMAND_END, sizeof(tEnd), &tEnd, nullptr));
    clReleaseEvent(profEvent);
    const double kernelMs = static_cast<double>(tEnd - tStart) * 1e-6;

    // 4. Zero-copy readback of Sobel-X and packed short2 BT bounds
    sobelOut.resize(numPixels_);
    sobelBoundsOut.resize(numPixels_);
    rawBoundsOut.resize(numPixels_);

    auto* pSobel = static_cast<int16_t*>(clEnqueueMapBuffer(queue_, bufSobelL_, CL_TRUE, CL_MAP_READ, 0, numPixels_ * sizeof(int16_t), 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memcpy(sobelOut.data(), pSobel, numPixels_ * sizeof(int16_t));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufSobelL_, pSobel, 0, nullptr, nullptr));

    auto* pSobelB = static_cast<cl_short2*>(clEnqueueMapBuffer(queue_, bufSobelBoundsL_, CL_TRUE, CL_MAP_READ, 0, numPixels_ * sizeof(cl_short2), 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memcpy(sobelBoundsOut.data(), pSobelB, numPixels_ * sizeof(cl_short2));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufSobelBoundsL_, pSobelB, 0, nullptr, nullptr));

    auto* pRawB = static_cast<cl_short2*>(clEnqueueMapBuffer(queue_, bufRawBoundsL_, CL_TRUE, CL_MAP_READ, 0, numPixels_ * sizeof(cl_short2), 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memcpy(rawBoundsOut.data(), pRawB, numPixels_ * sizeof(cl_short2));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawBoundsL_, pRawB, 0, nullptr, nullptr));

    return kernelMs;
  }

  double runBoxCost(const std::vector<uint8_t>& rawLeft, const std::vector<uint8_t>& rawRight, int preFilterCap, int minDisparity, int blockSize, std::vector<int16_t>& costOut) {
    cl_int err = CL_SUCCESS;

    // 1. Zero-copy upload of Left and Right raw images
    void* pInL = clEnqueueMapBuffer(queue_, bufRawL_, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, numPixels_ * sizeof(uint8_t), 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    std::memcpy(pInL, rawLeft.data(), numPixels_ * sizeof(uint8_t));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawL_, pInL, 0, nullptr, nullptr));

    void* pInR = clEnqueueMapBuffer(queue_, bufRawR_, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, numPixels_ * sizeof(uint8_t), 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    std::memcpy(pInR, rawRight.data(), numPixels_ * sizeof(uint8_t));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawR_, pInR, 0, nullptr, nullptr));

    // Stage 2A & 2B dispatch:
    const std::array<size_t, 3> pfLocal = {specs_.costWgW, specs_.costWgH, 1};
    const std::array<size_t, 3> pfGlobal = {((static_cast<size_t>(width_) + specs_.costWgH - 1) / specs_.costWgH) * specs_.costWgW, static_cast<size_t>(height_) * specs_.costWgH, 1};
    // Left prefilter + BT bounds
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &preFilterCap));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, nullptr));

    // 3A. Dispatch Kernel 2A: k_bt_pixel_cost (32x8 = 256 threads/WG)
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 1, sizeof(cl_mem), &bufRawR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 2, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 3, sizeof(cl_mem), &bufSobelR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 4, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 5, sizeof(cl_mem), &bufSobelBoundsR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 6, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 7, sizeof(cl_mem), &bufRawBoundsR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 8, sizeof(cl_mem), &bufPixelCost_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 9, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 10, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 11, sizeof(int), &minDisparity));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 12, sizeof(int), &numDisparities_));

    const std::array<size_t, 3> stage2Local_bt = {specs_.costWgW, specs_.costWgH, 1};
    const std::array<size_t, 3> stage2Global_bt = {static_cast<size_t>((width_ + specs_.costWgW - 1) / specs_.costWgW) * specs_.costWgW, static_cast<size_t>(height_) * specs_.costWgH, 1};

    cl_event btEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBtPixel_, 3, nullptr, stage2Global_bt.data(), stage2Local_bt.data(), 0, nullptr, &btEvent));

    // 3B. Dispatch Kernel 2B: k_box_filter_cost (32x8 = 256 threads/WG)
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 0, sizeof(cl_mem), &bufPixelCost_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 4, sizeof(int), &minDisparity));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 5, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 6, sizeof(int), &blockSize));

    const std::array<size_t, 3> stage2Local_box = {specs_.costWgW, specs_.costWgH, 1};
    const std::array<size_t, 3> stage2Global_box = {static_cast<size_t>((width_ + specs_.costWgW - 1) / specs_.costWgW) * specs_.costWgW, static_cast<size_t>(height_) * specs_.costWgH, 1};

    cl_event boxEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBoxFilter_, 3, nullptr, stage2Global_box.data(), stage2Local_box.data(), 0, nullptr, &boxEvent));
    CL_CHECK(clFinish(queue_));

    cl_ulong t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    CL_CHECK(clGetEventProfilingInfo(btEvent, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(btEvent, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
    CL_CHECK(clGetEventProfilingInfo(boxEvent, CL_PROFILING_COMMAND_START, sizeof(t2), &t2, nullptr));
    CL_CHECK(clGetEventProfilingInfo(boxEvent, CL_PROFILING_COMMAND_END, sizeof(t3), &t3, nullptr));
    clReleaseEvent(btEvent);
    clReleaseEvent(boxEvent);

    const double btMs = static_cast<double>(t1 - t0) * 1e-6;
    const double boxMs = static_cast<double>(t3 - t2) * 1e-6;
    return btMs + boxMs;
  }

  double runAggregate(int P1, int P2, std::vector<uint16_t>& aggOut) {
    struct Dir {
      int dx;
      int dy;
      size_t numScanlines;
    };

    const std::array<Dir, 8> dirs = {{{1, 0, static_cast<size_t>(height_)}, {-1, 0, static_cast<size_t>(height_)}, {0, 1, static_cast<size_t>(width_)}, {0, -1, static_cast<size_t>(width_)}, {1, 1, static_cast<size_t>(width_ + height_ - 1)}, {-1, 1, static_cast<size_t>(width_ + height_ - 1)}, {1, -1, static_cast<size_t>(width_ + height_ - 1)}, {-1, -1, static_cast<size_t>(width_ + height_ - 1)}}};

    CL_CHECK(clSetKernelArg(kernelAggDir_, 0, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 1, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 4, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 5, sizeof(int), &P1));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 6, sizeof(int), &P2));

    std::array<cl_event, 8> events{};
    const size_t localSize = specs_.sgmWgSize;

    for (size_t i = 0; i < dirs.size(); ++i) {
      const int isFirst = (i == 0) ? 1 : 0;
      CL_CHECK(clSetKernelArg(kernelAggDir_, 7, sizeof(int), &dirs[i].dx));
      CL_CHECK(clSetKernelArg(kernelAggDir_, 8, sizeof(int), &dirs[i].dy));
      CL_CHECK(clSetKernelArg(kernelAggDir_, 9, sizeof(int), &isFirst));

      const size_t globalSize = dirs[i].numScanlines * localSize;
      CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelAggDir_, 1, nullptr, &globalSize, &localSize, 0, nullptr, &events[i]));
    }

    CL_CHECK(clFinish(queue_));

    double totalAggMs = 0.0;
    for (size_t i = 0; i < dirs.size(); ++i) {
      cl_ulong t0 = 0, t1 = 0;
      CL_CHECK(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
      CL_CHECK(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
      clReleaseEvent(events[i]);
      totalAggMs += static_cast<double>(t1 - t0) * 1e-6;
    }

    cl_int err = CL_SUCCESS;
    const size_t totalElems = numPixels_ * static_cast<size_t>(numDisparities_);
    aggOut.resize(totalElems);
    auto* pAgg = static_cast<int32_t*>(clEnqueueMapBuffer(queue_, bufAggL_, CL_TRUE, CL_MAP_READ, 0, totalElems * sizeof(int32_t), 0, nullptr, nullptr, &err));
    CL_CHECK(err);
    std::memcpy(aggOut.data(), pAgg, totalElems * sizeof(int32_t));
    CL_CHECK(clEnqueueUnmapMemObject(queue_, bufAggL_, pAgg, 0, nullptr, nullptr));

    return totalAggMs;
  }

  double runDisparitySelection(const ClSgbmParams& p, std::vector<float>& dispLOut) {
    // 4A. Left WTA + non-adjacent uniqueness + parabolic sub-pixel
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 0, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 2, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 3, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 4, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 5, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 6, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 7, sizeof(int), &p.uniquenessRatio));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 8, sizeof(float), &p.invalidDisparity));

    const std::array<size_t, 2> wtaLocal = {specs_.sgmWgSize, 1};
    const std::array<size_t, 2> wtaGlobal = {static_cast<size_t>(width_) * specs_.sgmWgSize, static_cast<size_t>(height_)};

    cl_event evWtaL = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelWtaSubpixel_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWtaL));

    // 4B. Shared-volume Right WTA
    CL_CHECK(clSetKernelArg(kernelRightWta_, 0, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 2, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 3, sizeof(cl_mem), &bufDispR_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 6, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 7, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 8, sizeof(int), &p.disp12MaxDiff));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 9, sizeof(float), &p.invalidDisparity));

    cl_event evWtaR = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelRightWta_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWtaR));

    // 4C. Integer Left-Right consistency check
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 0, sizeof(cl_mem), &bufDispR_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 1, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 4, sizeof(int), &p.disp12MaxDiff));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 5, sizeof(float), &p.invalidDisparity));

    const std::array<size_t, 2> lrLocal = {specs_.pfWgW, specs_.pfWgH};
    const std::array<size_t, 2> lrGlobal = {((static_cast<size_t>(width_) + specs_.pfWgW - 1) / specs_.pfWgW) * specs_.pfWgW, ((static_cast<size_t>(height_) + specs_.pfWgH - 1) / specs_.pfWgH) * specs_.pfWgH};

    cl_event evLr = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelLrCheck_, 2, nullptr, lrGlobal.data(), lrLocal.data(), 0, nullptr, &evLr));
    CL_CHECK(clFinish(queue_));

    auto eventMs = [](cl_event ev) {
      cl_ulong t0 = 0, t1 = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
      clReleaseEvent(ev);
      return static_cast<double>(t1 - t0) * 1e-6;
    };

    const double wtaLrMs = eventMs(evWtaL) + eventMs(evWtaR) + eventMs(evLr);

    // Zero-copy map/read final 1.86 MB dispL buffer
    cl_int err = CL_SUCCESS;
    dispLOut.resize(numPixels_);
    if (specs_.unifiedMemory == CL_TRUE) {
      auto* pDisp = static_cast<float*>(clEnqueueMapBuffer(queue_, bufDispL_, CL_TRUE, CL_MAP_READ, 0, numPixels_ * sizeof(float), 0, nullptr, nullptr, &err));
      CL_CHECK(err);
      std::memcpy(dispLOut.data(), pDisp, numPixels_ * sizeof(float));
      CL_CHECK(clEnqueueUnmapMemObject(queue_, bufDispL_, pDisp, 0, nullptr, nullptr));
    } else {
      CL_CHECK(clEnqueueReadBuffer(queue_, bufDispL_, CL_TRUE, 0, numPixels_ * sizeof(float), dispLOut.data(), 0, nullptr, nullptr));
    }

    // Host speckle filter (~0.4 ms in -O3 Release)
    speckleFilterHost(dispLOut, width_, height_, p.speckleWindowSize, p.speckleRange, p.invalidDisparity);
    return wtaLrMs;
  }

  void compute(const std::vector<uint8_t>& rawLeft, const std::vector<uint8_t>& rawRight, const ClSgbmParams& p, std::vector<float>& dispLOut, ClSgbmTimings& timings) {
    const auto tWallStart = std::chrono::steady_clock::now();
    cl_int err = CL_SUCCESS;

    auto eventMs = [](cl_event ev) {
      cl_ulong t0 = 0, t1 = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
      clReleaseEvent(ev);
      return static_cast<double>(t1 - t0) * 1e-6;
    };

    // 1. Upload Left & Right images (zero-copy map on unified APUs)
    if (specs_.unifiedMemory == CL_TRUE) {
      void* pInL = clEnqueueMapBuffer(queue_, bufRawL_, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, numPixels_ * sizeof(uint8_t), 0, nullptr, nullptr, &err);
      CL_CHECK(err);
      std::memcpy(pInL, rawLeft.data(), numPixels_ * sizeof(uint8_t));
      CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawL_, pInL, 0, nullptr, nullptr));

      void* pInR = clEnqueueMapBuffer(queue_, bufRawR_, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, numPixels_ * sizeof(uint8_t), 0, nullptr, nullptr, &err);
      CL_CHECK(err);
      std::memcpy(pInR, rawRight.data(), numPixels_ * sizeof(uint8_t));
      CL_CHECK(clEnqueueUnmapMemObject(queue_, bufRawR_, pInR, 0, nullptr, nullptr));
    } else {
      CL_CHECK(clEnqueueWriteBuffer(queue_, bufRawL_, CL_FALSE, 0, numPixels_ * sizeof(uint8_t), rawLeft.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueWriteBuffer(queue_, bufRawR_, CL_FALSE, 0, numPixels_ * sizeof(uint8_t), rawRight.data(), 0, nullptr, nullptr));
    }

    // 2. Stage 1: Prefilter + BT Bounds (Left & Right)
    const std::array<size_t, 2> pfLocal = {specs_.pfWgW, specs_.pfWgH};
    const std::array<size_t, 2> pfGlobal = {((static_cast<size_t>(width_) + specs_.pfWgW - 1) / specs_.pfWgW) * specs_.pfWgW, ((static_cast<size_t>(height_) + specs_.pfWgH - 1) / specs_.pfWgH) * specs_.pfWgH};

    cl_event evPfL = nullptr, evPfR = nullptr;
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &p.preFilterCap));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evPfL));

    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &p.preFilterCap));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evPfR));

    // 3. Stage 2: 1x1 BT Pixel Cost + BxB Box Cost
    const std::array<size_t, 3> stage2Local = {specs_.costWgW, specs_.costWgH, 1};
    const std::array<size_t, 3> stage2Global = {((static_cast<size_t>(width_) + specs_.costWgH - 1) / specs_.costWgH) * specs_.costWgW, static_cast<size_t>(height_) * specs_.costWgH, 1};

    cl_event evBt = nullptr, evBox = nullptr;
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 1, sizeof(cl_mem), &bufRawR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 2, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 3, sizeof(cl_mem), &bufSobelR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 4, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 5, sizeof(cl_mem), &bufSobelBoundsR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 6, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 7, sizeof(cl_mem), &bufRawBoundsR_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 8, sizeof(cl_mem), &bufPixelCost_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 9, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 10, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 11, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelBtPixel_, 12, sizeof(int), &numDisparities_));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBtPixel_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &evBt));

    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 0, sizeof(cl_mem), &bufPixelCost_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 4, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 5, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 6, sizeof(int), &p.blockSize));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBoxFilter_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &evBox));

    // 4. Stage 3: 8-Path SGM Aggregation
    struct Dir {
      int dx;
      int dy;
      size_t numScanlines;
    };
    const std::array<Dir, 8> dirs = {{{1, 0, static_cast<size_t>(height_)}, {-1, 0, static_cast<size_t>(height_)}, {0, 1, static_cast<size_t>(width_)}, {0, -1, static_cast<size_t>(width_)}, {1, 1, static_cast<size_t>(width_ + height_ - 1)}, {-1, 1, static_cast<size_t>(width_ + height_ - 1)}, {1, -1, static_cast<size_t>(width_ + height_ - 1)}, {-1, -1, static_cast<size_t>(width_ + height_ - 1)}}};

    CL_CHECK(clSetKernelArg(kernelAggDir_, 0, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 1, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 4, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 5, sizeof(int), &p.P1));
    CL_CHECK(clSetKernelArg(kernelAggDir_, 6, sizeof(int), &p.P2));

    std::array<cl_event, 8> evAgg{};
    const size_t aggLocal = specs_.sgmWgSize;
    for (size_t i = 0; i < dirs.size(); ++i) {
      const int isFirst = (i == 0) ? 1 : 0;
      CL_CHECK(clSetKernelArg(kernelAggDir_, 7, sizeof(int), &dirs[i].dx));
      CL_CHECK(clSetKernelArg(kernelAggDir_, 8, sizeof(int), &dirs[i].dy));
      CL_CHECK(clSetKernelArg(kernelAggDir_, 9, sizeof(int), &isFirst));
      const size_t aggGlobal = dirs[i].numScanlines * aggLocal;
      CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelAggDir_, 1, nullptr, &aggGlobal, &aggLocal, 0, nullptr, &evAgg[i]));
    }

    // 5. Stage 4: Left WTA + Right WTA + LR Check
    const std::array<size_t, 2> wtaLocal = {specs_.sgmWgSize, specs_.costWgH};
    const std::array<size_t, 2> wtaGlobal = {
        ((static_cast<size_t>(width_) + specs_.costWgH - 1) / specs_.costWgH) * specs_.sgmWgSize,
        static_cast<size_t>(height_) * specs_.costWgH };

    cl_event evWtaL = nullptr, evWtaR = nullptr, evLr = nullptr;
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 0, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 2, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 3, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 4, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 5, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 6, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 7, sizeof(int), &p.uniquenessRatio));
    CL_CHECK(clSetKernelArg(kernelWtaSubpixel_, 8, sizeof(float), &p.invalidDisparity));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelWtaSubpixel_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWtaL));

    CL_CHECK(clSetKernelArg(kernelRightWta_, 0, sizeof(cl_mem), &bufAggL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 2, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 3, sizeof(cl_mem), &bufDispR_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 6, sizeof(int), &p.minDisparity));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 7, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 8, sizeof(int), &p.disp12MaxDiff));
    CL_CHECK(clSetKernelArg(kernelRightWta_, 9, sizeof(float), &p.invalidDisparity));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelRightWta_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWtaR));

    CL_CHECK(clSetKernelArg(kernelLrCheck_, 0, sizeof(cl_mem), &bufDispR_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 1, sizeof(cl_mem), &bufDispL_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 3, sizeof(int), &height_));

    CL_CHECK(clSetKernelArg(kernelLrCheck_, 4, sizeof(int), &p.disp12MaxDiff));
    CL_CHECK(clSetKernelArg(kernelLrCheck_, 5, sizeof(float), &p.invalidDisparity));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelLrCheck_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evLr));

    // Single synchronization point for the entire GPU pipeline
    CL_CHECK(clFinish(queue_));

    timings.prefilterMs = eventMs(evPfL) + eventMs(evPfR);
    timings.costVolumeMs = eventMs(evBt) + eventMs(evBox);
    timings.aggregationMs = 0.0;
    for (auto& ev : evAgg) {
      timings.aggregationMs += eventMs(ev);
    }
    timings.wtaLrMs = eventMs(evWtaL) + eventMs(evWtaR) + eventMs(evLr);
    timings.totalGpuKernelMs = timings.prefilterMs + timings.costVolumeMs + timings.aggregationMs + timings.wtaLrMs;

    // 6. Read back dispL & run host speckle filter
    dispLOut.resize(numPixels_);
    if (specs_.unifiedMemory == CL_TRUE) {
      auto* pDisp = static_cast<float*>(clEnqueueMapBuffer(queue_, bufDispL_, CL_TRUE, CL_MAP_READ, 0, numPixels_ * sizeof(float), 0, nullptr, nullptr, &err));
      CL_CHECK(err);
      std::memcpy(dispLOut.data(), pDisp, numPixels_ * sizeof(float));
      CL_CHECK(clEnqueueUnmapMemObject(queue_, bufDispL_, pDisp, 0, nullptr, nullptr));
    } else {
      CL_CHECK(clEnqueueReadBuffer(queue_, bufDispL_, CL_TRUE, 0, numPixels_ * sizeof(float), dispLOut.data(), 0, nullptr, nullptr));
    }

    const auto tSp0 = std::chrono::steady_clock::now();
    speckleFilterHost(dispLOut, width_, height_, p.speckleWindowSize, p.speckleRange, p.invalidDisparity);
    const auto tSp1 = std::chrono::steady_clock::now();

    timings.speckleMs = std::chrono::duration<double, std::milli>(tSp1 - tSp0).count();
    timings.endToEndMs = std::chrono::duration<double, std::milli>(tSp1 - tWallStart).count();
  }

 private:
  void initContextAndQueue() {
    cl_int err = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);

    // Check OpenCL device version so we use OpenCL 2.0 queue properties when supported
    // and cleanly fall back to OpenCL 1.2 clCreateCommandQueue on older drivers (e.g., NVIDIA OpenCL 1.2)
    std::array<char, 128> versionStr{};
    clGetDeviceInfo(device_, CL_DEVICE_VERSION, versionStr.size(), versionStr.data(), nullptr);
    const std::string ver(versionStr.data());

    if (ver.find("OpenCL 1.") == std::string::npos) {
      const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
      queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#pragma GCC diagnostic pop
    }
    CL_CHECK(err);
  }

  std::string detectOpenClCStandard(cl_device_id device) {
    std::array<char, 128> cVer{};
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, cVer.size(), cVer.data(), nullptr);
    const std::string verStr(cVer.data());
    if (verStr.find("OpenCL C 3.") != std::string::npos) return "-cl-std=CL3.0";
    if (verStr.find("OpenCL C 2.") != std::string::npos) return "-cl-std=CL2.0";
    return "-cl-std=CL1.2";
  }

  std::string buildCompilerOptions(const DeviceSpecs& specs, const TuningConfig& cfg, int numDisparities) {
    const int apron = 2;
    const int pfTileW = static_cast<int>(specs.pfWgW) + 2 * apron;
    const int pfTileH = static_cast<int>(specs.pfWgH) + 2 * apron;
    // Make LDS stride odd (coprime to power-of-2 LDS bank counts 16/32/64) to prevent bank conflicts
    const int pfLdsStride = (pfTileW % 2 == 0) ? (pfTileW + 1) : pfTileW;

    const int maxDPerLane = (numDisparities + static_cast<int>(specs.sgmWgSize) - 1) / static_cast<int>(specs.sgmWgSize);

    auto cl_ver = detectOpenClCStandard(device_);  // Ensure OpenCL C standard is detected before building options

    std::ostringstream opts;
    opts << cl_ver << " -cl-mad-enable -cl-fast-relaxed-math "
         << "-D WARP_WIDTH=" << specs.warpWidth << " "
         << "-D PF_WG_W=" << specs.pfWgW << " "
         << "-D PF_WG_H=" << specs.pfWgH << " "
         << "-D PF_TILE_W=" << pfTileW << " "
         << "-D PF_TILE_H=" << pfTileH << " "
         << "-D PF_LDS_STRIDE=" << pfLdsStride << " "
         << "-D COST_WG_W=" << specs.costWgW << " "
         << "-D COST_WG_H=" << specs.costWgH << " "
         << "-D SGM_WG_SIZE=" << specs.sgmWgSize << " "
         << "-D WTA_ROWS_PER_WG=" << specs.costWgH << " "
         << "-D NUM_DISPARITIES=" << numDisparities << " "
         << "-D D_PER_LANE=" << maxDPerLane << " "
         << "-D MAX_B2=" << (cfg.maxBlockSize / 2) << " "
         << "-D INVALID_COST=32767 ";
    return opts.str();
  }

  void buildProgram(const std::vector<std::string>& clFilePaths) {
    std::vector<std::string> srcStrings;
    std::string result;
    for (const auto& clFilePath : clFilePaths) {
      std::ifstream file(clFilePath);
      if (!file.is_open()) throw std::runtime_error("Failed to open kernel file: " + clFilePath);
      std::stringstream ss;
      ss << file.rdbuf();
      srcStrings.push_back(ss.str());
      result += srcStrings.back();
    }

    const std::string srcStr = result;
    const char* srcPtr = srcStr.c_str();
    const size_t srcLen = srcStr.size();

    cl_int err = CL_SUCCESS;
    program_ = clCreateProgramWithSource(context_, 1, &srcPtr, &srcLen, &err);
    CL_CHECK(err);

    // Target OpenCL C 2.0 on gfx1102 with fast math optimizations
    const char* buildOpts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    auto cflags = buildCompilerOptions(specs_, tuningConfig_, numDisparities_);  // Generate additional build options based on device specs and tuning
    if (cflags.size() > 0) {
      buildOpts = cflags.c_str();
    }
    std::cout << "Building OpenCL program with options: " << buildOpts << std::endl;

    err = clBuildProgram(program_, 1, &device_, buildOpts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize = 0;
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string buildLog(logSize, '\0');
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
      std::cerr << "OpenCL Build Log:\n" << buildLog << "\n";
      CL_CHECK(err);
    }

    kernelPrefilter_ = clCreateKernel(program_, "k_prefilter_and_bt_bounds", &err);
    CL_CHECK(err);
    kernelBtPixel_ = clCreateKernel(program_, "k_bt_pixel_cost", &err);
    CL_CHECK(err);

    kernelBoxFilter_ = clCreateKernel(program_, "k_box_filter_cost", &err);
    CL_CHECK(err);

    kernelAggDir_ = clCreateKernel(program_, "k_sgm_aggregate_dir", &err);
    CL_CHECK(err);

    // In buildProgram() (and release in refineWarpWidthFromKernel() / ~SgbmOpenClRunner()):
    kernelWtaSubpixel_ = clCreateKernel(program_, "k_wta_subpixel", &err);
    CL_CHECK(err);
    kernelRightWta_ = clCreateKernel(program_, "k_right_wta_and_lr_check", &err);
    CL_CHECK(err);
    kernelLrCheck_ = clCreateKernel(program_, "k_lr_check", &err);
    CL_CHECK(err);
  }

  void allocateBuffers() {
    cl_int err = CL_SUCCESS;

    const cl_mem_flags hostAllocFlag = (specs_.unifiedMemory == CL_TRUE) ? CL_MEM_ALLOC_HOST_PTR : 0;

    // Host-facing I/O buffers: zero-copy pinned memory on unified APUs, standard on discrete GPUs
    const cl_mem_flags hostReadFlags = CL_MEM_READ_ONLY | hostAllocFlag;
    const cl_mem_flags hostWriteFlags = CL_MEM_WRITE_ONLY | hostAllocFlag;

    // Pure GPU-resident working buffers: plain CL_MEM_READ_WRITE so AMD gfx1102 places them
    // in L2/MALL-cached GPU carve-out memory rather than uncached GTT host memory
    const cl_mem_flags gpuDeviceFlags = CL_MEM_READ_WRITE;

    const size_t rawBytes = numPixels_ * sizeof(uint8_t);
    const size_t sobelBytes = numPixels_ * sizeof(int16_t);
    const size_t boundsBytes = numPixels_ * sizeof(cl_short2);
    const size_t costVolumeBytes = numPixels_ * static_cast<size_t>(numDisparities_) * sizeof(int16_t);
    const size_t aggVolumeBytes = numPixels_ * static_cast<size_t>(numDisparities_) * sizeof(uint16_t);

    // Host-uploaded input images
    bufRawL_ = clCreateBuffer(context_, hostReadFlags, rawBytes, nullptr, &err);
    CL_CHECK(err);
    bufRawR_ = clCreateBuffer(context_, hostReadFlags, rawBytes, nullptr, &err);
    CL_CHECK(err);

    // GPU-resident Stage 1 outputs (Left & Right)
    bufSobelL_ = clCreateBuffer(context_, gpuDeviceFlags, sobelBytes, nullptr, &err);
    CL_CHECK(err);
    bufSobelBoundsL_ = clCreateBuffer(context_, gpuDeviceFlags, boundsBytes, nullptr, &err);
    CL_CHECK(err);
    bufRawBoundsL_ = clCreateBuffer(context_, gpuDeviceFlags, boundsBytes, nullptr, &err);
    CL_CHECK(err);

    bufSobelR_ = clCreateBuffer(context_, gpuDeviceFlags, sobelBytes, nullptr, &err);
    CL_CHECK(err);
    bufSobelBoundsR_ = clCreateBuffer(context_, gpuDeviceFlags, boundsBytes, nullptr, &err);
    CL_CHECK(err);
    bufRawBoundsR_ = clCreateBuffer(context_, gpuDeviceFlags, boundsBytes, nullptr, &err);
    CL_CHECK(err);

    // GPU-resident Stage 2 & Stage 3 volumes
    bufPixelCost_ = clCreateBuffer(context_, gpuDeviceFlags, costVolumeBytes, nullptr, &err);
    CL_CHECK(err);
    bufCostL_ = clCreateBuffer(context_, gpuDeviceFlags, costVolumeBytes, nullptr, &err);
    CL_CHECK(err);
    bufAggL_ = clCreateBuffer(context_, gpuDeviceFlags, aggVolumeBytes, nullptr, &err);
    CL_CHECK(err);

    (void)hostWriteFlags;  // Ready for final dispL / dispR output buffers in Stage 4

    // In allocateBuffers():
    const size_t dispBytes = numPixels_ * sizeof(float);
    // GPU-resident working buffer for dispR; host-mapped zero-copy buffer for final dispL
    bufDispR_ = clCreateBuffer(context_, gpuDeviceFlags, dispBytes, nullptr, &err);
    CL_CHECK(err);
    bufDispL_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | hostAllocFlag, dispBytes, nullptr, &err);
    CL_CHECK(err);
  }

  void speckleFilterHost(std::vector<float>& disp, int width, int height, int speckleWindowSize, int speckleRange, float invalidDisparity) {
    if (speckleWindowSize <= 0) return;

    const float maxDiff = static_cast<float>(speckleRange);
    const size_t total = static_cast<size_t>(width) * height;

    speckleVisited_.assign(total, 0);
    speckleStack_.clear();
    speckleComponent_.clear();

    const std::array<std::pair<int, int>, 4> nb = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t idx = static_cast<size_t>(y) * width + x;
        if (speckleVisited_[idx]) continue;

        if (disp[idx] == invalidDisparity || !std::isfinite(disp[idx])) {
          speckleVisited_[idx] = 1;
          continue;
        }

        speckleStack_.clear();
        speckleComponent_.clear();
        speckleStack_.push_back({x, y});
        speckleVisited_[idx] = 1;

        while (!speckleStack_.empty()) {
          const auto [cx, cy] = speckleStack_.back();
          speckleStack_.pop_back();
          speckleComponent_.push_back({cx, cy});

          const float ref = disp[static_cast<size_t>(cy) * width + cx];

          for (const auto& [dx, dy] : nb) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            const size_t nIdx = static_cast<size_t>(ny) * width + nx;
            if (speckleVisited_[nIdx]) continue;

            const float v = disp[nIdx];
            if (v == invalidDisparity || !std::isfinite(v)) {
              speckleVisited_[nIdx] = 1;
              continue;
            }

            if (std::abs(v - ref) <= maxDiff) {
              speckleVisited_[nIdx] = 1;
              speckleStack_.push_back({nx, ny});
            }
          }
        }

        if (static_cast<int>(speckleComponent_.size()) < speckleWindowSize) {
          for (const auto& [cx, cy] : speckleComponent_) {
            disp[static_cast<size_t>(cy) * width + cx] = invalidDisparity;
          }
        }
      }
    }
  }

  int width_ = 0;
  int height_ = 0;
  size_t numPixels_ = 0;
  int numDisparities_ = 0;
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;

  cl_kernel kernelPrefilter_ = nullptr;
  cl_kernel kernelBtPixel_ = nullptr;
  cl_kernel kernelBoxFilter_ = nullptr;
  cl_kernel kernelAggDir_ = nullptr;
  cl_kernel kernelWtaSubpixel_ = nullptr;
  cl_kernel kernelRightWta_ = nullptr;
  cl_kernel kernelLrCheck_ = nullptr;

  cl_mem bufRawL_ = nullptr;
  cl_mem bufRawR_ = nullptr;
  cl_mem bufSobelL_ = nullptr;
  cl_mem bufSobelR_ = nullptr;
  cl_mem bufSobelBoundsL_ = nullptr;
  cl_mem bufSobelBoundsR_ = nullptr;
  cl_mem bufRawBoundsL_ = nullptr;
  cl_mem bufRawBoundsR_ = nullptr;
  cl_mem bufCostL_ = nullptr;
  cl_mem bufPixelCost_ = nullptr;
  cl_mem bufAggL_ = nullptr;
  cl_mem bufDispL_ = nullptr;
  cl_mem bufDispR_ = nullptr;

  DeviceSpecs specs_;
  TuningConfig tuningConfig_;
  std::vector<std::string> clFilePaths_;
  std::vector<uint8_t> speckleVisited_;
  std::vector<std::pair<int, int>> speckleStack_;
  std::vector<std::pair<int, int>> speckleComponent_;
};
