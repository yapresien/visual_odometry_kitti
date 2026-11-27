
#include "sgbm_opencl.h"

#define CL_CHECK(err)                                                                                                        \
  do {                                                                                                                       \
    cl_int _err = (err);                                                                                                     \
    if (_err != CL_SUCCESS) {                                                                                                \
      throw std::runtime_error("OpenCL error " + std::to_string(_err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                                                                        \
  } while (0)

SgbmOpenClRunner::SgbmOpenClRunner(const std::vector<std::string>& clFilePaths, int width, int height, int numDisparities, const TuningConfig& tuning ) : clFilePaths_(clFilePaths), width_(width), height_(height), numDisparities_(numDisparities), numPixels_(static_cast<size_t>(width) * height), tuningConfig_(tuning) {
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

SgbmOpenClRunner::~SgbmOpenClRunner() {
  if (bufRawL_) clReleaseMemObject(bufRawL_);
  if (bufSobelL_) clReleaseMemObject(bufSobelL_);
  if (bufSobelBoundsL_) clReleaseMemObject(bufSobelBoundsL_);
  if (bufRawBoundsL_) clReleaseMemObject(bufRawBoundsL_);
  if(kernelClearUint_) clReleaseKernel(kernelClearUint_);

  if (kernelPrefilter_) clReleaseKernel(kernelPrefilter_);
  if (kernelBtPixel_) clReleaseKernel(kernelBtPixel_);
  if (kernelBoxFilter_) clReleaseKernel(kernelBoxFilter_);
  if (kernelAggDir_) clReleaseKernel(kernelAggDir_);
  if (kernelWtaSubpixel_) clReleaseKernel(kernelWtaSubpixel_);
  if (kernelRightWta_) clReleaseKernel(kernelRightWta_);
  if (kernelLrCheck_) clReleaseKernel(kernelLrCheck_);

  if (bufDispL_) clReleaseMemObject(bufDispL_);
  if (bufRightMinPacked_) clReleaseMemObject(bufRightMinPacked_);

  if (program_) clReleaseProgram(program_);
  if (queue_) clReleaseCommandQueue(queue_);
  if (context_) clReleaseContext(context_);
}

std::vector<cl_device_id> SgbmOpenClRunner::selectDevice(const std::string& preferredVendor ) {
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

void SgbmOpenClRunner::refineWarpWidthFromKernel() {
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

DeviceSpecs SgbmOpenClRunner::queryAndTuneDevice(cl_device_id device, const TuningConfig& cfg) {
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

double SgbmOpenClRunner::runPrefilter(const std::vector<uint8_t>& rawLeft, int preFilterCap, std::vector<int16_t>& sobelOut, std::vector<cl_short2>& sobelBoundsOut, std::vector<cl_short2>& rawBoundsOut) {
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

double SgbmOpenClRunner::runBoxCost(const std::vector<uint8_t>& rawLeft, const std::vector<uint8_t>& rawRight, int preFilterCap, int minDisparity, int blockSize, std::vector<short>& costOut) {
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

double SgbmOpenClRunner::runAggregate(int P1, int P2, std::vector<uint16_t>& aggOut) {
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

double SgbmOpenClRunner::runDisparitySelection(const ClSgbmParams& p, std::vector<float>& dispLOut) {
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
  CL_CHECK(clSetKernelArg(kernelRightWta_, 3, sizeof(cl_mem), &bufRightMinPacked_));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 4, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 5, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 6, sizeof(int), &p.minDisparity));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 7, sizeof(int), &numDisparities_));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 8, sizeof(int), &p.disp12MaxDiff));
  CL_CHECK(clSetKernelArg(kernelRightWta_, 9, sizeof(float), &p.invalidDisparity));

  cl_event evWtaR = nullptr;
  CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelRightWta_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWtaR));

  // 4C. Integer Left-Right consistency check
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 0, sizeof(cl_mem), &bufRightMinPacked_));
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

void SgbmOpenClRunner::compute(
    const std::vector<uint8_t>& rawLeft,
    const std::vector<uint8_t>& rawRight,
    const ClSgbmParams& p,
    std::vector<float>& dispLOut,
    ClSgbmTimings& timings) {
  const auto tWallStart = std::chrono::steady_clock::now();

  auto eventMs = [](cl_event ev) {
    if (!ev) return 0.0;
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    clReleaseEvent(ev);
    return static_cast<double>(t1 - t0) * 1e-6;
  };

  // 1. Non-blocking image uploads (zero map/unmap page-table faults)
  CL_CHECK(clEnqueueWriteBuffer(
      queue_, bufRawL_, CL_FALSE, 0, numPixels_ * sizeof(uint8_t), rawLeft.data(), 0, nullptr, nullptr));
  CL_CHECK(clEnqueueWriteBuffer(
      queue_, bufRawR_, CL_FALSE, 0, numPixels_ * sizeof(uint8_t), rawRight.data(), 0, nullptr, nullptr));

  // 2. Stage 1: Prefilter + Birchfield-Tomasi Bounds (Left & Right)
  const std::array<size_t, 2> pfLocal = {specs_.pfWgW, specs_.pfWgH};
  const std::array<size_t, 2> pfGlobal = {
      ((static_cast<size_t>(width_) + specs_.pfWgW - 1) / specs_.pfWgW) * specs_.pfWgW,
      ((static_cast<size_t>(height_) + specs_.pfWgH - 1) / specs_.pfWgH) * specs_.pfWgH};

  cl_event evPfL = nullptr, evPfR = nullptr;
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawL_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelL_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsL_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsL_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &p.preFilterCap));
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evPfL));

  CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawR_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelR_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsR_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsR_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &p.preFilterCap));
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evPfR));

  // 3. Stage 2: 8-bit Dual-BT Pixel Cost + BxB Box Filter
  const std::array<size_t, 3> stage2Local = {specs_.costWgW, specs_.costWgH, 1};
  const std::array<size_t, 3> stage2Global = {
      ((static_cast<size_t>(width_) + specs_.costWgH - 1) / specs_.costWgH) * specs_.costWgW,
      static_cast<size_t>(height_) * specs_.costWgH,
      1};

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
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelBtPixel_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &evBt));

  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 0, sizeof(cl_mem), &bufPixelCost_));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 1, sizeof(cl_mem), &bufCostL_));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 2, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 3, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 4, sizeof(int), &p.minDisparity));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 5, sizeof(int), &numDisparities_));
  CL_CHECK(clSetKernelArg(kernelBoxFilter_, 6, sizeof(int), &p.blockSize));
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelBoxFilter_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &evBox));

  // 4. Stage 3: 4-Pass Bidirectional 256-Thread SGM Aggregation
  struct Axis {
    int dx;
    int dy;
    size_t numScanlines;
    int doBackward;
  };

  const int doBwd = p.full8Paths ? 1 : 0;
  const std::array<Axis, 4> axes = {{
      {1,  0, static_cast<size_t>(height_), 1},
      {0,  1, static_cast<size_t>(width_), doBwd},
      {1,  1, static_cast<size_t>(width_ + height_ - 1), doBwd},
      {-1, 1, static_cast<size_t>(width_ + height_ - 1), doBwd}
  }};

  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 0, sizeof(cl_mem), &bufCostL_));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 1, sizeof(cl_mem), &bufAggL_));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 2, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 3, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 4, sizeof(int), &numDisparities_));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 5, sizeof(int), &p.P1));
  CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 6, sizeof(int), &p.P2));

  std::array<cl_event, 4> evAxes{};
  const std::array<size_t, 2> aggLocal = {specs_.sgmWgSize, specs_.costWgH};

  for (size_t i = 0; i < axes.size(); ++i) {
    const int isFirst = (i == 0) ? 1 : 0;
    CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 7, sizeof(int), &axes[i].dx));
    CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 8, sizeof(int), &axes[i].dy));
    CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 9, sizeof(int), &isFirst));
    CL_CHECK(clSetKernelArg(kernelAggBidiAxis_, 10, sizeof(int), &axes[i].doBackward));

    const std::array<size_t, 2> aggGlobal = {
        ((axes[i].numScanlines + specs_.costWgH - 1) / specs_.costWgH) * specs_.sgmWgSize,
        specs_.costWgH};

    CL_CHECK(clEnqueueNDRangeKernel(
        queue_, kernelAggBidiAxis_, 2, nullptr, aggGlobal.data(), aggLocal.data(), 0, nullptr, &evAxes[i]));
  }

  // 5. Stage 4: Compute Clear (Zero SDMA) -> Fused Left WTA + Right Scatter -> LR Check
  const cl_uint fillSentinel = 0xFFFFFFFFu;
  const cl_uint pixCount = static_cast<cl_uint>(numPixels_);
  CL_CHECK(clSetKernelArg(kernelClearUint_, 0, sizeof(cl_mem), &bufRightMinPacked_));
  CL_CHECK(clSetKernelArg(kernelClearUint_, 1, sizeof(cl_uint), &fillSentinel));
  CL_CHECK(clSetKernelArg(kernelClearUint_, 2, sizeof(cl_uint), &pixCount));
  const size_t clearLocal = 256;
  const size_t clearGlobal = ((numPixels_ + 255) / 256) * 256;
  cl_event evClear = nullptr, evWta = nullptr, evLr = nullptr;
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelClearUint_, 1, nullptr, &clearGlobal, &clearLocal, 0, nullptr, &evClear));

  const std::array<size_t, 2> wtaLocal = {specs_.sgmWgSize, specs_.costWgH};
  const std::array<size_t, 2> wtaGlobal = {
      ((static_cast<size_t>(width_) + specs_.costWgH - 1) / specs_.costWgH) * specs_.sgmWgSize,
      static_cast<size_t>(height_) * specs_.costWgH};

  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 0, sizeof(cl_mem), &bufAggL_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 1, sizeof(cl_mem), &bufCostL_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 2, sizeof(cl_mem), &bufDispL_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 3, sizeof(cl_mem), &bufRightMinPacked_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 4, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 5, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 6, sizeof(int), &p.minDisparity));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 7, sizeof(int), &numDisparities_));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 8, sizeof(int), &p.uniquenessRatio));
  CL_CHECK(clSetKernelArg(kernelWtaScatter_, 9, sizeof(float), &p.invalidDisparity));
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelWtaScatter_, 2, nullptr, wtaGlobal.data(), wtaLocal.data(), 0, nullptr, &evWta));

  CL_CHECK(clSetKernelArg(kernelLrCheck_, 0, sizeof(cl_mem), &bufRightMinPacked_));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 1, sizeof(cl_mem), &bufDispL_));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 2, sizeof(int), &width_));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 3, sizeof(int), &height_));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 4, sizeof(int), &p.minDisparity));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 5, sizeof(int), &p.disp12MaxDiff));
  CL_CHECK(clSetKernelArg(kernelLrCheck_, 6, sizeof(float), &p.invalidDisparity));
  CL_CHECK(clEnqueueNDRangeKernel(
      queue_, kernelLrCheck_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, &evLr));

  // 6. Blocking Readback (implicitly drains queue_ cleanly without separate clFinish)
  dispLOut.resize(numPixels_);
  CL_CHECK(clEnqueueReadBuffer(
      queue_, bufDispL_, CL_TRUE, 0, numPixels_ * sizeof(float), dispLOut.data(), 0, nullptr, nullptr));

  // Record accurate per-stage GPU profiling timestamps
  timings.prefilterMs = eventMs(evPfL) + eventMs(evPfR);
  timings.costVolumeMs = eventMs(evBt) + eventMs(evBox);
  timings.aggregationMs = 0.0;
  for (auto& ev : evAxes) timings.aggregationMs += eventMs(ev);
  timings.wtaLrMs = eventMs(evClear) + eventMs(evWta) + eventMs(evLr);
  timings.totalGpuKernelMs = timings.prefilterMs + timings.costVolumeMs + timings.aggregationMs + timings.wtaLrMs;
  // 7. Host speckle filter
  const auto tSp0 = std::chrono::steady_clock::now();
  speckleFilterHost(dispLOut, width_, height_, p.speckleWindowSize, p.speckleRange, p.invalidDisparity);
  const auto tSp1 = std::chrono::steady_clock::now();
  timings.speckleMs = std::chrono::duration<double, std::milli>(tSp1 - tSp0).count();
  timings.endToEndMs = std::chrono::duration<double, std::milli>(tSp1 - tWallStart).count();
}

void SgbmOpenClRunner::initContextAndQueue()
{
    cl_int err = CL_SUCCESS;

    context_ =
        clCreateContext(
            nullptr,
            1,
            &device_,
            nullptr,
            nullptr,
            &err);

    CL_CHECK(err);

#if CL_TARGET_OPENCL_VERSION >= 200

    const cl_queue_properties props[] =
    {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0
    };

    queue_ =
        clCreateCommandQueueWithProperties(
            context_,
            device_,
            props,
            &err);

#else

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    queue_ =
        clCreateCommandQueue(
            context_,
            device_,
            CL_QUEUE_PROFILING_ENABLE,
            &err);

#pragma GCC diagnostic pop

#endif

    CL_CHECK(err);
}

static std::string KernelLanguageOption()
{
#if OPENCL_TARGET_VERSION >= 300
    return "-cl-std=CL3.0";
#elif OPENCL_TARGET_VERSION >= 200
    return "-cl-std=CL2.0";
#elif OPENCL_TARGET_VERSION >= 120
    return "-cl-std=CL1.2";
#elif OPENCL_TARGET_VERSION >= 110
    return "-cl-std=CL1.1";
#else
    return "-cl-std=CL1.0";
#endif
}

std::string SgbmOpenClRunner::detectOpenClCStandard(cl_device_id device) {
  std::array<char, 128> cVer{};
  clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, cVer.size(), cVer.data(), nullptr);
  const std::string verStr(cVer.data());
  if (verStr.find("OpenCL C 3.") != std::string::npos) return "-cl-std=CL3.0";
  if (verStr.find("OpenCL C 2.") != std::string::npos) return "-cl-std=CL2.0";
  return "-cl-std=CL1.2";
}

std::string SgbmOpenClRunner::buildCompilerOptions(const DeviceSpecs& specs, const TuningConfig& cfg, int numDisparities) {
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

void SgbmOpenClRunner::buildProgram(const std::vector<std::string>& clFilePaths) {
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

  std::string buildOpts = KernelLanguageOption();

  buildOpts +=
      " -cl-mad-enable"
      " -cl-fast-relaxed-math";

  auto cflags = buildCompilerOptions(specs_, tuningConfig_, numDisparities_);  // Generate additional build options based on device specs and tuning
  if (!cflags.empty())
  {
      buildOpts += " ";
      buildOpts += cflags;
  }
  std::cout << "Building OpenCL program with options: " << buildOpts << std::endl;

  err = clBuildProgram(program_, 1, &device_, buildOpts.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::string buildLog(logSize, '\0');
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
    std::cerr << "OpenCL Build Log:\n" << buildLog << "\n";
    CL_CHECK(err);
  }
  
  //Stage 1 kernels: Prefilter + BT bounds
  kernelClearUint_ = clCreateKernel(program_, "k_clear_uint", &err);
  CL_CHECK(err);

  kernelPrefilter_ = clCreateKernel(program_, "k_prefilter_and_bt_bounds", &err);
  CL_CHECK(err);

  //Stage 2 kernels: BT pixel cost + box filter cost
  kernelBtPixel_ = clCreateKernel(program_, "k_bt_pixel_cost", &err);
  CL_CHECK(err);
  kernelBoxFilter_ = clCreateKernel(program_, "k_box_filter_cost", &err);
  CL_CHECK(err);

  //Stage 3 kernels: 4-pass bidirectional SGM aggregation
  kernelAggDir_ = clCreateKernel(program_, "k_sgm_aggregate_dir", &err);
  CL_CHECK(err);

  //Stage 4 kernels: WTA + Right scatter + LR check
  kernelLrCheck_ = clCreateKernel(program_, "k_lr_check_packed", &err);
  CL_CHECK(err);
  kernelAggBidiAxis_ = clCreateKernel(program_, "k_sgm_bidi_axis", &err);
  CL_CHECK(err);
  kernelWtaScatter_ = clCreateKernel(program_, "k_wta_subpixel_and_right_scatter", &err);
  CL_CHECK(err);


}

  void SgbmOpenClRunner::allocateBuffers() {
    cl_int err = CL_SUCCESS;

    const size_t rawBytes        = numPixels_ * sizeof(uint8_t);
    const size_t sobelBytes      = numPixels_ * sizeof(int16_t);
    const size_t boundsBytes     = numPixels_ * sizeof(cl_short2);
    const size_t costVolumeBytes = numPixels_ * static_cast<size_t>(numDisparities_) * sizeof(int16_t);
    const size_t aggVolumeBytes  = numPixels_ * static_cast<size_t>(numDisparities_) * sizeof(uint16_t);  // 16-bit ushort
    const size_t dispBytes       = numPixels_ * sizeof(float);
    const size_t packedBytes     = numPixels_ * sizeof(uint32_t);

    bufRawL_           = clCreateBuffer(context_, CL_MEM_READ_ONLY,  rawBytes,        nullptr, &err); CL_CHECK(err);
    bufRawR_           = clCreateBuffer(context_, CL_MEM_READ_ONLY,  rawBytes,        nullptr, &err); CL_CHECK(err);

    bufSobelL_         = clCreateBuffer(context_, CL_MEM_READ_WRITE, sobelBytes,      nullptr, &err); CL_CHECK(err);
    bufSobelBoundsL_   = clCreateBuffer(context_, CL_MEM_READ_WRITE, boundsBytes,     nullptr, &err); CL_CHECK(err);
    bufRawBoundsL_     = clCreateBuffer(context_, CL_MEM_READ_WRITE, boundsBytes,     nullptr, &err); CL_CHECK(err);

    bufSobelR_         = clCreateBuffer(context_, CL_MEM_READ_WRITE, sobelBytes,      nullptr, &err); CL_CHECK(err);
    bufSobelBoundsR_   = clCreateBuffer(context_, CL_MEM_READ_WRITE, boundsBytes,     nullptr, &err); CL_CHECK(err);
    bufRawBoundsR_     = clCreateBuffer(context_, CL_MEM_READ_WRITE, boundsBytes,     nullptr, &err); CL_CHECK(err);

    bufPixelCost_      = clCreateBuffer(context_, CL_MEM_READ_WRITE, costVolumeBytes, nullptr, &err); CL_CHECK(err);
    bufCostL_          = clCreateBuffer(context_, CL_MEM_READ_WRITE, costVolumeBytes, nullptr, &err); CL_CHECK(err);
    bufAggL_           = clCreateBuffer(context_, CL_MEM_READ_WRITE, aggVolumeBytes,  nullptr, &err); CL_CHECK(err);

    bufRightMinPacked_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, packedBytes,     nullptr, &err); CL_CHECK(err);
    bufDispL_          = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, dispBytes,       nullptr, &err); CL_CHECK(err);
  }


void SgbmOpenClRunner::speckleFilterHost(std::vector<float>& disp, int width, int height, int speckleWindowSize, int speckleRange, float invalidDisparity) {
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
