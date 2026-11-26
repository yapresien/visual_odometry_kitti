// #define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <array>
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

class SgbmOpenClRunner {
 public:
  SgbmOpenClRunner(const std::vector<std::string>& clFilePaths, int width, int height, int numDisparities)
      : width_(width),
        height_(height),
        numDisparities_(numDisparities),
        numPixels_(static_cast<size_t>(width) * height) {
    initDevice();
    buildProgram(clFilePaths);
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
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
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
    const std::array<size_t, 2> localSize = {32, 8};
    const std::array<size_t, 2> globalSize = {static_cast<size_t>((width_ + 31) / 32) * 32, static_cast<size_t>((height_ + 7) / 8) * 8};

    cl_event profEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, globalSize.data(), localSize.data(), 0, nullptr, &profEvent));

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

    // 2. Dispatch Stage 1 (k_prefilter_and_bt_bounds) for Left and Right images
    const std::array<size_t, 2> pfLocal = {32, 8};
    const std::array<size_t, 2> pfGlobal = {static_cast<size_t>((width_ + 31) / 32) * 32, static_cast<size_t>((height_ + 7) / 8) * 8};

    // Left prefilter + BT bounds
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsL_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 4, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 5, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 6, sizeof(int), &preFilterCap));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelPrefilter_, 2, nullptr, pfGlobal.data(), pfLocal.data(), 0, nullptr, nullptr));

    // Right prefilter + BT bounds
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 0, sizeof(cl_mem), &bufRawR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 1, sizeof(cl_mem), &bufSobelR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 2, sizeof(cl_mem), &bufSobelBoundsR_));
    CL_CHECK(clSetKernelArg(kernelPrefilter_, 3, sizeof(cl_mem), &bufRawBoundsR_));
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

    const std::array<size_t, 3> stage2Local = {32, 8, 1};
    const std::array<size_t, 3> stage2Global = {static_cast<size_t>((width_ + 7) / 8) * 32, static_cast<size_t>(height_), 1};

    cl_event btEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBtPixel_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &btEvent));

    // 3B. Dispatch Kernel 2B: k_box_filter_cost (32x8 = 256 threads/WG)
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 0, sizeof(cl_mem), &bufPixelCost_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 1, sizeof(cl_mem), &bufCostL_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 2, sizeof(int), &width_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 3, sizeof(int), &height_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 4, sizeof(int), &minDisparity));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 5, sizeof(int), &numDisparities_));
    CL_CHECK(clSetKernelArg(kernelBoxFilter_, 6, sizeof(int), &blockSize));

    cl_event boxEvent = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernelBoxFilter_, 3, nullptr, stage2Global.data(), stage2Local.data(), 0, nullptr, &boxEvent));
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

 private:
  void initDevice() {
    cl_uint numPlatforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &numPlatforms));
    std::vector<cl_platform_id> platforms(numPlatforms);
    CL_CHECK(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

    // Select AMD platform (skip Xilinx embedded platform)
    for (auto plat : platforms) {
      std::array<char, 256> vendor{};
      clGetPlatformInfo(plat, CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr);
      if (std::string(vendor.data()).find("Advanced Micro Devices") != std::string::npos) {
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr) == CL_SUCCESS) {
          platform_ = plat;
          break;
        }
      }
    }
    if (!device_) throw std::runtime_error("AMD GPU device (gfx1102) not found");

    cl_int err = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);

    // OpenCL 2.0 command queue with profiling enabled
    const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
    CL_CHECK(err);
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
  }

  void allocateBuffers() {
    cl_int err = CL_SUCCESS;
    // CL_MEM_ALLOC_HOST_PTR allocates 256-byte aligned pinned unified memory on AMD APU
    bufRawL_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(uint8_t), nullptr, &err);
    CL_CHECK(err);

    bufSobelL_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(int16_t), nullptr, &err);
    CL_CHECK(err);

    bufSobelBoundsL_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(cl_short2), nullptr, &err);
    CL_CHECK(err);

    bufRawBoundsL_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(cl_short2), nullptr, &err);
    CL_CHECK(err);

    const size_t costVolumeBytes = numPixels_ * static_cast<size_t>(numDisparities_) * sizeof(int16_t);

    bufRawR_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(uint8_t), nullptr, &err);
    CL_CHECK(err);
    bufSobelR_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(int16_t), nullptr, &err);
    CL_CHECK(err);
    bufSobelBoundsR_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(cl_short2), nullptr, &err);
    CL_CHECK(err);
    bufRawBoundsR_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, numPixels_ * sizeof(cl_short2), nullptr, &err);
    CL_CHECK(err);

  //Note: Change bufSobelL_, bufSobelBoundsL_, bufRawBoundsL_ flags from CL_MEM_WRITE_ONLY to CL_MEM_READ_WRITE so k_bt_box_cost can read them on the GPU without host roundtrips.
    bufCostL_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, costVolumeBytes, nullptr, &err);
    CL_CHECK(err);

    bufPixelCost_ = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  costVolumeBytes, nullptr, &err);
    CL_CHECK(err);

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
};
