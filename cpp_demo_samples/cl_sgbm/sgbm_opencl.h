#ifndef OPENCL_TARGET_VERSION
#define OPENCL_TARGET_VERSION 120
#endif

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
  bool full8Paths = false;
};


// Add this small helper struct inside SgbmOpenClRunner (or above it):
struct AlignedHostMem {
  void* ptr = nullptr;
  size_t bytes = 0;

  void allocate(size_t reqBytes, size_t alignment = 256) {
    release();
    bytes = ((reqBytes + alignment - 1) / alignment) * alignment;
    if (posix_memalign(&ptr, alignment, bytes) != 0 || !ptr) {
      throw std::bad_alloc();
    }
    std::memset(ptr, 0, bytes);
  }

  void release() {
    if (ptr) {
      std::free(ptr);
      ptr = nullptr;
      bytes = 0;
    }
  }

  ~AlignedHostMem() { release(); }
};


class SgbmOpenClRunner {
 public:
  SgbmOpenClRunner(const std::vector<std::string>&, int, int, int, const TuningConfig& tuning = TuningConfig{});

  ~SgbmOpenClRunner();

  std::vector<cl_device_id> selectDevice(const std::string& preferredVendor = "");

  void refineWarpWidthFromKernel();

  DeviceSpecs queryAndTuneDevice(cl_device_id device, const TuningConfig& cfg);

  double runPrefilter(const std::vector<uint8_t>& rawLeft, int preFilterCap, std::vector<int16_t>& sobelOut, std::vector<cl_short2>& sobelBoundsOut, std::vector<cl_short2>& rawBoundsOut);

  double runBoxCost(const std::vector<uint8_t>& rawLeft, const std::vector<uint8_t>& rawRight, int preFilterCap, int minDisparity, int blockSize, std::vector<short>& costOut);

  double runAggregate(int P1, int P2, std::vector<uint16_t>& aggOut);

  double runDisparitySelection(const ClSgbmParams& p, std::vector<float>& dispLOut);

  void compute(const std::vector<uint8_t>& rawLeft, const std::vector<uint8_t>& rawRight, const ClSgbmParams& p, std::vector<float>& dispLOut, ClSgbmTimings& timings);

 private:
  void initContextAndQueue();

  std::string detectOpenClCStandard(cl_device_id device);

  std::string buildCompilerOptions(const DeviceSpecs& specs, const TuningConfig& cfg, int numDisparities);

  void buildProgram(const std::vector<std::string>& clFilePaths);

  void allocateBuffers();

  void speckleFilterHost(std::vector<float>& disp, int width, int height, int speckleWindowSize, int speckleRange, float invalidDisparity);

 private:
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
  cl_kernel kernelBoxH_ = nullptr;
  cl_kernel kernelBoxV_ = nullptr;
  cl_kernel kernelAggDir_ = nullptr;
  cl_kernel kernelWtaSubpixel_ = nullptr;
  cl_kernel kernelRightWta_ = nullptr;
  cl_kernel kernelLrCheck_ = nullptr;
  cl_kernel kernelAggBidiAxis_ = nullptr;
  cl_kernel kernelWtaScatter_ = nullptr;
  cl_kernel kernelClearUint_ = nullptr;

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
  cl_mem bufBoxTemp_ = nullptr;
  cl_mem bufAggL_ = nullptr;
  cl_mem bufDispL_ = nullptr;
  cl_mem bufRightMinPacked_ = nullptr;

  cl_mem bufRowPrevTB_ = nullptr;
  cl_mem bufRowPrevDL_ = nullptr;
  cl_mem bufRowPrevDR_ = nullptr;
  cl_mem bufRowCurrTB_ = nullptr;
  cl_mem bufRowCurrDL_ = nullptr;
  cl_mem bufRowCurrDR_ = nullptr;

  DeviceSpecs specs_;
  TuningConfig tuningConfig_;
  std::vector<std::string> clFilePaths_;
  std::vector<uint8_t> speckleVisited_;
  std::vector<std::pair<int, int>> speckleStack_;
  std::vector<std::pair<int, int>> speckleComponent_;

};
