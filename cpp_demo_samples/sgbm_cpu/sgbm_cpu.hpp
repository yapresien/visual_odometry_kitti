#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sgbm {

struct Params {
  int minDisparity = 0;
  int numDisparities = 128;  // divisible by 16
  int blockSize = 5;         // odd, >=3
  int P1 = 200;
  int P2 = 400;
  int disp12MaxDiff = 1;        // -1 disables
  int preFilterCap = 63;        // 0 disables
  int uniquenessRatio = 3;      // 0 disables
  int speckleWindowSize = 100;  // 0 disables
  int speckleRange = 32;
};

struct StereoOutputs
{
    std::vector<int16_t> preFilterL;
    std::vector<int16_t> preFilterR;

    std::vector<int16_t> costL;
    std::vector<int16_t> costR;

    std::vector<int32_t> pathLR;
    std::vector<int32_t> pathRL;
    std::vector<int32_t> pathTB;
    std::vector<int32_t> pathBT;

    std::vector<int32_t> pathTLBR;
    std::vector<int32_t> pathBRTL;
    std::vector<int32_t> pathTRBL;
    std::vector<int32_t> pathBLTR;

    std::vector<int32_t> aggL;
    std::vector<int32_t> aggR;

    std::vector<float> dispLRaw;
    std::vector<float> dispRRaw;

    std::vector<float> dispL;
    std::vector<float> dispR;
};


struct ErrorStats
{
    double mae;
    double rmse;
    double maxError;
    size_t mismatches;
};


class SgbmCpu {
 public:
  explicit SgbmCpu(const Params& p = Params());

  // Input: grayscale, width*height. Output: disparity in pixels, same size.
  // Invalid pixels (occlusions, low-texture, etc.) are set to the invalid value.
  static constexpr float kInvalidDisparity = -1.0f;

  StereoOutputs compute(const std::vector<uint8_t>& left, const std::vector<uint8_t>& right, int width, int height) const;

  const Params& params() const { return params_; }

 private:
  Params params_;

  void validate(int width, int height, size_t leftSize, size_t rightSize) const;

  // Stage 1: optional pre-filter (Sobel-like) — matches OpenCV's preFilterCap
  void preFilter(const std::vector<uint8_t>& src, std::vector<int16_t>& dst, int width, int height) const;

  // Stage 2: matching cost (BT / SAD-style block matching per pixel-disparity)
  void computeCost(const std::vector<int16_t>& left, const std::vector<int16_t>& right, int width, int height, std::vector<int16_t>& cost) const;

  // Stage 3: path aggregation (5 paths: L->R, R->L, T->B, B->T, diagonal)
  void aggregate(const std::vector<int16_t>& cost, int width, int height, std::vector<int32_t>& aggregated) const;

  // Stage 4: WTA + uniqueness + subpixel + LR check
  void selectDisparity(const std::vector<int32_t>& aggregated, const std::vector<int16_t>& cost, int width, int height, std::vector<float>& disp) const;

  // Stage 5: left-right consistency check
  void lrCheck(const std::vector<float>& dispRight, int width, int height, std::vector<float>& dispLeft) const;

  // Stage 6: speckle filtering
  void speckleFilter(std::vector<float>& disp, int width, int height) const;

#ifdef UNIT_TESTS
public:
    void preFilterForTest(
        const std::vector<uint8_t>& src,
        std::vector<int16_t>& dst,
        int width,
        int height) const
    {
        preFilter(src, dst, width, height);
    }
#endif
  
};

}  // namespace sgbm