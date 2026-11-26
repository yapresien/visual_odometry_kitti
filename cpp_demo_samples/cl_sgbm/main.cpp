#include <iostream>
#include <vector>
#include "sgbm_opencl.hpp"

int main(int argc, char** argv) {
  const int width = 1241;
  const int height = 376;
  const int preFilterCap = 63;
  const int minDisparity = 0;
  const int numDisparities = 128;
  const int blockSize = 5;

  SgbmOpenClRunner runner(
      {"../kernels/sgbm_prefilter_gfx1102.cl",
       "../kernels/sgbm_kernels.cl"},
      width,
      height,
      numDisparities);

  std::vector<uint8_t> rawLeft(static_cast<size_t>(width) * height, 128);
  std::vector<uint8_t> rawRight(static_cast<size_t>(width) * height, 128);
  std::vector<int16_t> sobelOut;
  std::vector<cl_short2> sobelBoundsOut;
  std::vector<cl_short2> rawBoundsOut;
  std::vector<int16_t> costOut;

  // Stage 1: Prefilter + BT bounds warmup & timing
  runner.runPrefilter(rawLeft, preFilterCap, sobelOut, sobelBoundsOut, rawBoundsOut);
  double pfMs = runner.runPrefilter(rawLeft, preFilterCap, sobelOut, sobelBoundsOut, rawBoundsOut);
  std::cout << "gfx1102 k_prefilter_and_bt_bounds kernel time: " << pfMs << " ms\n";

  // Stage 2: Box cost volume warmup & timing
  runner.runBoxCost(rawLeft, rawRight, preFilterCap, minDisparity, blockSize, costOut);
  double boxMs = runner.runBoxCost(rawLeft, rawRight, preFilterCap, minDisparity, blockSize, costOut);
  std::cout << "gfx1102 k_bt_box_cost kernel time: " << boxMs << " ms\n";

  return 0;
}
