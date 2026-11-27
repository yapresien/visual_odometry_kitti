#include <iostream>
#include <vector>
#include "sgbm_opencl.h"

int main(int argc, char** argv) {
  const int width = 1241;
  const int height = 376;
  const int preFilterCap = 63;
  const int minDisparity = 0;
  const int numDisparities = 128;
  const int blockSize = 5;

  SgbmOpenClRunner runner(
      {"../kernels/sgbm_prefilter_gfx1102.cl",
       "../kernels/sgbm_kernels.cl",
       "../kernels/sgbm_wta_lrcheck.cl",
       "../kernels/sgbm_aggregate_dir.cl"
      },
      width,
      height,
      numDisparities);

  std::vector<uint8_t> rawLeft(static_cast<size_t>(width) * height, 128);
  std::vector<uint8_t> rawRight(static_cast<size_t>(width) * height, 128);
  std::vector<int16_t> sobelOut;
  std::vector<cl_short2> sobelBoundsOut;
  std::vector<cl_short2> rawBoundsOut;
  std::vector<short> costOut;

  // Stage 1: Prefilter + BT bounds warmup & timing
  runner.runPrefilter(rawLeft, preFilterCap, sobelOut, sobelBoundsOut, rawBoundsOut);
  double pfMs = runner.runPrefilter(rawLeft, preFilterCap, sobelOut, sobelBoundsOut, rawBoundsOut);
  std::cout << "gfx1102 k_prefilter_and_bt_bounds kernel time: " << pfMs << " ms\n";

  // Stage 2: Box cost volume warmup & timing
  runner.runBoxCost(rawLeft, rawRight, preFilterCap, minDisparity, blockSize, costOut);
  double boxMs = runner.runBoxCost(rawLeft, rawRight, preFilterCap, minDisparity, blockSize, costOut);
  std::cout << "gfx1102 k_bt_box_cost kernel time: " << boxMs << " ms\n";

  const int P1 = 8 * blockSize * blockSize;   // 200 for blockSize = 5
  const int P2 = 32 * blockSize * blockSize;  // 800 for blockSize = 5
  std::vector<uint16_t> aggOut;

  // Warmup + timed 8-path GPU aggregation
  runner.runAggregate(P1, P2, aggOut);
  double aggMs = runner.runAggregate(P1, P2, aggOut);
  std::cout << "gfx1102 k_sgm_aggregate_dir (8 paths) kernel time: " << aggMs << " ms\n";

  return 0;
}
