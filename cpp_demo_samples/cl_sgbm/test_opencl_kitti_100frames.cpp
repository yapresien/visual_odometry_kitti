#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "sgbm_opencl.hpp"

TEST(OpenClSgbmBenchmark, First100KittiFramesVsOpenCv) {
  const std::string seqDir = "/home/master/myspace/projects/dataset_kitti/sequences/00";
  const int numFrames = 100;
  const int numDisparities = 128;

  cv::Mat firstL = cv::imread(seqDir + "/image_0/000000.png", cv::IMREAD_GRAYSCALE);
  if (firstL.empty()) {
    GTEST_SKIP() << "KITTI sequence not found at " << seqDir;
  }

  const int width = firstL.cols;
  const int height = firstL.rows;

  SgbmOpenClRunner clSgbm(
      {"../kernels/sgbm_prefilter_gfx1102.cl",
       "../kernels/sgbm_kernels.cl",
       "../kernels/sgbm_wta_lrcheck.cl",
       "../kernels/sgbm_aggregate_dir.cl"
      },
      width,
      height,
      numDisparities);

  ClSgbmParams p;
  p.minDisparity = 0;
  p.blockSize = 5;
  p.P1 = 8 * p.blockSize * p.blockSize;
  p.P2 = 32 * p.blockSize * p.blockSize;
  p.preFilterCap = 63;
  p.uniquenessRatio = 10;
  p.speckleWindowSize = 100;
  p.speckleRange = 2;
  p.disp12MaxDiff = 1;
  p.invalidDisparity = -1.0f;

  auto ocvSgbm = cv::StereoSGBM::create(
      p.minDisparity, numDisparities, p.blockSize,
      p.P1, p.P2, p.disp12MaxDiff, p.preFilterCap,
      p.uniquenessRatio, p.speckleWindowSize, p.speckleRange,
      cv::StereoSGBM::MODE_HH);

  std::vector<float> clDispL;
  ClSgbmTimings t;

  // Warmup pass on frame 0 so gfx1102 boosts to 2,900 MHz
  std::vector<uint8_t> warmL(firstL.datastart, firstL.dataend);
  clSgbm.compute(warmL, warmL, p, clDispL, t);

  double sumEndToEndMs = 0.0, sumGpuMs = 0.0, sumOcvMs = 0.0;
  double sumMae = 0.0, sumBad3 = 0.0, sumValidRatio = 0.0;
  int processed = 0;

  for (int i = 0; i < numFrames; ++i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.png", i);
    cv::Mat leftMat = cv::imread(seqDir + "/image_0/" + buf, cv::IMREAD_GRAYSCALE);
    cv::Mat rightMat = cv::imread(seqDir + "/image_1/" + buf, cv::IMREAD_GRAYSCALE);
    if (leftMat.empty() || rightMat.empty()) break;

    std::vector<uint8_t> rawL(leftMat.datastart, leftMat.dataend);
    std::vector<uint8_t> rawR(rightMat.datastart, rightMat.dataend);

    clSgbm.compute(rawL, rawR, p, clDispL, t);

    cv::Mat ocvDisp16;
    auto ocvT0 = std::chrono::steady_clock::now();
    ocvSgbm->compute(leftMat, rightMat, ocvDisp16);
    auto ocvT1 = std::chrono::steady_clock::now();
    const double ocvMs = std::chrono::duration<double, std::milli>(ocvT1 - ocvT0).count();

    const int16_t minValid16 = static_cast<int16_t>(p.minDisparity * 16);
    size_t clValid = 0, ocvValid = 0, bothValid = 0, bad3 = 0;
    double sumAbsErr = 0.0;

    for (int y = 5; y < height - 5; ++y) {
      const int16_t* ocvRow = ocvDisp16.ptr<int16_t>(y);
      for (int x = 5; x < width - 5; ++x) {
        const float dCl = clDispL[static_cast<size_t>(y) * width + x];
        const int16_t dOcv16 = ocvRow[x];
        const bool vCl = (dCl != p.invalidDisparity) && std::isfinite(dCl);
        const bool vOcv = (dOcv16 >= minValid16);

        if (vCl) ++clValid;
        if (vOcv) ++ocvValid;
        if (vCl && vOcv) {
          ++bothValid;
          const double err = std::abs(static_cast<double>(dCl) - static_cast<double>(dOcv16) / 16.0);
          sumAbsErr += err;
          if (err > 3.0) ++bad3;
        }
      }
    }

    const double mae = (bothValid > 0) ? (sumAbsErr / bothValid) : 0.0;
    const double bad3Pct = (bothValid > 0) ? (100.0 * bad3 / bothValid) : 0.0;
    const double validRatio = (ocvValid > 0) ? (static_cast<double>(clValid) / ocvValid) : 0.0;

    sumEndToEndMs += t.endToEndMs;
    sumGpuMs += t.totalGpuKernelMs;
    sumOcvMs += ocvMs;
    sumMae += mae;
    sumBad3 += bad3Pct;
    sumValidRatio += validRatio;
    ++processed;

    if (i % 10 == 0) {
      std::cout << "[Frame " << i << "] OpenCL E2E=" << t.endToEndMs << " ms (GPU=" << t.totalGpuKernelMs
                << " ms) | OCV=" << ocvMs << " ms | MAE=" << mae << " px | Bad3=" << bad3Pct << "%\n";
    }
  }

  ASSERT_GT(processed, 0);
  std::cout << "\n=== KITTI 100-Frame Production OpenCL SGBM Summary (n=" << processed << ") ===\n"
            << "OpenCL End-to-End : mean=" << (sumEndToEndMs / processed) << " ms ("
            << (1000.0 / (sumEndToEndMs / processed)) << " FPS)\n"
            << "OpenCL GPU Kernels: mean=" << (sumGpuMs / processed) << " ms\n"
            << "OpenCV CPU MODE_HH: mean=" << (sumOcvMs / processed) << " ms ("
            << (1000.0 / (sumOcvMs / processed)) << " FPS)\n"
            << "Mean MAE vs OCV   : " << (sumMae / processed) << " px\n"
            << "Mean Bad > 3px    : " << (sumBad3 / processed) << "%\n"
            << "Mean Valid Ratio  : " << (sumValidRatio / processed) << "x OpenCV\n";

  EXPECT_LT(sumMae / processed, 0.35);
  EXPECT_LT(sumBad3 / processed, 1.5);
  EXPECT_GT(sumValidRatio / processed, 0.90);
}
