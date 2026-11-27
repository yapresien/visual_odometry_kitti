#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "sgbm_opencl.h"

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

  // Preload all 100 stereo pairs into RAM so disk I/O never skews GPU/CPU timings
  std::vector<cv::Mat> leftMats, rightMats;
  std::vector<std::vector<uint8_t>> rawLefts, rawRights;
  leftMats.reserve(numFrames);
  rightMats.reserve(numFrames);
  rawLefts.reserve(numFrames);
  rawRights.reserve(numFrames);

  for (int i = 0; i < numFrames; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%06d.png", i);
    cv::Mat l = cv::imread(seqDir + "/image_0/" + buf, cv::IMREAD_GRAYSCALE);
    cv::Mat r = cv::imread(seqDir + "/image_1/" + buf, cv::IMREAD_GRAYSCALE);
    if (l.empty() || r.empty()) break;

    leftMats.push_back(l);
    rightMats.push_back(r);
    rawLefts.emplace_back(l.datastart, l.dataend);
    rawRights.emplace_back(r.datastart, r.dataend);
  }

  const int processed = static_cast<int>(leftMats.size());
  ASSERT_GT(processed, 0);

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
  p.full8Paths = false;  // 5-path MODE_SGBM (set true for 8-path MODE_HH)

  // ─── Pass 1: Standalone OpenCL Throughput Benchmark (Full GPU TDP, Zero CPU Contention) ───
  std::vector<std::vector<float>> clDisparities(processed);
  std::vector<ClSgbmTimings> clTimings(processed);

  // Warmup 5 iterations so gfx1102 reaches peak boost clock
  // Warmup 5 iterations so gfx1102 reaches peak boost clock
  for (int w = 0; w < 5; ++w) {
    clSgbm.compute(rawLefts.front(), rawRights.front(), p, clDisparities.front(), clTimings.front());
  }


  double sumEndToEndMs = 0.0, sumGpuMs = 0.0;
  double sumPfMs = 0.0, sumCostMs = 0.0, sumAggMs = 0.0, sumWtaMs = 0.0, sumSpeckleMs = 0.0;

  for (int i = 0; i < processed; ++i) {
    clSgbm.compute(rawLefts[i], rawRights[i], p, clDisparities[i], clTimings[i]);

    const auto& t = clTimings[i];
    sumEndToEndMs += t.endToEndMs;
    sumGpuMs += t.totalGpuKernelMs;
    sumPfMs += t.prefilterMs;
    sumCostMs += t.costVolumeMs;
    sumAggMs += t.aggregationMs;
    sumWtaMs += t.wtaLrMs;
    sumSpeckleMs += t.speckleMs;
  }

  // ─── Pass 2: OpenCV Baseline & Disparity Accuracy Verification ──────────────────────────
  const int ocvMode = p.full8Paths ? cv::StereoSGBM::MODE_HH : cv::StereoSGBM::MODE_SGBM;
  auto ocvSgbm = cv::StereoSGBM::create(
      p.minDisparity, numDisparities, p.blockSize,
      p.P1, p.P2, p.disp12MaxDiff, p.preFilterCap,
      p.uniquenessRatio, p.speckleWindowSize, p.speckleRange,
      ocvMode);

  double sumOcvMs = 0.0, sumMae = 0.0, sumBad3 = 0.0, sumValidRatio = 0.0;
  const int16_t minValid16 = static_cast<int16_t>(p.minDisparity * 16);

  for (int i = 0; i < processed; ++i) {
    cv::Mat ocvDisp16;
    auto ocvT0 = std::chrono::steady_clock::now();
    ocvSgbm->compute(leftMats[i], rightMats[i], ocvDisp16);
    auto ocvT1 = std::chrono::steady_clock::now();
    const double ocvMs = std::chrono::duration<double, std::milli>(ocvT1 - ocvT0).count();

    size_t clValid = 0, ocvValid = 0, bothValid = 0, bad3 = 0;
    double sumAbsErr = 0.0;
    const auto& clDispL = clDisparities[i];

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

    sumOcvMs += ocvMs;
    sumMae += mae;
    sumBad3 += bad3Pct;
    sumValidRatio += validRatio;

    if (i % 10 == 0) {
      std::cout << "[Frame " << i << "] OpenCL E2E=" << clTimings[i].endToEndMs
                << " ms (GPU=" << clTimings[i].totalGpuKernelMs << " ms) | OCV=" << ocvMs
                << " ms | MAE=" << mae << " px | Bad3=" << bad3Pct << "%\n";
    }
  }

  std::cout << "\n=== KITTI 100-Frame Production OpenCL SGBM Summary (n=" << processed << ", "
            << (p.full8Paths ? "8-path" : "5-path") << ") ===\n"
            << "OpenCL End-to-End : mean=" << (sumEndToEndMs / processed) << " ms ("
            << (1000.0 / (sumEndToEndMs / processed)) << " FPS)\n"
            << "OpenCL GPU Kernels: mean=" << (sumGpuMs / processed) << " ms\n"
            << "  [1] Prefilter   : " << (sumPfMs / processed) << " ms\n"
            << "  [2] Cost Volume : " << (sumCostMs / processed) << " ms\n"
            << "  [3] SGM Agg     : " << (sumAggMs / processed) << " ms\n"
            << "  [4] WTA + LR    : " << (sumWtaMs / processed) << " ms\n"
            << "  [5] Host Speckle: " << (sumSpeckleMs / processed) << " ms\n"
            << "  [6] Upload/Sync : "
            << ((sumEndToEndMs - sumGpuMs - sumSpeckleMs) / processed) << " ms\n"
            << "OpenCV CPU        : mean=" << (sumOcvMs / processed) << " ms ("
            << (1000.0 / (sumOcvMs / processed)) << " FPS)\n"
            << "Mean MAE vs OCV   : " << (sumMae / processed) << " px\n"
            << "Mean Bad > 3px    : " << (sumBad3 / processed) << "%\n"
            << "Mean Valid Ratio  : " << (sumValidRatio / processed) << "x OpenCV\n";

  EXPECT_LT(sumMae / processed, 0.35);
  EXPECT_LT(sumBad3 / processed, 1.5);
  EXPECT_GT(sumValidRatio / processed, 0.90);
}
