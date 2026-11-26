#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "sgbm_cpu.hpp"

namespace {

struct FrameStats {
  double cpuMs = 0.0;
  double ocvMs = 0.0;
  double mae = 0.0;
  double bad3px = 0.0;
  double validRatio = 0.0;  // cpuValid / ocvValid
};

// Parses KITTI calib.txt P0 (left) and P1 (right) to extract focal length fx and baseline B
bool loadKittiCalibration(const std::string& calibPath, double& fx, double& baselineMeters) {
  std::ifstream file(calibPath);
  if (!file.is_open()) return false;

  std::string line;
  double p0_0 = 0.0, p1_0 = 0.0, p1_3 = 0.0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "P0:") {
      iss >> p0_0;
    } else if (tag == "P1:") {
      double dummy;
      iss >> p1_0 >> dummy >> dummy >> p1_3;
    }
  }
  if (p0_0 <= 0.0) return false;
  fx = p0_0;
  baselineMeters = std::abs(p1_3) / fx;  // P1(0,3) = -fx * baseline
  return true;
}

FrameStats evaluateSingleFrame(
    const sgbm::SgbmCpu& cpuSgbm,
    const cv::Ptr<cv::StereoSGBM>& ocvSgbm,
    const cv::Mat& leftMat,
    const cv::Mat& rightMat,
    int minDisparity) {
  const int width = leftMat.cols;
  const int height = leftMat.rows;
  std::vector<uint8_t> left(leftMat.datastart, leftMat.dataend);
  std::vector<uint8_t> right(rightMat.datastart, rightMat.dataend);

  // 1. Time CPU SGBM
  auto t0 = std::chrono::steady_clock::now();
  auto out = cpuSgbm.compute(left, right, width, height);
  auto t1 = std::chrono::steady_clock::now();

  // 2. Time OpenCV SGBM
  cv::Mat ocvDisp16;
  auto t2 = std::chrono::steady_clock::now();
  ocvSgbm->compute(leftMat, rightMat, ocvDisp16);
  auto t3 = std::chrono::steady_clock::now();

  FrameStats stats;
  stats.cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  stats.ocvMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

  const int16_t minValid16 = static_cast<int16_t>(minDisparity * 16);
  size_t cpuValid = 0, ocvValid = 0, bothValid = 0, bad3 = 0;
  double sumAbsErr = 0.0;

  for (int y = 5; y < height - 5; ++y) {
    const int16_t* ocvRow = ocvDisp16.ptr<int16_t>(y);
    for (int x = 5; x < width - 5; ++x) {
      const float dCpu = out.dispL[static_cast<size_t>(y) * width + x];
      const int16_t dOcv16 = ocvRow[x];

      const bool vCpu = (dCpu != sgbm::SgbmCpu::kInvalidDisparity) && std::isfinite(dCpu);
      const bool vOcv = (dOcv16 >= minValid16);

      if (vCpu) ++cpuValid;
      if (vOcv) ++ocvValid;
      if (vCpu && vOcv) {
        ++bothValid;
        const double err = std::abs(static_cast<double>(dCpu) - static_cast<double>(dOcv16) / 16.0);
        sumAbsErr += err;
        if (err > 3.0) ++bad3;
      }
    }
  }

  if (bothValid > 0) {
    stats.mae = sumAbsErr / static_cast<double>(bothValid);
    stats.bad3px = 100.0 * static_cast<double>(bad3) / static_cast<double>(bothValid);
  }
  if (ocvValid > 0) {
    stats.validRatio = static_cast<double>(cpuValid) / static_cast<double>(ocvValid);
  }
  return stats;
}

}  // namespace

TEST(KittiVoBenchmark, First100FramesThroughputAndAccuracy) {
  // Update to your KITTI Odometry sequence root (expects image_0/000000.png, image_1/000000.png)
  const std::string seqDir = "/home/master/myspace/projects/dataset_kitti/sequences/00";
  const int numFrames = 100;

  double fx = 718.856, baseline = 0.5371657;
  loadKittiCalibration(seqDir + "/calib.txt", fx, baseline);

  sgbm::Params p;
  p.minDisparity = 0;
  p.numDisparities = 128;
  p.blockSize = 5;
  p.P1 = 8 * p.blockSize * p.blockSize;
  p.P2 = 32 * p.blockSize * p.blockSize;
  p.preFilterCap = 63;
  p.uniquenessRatio = 10;
  p.speckleWindowSize = 100;
  p.speckleRange = 2;
  p.disp12MaxDiff = 1;

  sgbm::SgbmCpu cpuSgbm(p);
  auto ocvSgbm = cv::StereoSGBM::create(
      p.minDisparity, p.numDisparities, p.blockSize,
      p.P1, p.P2, p.disp12MaxDiff, p.preFilterCap,
      p.uniquenessRatio, p.speckleWindowSize, p.speckleRange,
      cv::StereoSGBM::MODE_HH);

  std::vector<FrameStats> allStats;
  allStats.reserve(numFrames);

  for (int i = 0; i < numFrames; ++i) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%06d.png", i);
    const std::string leftPath = seqDir + "/image_0/" + buf;
    const std::string rightPath = seqDir + "/image_1/" + buf;

    cv::Mat leftMat = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightMat = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
    if (leftMat.empty() || rightMat.empty()) {
      if (i == 0) {
        GTEST_SKIP() << "KITTI sequence not found at " << seqDir;
      }
      break;
    }

    auto st = evaluateSingleFrame(cpuSgbm, ocvSgbm, leftMat, rightMat, p.minDisparity);
    allStats.push_back(st);

    if (i % 10 == 0) {
      std::cout << "[Frame " << i << "] CPU=" << st.cpuMs << " ms | OCV=" << st.ocvMs
                << " ms | MAE=" << st.mae << " px | Bad3=" << st.bad3px << "%\n";
    }
  }

  ASSERT_FALSE(allStats.empty());

  double sumCpuMs = 0.0, sumOcvMs = 0.0, sumMae = 0.0, sumBad3 = 0.0, sumRatio = 0.0;
  std::vector<double> cpuTimes;
  for (const auto& s : allStats) {
    sumCpuMs += s.cpuMs;
    sumOcvMs += s.ocvMs;
    sumMae += s.mae;
    sumBad3 += s.bad3px;
    sumRatio += s.validRatio;
    cpuTimes.push_back(s.cpuMs);
  }

  std::sort(cpuTimes.begin(), cpuTimes.end());
  const size_t n = allStats.size();
  const double meanCpuMs = sumCpuMs / n;
  const double p95CpuMs = cpuTimes[static_cast<size_t>(0.95 * (n - 1))];

  std::cout << "\n=== KITTI 100-Frame SGBM Benchmark Summary (n=" << n << ") ===\n"
            << "Calibration      : fx=" << fx << " px, baseline=" << baseline << " m\n"
            << "CPU SGBM Latency : mean=" << meanCpuMs << " ms (" << (1000.0 / meanCpuMs)
            << " FPS), p95=" << p95CpuMs << " ms\n"
            << "OpenCV Latency   : mean=" << (sumOcvMs / n) << " ms (" << (1000.0 / (sumOcvMs / n)) << " FPS)\n"
            << "Mean MAE vs OCV  : " << (sumMae / n) << " px\n"
            << "Mean Bad > 3px   : " << (sumBad3 / n) << "%\n"
            << "Mean Valid Ratio : " << (sumRatio / n) << "x OpenCV\n";

  EXPECT_LT(sumMae / n, 0.35);
  EXPECT_LT(sumBad3 / n, 1.5);
  EXPECT_GT(sumRatio / n, 0.90);
}
