#include <gtest/gtest.h>

#include <cassert>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

#include "helper_sgbm.hpp"
#include "sgbm_cpu.hpp"
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <iostream>
#include <vector>

#include "sgbm_cpu.hpp"

namespace {

struct DisparityMetrics {
  size_t totalPixels = 0;
  size_t cpuValid = 0;
  size_t ocvValid = 0;
  size_t bothValid = 0;
  double mae = 0.0;
  double rmse = 0.0;
  double bad1pxRate = 0.0;  // % of mutually valid pixels with |err| > 1.0 px
  double bad3pxRate = 0.0;  // % of mutually valid pixels with |err| > 3.0 px (KITTI metric)
  double maxErr = 0.0;
};
DisparityMetrics compareDisparityMaps(
    const std::vector<float>& cpuDisp,
    const cv::Mat& ocvDisp16S,
    int minDisparity,
    int borderMargin = 0) {
  DisparityMetrics m;
  const int width = ocvDisp16S.cols;
  const int height = ocvDisp16S.rows;
  const int16_t minValid16 = static_cast<int16_t>(minDisparity * 16);

  double sumAbsErr = 0.0;
  double sumSqErr = 0.0;
  size_t bad1 = 0;
  size_t bad3 = 0;

  for (int y = borderMargin; y < height - borderMargin; ++y) {
    const int16_t* ocvRow = ocvDisp16S.ptr<int16_t>(y);
    for (int x = borderMargin; x < width - borderMargin; ++x) {
      ++m.totalPixels;
      const float dCpu = cpuDisp.at(static_cast<size_t>(y) * width + x);
      const int16_t dOcv16 = ocvRow[x];

      const bool validCpu = (dCpu != sgbm::SgbmCpu::kInvalidDisparity) && std::isfinite(dCpu);
      const bool validOcv = (dOcv16 >= minValid16);

      if (validCpu) ++m.cpuValid;
      if (validOcv) ++m.ocvValid;

      if (validCpu && validOcv) {
        ++m.bothValid;
        const double dOcv = static_cast<double>(dOcv16) / 16.0;
        const double err = std::abs(static_cast<double>(dCpu) - dOcv);
        sumAbsErr += err;
        sumSqErr += err * err;
        m.maxErr = std::max(m.maxErr, err);
        if (err > 1.0) ++bad1;
        if (err > 3.0) ++bad3;
      }
    }
  }

  if (m.bothValid > 0) {
    m.mae = sumAbsErr / static_cast<double>(m.bothValid);
    m.rmse = std::sqrt(sumSqErr / static_cast<double>(m.bothValid));
    m.bad1pxRate = 100.0 * static_cast<double>(bad1) / static_cast<double>(m.bothValid);
    m.bad3pxRate = 100.0 * static_cast<double>(bad3) / static_cast<double>(m.bothValid);
  }
  return m;
}

// Helper to synthesize a textured stereo pair with exact known integer shift
void makeSyntheticStereoPair(
    int width,
    int height,
    int trueShift,
    std::vector<uint8_t>& left,
    std::vector<uint8_t>& right) {
  left.resize(static_cast<size_t>(width) * height);
  right.resize(static_cast<size_t>(width) * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Multi-frequency sinusoidal + checkerboard texture so Sobel & BT have rich signal
      double val = 128.0
                 + 50.0 * std::sin(x * 0.17) * std::cos(y * 0.13)
                 + 35.0 * std::sin((x + y) * 0.31)
                 + (((x / 4 + y / 4) % 2 == 0) ? 20.0 : -20.0);
      uint8_t px = static_cast<uint8_t>(std::max(0.0, std::min(255.0, val)));
      left.at(static_cast<size_t>(y) * width + x) = px;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int xl = std::min(width - 1, x + trueShift);
      right.at(static_cast<size_t>(y) * width + x) = left.at(static_cast<size_t>(y) * width + xl);
    }
  }
}

}  // namespace


TEST(HelloTest, BasicAssertions) {
  EXPECT_STRNE("hello", "world");
  EXPECT_EQ(7 * 6, 42);
}

TEST(SgbmPrefilter, PreservesNegativeSign) {
  sgbm::Params p;
  p.preFilterCap = 63;

  sgbm::SgbmCpu sgbm(p);

  constexpr int W = 3;
  constexpr int H = 3;

  // Strong negative horizontal gradient

  std::vector<uint8_t> img = {255, 128, 0, 255, 128, 0, 255, 128, 0};

  std::vector<int16_t> out;

  sgbm.preFilterForTest(img, out, W, H);

  const int16_t center = out[1 * W + 1];

  EXPECT_LT(center, 0);
}

TEST(SgbmPrefilter, PreservesPositiveSign) {
  sgbm::Params p;
  p.preFilterCap = 63;

  sgbm::SgbmCpu sgbm(p);

  constexpr int W = 3;
  constexpr int H = 3;

  std::vector<uint8_t> img = {0, 128, 255, 0, 128, 255, 0, 128, 255};

  std::vector<int16_t> out;

  sgbm.preFilterForTest(img, out, W, H);

  EXPECT_GT(out[1 * W + 1], 0);
}

TEST(SgbmPrefilter, PositiveClip) {
  sgbm::Params p;
  p.preFilterCap = 20;

  sgbm::SgbmCpu sgbm(p);

  constexpr int W = 3;
  constexpr int H = 3;

  std::vector<uint8_t> img = {0, 0, 255, 0, 0, 255, 0, 0, 255};

  std::vector<int16_t> out;

  sgbm.preFilterForTest(img, out, W, H);

  EXPECT_EQ(out[1 * W + 1], 20);
}

TEST(SgbmPrefilter, NegativeClip) {
  sgbm::Params p;
  p.preFilterCap = 20;

  sgbm::SgbmCpu sgbm(p);

  constexpr int W = 3;
  constexpr int H = 3;

  std::vector<uint8_t> img = {255, 0, 0, 255, 0, 0, 255, 0, 0};

  std::vector<int16_t> out;

  sgbm.preFilterForTest(img, out, W, H);

  EXPECT_EQ(out[1 * W + 1], -20);
}

TEST(SgbmPrefilter, NotAbsoluteValue) {
  sgbm::Params p;
  p.preFilterCap = 63;

  sgbm::SgbmCpu sgbm(p);

  constexpr int W = 3;
  constexpr int H = 3;

  std::vector<uint8_t> img = {255, 128, 0, 255, 128, 0, 255, 128, 0};

  std::vector<int16_t> out;

  sgbm.preFilterForTest(img, out, W, H);

  EXPECT_NE(out[1 * W + 1], std::abs(out[1 * W + 1]));

  EXPECT_LT(out[1 * W + 1], 0);
}

TEST(SgbmPrefilter, MatchesOpenCvXSobel) {

  cv::Mat left = cv::imread(KITTI_LEFT, cv::IMREAD_GRAYSCALE);

  assert(!left.empty());

  sgbm::Params p;

  p.preFilterCap = 63;

  sgbm::SgbmCpu sgbm(p);

  sgbm::StereoOutputs out = sgbm.compute(matToVector(left), matToVector(left), left.cols, left.rows);

  auto reference = ComputeOpenCVReferencePreFilter(left, p.preFilterCap);

  auto stats = CompareBuffers(reference, out.preFilterL);

  PrintDiff(reference, out.preFilterL, left.cols, left.rows, "Prefilter", stats);

  assert(stats.mae < 1.0);
  assert(stats.mismatches < reference.size() * 0.01);
}




// ─── Test 1: Synthetic Constant-Shift Accuracy vs. OpenCV ────────────────────
TEST(SgbmVsOpenCv, SyntheticConstantDisparityMatchesOpenCv) {
  const int width = 320;
  const int height = 160;
  const int trueDisp = 18;

  std::vector<uint8_t> left, right;
  makeSyntheticStereoPair(width, height, trueDisp, left, right);

  sgbm::Params p;
  p.minDisparity = 0;
  p.numDisparities = 64;
  p.blockSize = 5;
  p.P1 = 8 * p.blockSize * p.blockSize;
  p.P2 = 32 * p.blockSize * p.blockSize;
  p.preFilterCap = 63;
  p.uniquenessRatio = 10;
  p.speckleWindowSize = 50;
  p.speckleRange = 2;
  p.disp12MaxDiff = 1;

  sgbm::SgbmCpu cpuSgbm(p);
  auto out = cpuSgbm.compute(left, right, width, height);

  cv::Mat leftMat(height, width, CV_8UC1, left.data());
  cv::Mat rightMat(height, width, CV_8UC1, right.data());
  auto ocvSgbm = cv::StereoSGBM::create(
      p.minDisparity, p.numDisparities, p.blockSize,
      p.P1, p.P2, p.disp12MaxDiff, p.preFilterCap,
      p.uniquenessRatio, p.speckleWindowSize, p.speckleRange,
      cv::StereoSGBM::MODE_HH);

  cv::Mat ocvDisp16;
  ocvSgbm->compute(leftMat, rightMat, ocvDisp16);

  // Exclude left occluded region (x < numDisparities) and block border
  auto metrics = compareDisparityMaps(out.dispL, ocvDisp16, p.minDisparity, p.numDisparities);

  std::cout << "[Synthetic] bothValid=" << metrics.bothValid
            << " MAE=" << metrics.mae
            << " RMSE=" << metrics.rmse
            << " Bad1px=" << metrics.bad1pxRate << "%\n";

  EXPECT_GT(metrics.bothValid, 0u);
  EXPECT_LT(metrics.mae, 0.15);
  EXPECT_LT(metrics.bad1pxRate, 1.0);
}

// ─── Test 2: Real KITTI Stereo Frame Comparison vs. OpenCV ───────────────────
TEST(SgbmVsOpenCv, KittiFrameMatchesOpenCv) {
  // Update paths to match your local KITTI test image directory

  cv::Mat leftMat = cv::imread(KITTI_LEFT, cv::IMREAD_GRAYSCALE);
  cv::Mat rightMat = cv::imread(KITTI_RIGHT, cv::IMREAD_GRAYSCALE);
  if (leftMat.empty() || rightMat.empty()) {
    GTEST_SKIP() << "KITTI test images not found at " << KITTI_LEFT;
  }

  const int width = leftMat.cols;
  const int height = leftMat.rows;
  std::vector<uint8_t> left(leftMat.datastart, leftMat.dataend);
  std::vector<uint8_t> right(rightMat.datastart, rightMat.dataend);

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
  auto out = cpuSgbm.compute(left, right, width, height);

  auto ocvSgbm = cv::StereoSGBM::create(
      p.minDisparity, p.numDisparities, p.blockSize,
      p.P1, p.P2, p.disp12MaxDiff, p.preFilterCap,
      p.uniquenessRatio, p.speckleWindowSize, p.speckleRange,
      cv::StereoSGBM::MODE_HH);

  cv::Mat ocvDisp16;
  ocvSgbm->compute(leftMat, rightMat, ocvDisp16);

  // Compare valid region (x >= numDisparities)
  auto metrics = compareDisparityMaps(out.dispL, ocvDisp16, p.minDisparity, 5);

  std::cout << "\n=== KITTI Full SGBM Comparison vs OpenCV (MODE_HH) ===\n"
            << "Total pixels : " << metrics.totalPixels << "\n"
            << "CPU valid    : " << metrics.cpuValid << "\n"
            << "OpenCV valid : " << metrics.ocvValid << "\n"
            << "Both valid   : " << metrics.bothValid << "\n"
            << "MAE (px)     : " << metrics.mae << "\n"
            << "RMSE (px)    : " << metrics.rmse << "\n"
            << "Bad > 1px    : " << metrics.bad1pxRate << "%\n"
            << "Bad > 3px    : " << metrics.bad3pxRate << "%\n"
            << "Max err (px) : " << metrics.maxErr << "\n";

  EXPECT_GT(metrics.bothValid, metrics.ocvValid * 0.85);
  EXPECT_LT(metrics.mae, 0.5);
  EXPECT_LT(metrics.bad3pxRate, 3.0);
}
