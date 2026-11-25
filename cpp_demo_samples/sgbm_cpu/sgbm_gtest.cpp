#include <gtest/gtest.h>

#include <cassert>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

#include "helper_sgbm.hpp"
#include "sgbm_cpu.hpp"

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
//   cv::Mat image = cv::imread(KITTI_LEFT, cv::IMREAD_GRAYSCALE);

//   ASSERT_FALSE(image.empty());

//   auto input = matToVector(image);

//   sgbm::Params p;
//   p.preFilterCap = 63;

//   sgbm::SgbmCpu sgbm(p);

//   std::vector<int16_t> ours;

//   sgbm.preFilterForTest(input, ours, image.cols, image.rows);

//   auto ocv = ComputeOpenCVReferencePreFilter(image, 63);
//   size_t zeroVsNonZero = 0;
//   size_t signMismatch = 0;
//   size_t valueMismatch = 0;
//   if (!CompareZeroVectors(ours, ocv, image.cols, image.rows, "PreFilter"))
//     ++zeroVsNonZero;
//   else if (!CompareNegativeVectors(ours, ocv, image.cols, image.rows, "PreFilter")) 
//     ++signMismatch;
//   else
//     ++valueMismatch;

//   std::cout << "zeroVsNonZero=" << zeroVsNonZero << " signMismatch=" << signMismatch << " valueMismatch=" << valueMismatch << "\n";

//   //ASSERT_EQ(ours.size(), ocv.size());

//   for (size_t i = 0; i < ours.size(); ++i) {
//     // EXPECT_EQ(ours[i], ocv[i]) << "pixel=" << i;
//     if (ours[i] != ocv[i]) {
//       int x = i % image.cols;
//       int y = i / image.cols;

//       std::cout << "Mismatch at " << x << "," << y << "\n";


//       int yy0 = std::max(0, y - 1);
//       int yy1 = std::min(image.rows - 1, y + 1);
//       int xx0 = std::max(0, x - 1);
//       int xx1 = std::min(image.cols - 1, x + 1);

//       for (int yy = yy0; yy <= yy1; yy++) {
//         for (int xx = xx0; xx <= xx1; xx++) {
//           std::cout << static_cast<int>(image.at<uchar>(yy, xx)) << " ";
//         }
//         std::cout << "\n";
//       }

//       std::cout << "ours=" << ours[i] << " opencv=" << ocv[i] << "\n";

//       break;
//     }
//   }
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
