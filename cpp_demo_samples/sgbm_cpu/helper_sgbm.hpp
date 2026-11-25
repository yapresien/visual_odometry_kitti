#include <cassert>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <random>
// #include <opencv2/cudafeatures2d.hpp>
// #include <opencv2/cudastereo.hpp>
// #include <opencv2/cudaoptflow.hpp>
// #include <opencv2/cudaimgproc.hpp>

#include "sgbm_cpu.hpp"

const std::string KITTI_LEFT = "/home/master/myspace/projects/dataset_kitti/sequences/04/image_0/000000.png";
const std::string KITTI_RIGHT = "/home/master/myspace/projects/dataset_kitti/sequences/04/image_1/000000.png";

struct DiffStats {
  size_t total = 0;

  size_t mismatches = 0;

  double mae = 0.0;
  double rmse = 0.0;

  double maxError = 0.0;

  size_t firstBadIndex = std::numeric_limits<size_t>::max();
};

template <typename T>
void PrintDiff(const std::vector<T>& expected, const std::vector<T>& actual, int width, int height, const std::string& name, const DiffStats& stats) {
  std::cout << "\n=== " << name << " ===\n";

  std::cout << "elements=" << stats.total << "\n";

  std::cout << "mismatches=" << stats.mismatches << "\n";

  std::cout << "mae=" << stats.mae << "\n";

  std::cout << "rmse=" << stats.rmse << "\n";

  std::cout << "max=" << stats.maxError << "\n";

  if (stats.firstBadIndex == std::numeric_limits<size_t>::max()) {
    return;
  }

  auto i = stats.firstBadIndex;

  int x = static_cast<int>(i % width);

  int y = static_cast<int>(i / width);

  std::cout << "first mismatch @ (" << x << "," << y << ") expected=" << expected[i] << " actual=" << actual[i] << "\n";
}

template <typename T>
DiffStats CompareBuffers(const std::vector<T>& expected, const std::vector<T>& actual, double tolerance = 0.0) {
  assert(expected.size() == actual.size());

  DiffStats s;

  s.total = expected.size();

  double sq = 0.0;

  for (size_t i = 0; i < expected.size(); ++i) {
    double err = std::abs(double(expected[i]) - double(actual[i]));

    s.mae += err;
    sq += err * err;

    s.maxError = std::max(s.maxError, err);

    if (err > tolerance) {
      ++s.mismatches;

      if (s.firstBadIndex == std::numeric_limits<size_t>::max()) {
        s.firstBadIndex = i;
      }
    }
  }

  s.mae /= s.total;

  s.rmse = std::sqrt(sq / s.total);

  return s;
}

template <typename T>
static void CompareVectors(const std::vector<T>& expected, const std::vector<T>& actual, int width, int height, const std::string& name) {
  assert(expected.size() == actual.size());

  size_t mismatches = 0;

  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      const int x = static_cast<int>(i % width);

      const int y = static_cast<int>(i / width);

      if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        continue;
      }
      std::cerr << name << " mismatch at (" << x << "," << y << ") expected=" << expected[i] << " actual=" << actual[i] << "\n";

      ++mismatches;

      if (mismatches > 20) break;
    }
  }

  if (mismatches) {
    std::cerr << name << " total mismatches (CompareZeroVectors)=" << mismatches << "\n";
  }

  assert(mismatches == 0);
}

template <typename T>
static bool CompareNegativeVectors(const std::vector<T>& expected, const std::vector<T>& actual, int width, int height, const std::string& name) {
  assert(expected.size() == actual.size());

  size_t mismatches = 0;

  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != -1 * actual[i]) {
      const int x = static_cast<int>(i % width);

      const int y = static_cast<int>(i / width);

      if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        continue;
      }
      // std::cerr << name << " CompareNegativeVectors: mismatch at (" << x << "," << y << ") expected=" << expected[i] << " actual=" << actual[i] << "\n";

      ++mismatches;

      // if (mismatches > 20) break;
    }
  }

  if (mismatches) {
    std::cerr << name << " total mismatches (CompareNegativeVectors)=" << mismatches << "\n";
  }

  return mismatches == 0;
}

template <typename T>
static bool CompareZeroVectors(const std::vector<T>& ours, const std::vector<T>& ocv, int width, int height, const std::string& name) {
  assert(ours.size() == ocv.size());

  size_t mismatches = 0;

  for (size_t i = 0; i < ours.size(); ++i) {
    if (ours[i] == 0 && ocv[i] != 0) {
      const int x = static_cast<int>(i % width);

      const int y = static_cast<int>(i / width);

      if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        continue;
      }
      std::cerr << name << " CompareZeroVectors: mismatch at (" << x << "," << y << ") expected=" << ours[i] << " actual=" << ocv[i] << "\n";

      ++mismatches;

      // if (mismatches > 20) break;
    }
  }

  if (mismatches) {
    std::cerr << name << " total mismatches (CompareZeroVectors)=" << mismatches << "\n";
  }

  return mismatches == 0;
}

static std::vector<int16_t> ComputeOpenCVReferencePreFilter(const cv::Mat& img, int cap) {
  cv::Mat sobel;

  cv::Sobel(img, sobel, CV_16S, 1, 0, 3);

  std::vector<int16_t> result(img.rows * img.cols);

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      int v = sobel.at<int16_t>(y, x);

      v = std::max(-cap, std::min(cap, v));

      result[y * img.cols + x] = static_cast<int16_t>(v);
    }
  }

  return result;
}

static void makeSyntheticStereo(std::vector<uint8_t>& left, std::vector<uint8_t>& right, int w, int h, int disp = 16) {
  left.assign((size_t)w * h, 0);
  // Non-periodic intensity ramp with mild noise, restricted to a rectangle.
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> jitter(-6, 6);
  for (int y = 30; y < 90; ++y) {
    for (int x = 80; x < 200; ++x) {
      int val = 40 + (x - 80) * 2 + jitter(rng);  // unique per column
      val = std::min(250, std::max(5, val));
      left[(size_t)y * w + x] = (uint8_t)val;
    }
  }
  right.assign((size_t)w * h, 0);
  for (int y = 30; y < 90; ++y)
    for (int x = 80; x < 200; ++x) {
      int xr = x - disp;
      if (xr >= 0) right[(size_t)y * w + xr] = left[(size_t)y * w + x];
    }
}

static float meanValidInROI(const std::vector<float>& disp, int w, int h, int rx, int ry, int rw, int rh) {
  double sum = 0;
  int n = 0;
  for (int y = ry; y < ry + rh; ++y)
    for (int x = rx; x < rx + rw; ++x) {
      float v = disp[(size_t)y * w + x];
      if (v >= 0) {
        sum += v;
        ++n;
      }
    }
  return n ? (float)(sum / n) : -1.0f;
}
static std::vector<uint8_t> matToVector(const cv::Mat& m) {
  CV_Assert(m.type() == CV_8UC1 && m.isContinuous());
  return std::vector<uint8_t>(m.data, m.data + (size_t)m.total());
}

void compareAndTestCVSgbm() {
  // Load one KITTI stereo pair
  cv::Mat L = cv::imread(KITTI_LEFT, cv::IMREAD_GRAYSCALE);
  cv::Mat R = cv::imread(KITTI_RIGHT, cv::IMREAD_GRAYSCALE);

  // Run cv::StereoSGBM with same params
  cv::Ptr<cv::StereoSGBM> cvSgbm = cv::StereoSGBM::create(0, 128, 5, 200, 400, -1, 63, 0, 0, 32, cv::StereoSGBM::MODE_SGBM);
  cv::Mat cvDisp16;
  cvSgbm->compute(L, R, cvDisp16);
  cv::Mat cvDisp;
  cvDisp16.convertTo(cvDisp, CV_32F, 1.0 / 16.0);

  // Run your SGBM
  sgbm::Params p;
  p.numDisparities = 128;
  p.blockSize = 5;
  p.P1 = 200;
  p.P2 = 400;
  p.disp12MaxDiff = -1;
  p.uniquenessRatio = 0;
  p.speckleWindowSize = 0;
  sgbm::SgbmCpu mine(p);
  auto result = mine.compute(matToVector(L), matToVector(R), L.cols, L.rows);

  // Compute metrics
  int valid = 0, bad1 = 0, bad2 = 0;
  double sumAE = 0;
  for (int y = 0; y < L.rows; ++y)
    for (int x = 0; x < L.cols; ++x) {
      float cv = cvDisp.at<float>(y, x);
      float my = result.dispL[y * L.cols + x];
      if (cv > 0.5f && my >= 0.0f) {
        ++valid;
        float ae = std::abs(cv - my);
        sumAE += ae;
        if (ae > 1.0f) ++bad1;
        if (ae > 2.0f) ++bad2;
      }
    }
  std::cout << "valid=" << valid << " MAE=" << sumAE / valid << " bad@1=" << 100.0 * bad1 / valid << "%"
            << " bad@2=" << 100.0 * bad2 / valid << "%\n";
}

void TestPrefilterVsOpenCV() {
  cv::Mat left = cv::imread(KITTI_LEFT, cv::IMREAD_GRAYSCALE);

  assert(!left.empty());

  sgbm::Params p;

  p.preFilterCap = 63;

  sgbm::SgbmCpu sgbm(p);

  sgbm::StereoOutputs out = sgbm.compute(matToVector(left), matToVector(left), left.cols, left.rows);

  auto reference = ComputeOpenCVReferencePreFilter(left, p.preFilterCap);

  // size_t mismatches = 0;
  // double mae = 0.0;
  // double maxError = 0.0;
  // for (size_t i = 0; i < reference.size(); i++) {
  //   double e = std::abs(double(reference[i]) - double(out.preFilterL[i]));
  //   mae += e;
  //   maxError = std::max(maxError, e);
  //   if (e != 0) mismatches++;
  // }
  // std::cout << "Prefilter\n"
  //           << "MAE=" << mae / reference.size() << "\n"
  //           << "max=" << maxError << "\n"
  //           << "mismatch=" << 100.0 * mismatches / reference.size() << "%\n";
  // CompareVectors(reference, out.preFilterL, left.cols, left.rows, "prefilter");
  auto stats = CompareBuffers(reference, out.preFilterL);

  PrintDiff(reference, out.preFilterL, left.cols, left.rows, "Prefilter", stats);
}

// int main() {
//   const int w = 320, h = 120, disp = 16;

//   std::vector<uint8_t> left, right;
//   makeSyntheticStereo(left, right, w, h, disp);

//   //   sgbm::Params p;
//   //   p.numDisparities = 64;
//   //   p.blockSize = 5;
//   //   p.disp12MaxDiff = -1;
//   //   p.speckleWindowSize = 0;
//   //   p.uniquenessRatio = 0;
//   //   p.P1 = 200;
//   //   p.P2 = 400;
//   //   sgbm::SgbmCpu sgbm(p);

//   //   auto outputs = sgbm.compute(left, right, w, h);

//   //   // Test 1: output size
//   //   assert(outputs.dispL.size() == (size_t)w * h);

//   //   assert(outputs.dispR.size() == (size_t)w * h);

//   //   assert(outputs.dispLRaw.size() == (size_t)w * h);

//   //   assert(outputs.dispRRaw.size() == (size_t)w * h);

//   //   // Test 2: no NaN/Inf
//   //   for (float v : outputs.dispL) {
//   //     assert(std::isfinite(v) || v == sgbm::SgbmCpu::kInvalidDisparity);
//   //   }

//   //   for (float v : outputs.dispR) {
//   //     assert(std::isfinite(v) || v == sgbm::SgbmCpu::kInvalidDisparity);
//   //   }

//   //   for (float v : outputs.dispLRaw) {
//   //     assert(std::isfinite(v) || v == sgbm::SgbmCpu::kInvalidDisparity);
//   //   }

//   //   for (float v : outputs.dispRRaw) {
//   //     assert(std::isfinite(v) || v == sgbm::SgbmCpu::kInvalidDisparity);
//   //   }

//   //   // Test 3: mean disparity in textured ROI should be ~16
//   //   float mean = meanValidInROI(outputs.dispLRaw, w, h, 100, 40, 80, 40);

//   //   std::cout << "[test] ROI mean disparity = " << mean << " (expected ~" << disp << ")\n";

//   //   assert(std::abs(mean - disp) < 4.0f);

//   //   // Test 4: invalid input throws
//   //   size_t validRaw = 0;
//   //   size_t validChecked = 0;

//   //   for (float v : outputs.dispLRaw) {
//   //     if (v != sgbm::SgbmCpu::kInvalidDisparity) ++validRaw;
//   //   }

//   //   for (float v : outputs.dispL) {
//   //     if (v != sgbm::SgbmCpu::kInvalidDisparity) ++validChecked;
//   //   }

//   //   assert(validChecked <= validRaw);

//   //   assert(outputs.aggL.size() == outputs.costL.size());

//   //   assert(outputs.aggR.size() == outputs.costR.size());

//   //   assert(!outputs.aggL.empty());
//   //   assert(!outputs.aggR.empty());

//   //   bool thrown = false;

//   //   try {
//   //     sgbm.compute({}, {}, w, h);
//   //   } catch (const std::invalid_argument&) {
//   //     thrown = true;
//   //   }

//   //   assert(thrown);

//   //   float meanL = meanValidInROI(outputs.dispLRaw, w, h, 100, 40, 80, 40);

//   //   float meanR = meanValidInROI(outputs.dispRRaw, w, h, 100, 40, 80, 40);

//   //   std::cout << "[test] meanL=" << meanL << " meanR=" << meanR << '\n';
//   //   assert(meanL > 10.0f);
//   //   assert(meanL < 22.0f);
//   //   assert(std::abs(meanL - disp) < 4.0f);

//   //   compareAndTestCVSgbm();

//   //   std::cout << "All SGBM tests passed.\n";

//   TestPrefilterVsOpenCV();

//   return 0;
// }