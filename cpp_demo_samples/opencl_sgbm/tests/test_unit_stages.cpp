// =============================================================================
// Project     : opencl_sgbm — Hardware-Agnostic OpenCL Semi-Global Block Matching
// File        : test_unit_stages.cpp
// Author      : <Your Name> <<your.email@example.com>>
// Created     : 2026-11-28
// License     : MIT (SPDX-License-Identifier: MIT)
// Description : Self-contained GoogleTest suite verifying target auto-tuning,
//               persistent binary caching, 1x full-res, and 2x downsampled modes.
// =============================================================================

#include <gtest/gtest.h>
#include <opencl_sgbm/binary_cache.h>
#include <opencl_sgbm/opencl_sgbm.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace {

void generateTexturedStereoPair(
    int width,
    int height,
    int trueShift,
    std::vector<uint8_t>& leftOut,
    std::vector<uint8_t>& rightOut) {
  leftOut.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  rightOut.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Rich multi-frequency pseudo-random texture (avoids repetitive aliasing)
      uint32_t h = static_cast<uint32_t>(x * 374761393u + y * 668265263u);
      h = (h ^ (h >> 13)) * 1274126177u;
      const uint8_t tex = static_cast<uint8_t>((h >> 24) & 0xFFu);
      const uint8_t grad = static_cast<uint8_t>((x * 3 + y * 5) & 0xFF);
      leftOut[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>((tex >> 1) + (grad >> 1));
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int xl = x + trueShift;
      if (xl >= 0 && xl < width) {
        rightOut[static_cast<size_t>(y) * width + x] = leftOut[static_cast<size_t>(y) * width + xl];
      } else {
        rightOut[static_cast<size_t>(y) * width + x] = 0;
      }
    }
  }
}

}  // namespace

TEST(OpenClSgbmUnit, HostPlatformAndBinaryCache) {
  const auto host = opencl_sgbm::detectHostPlatform();
  EXPECT_FALSE(host.osName.empty());
  EXPECT_FALSE(host.cpuArch.empty());

  const std::string key1 = opencl_sgbm::buildCacheKeyPath(host, "TestGPU", "1.0", "-D WARP_WIDTH=32", "__kernel void k(){}");
  const std::string key2 = opencl_sgbm::buildCacheKeyPath(host, "TestGPU", "1.0", "-D WARP_WIDTH=64", "__kernel void k(){}");
  EXPECT_NE(key1, key2);
  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(key1).parent_path()));
}

TEST(OpenClSgbmUnit, FullResSyntheticStereoAccuracy) {
  const int width = 640;
  const int height = 480;
  const int numDisp = 64;
  const int trueShift = 22;

  std::vector<uint8_t> leftImg, rightImg;
  generateTexturedStereoPair(width, height, trueShift, leftImg, rightImg);

  // First instantiation compiles & caches binary; second loads precompiled binary
  OpenClSgbm sgbm1(width, height, numDisp);
  OpenClSgbm sgbm2(width, height, numDisp);

  const auto& specs = sgbm2.deviceSpecs();
  EXPECT_GT(specs.maxComputeUnits, 0u);
  EXPECT_TRUE(specs.warpWidth == 32 || specs.warpWidth == 64 || specs.warpWidth == 16);

  ClSgbmParams p;
  p.minDisparity = 0;
  p.blockSize = 5;
  p.sgmPaths = 5;
  p.downsampleFactor = 1;

  std::vector<float> dispOut;
  ClSgbmTimings timings;
  sgbm2.compute(leftImg, rightImg, p, dispOut, timings);

  ASSERT_EQ(dispOut.size(), static_cast<size_t>(width) * height);

  double sumErr = 0.0;
  size_t validCount = 0;
  for (int y = 20; y < height - 20; ++y) {
    for (int x = trueShift + 20; x < width - 20; ++x) {
      const float d = dispOut[static_cast<size_t>(y) * width + x];
      if (d != p.invalidDisparity && std::isfinite(d)) {
        sumErr += std::abs(static_cast<double>(d) - static_cast<double>(trueShift));
        ++validCount;
      }
    }
  }

  ASSERT_GT(validCount, static_cast<size_t>((width - 60) * (height - 40) * 0.90));
  EXPECT_LT(sumErr / static_cast<double>(validCount), 0.10);
}

TEST(OpenClSgbmUnit, Downsample2xSyntheticStereoAccuracy) {
  const int fullW = 1240;
  const int fullH = 376;
  const int fullD = 128;
  const int trueShift = 36;
  const int ds = 2;

  std::vector<uint8_t> leftImg, rightImg;
  generateTexturedStereoPair(fullW, fullH, trueShift, leftImg, rightImg);

  OpenClSgbm sgbmHalf(fullW / ds, fullH / ds, fullD / ds);

  ClSgbmParams p;
  p.minDisparity = 0;
  p.blockSize = 5;
  p.sgmPaths = 5;
  p.downsampleFactor = ds;

  std::vector<float> dispOut;
  ClSgbmTimings timings;
  sgbmHalf.compute(leftImg, rightImg, p, dispOut, timings);

  ASSERT_EQ(dispOut.size(), static_cast<size_t>(fullW) * fullH);

  double sumErr = 0.0;
  size_t validCount = 0;
  for (int y = 20; y < fullH - 20; ++y) {
    for (int x = trueShift + 24; x < fullW - 24; ++x) {
      const float d = dispOut[static_cast<size_t>(y) * fullW + x];
      if (d != p.invalidDisparity && std::isfinite(d)) {
        sumErr += std::abs(static_cast<double>(d) - static_cast<double>(trueShift));
        ++validCount;
      }
    }
  }

  ASSERT_GT(validCount, static_cast<size_t>((fullW - 80) * (fullH - 40) * 0.85));
  EXPECT_LT(sumErr / static_cast<double>(validCount), 0.25);
}
