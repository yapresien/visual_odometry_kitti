// =============================================================================
// Project     : opencl_sgbm — CLI Hardware Profiler & Target Benchmark
// File        : opencl_sgbm_profile.cpp
// Author      : <Your Name> <<your.email@example.com>>
// Created     : 2026-11-28
// License     : MIT (SPDX-License-Identifier: MIT)
// =============================================================================

#include <opencl_sgbm/opencl_sgbm.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
  int width = 1241;
  int height = 376;
  int numDisparities = 128;
  int iterations = 50;
  int sgmPaths = 5;
  std::string vendorFilter;
  bool jsonOutput = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) width = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) height = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--disp") == 0 && i + 1 < argc) numDisparities = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--paths") == 0 && i + 1 < argc) sgmPaths = std::stoi(argv[++i]);
    else if (std::strcmp(argv[i], "--vendor") == 0 && i + 1 < argc) vendorFilter = argv[++i];
    else if (std::strcmp(argv[i], "--json") == 0) jsonOutput = true;
  }

  TuningConfig cfg;
  cfg.preferredVendor = vendorFilter;

  OpenClSgbm sgbm(width, height, numDisparities, cfg);
  const auto& specs = sgbm.deviceSpecs();

  // Deterministic textured stereo pair (horizontal gradient + sinusoidal texture)
  std::vector<uint8_t> leftImg(static_cast<size_t>(width) * height);
  std::vector<uint8_t> rightImg(static_cast<size_t>(width) * height);
  const int gtShift = 24;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const uint8_t val = static_cast<uint8_t>((x * 7 + y * 13 + ((x ^ y) & 31)) & 0xFF);
      leftImg[static_cast<size_t>(y) * width + x] = val;
      const int xr = x - gtShift;
      rightImg[static_cast<size_t>(y) * width + x] = (xr >= 0) ? leftImg[static_cast<size_t>(y) * width + xr] : 0;
    }
  }

  ClSgbmParams p;
  p.sgmPaths = sgmPaths;

  std::vector<float> dispOut;
  ClSgbmTimings t;

  // Warmup (also populates persistent ISA binary cache on first run)
  for (int w = 0; w < 5; ++w) sgbm.compute(leftImg, rightImg, p, dispOut, t);

  double sumPf = 0, sumCost = 0, sumAgg = 0, sumWta = 0, sumSp = 0, sumGpu = 0, sumE2E = 0;
  for (int it = 0; it < iterations; ++it) {
    sgbm.compute(leftImg, rightImg, p, dispOut, t);
    sumPf += t.prefilterMs;
    sumCost += t.costVolumeMs;
    sumAgg += t.aggregationMs;
    sumWta += t.wtaLrMs;
    sumSp += t.speckleMs;
    sumGpu += t.totalGpuKernelMs;
    sumE2E += t.endToEndMs;
  }

  const double meanE2E = sumE2E / iterations;
  if (jsonOutput) {
    std::cout << "{\n"
              << "  \"device\": \"" << specs.deviceName << "\",\n"
              << "  \"compute_units\": " << specs.maxComputeUnits << ",\n"
              << "  \"warp_width\": " << specs.warpWidth << ",\n"
              << "  \"resolution\": \"" << width << "x" << height << "\",\n"
              << "  \"disparities\": " << numDisparities << ",\n"
              << "  \"paths\": " << sgmPaths << ",\n"
              << "  \"prefilter_ms\": " << (sumPf / iterations) << ",\n"
              << "  \"cost_volume_ms\": " << (sumCost / iterations) << ",\n"
              << "  \"sgm_agg_ms\": " << (sumAgg / iterations) << ",\n"
              << "  \"wta_lr_ms\": " << (sumWta / iterations) << ",\n"
              << "  \"speckle_ms\": " << (sumSp / iterations) << ",\n"
              << "  \"gpu_total_ms\": " << (sumGpu / iterations) << ",\n"
              << "  \"e2e_ms\": " << meanE2E << ",\n"
              << "  \"fps\": " << (1000.0 / meanE2E) << "\n"
              << "}\n";
  } else {
    std::cout << "=== opencl_sgbm Target Profiler ===\n"
              << "Device        : " << specs.deviceName << " (" << specs.maxComputeUnits
              << " CUs, Warp=" << specs.warpWidth << ", UnifiedMem=" << specs.unifiedMemory << ")\n"
              << "Workload      : " << width << "x" << height << " x D=" << numDisparities
              << " (" << sgmPaths << "-path, " << iterations << " iters)\n"
              << "Prefilter     : " << (sumPf / iterations) << " ms\n"
              << "Cost Volume   : " << (sumCost / iterations) << " ms\n"
              << "SGM Agg       : " << (sumAgg / iterations) << " ms\n"
              << "WTA + LR      : " << (sumWta / iterations) << " ms\n"
              << "Host Speckle  : " << (sumSp / iterations) << " ms\n"
              << "Total GPU     : " << (sumGpu / iterations) << " ms\n"
              << "End-to-End    : " << meanE2E << " ms (" << (1000.0 / meanE2E) << " FPS)\n";
  }
  return 0;
}
