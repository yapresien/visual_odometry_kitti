#include "sgbm_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <array>

namespace sgbm {

static constexpr int16_t kInvalidCost = std::numeric_limits<int16_t>::max();
static constexpr int32_t kMaxPathCost = std::numeric_limits<int32_t>::max() / 4;

// ─── Construction / validation ───────────────────────────────────────────────
SgbmCpu::SgbmCpu(const Params& p) : params_(p) {
  if (p.numDisparities <= 0 || p.numDisparities % 16 != 0)
    throw std::invalid_argument("numDisparities must be >0 and divisible by 16");
  if (p.blockSize < 3 || p.blockSize % 2 == 0)
    throw std::invalid_argument("blockSize must be odd and >=3");
  if (p.P2 <= p.P1)
    throw std::invalid_argument("P2 must be > P1");
}

void SgbmCpu::validate(int width, int height, size_t leftSize, size_t rightSize) const {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("invalid image dimensions");
  if (leftSize != static_cast<size_t>(width) * height || rightSize != leftSize)
    throw std::invalid_argument("input size does not match width*height");
  if (width < params_.numDisparities)
    throw std::invalid_argument("width must be >= numDisparities");
}

// ─── Stage 1: Sobel-X Prefilter (Bit-exact with OpenCV prefilterXSobel) ──────
void SgbmCpu::preFilter(const std::vector<uint8_t>& src, std::vector<int16_t>& dst, int width, int height) const {
  dst.assign(static_cast<size_t>(width) * height, 0);

  if (params_.preFilterCap <= 0) {
    for (size_t i = 0; i < src.size(); ++i) {
      dst[i] = static_cast<int16_t>(src[i]);
    }
    return;
  }

  const int cap = params_.preFilterCap;

  for (int y = 0; y < height; ++y) {
    const int y_prev = (y > 0) ? (y - 1) : std::min(1, height - 1);
    const int y_next = (y < height - 1) ? (y + 1) : std::max(0, height - 2);

    const uint8_t* row_prev = &src[y_prev * width];
    const uint8_t* row_curr = &src[y * width];
    const uint8_t* row_next = &src[y_next * width];
    int16_t* dst_row = &dst[y * width];

    for (int x = 1; x < width - 1; ++x) {
      int s = (row_prev[x + 1] - row_prev[x - 1])
            + 2 * (row_curr[x + 1] - row_curr[x - 1])
            + (row_next[x + 1] - row_next[x - 1]);

      dst_row[x] = static_cast<int16_t>(std::max(-cap, std::min(cap, s)));
    }
  }
}

// ─── Stage 2: Birchfield-Tomasi + Raw Intensity Cost Volume ──────────────────
void SgbmCpu::computeCost(
    const std::vector<uint8_t>& rawLeft,
    const std::vector<uint8_t>& rawRight,
    const std::vector<int16_t>& sobelLeft,
    const std::vector<int16_t>& sobelRight,
    int width,
    int height,
    std::vector<int16_t>& cost) const {
  const int D = params_.numDisparities;
  const int B = params_.blockSize;
  const int B2 = B / 2;
  const int dMin = params_.minDisparity;
  const int dMax = dMin + D - 1;

  const size_t numPixels = static_cast<size_t>(width) * height;
  cost.assign(numPixels * D, kInvalidCost);

  std::vector<int16_t> sobelLMin(numPixels), sobelLMax(numPixels);
  std::vector<int16_t> sobelRMin(numPixels), sobelRMax(numPixels);
  std::vector<int16_t> rawLMin(numPixels), rawLMax(numPixels);
  std::vector<int16_t> rawRMin(numPixels), rawRMax(numPixels);

  auto computeRowBounds = [width](const auto* row, int16_t* minOut, int16_t* maxOut) {
    minOut[0] = maxOut[0] = static_cast<int16_t>(row[0]);
    minOut[width - 1] = maxOut[width - 1] = static_cast<int16_t>(row[width - 1]);

    for (int x = 1; x < width - 1; ++x) {
      const int v = row[x];
      const int vPrev = (v + row[x - 1]) >> 1;
      const int vNext = (v + row[x + 1]) >> 1;
      minOut[x] = static_cast<int16_t>(std::min(v, std::min(vPrev, vNext)));
      maxOut[x] = static_cast<int16_t>(std::max(v, std::max(vPrev, vNext)));
    }
  };

  for (int y = 0; y < height; ++y) {
    const size_t rowOff = static_cast<size_t>(y) * width;
    computeRowBounds(&sobelLeft[rowOff], &sobelLMin[rowOff], &sobelLMax[rowOff]);
    computeRowBounds(&sobelRight[rowOff], &sobelRMin[rowOff], &sobelRMax[rowOff]);
    computeRowBounds(&rawLeft[rowOff], &rawLMin[rowOff], &rawLMax[rowOff]);
    computeRowBounds(&rawRight[rowOff], &rawRMin[rowOff], &rawRMax[rowOff]);
  }

  std::vector<int32_t> pixelCost(numPixels * D, 0);

  for (int y = 0; y < height; ++y) {
    const size_t rowOff = static_cast<size_t>(y) * width;
    for (int x = 1; x < width - 1; ++x) {
      const size_t idxL = rowOff + x;
      const int sL = sobelLeft[idxL], sLMin = sobelLMin[idxL], sLMax = sobelLMax[idxL];
      const int rL = rawLeft[idxL],   rLMin = rawLMin[idxL],   rLMax = rawLMax[idxL];

      for (int d = dMin; d <= dMax; ++d) {
        const int xr = x - d;
        if (xr < 1 || xr >= width - 1) continue;

        const size_t idxR = rowOff + xr;

        const int sR = sobelRight[idxR];
        const int sDiffLR = std::max(0, std::max(sL - sobelRMax[idxR], sobelRMin[idxR] - sL));
        const int sDiffRL = std::max(0, std::max(sR - sLMax, sLMin - sR));
        const int btSobel = std::min(sDiffLR, sDiffRL);

        const int rR = rawRight[idxR];
        const int rDiffLR = std::max(0, std::max(rL - rawRMax[idxR], rawRMin[idxR] - rL));
        const int rDiffRL = std::max(0, std::max(rR - rLMax, rLMin - rR));
        const int btRaw = std::min(rDiffLR, rDiffRL);

        pixelCost[idxL * D + (d - dMin)] = btSobel + btRaw;
      }
    }
  }

  for (int y = B2; y < height - B2; ++y) {
    for (int x = B2 + 1; x < width - B2 - 1; ++x) {
      for (int d = dMin; d <= dMax; ++d) {
        const int xr = x - d;
        if (xr - B2 < 1 || xr + B2 >= width - 1) continue;

        const int dIdx = d - dMin;
        int32_t blockSum = 0;
        for (int by = -B2; by <= B2; ++by) {
          const size_t rowBase = (static_cast<size_t>(y + by) * width + x) * D + dIdx;
          for (int bx = -B2; bx <= B2; ++bx) {
            blockSum += pixelCost[rowBase + bx * D];
          }
        }

        cost[(static_cast<size_t>(y) * width + x) * D + dIdx] =
            static_cast<int16_t>(std::min<int32_t>(blockSum, kInvalidCost - 1));
      }
    }
  }
}

// ─── Stage 3: 8-Path Cost Aggregation ────────────────────────────────────────
void SgbmCpu::aggregate(const std::vector<int16_t>& cost, int width, int height, std::vector<int32_t>& agg) const {
  const int D = params_.numDisparities;
  const int P1 = params_.P1;
  const int P2 = params_.P2;
  const size_t volumeSize = cost.size();

  agg.assign(volumeSize, 0);
  std::vector<int32_t> path(volumeSize, kMaxPathCost);

  struct Dir {
    int dx;
    int dy;
  };

  const std::array<Dir, 8> dirs = {{
    {1, 0},    // LR
    {-1, 0},   // RL
    {0, 1},    // TB
    {0, -1},   // BT
    {1, 1},    // TL -> BR
    {-1, 1},   // TR -> BL
    {1, -1},   // BL -> TR
    {-1, -1}   // BR -> TL
}};


  for (const auto& dir : dirs) {
    std::fill(path.begin(), path.end(), kMaxPathCost);

    const int dx = dir.dx;
    const int dy = dir.dy;

    const int xStart = (dx < 0) ? (width - 1) : 0;
    const int xEnd   = (dx < 0) ? -1 : width;
    const int xStep  = (dx < 0) ? -1 : 1;

    const int yStart = (dy < 0) ? (height - 1) : 0;
    const int yEnd   = (dy < 0) ? -1 : height;
    const int yStep  = (dy < 0) ? -1 : 1;

    for (int y = yStart; y != yEnd; y += yStep) {
      for (int x = xStart; x != xEnd; x += xStep) {
        const size_t currBase = (static_cast<size_t>(y) * width + x) * D;
        const int px = x - dx;
        const int py = y - dy;
        const bool hasPrev = (px >= 0 && px < width && py >= 0 && py < height);

        if (!hasPrev) {
          for (int d = 0; d < D; ++d) {
            const int16_t c = cost[currBase + d];
            if (c != kInvalidCost) {
              path[currBase + d] = c;
            }
          }
          continue;
        }

        const size_t prevBase = (static_cast<size_t>(py) * width + px) * D;

        int32_t minPrev = kMaxPathCost;
        for (int d = 0; d < D; ++d) {
          if (cost[prevBase + d] != kInvalidCost) {
            minPrev = std::min(minPrev, path[prevBase + d]);
          }
        }

        if (minPrev == kMaxPathCost) {
          for (int d = 0; d < D; ++d) {
            const int16_t c = cost[currBase + d];
            if (c != kInvalidCost) {
              path[currBase + d] = c;
            }
          }
          continue;
        }

        for (int d = 0; d < D; ++d) {
          const int16_t c = cost[currBase + d];
          if (c == kInvalidCost) continue;

          const int32_t l0 = path[prevBase + d];
          const int32_t l1 = (d > 0) ? (path[prevBase + d - 1] + P1) : kMaxPathCost;
          const int32_t l2 = (d < D - 1) ? (path[prevBase + d + 1] + P1) : kMaxPathCost;
          const int32_t l3 = minPrev + P2;

          const int32_t best = std::min(std::min(l0, l1), std::min(l2, l3));
          path[currBase + d] = static_cast<int32_t>(c) + best - minPrev;
        }
      }
    }

    for (size_t i = 0; i < volumeSize; ++i) {
      if (cost[i] != kInvalidCost) {
        agg[i] += path[i];
      }
    }
  }
}

// ─── Stage 4: Left WTA + Non-Adjacent Uniqueness + Subpixel ──────────────────
void SgbmCpu::selectDisparity(const std::vector<int32_t>& agg, const std::vector<int16_t>& cost, int width, int height, std::vector<float>& disp) const {
  const int D = params_.numDisparities;
  const int dMin = params_.minDisparity;
  disp.assign(static_cast<size_t>(width) * height, kInvalidDisparity);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t base = (static_cast<size_t>(y) * width + x) * D;
      const int32_t* row = &agg[base];
      const int16_t* costRow = &cost[base];

      int bestD = -1;
      int32_t bestC = std::numeric_limits<int32_t>::max();
      for (int d = 0; d < D; ++d) {
        if (costRow[d] == kInvalidCost) continue;
        if (row[d] < bestC) {
          bestC = row[d];
          bestD = d;
        }
      }
      if (bestD < 0) continue;

      // OpenCV uniqueness check: compare against non-adjacent disparities (|d - bestD| > 1)
      if (params_.uniquenessRatio > 0) {
        int32_t minOther = std::numeric_limits<int32_t>::max();
        for (int d = 0; d < D; ++d) {
          if (std::abs(d - bestD) <= 1) continue;
          if (costRow[d] == kInvalidCost) continue;
          minOther = std::min(minOther, row[d]);
        }

        if (minOther != std::numeric_limits<int32_t>::max()) {
          if (static_cast<int64_t>(bestC) * (100 + params_.uniquenessRatio) >
              static_cast<int64_t>(minOther) * 100) {
            continue;
          }
        }
      }

      // Parabolic sub-pixel interpolation
      float subpixelOffset = 0.0f;
      if (bestD > 0 && bestD < D - 1 &&
          costRow[bestD - 1] != kInvalidCost &&
          costRow[bestD + 1] != kInvalidCost) {
        const int32_t cPrev = row[bestD - 1];
        const int32_t cCurr = row[bestD];
        const int32_t cNext = row[bestD + 1];
        const int32_t denom = 2 * (cPrev - 2 * cCurr + cNext);
        if (denom > 0) {
          subpixelOffset = static_cast<float>(cPrev - cNext) / static_cast<float>(denom);
          subpixelOffset = std::max(-0.5f, std::min(0.5f, subpixelOffset));
        }
      }

      disp[static_cast<size_t>(y) * width + x] = static_cast<float>(dMin + bestD) + subpixelOffset;
    }
  }
}

// ─── Stage 5: Shared-Volume Right Disparity & LR Consistency Check ───────────
void SgbmCpu::selectRightDisparity(
    const std::vector<int32_t>& aggL,
    const std::vector<int16_t>& costL,
    int width,
    int height,
    std::vector<float>& dispR) const {
  const int D = params_.numDisparities;
  const int dMin = params_.minDisparity;
  dispR.assign(static_cast<size_t>(width) * height, kInvalidDisparity);

  for (int y = 0; y < height; ++y) {
    for (int xr = 0; xr < width; ++xr) {
      int bestD = -1;
      int32_t bestC = std::numeric_limits<int32_t>::max();

      for (int dIdx = 0; dIdx < D; ++dIdx) {
        const int d = dMin + dIdx;
        const int xl = xr + d;
        if (xl < 0 || xl >= width) continue;

        const size_t idx = (static_cast<size_t>(y) * width + xl) * D + dIdx;
        if (costL[idx] == kInvalidCost) continue;

        if (aggL[idx] < bestC) {
          bestC = aggL[idx];
          bestD = dIdx;
        }
      }

      if (bestD >= 0) {
        dispR[static_cast<size_t>(y) * width + xr] = static_cast<float>(dMin + bestD);
      }
    }
  }
}

void SgbmCpu::lrCheck(
    const std::vector<float>& dispR,
    int width,
    int height,
    std::vector<float>& dispL) const {
  if (params_.disp12MaxDiff < 0) return;

  const float maxDiff = static_cast<float>(params_.disp12MaxDiff);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idxL = static_cast<size_t>(y) * width + x;
      const float dL = dispL[idxL];
      if (dL == kInvalidDisparity) continue;

      const int xr = static_cast<int>(std::lround(x - dL));
      if (xr < 0 || xr >= width) {
        dispL[idxL] = kInvalidDisparity;
        continue;
      }

      const int dLInt = static_cast<int>(std::lround(dL));
      const float dR = dispR[static_cast<size_t>(y) * width + xr];
      if (dR == kInvalidDisparity || std::abs(static_cast<float>(dLInt) - dR) > maxDiff) {
        dispL[idxL] = kInvalidDisparity;
      }
    }
  }
}

// ─── Stage 6: Speckle Filtering (OpenCV 1/16th-pixel threshold scale) ────────
void SgbmCpu::speckleFilter(std::vector<float>& disp, int width, int height) const {
  if (params_.speckleWindowSize <= 0) return;

  const float maxDiff = static_cast<float>(params_.speckleRange) ;
  const int winSize = params_.speckleWindowSize;

  std::vector<uint8_t> visited(static_cast<size_t>(width) * height, 0);
  std::vector<std::pair<int, int>> stack;
  std::vector<std::pair<int, int>> component;

  const std::array<std::pair<int, int>, 4> nb = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      if (visited[idx]) continue;
      if (disp[idx] == kInvalidDisparity) {
        visited[idx] = 1;
        continue;
      }

      stack.clear();
      component.clear();
      stack.push_back({x, y});
      visited[idx] = 1;

      while (!stack.empty()) {
        auto [cx, cy] = stack.back();
        stack.pop_back();
        component.push_back({cx, cy});
        const float ref = disp[static_cast<size_t>(cy) * width + cx];

        
        for (const auto& [dx, dy] : nb) {
            const int nx = cx + dx;
            const int ny = cy + dy;

          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

          const size_t nIdx = static_cast<size_t>(ny) * width + nx;
          if (visited[nIdx]) continue;

          const float v = disp[nIdx];
          if (v == kInvalidDisparity) {
            visited[nIdx] = 1;
            continue;
          }

          if (std::abs(v - ref) <= maxDiff) {
            visited[nIdx] = 1;
            stack.push_back({nx, ny});
          }
        }
      }

      if (static_cast<int>(component.size()) < winSize) {
        for (const auto& [cx, cy] : component) {
          disp[static_cast<size_t>(cy) * width + cx] = kInvalidDisparity;
        }
      }
    }
  }
}

// ─── Main entry point ────────────────────────────────────────────────────────
StereoOutputs SgbmCpu::compute(
    const std::vector<uint8_t>& left,
    const std::vector<uint8_t>& right,
    int width,
    int height) const {
  validate(width, height, left.size(), right.size());

  std::vector<int16_t> lf, rf;
  preFilter(left, lf, width, height);
  preFilter(right, rf, width, height);

  StereoOutputs outputs;
  outputs.preFilterL = lf;
  outputs.preFilterR = rf;

  // Single left-to-right cost + 8-path aggregation (matches OpenCV StereoSGBM)
  computeCost(left, right, lf, rf, width, height, outputs.costL);
  aggregate(outputs.costL, width, height, outputs.aggL);

  selectDisparity(outputs.aggL, outputs.costL, width, height, outputs.dispLRaw);
  selectRightDisparity(outputs.aggL, outputs.costL, width, height, outputs.dispRRaw);

  outputs.dispL = outputs.dispLRaw;
  outputs.dispR = outputs.dispRRaw;

  lrCheck(outputs.dispR, width, height, outputs.dispL);
  speckleFilter(outputs.dispL, width, height);

  return outputs;
}

}  // namespace sgbm
