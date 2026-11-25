#include "sgbm_cpu.hpp"

#include <cstring>
#include <iostream>

namespace sgbm {

static constexpr int16_t kInvalidCost = std::numeric_limits<int16_t>::max();

// ─── Construction / validation ───────────────────────────────────────────────
SgbmCpu::SgbmCpu(const Params& p) : params_(p) {
  if (p.numDisparities <= 0 || p.numDisparities % 16 != 0) throw std::invalid_argument("numDisparities must be >0 and divisible by 16");
  if (p.blockSize < 3 || p.blockSize % 2 == 0) throw std::invalid_argument("blockSize must be odd and >=3");
  if (p.P2 <= p.P1) throw std::invalid_argument("P2 must be > P1");
}

void SgbmCpu::validate(int width, int height, size_t leftSize, size_t rightSize) const {
  if (width <= 0 || height <= 0) throw std::invalid_argument("invalid image dimensions");
  if (leftSize != (size_t)width * height || rightSize != leftSize) throw std::invalid_argument("input size does not match width*height");
  if (width < params_.numDisparities) throw std::invalid_argument("width must be >= numDisparities");
}
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
    const int y_prev = (y > 0) ? (y - 1) : 1;
    const int y_next = (y < height - 1) ? (y + 1) : (height - 2);

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

// ─── Stage 2: matching cost (SAD over blockSize window) ──────────────────────
void SgbmCpu::computeCost(const std::vector<int16_t>& left, const std::vector<int16_t>& right, int width, int height, std::vector<int16_t>& cost) const {
  const int D = params_.numDisparities;
  const int B = params_.blockSize;
  const int B2 = B / 2;

  // cost[(y*width + x)*D + d] — too big for int16 in some cases, but
  // with preFilterCap <= 63 and B=5 the SAD max is 63*25 = 1575, fits.
  cost.assign((size_t)width * height * D, kInvalidCost);

  for (int y = B2; y < height - B2; ++y) {
    for (int x = B2; x < width - B2; ++x) {
      const int dMin = params_.minDisparity;
      const int dMax = dMin + D - 1;
      for (int d = dMin; d <= dMax; ++d) {
        const int xr = x - d;
        if (xr - B2 < 0 || xr + B2 >= width) continue;
        int sad = 0;
        for (int by = -B2; by <= B2; ++by) {
          for (int bx = -B2; bx <= B2; ++bx) {
            int lv = left[(y + by) * width + (x + bx)];
            int rv = right[(y + by) * width + (xr + bx)];
            sad += std::abs(lv - rv);
          }
        }
        if (sad < std::numeric_limits<int16_t>::max()) cost[((size_t)y * width + x) * D + (d - dMin)] = (int16_t)sad;
      }
    }
  }
}

// ─── Stage 3: path aggregation (4 paths, in-place recursive) ─────────────────
void SgbmCpu::aggregate(const std::vector<int16_t>& cost, int width, int height, std::vector<int32_t>& agg) const {
  const int D = params_.numDisparities;
  const int P1 = params_.P1;
  const int P2 = params_.P2;

  const size_t volumeSize = cost.size();

  agg.assign(volumeSize, 0);

  struct Dir {
    int dx;
    int dy;
  };

  const Dir dirs[4] = {
      {1, 0},   // LR
      {-1, 0},  // RL
      {0, 1},   // TB
      {0, -1}   // BT
  };

  for (const auto& dir : dirs) {
    std::vector<int32_t> path(volumeSize, 0);

    const int dx = dir.dx;
    const int dy = dir.dy;

    const int xStart = (dx > 0) ? 0 : (dx < 0) ? width - 1 : 0;

    const int xEnd = (dx > 0) ? width : (dx < 0) ? -1 : width;

    const int xStep = (dx > 0) ? 1 : (dx < 0) ? -1 : 1;

    const int yStart = (dy > 0) ? 0 : (dy < 0) ? height - 1 : 0;

    const int yEnd = (dy > 0) ? height : (dy < 0) ? -1 : height;

    const int yStep = (dy > 0) ? 1 : (dy < 0) ? -1 : 1;

    for (int y = yStart; y != yEnd; y += yStep) {
      for (int x = xStart; x != xEnd; x += xStep) {
        const int px = x - dx;
        const int py = y - dy;

        const bool hasPrev = px >= 0 && px < width && py >= 0 && py < height;

        if (!hasPrev) {
          for (int d = 0; d < D; ++d) {
            const int16_t c = cost[((size_t)y * width + x) * D + d];

            if (c == kInvalidCost) continue;

            path[((size_t)y * width + x) * D + d] = c;
          }

          continue;
        }

        int32_t minPrev = std::numeric_limits<int32_t>::max();

        for (int d = 0; d < D; ++d) {
          const int16_t cPrev = cost[((size_t)py * width + px) * D + d];

          if (cPrev == kInvalidCost) continue;

          minPrev = std::min(minPrev, path[((size_t)py * width + px) * D + d]);
        }

        if (minPrev == std::numeric_limits<int32_t>::max()) {
          minPrev = 0;
        }

        for (int d = 0; d < D; ++d) {
          const int16_t c = cost[((size_t)y * width + x) * D + d];

          if (c == kInvalidCost) continue;

          const int prevBase = ((size_t)py * width + px) * D;

          int32_t l0 = path[prevBase + d];

          int32_t l1 = (d > 0) ? path[prevBase + d - 1] + P1 : std::numeric_limits<int32_t>::max() / 4;

          int32_t l2 = (d < D - 1) ? path[prevBase + d + 1] + P1 : std::numeric_limits<int32_t>::max() / 4;

          int32_t l3 = minPrev + P2;

          int32_t best = std::min(std::min(l0, l1), std::min(l2, l3));

          path[((size_t)y * width + x) * D + d] = static_cast<int32_t>(c) + best - minPrev;
        }
      }
    }

    for (size_t i = 0; i < volumeSize; ++i) {
      agg[i] += path[i];
    }
  }
}
// ─── Stage 4: WTA + subpixel + uniqueness ────────────────────────────────────
void SgbmCpu::selectDisparity(const std::vector<int32_t>& agg, const std::vector<int16_t>& cost, int width, int height, std::vector<float>& disp) const {
  const int D = params_.numDisparities;
  disp.assign((size_t)width * height, kInvalidDisparity);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int32_t* row = &agg[((size_t)y * width + x) * D];
      // WTA — find best disparity, but only over disparities whose cost is valid
      int bestD = -1;
      int32_t bestC = std::numeric_limits<int32_t>::max();
      const int16_t* costRow = &cost[((size_t)y * width + x) * D];

      for (int d = 0; d < D; ++d) {
        if (costRow[d] == kInvalidCost) continue;
        if (row[d] < bestC) {
          bestC = row[d];
          bestD = d;
        }
      }
      if (bestD < 0) continue;

      // after the WTA loop, before the uniqueness check:
      if (y == 60 && x == 140) {  // a pixel in the middle of the ROI
        std::cout << "pixel(" << x << "," << y << ") bestD=" << bestD << " bestC=" << bestC << " cost[bestD-1]=" << (int)costRow[std::max(0, bestD - 1)] << " cost[bestD]=" << (int)costRow[bestD] << " cost[bestD+1]=" << (int)costRow[std::min(D - 1, bestD + 1)] << " agg[bestD-1]=" << row[std::max(0, bestD - 1)] << " agg[bestD]=" << row[bestD] << " agg[bestD+1]=" << row[std::min(D - 1, bestD + 1)] << "\n";
      }
      // Uniqueness: OpenCV-style, compare best against immediate neighbors only.
      // Both neighbors must themselves have valid cost entries.
      if (params_.uniquenessRatio > 0 && bestD > 0 && bestD < D - 1) {
        const bool nLValid = (costRow[bestD - 1] != kInvalidCost);
        const bool nRValid = (costRow[bestD + 1] != kInvalidCost);
        int32_t cMin2 = std::numeric_limits<int32_t>::max();
        if (nLValid || nRValid) {
          if (nLValid) cMin2 = std::min(cMin2, row[bestD - 1]);
          if (nRValid) cMin2 = std::min(cMin2, row[bestD + 1]);
          if ((int64_t)bestC * (100 + params_.uniquenessRatio) > (int64_t)cMin2 * 100) {
            continue;  // reject — best is not sufficiently unique
          }
        }
      }

      // Subpixel refinement — unchanged
      float sub = 0.0f;
      if (bestD > 0 && bestD < D - 1) {
        const int32_t c0 = row[bestD - 1];
        const int32_t c1 = row[bestD];
        const int32_t c2 = row[bestD + 1];
        const double denom = (double)(c0 - 2 * c1 + c2);
        if (std::abs(denom) > 1e-9) sub = (float)(0.5 * (c0 - c2) / denom);
        if (sub < -0.5f) sub = -0.5f;
        if (sub > 0.5f) sub = 0.5f;
      }

      disp[(size_t)y * width + x] = (float)(bestD + params_.minDisparity) + sub;
    }
  }
}

// ─── Stage 5: left-right consistency check ───────────────────────────────────
void SgbmCpu::lrCheck(const std::vector<float>& dispRight, int width, int height, std::vector<float>& dispLeft) const {
  if (params_.disp12MaxDiff < 0) return;

  const float maxDiff = static_cast<float>(params_.disp12MaxDiff);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float dL = dispLeft[(size_t)y * width + x];

      if (dL < 0.0f) continue;

      int xr = static_cast<int>(std::lround(x - dL));

      if (xr < 0 || xr >= width) {
        dispLeft[(size_t)y * width + x] = kInvalidDisparity;

        continue;
      }

      float dR = dispRight[(size_t)y * width + xr];

      if (dR < 0.0f) {
        dispLeft[(size_t)y * width + x] = kInvalidDisparity;

        continue;
      }

      if (std::abs(dL - dR) > maxDiff) {
        dispLeft[(size_t)y * width + x] = kInvalidDisparity;
      }
    }
  }
}

// ─── Stage 6: speckle filtering ──────────────────────────────────────────────
void SgbmCpu::speckleFilter(std::vector<float>& disp, int width, int height) const {
  if (params_.speckleWindowSize <= 0) return;

  const int maxDiff = params_.speckleRange;
  const int winSize = params_.speckleWindowSize;

  // Simple flood-fill-based speckle removal (connected components with
  // intensity difference threshold). Not as fast as OpenCV's, but correct.
  std::vector<uint8_t> visited((size_t)width * height, 0);
  std::vector<std::pair<int, int>> stack;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = (size_t)y * width + x;
      if (visited[idx]) continue;
      if (disp[idx] < 0) {
        visited[idx] = 1;
        continue;
      }

      stack.clear();
      stack.push_back({x, y});
      visited[idx] = 1;
      std::vector<std::pair<int, int>> component;

      while (!stack.empty()) {
        auto [cx, cy] = stack.back();
        stack.pop_back();
        component.push_back({cx, cy});
        const float ref = disp[(size_t)cy * width + cx];
        const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (auto& n : nb) {
          int nx = cx + n[0], ny = cy + n[1];
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          size_t nIdx = (size_t)ny * width + nx;
          if (visited[nIdx]) continue;
          float v = disp[nIdx];
          if (v < 0) {
            visited[nIdx] = 1;
            continue;
          }
          if (std::abs(v - ref) <= maxDiff) {
            visited[nIdx] = 1;
            stack.push_back({nx, ny});
          }
        }
      }

      if ((int)component.size() < winSize) {
        for (auto& [cx, cy] : component) disp[(size_t)cy * width + cx] = kInvalidDisparity;
      }
    }
  }
}

// ─── Main entry point ────────────────────────────────────────────────────────
StereoOutputs SgbmCpu::compute(const std::vector<uint8_t>& left, const std::vector<uint8_t>& right, int width, int height) const {
  validate(width, height, left.size(), right.size());

  std::vector<int16_t> lf, rf;
  preFilter(left, lf, width, height);
  preFilter(right, rf, width, height);

  StereoOutputs outputs;
  outputs.preFilterL = lf;
  outputs.preFilterR = rf;

  // left-to-right
  computeCost(lf, rf, width, height, outputs.costL);
  aggregate(outputs.costL, width, height, outputs.aggL);
  selectDisparity(outputs.aggL, outputs.costL, width, height, outputs.dispLRaw);
  // right-to-left
  computeCost(rf, lf, width, height, outputs.costR);
  aggregate(outputs.costR, width, height, outputs.aggR);
  selectDisparity(outputs.aggR, outputs.costR, width, height, outputs.dispRRaw);

  // Working copies
  outputs.dispL = outputs.dispLRaw;
  outputs.dispR = outputs.dispRRaw;

  lrCheck(outputs.dispR, width, height, outputs.dispL);
  speckleFilter(outputs.dispL, width, height);
  speckleFilter(outputs.dispR, width, height);
  return outputs;
}

}  // namespace sgbm