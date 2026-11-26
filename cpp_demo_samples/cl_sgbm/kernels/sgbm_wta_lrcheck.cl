// ─── Kernel 4A: Left WTA + Non-Adjacent Uniqueness + Subpixel Parabola ───────
// 1 work-group of SGM_WG_SIZE lanes per (x, y) pixel -> 100% coalesced reads of aggIn!
__attribute__((reqd_work_group_size(SGM_WG_SIZE, 1, 1)))
__kernel void k_wta_subpixel(
    __global const int* restrict aggIn,
    __global const short* restrict costIn,
    __global float* restrict dispLOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int uniquenessRatio,
    const float invalidDisparity) {
  const int lane = get_local_id(0);
  const int x = get_group_id(0);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const size_t base = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  __local int sharedAgg[NUM_DISPARITIES];
  __local short sharedCost[NUM_DISPARITIES];
  __local int minCostBuf[SGM_WG_SIZE];
  __local int minDispBuf[SGM_WG_SIZE];

  int localBestC = INT_MAX;
  int localBestD = -1;

  // 1. Coalesced load of aggIn & costIn into LDS + lane-local minimum search
  for (int k = 0; k < D_PER_LANE; ++k) {
    const int d = k * SGM_WG_SIZE + lane;
    if (d < numDisparities) {
      const short c = costIn[base + (size_t)d];
      const int a = aggIn[base + (size_t)d];
      sharedCost[d] = c;
      sharedAgg[d] = a;
      if (c != INVALID_COST && a < localBestC) {
        localBestC = a;
        localBestD = d;
      }
    }
  }

  minCostBuf[lane] = localBestC;
  minDispBuf[lane] = localBestD;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Parallel reduction for global bestD (tie-breaking toward smaller d, matching CPU loop)
  for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
    if (lane < stride) {
      const int cOther = minCostBuf[lane + stride];
      const int dOther = minDispBuf[lane + stride];
      const int cCurr = minCostBuf[lane];
      const int dCurr = minDispBuf[lane];
      if (cOther < cCurr || (cOther == cCurr && dOther >= 0 && (dCurr < 0 || dOther < dCurr))) {
        minCostBuf[lane] = cOther;
        minDispBuf[lane] = dOther;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const int bestC = *minCostBuf;
  const int bestD = *minDispBuf;

  if (bestD < 0) {
    if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
    return;
  }

  // 2. Non-adjacent uniqueness check (|d - bestD| > 1)
  if (uniquenessRatio > 0) {
    int localMinOther = INT_MAX;
    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities && abs(d - bestD) > 1 && sharedCost[d] != INVALID_COST) {
        localMinOther = min(localMinOther, sharedAgg[d]);
      }
    }

    minCostBuf[lane] = localMinOther;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) {
        minCostBuf[lane] = min(minCostBuf[lane], minCostBuf[lane + stride]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int minOther = *minCostBuf;
    if (minOther != INT_MAX) {
      if ((long)bestC * (long)(100 + uniquenessRatio) > (long)minOther * 100L) {
        if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
        return;
      }
    }
  }

  // 3. Parabolic sub-pixel interpolation (lane 0 writes final float disparity)
  if (lane == 0) {
    float subpixelOffset = 0.0f;
    if (bestD > 0 && bestD < numDisparities - 1 &&
        sharedCost[bestD - 1] != INVALID_COST &&
        sharedCost[bestD + 1] != INVALID_COST) {
      const int cPrev = sharedAgg[bestD - 1];
      const int cCurr = sharedAgg[bestD];
      const int cNext = sharedAgg[bestD + 1];
      const int denom = 2 * (cPrev - 2 * cCurr + cNext);
      if (denom > 0) {
        subpixelOffset = clamp((float)(cPrev - cNext) / (float)denom, -0.5f, 0.5f);
      }
    }
    dispLOut[(size_t)y * (size_t)width + (size_t)x] = (float)(minDisparity + bestD) + subpixelOffset;
  }
}

// ─── Kernel 4B: Shared-Volume Right WTA + Integer Left-Right Consistency Check ─
// Computes dispR from diagonal slices of aggIn, then validates dispL against dispR.
__attribute__((reqd_work_group_size(SGM_WG_SIZE, 1, 1)))
__kernel void k_right_wta_and_lr_check(
    __global const int* restrict aggIn,
    __global const short* restrict costIn,
    __global float* restrict dispLInOut,
    __global float* restrict dispROut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int disp12MaxDiff,
    const float invalidDisparity) {
  const int lane = get_local_id(0);
  const int xr = get_group_id(0);
  const int y = get_group_id(1);

  if (xr >= width || y >= height) return;

  __local int minCostBuf[SGM_WG_SIZE];
  __local int minDispBuf[SGM_WG_SIZE];

  int localBestC = INT_MAX;
  int localBestD = -1;

  for (int k = 0; k < D_PER_LANE; ++k) {
    const int dIdx = k * SGM_WG_SIZE + lane;
    if (dIdx < numDisparities) {
      const int d = minDisparity + dIdx;
      const int xl = xr + d;
      if (xl >= 0 && xl < width) {
        const size_t idx = ((size_t)y * (size_t)width + (size_t)xl) * (size_t)numDisparities + (size_t)dIdx;
        if (costIn[idx] != INVALID_COST) {
          const int a = aggIn[idx];
          if (a < localBestC) {
            localBestC = a;
            localBestD = dIdx;
          }
        }
      }
    }
  }

  minCostBuf[lane] = localBestC;
  minDispBuf[lane] = localBestD;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
    if (lane < stride) {
      const int cOther = minCostBuf[lane + stride];
      const int dOther = minDispBuf[lane + stride];
      const int cCurr = minCostBuf[lane];
      const int dCurr = minDispBuf[lane];
      if (cOther < cCurr || (cOther == cCurr && dOther >= 0 && (dCurr < 0 || dOther < dCurr))) {
        minCostBuf[lane] = cOther;
        minDispBuf[lane] = dOther;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lane == 0) {
    const int bestD = *minDispBuf;
    dispROut[(size_t)y * (size_t)width + (size_t)xr] =
        (bestD >= 0) ? (float)(minDisparity + bestD) : invalidDisparity;
  }
}

// ─── Kernel 4C: Left-Right Consistency Filter (Matches SgbmCpu::lrCheck) ──────
__attribute__((reqd_work_group_size(PF_WG_W, PF_WG_H, 1)))
__kernel void k_lr_check(
    __global const float* restrict dispR,
    __global float* restrict dispLInOut,
    const int width,
    const int height,
    const int disp12MaxDiff,
    const float invalidDisparity) {
  if (disp12MaxDiff < 0) return;

  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= width || y >= height) return;

  const size_t idxL = (size_t)y * (size_t)width + (size_t)x;
  const float dL = dispLInOut[idxL];
  if (dL == invalidDisparity) return;

  const int dLInt = (int)round(dL);
  const int xr = x - dLInt;
  if (xr < 0 || xr >= width) {
    dispLInOut[idxL] = invalidDisparity;
    return;
  }

  const float dR = dispR[(size_t)y * (size_t)width + (size_t)xr];
  if (dR == invalidDisparity || fabs((float)dLInt - dR) > (float)disp12MaxDiff) {
    dispLInOut[idxL] = invalidDisparity;
  }
}
