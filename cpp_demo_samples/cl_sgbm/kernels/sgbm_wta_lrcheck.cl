
// ─── Kernel 4A (256-Thread WG, ushort aggIn): Left WTA + Uniqueness + Subpixel ─
__attribute__((reqd_work_group_size(SGM_WG_SIZE, WTA_ROWS_PER_WG, 1)))
__kernel void k_wta_subpixel(
    __global const ushort* restrict aggIn,
    __global const short* restrict costIn,
    __global float* restrict dispLOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int uniquenessRatio,
    const float invalidDisparity) {
  const int lane = get_local_id(0);
  const int sub = get_local_id(1);
  const int x = get_group_id(0) * WTA_ROWS_PER_WG + sub;
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const size_t base = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  __local ushort sharedAgg[WTA_ROWS_PER_WG][NUM_DISPARITIES];
  __local short sharedCost[WTA_ROWS_PER_WG][NUM_DISPARITIES];
  __local int minCostBuf[WTA_ROWS_PER_WG][SGM_WG_SIZE];
  __local int minDispBuf[WTA_ROWS_PER_WG][SGM_WG_SIZE];

  int localBestC = INT_MAX;
  int localBestD = -1;

  for (int k = 0; k < D_PER_LANE; ++k) {
    const int d = k * SGM_WG_SIZE + lane;
    if (d < numDisparities) {
      const short c = costIn[base + (size_t)d];
      const ushort a = aggIn[base + (size_t)d];
      sharedCost[sub][d] = c;
      sharedAgg[sub][d] = a;
      if (c != INVALID_COST && (int)a < localBestC) {
        localBestC = (int)a;
        localBestD = d;
      }
    }
  }

  minCostBuf[sub][lane] = localBestC;
  minDispBuf[sub][lane] = localBestD;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
    if (lane < stride) {
      const int cOther = minCostBuf[sub][lane + stride];
      const int dOther = minDispBuf[sub][lane + stride];
      const int cCurr = minCostBuf[sub][lane];
      const int dCurr = minDispBuf[sub][lane];
      if (cOther < cCurr || (cOther == cCurr && dOther >= 0 && (dCurr < 0 || dOther < dCurr))) {
        minCostBuf[sub][lane] = cOther;
        minDispBuf[sub][lane] = dOther;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const int bestC = *minCostBuf[sub];
  const int bestD = *minDispBuf[sub];

  if (bestD < 0) {
    if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
    return;
  }

  if (uniquenessRatio > 0) {
    int localMinOther = INT_MAX;
    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities && abs(d - bestD) > 1 && sharedCost[sub][d] != INVALID_COST) {
        localMinOther = min(localMinOther, (int)sharedAgg[sub][d]);
      }
    }

    minCostBuf[sub][lane] = localMinOther;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) {
        minCostBuf[sub][lane] = min(minCostBuf[sub][lane], minCostBuf[sub][lane + stride]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int minOther = *minCostBuf[sub];
    if (minOther != INT_MAX) {
      if ((long)bestC * (long)(100 + uniquenessRatio) > (long)minOther * 100L) {
        if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
        return;
      }
    }
  }

  if (lane == 0) {
    float subpixelOffset = 0.0f;
    if (bestD > 0 && bestD < numDisparities - 1 &&
        sharedCost[sub][bestD - 1] != INVALID_COST &&
        sharedCost[sub][bestD + 1] != INVALID_COST) {
      const int cPrev = (int)sharedAgg[sub][bestD - 1];
      const int cCurr = (int)sharedAgg[sub][bestD];
      const int cNext = (int)sharedAgg[sub][bestD + 1];
      const int denom = 2 * (cPrev - 2 * cCurr + cNext);
      if (denom > 0) {
        subpixelOffset = clamp((float)(cPrev - cNext) / (float)denom, -0.5f, 0.5f);
      }
    }
    dispLOut[(size_t)y * (size_t)width + (size_t)x] = (float)(minDisparity + bestD) + subpixelOffset;
  }
}

// ─── Kernel 4B (256-Thread WG, ushort aggIn): Shared-Volume Right WTA ─────────
__attribute__((reqd_work_group_size(SGM_WG_SIZE, WTA_ROWS_PER_WG, 1)))
__kernel void k_right_wta_and_lr_check(
    __global const ushort* restrict aggIn,
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
  const int sub = get_local_id(1);
  const int xr = get_group_id(0) * WTA_ROWS_PER_WG + sub;
  const int y = get_group_id(1);

  if (xr >= width || y >= height) return;

  __local int minCostBuf[WTA_ROWS_PER_WG][SGM_WG_SIZE];
  __local int minDispBuf[WTA_ROWS_PER_WG][SGM_WG_SIZE];

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
          const int a = (int)aggIn[idx];
          if (a < localBestC) {
            localBestC = a;
            localBestD = dIdx;
          }
        }
      }
    }
  }

  minCostBuf[sub][lane] = localBestC;
  minDispBuf[sub][lane] = localBestD;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
    if (lane < stride) {
      const int cOther = minCostBuf[sub][lane + stride];
      const int dOther = minDispBuf[sub][lane + stride];
      const int cCurr = minCostBuf[sub][lane];
      const int dCurr = minDispBuf[sub][lane];
      if (cOther < cCurr || (cOther == cCurr && dOther >= 0 && (dCurr < 0 || dOther < dCurr))) {
        minCostBuf[sub][lane] = cOther;
        minDispBuf[sub][lane] = dOther;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lane == 0) {
    const int bestD = *minDispBuf[sub];
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
