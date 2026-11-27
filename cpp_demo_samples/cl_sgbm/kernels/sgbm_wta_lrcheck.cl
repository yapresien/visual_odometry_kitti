#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// ─── Kernel 4A (Subgroup-Accelerated WTA + Atomic Right Scatter) ─────────────
__attribute__((reqd_work_group_size(SGM_WG_SIZE, WTA_ROWS_PER_WG, 1)))
__kernel void k_wta_subpixel_and_right_scatter(
    __global const ushort* restrict aggIn,
    __global const short* restrict costIn,
    __global float* restrict dispLOut,
    __global volatile uint* restrict rightMinPacked,
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

  // Pack (aggCost << 8) | dIdx so sub_group_reduce_min finds min cost AND breaks ties toward smaller dIdx!
  int localBestPacked = INT_MAX;

  for (int k = 0; k < D_PER_LANE; ++k) {
    const int dIdx = k * SGM_WG_SIZE + lane;
    if (dIdx < numDisparities) {
      const short c = costIn[base + (size_t)dIdx];
      const ushort a = aggIn[base + (size_t)dIdx];
      sharedCost[sub][dIdx] = c;
      sharedAgg[sub][dIdx] = a;

      if (c != INVALID_COST) {
        const int packed = (((int)a) << 8) | dIdx;
        localBestPacked = min(localBestPacked, packed);
      }
    }
  }

  // 1 hardware Wave32 instruction replaces 5 barriers + 2 LDS arrays!
  const int globalBestPacked = sub_group_reduce_min(localBestPacked);
  if (globalBestPacked == INT_MAX) {
    if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
    return;
  }

  const int bestC = globalBestPacked >> 8;
  const int bestD = globalBestPacked & 0xFF;

  // Ensure sharedAgg/sharedCost writes from all lanes in this subgroup are visible to lane 0
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (uniquenessRatio > 0) {
    int localMinOther = INT_MAX;
    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities && abs(d - bestD) > 1 && sharedCost[sub][d] != INVALID_COST) {
        localMinOther = min(localMinOther, (int)sharedAgg[sub][d]);
      }
    }

    // 1 hardware Wave32 instruction replaces another 5 barriers!
    const int minOther = sub_group_reduce_min(localMinOther);
    if (minOther != INT_MAX) {
      if ((long)bestC * (long)(100 + uniquenessRatio) > (long)minOther * 100L) {
        if (lane == 0) dispLOut[(size_t)y * (size_t)width + (size_t)x] = invalidDisparity;
        return;
      }
    }
  }

  if (lane == 0) {
    const int dStart = max(0, bestD - 1);
    const int dEnd = min(numDisparities - 1, bestD + 1);
    for (int dCand = dStart; dCand <= dEnd; ++dCand) {
      if (sharedCost[sub][dCand] != INVALID_COST) {
        const int xr = x - (minDisparity + dCand);
        if (xr >= 0 && xr < width) {
          const uint packed = (((uint)sharedAgg[sub][dCand]) << 16) | (uint)dCand;
          atomic_min(&rightMinPacked[(size_t)y * (size_t)width + (size_t)xr], packed);
        }
      }
    }

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

// ─── Kernel 4B: Fast LR Consistency Check from Packed Right Minima (~0.1 ms) ───
__attribute__((reqd_work_group_size(PF_WG_W, PF_WG_H, 1)))
__kernel void k_lr_check_packed(
    __global const uint* restrict rightMinPacked,
    __global float* restrict dispLInOut,
    const int width,
    const int height,
    const int minDisparity,
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

  const uint packedR = rightMinPacked[(size_t)y * (size_t)width + (size_t)xr];
  if (packedR == 0xFFFFFFFFu) {
    dispLInOut[idxL] = invalidDisparity;
    return;
  }

  const int dRInt = minDisparity + (int)(packedR & 0xFFFFu);
  if (abs(dLInt - dRInt) > disp12MaxDiff) {
    dispLInOut[idxL] = invalidDisparity;
  }
}
