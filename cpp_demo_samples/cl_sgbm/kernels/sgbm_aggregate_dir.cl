// ─── Kernel 3 (16-bit Saturated aggOut): Scanline-Parallel SGM Aggregation ───
__attribute__((reqd_work_group_size(SGM_WG_SIZE, 1, 1)))
__kernel void k_sgm_aggregate_dir(
    __global const short* restrict costIn,
    __global ushort* restrict aggOut,
    const int width,
    const int height,
    const int numDisparities,
    const int P1,
    const int P2,
    const int dx,
    const int dy,
    const int isFirstPass) {
  const int lane = get_local_id(0);
  const int scanId = get_group_id(0);

  int x = 0;
  int y = 0;
  int steps = 0;

  if (dy == 0) {
    if (scanId >= height) return;
    y = scanId;
    x = (dx > 0) ? 0 : (width - 1);
    steps = width;
  } else if (dx == 0) {
    if (scanId >= width) return;
    x = scanId;
    y = (dy > 0) ? 0 : (height - 1);
    steps = height;
  } else {
    if (scanId >= width + height - 1) return;
    if (dx > 0 && dy > 0) {
      x = (scanId < height) ? 0 : (scanId - height + 1);
      y = (scanId < height) ? (height - 1 - scanId) : 0;
      steps = min(width - x, height - y);
    } else if (dx < 0 && dy > 0) {
      x = (scanId < width) ? scanId : (width - 1);
      y = (scanId < width) ? 0 : (scanId - width + 1);
      steps = min(x + 1, height - y);
    } else if (dx > 0 && dy < 0) {
      x = (scanId < width) ? scanId : 0;
      y = (scanId < width) ? (height - 1) : (height - 1 - (scanId - width + 1));
      steps = min(width - x, y + 1);
    } else {
      x = (scanId < height) ? (width - 1) : (width - 1 - (scanId - height + 1));
      y = (scanId < height) ? scanId : (height - 1);
      steps = min(x + 1, y + 1);
    }
  }

  __local int sharedLPrev[NUM_DISPARITIES + 2];
  __local int sharedMin[SGM_WG_SIZE];

  if (lane == 0) {
    *sharedLPrev = MAX_PATH_INT;
    sharedLPrev[numDisparities + 1] = MAX_PATH_INT;
  }

  int lPrev[D_PER_LANE];
  for (int k = 0; k < D_PER_LANE; ++k) {
    lPrev[k] = MAX_PATH_INT;
  }

  bool prevValid = false;

  for (int s = 0; s < steps; ++s, x += dx, y += dy) {
    const size_t baseIdx = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

    int cVal[D_PER_LANE];
    int laneMinPrev = MAX_PATH_INT;
    bool hasAnyValid = false;

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities) {
        const short c = costIn[baseIdx + (size_t)d];
        cVal[k] = (int)c;
        if (c != INVALID_COST) hasAnyValid = true;
        sharedLPrev[d + 1] = lPrev[k];
        laneMinPrev = min(laneMinPrev, lPrev[k]);
      }
    }

    sharedMin[lane] = laneMinPrev;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) {
        sharedMin[lane] = min(sharedMin[lane], sharedMin[lane + stride]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int minPrev = *sharedMin;

    sharedMin[lane] = hasAnyValid ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) {
        sharedMin[lane] |= sharedMin[lane + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const bool currHasValid = (*sharedMin != 0);

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d >= numDisparities) break;

      const int c = cVal[k];
      int lCurr = MAX_PATH_INT;

      if (c != INVALID_COST) {
        if (!prevValid || minPrev >= MAX_PATH_INT) {
          lCurr = c;
        } else {
          const int l0 = sharedLPrev[d + 1];
          const int l1 = sharedLPrev[d] + P1;
          const int l2 = sharedLPrev[d + 2] + P1;
          const int l3 = minPrev + P2;
          const int best = min(min(l0, l1), min(l2, l3));
          lCurr = c + best - minPrev;
        }
      }

      lPrev[k] = lCurr;

      const int addVal = (c != INVALID_COST) ? lCurr : 0;
      if (isFirstPass) {
        aggOut[baseIdx + (size_t)d] = convert_ushort_sat(addVal);
      } else {
        const int prevSum = (int)aggOut[baseIdx + (size_t)d];
        aggOut[baseIdx + (size_t)d] = convert_ushort_sat(prevSum + addVal);
      }
    }

    prevValid = currHasValid;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
