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

// ─── Kernel 3 (4-Pass Bidirectional, 256-Thread WG): SGM Axis Aggregation ────
// Packs SGM_ROWS_PER_WG (8) scanlines per 256-thread WG (100% CU occupancy on gfx1102)
// and sweeps BOTH forward (+dx, +dy) and backward (-dx, -dy) in a single pass!
__attribute__((reqd_work_group_size(SGM_WG_SIZE, WTA_ROWS_PER_WG, 1)))
__kernel void k_sgm_bidi_axis(
    __global const short* restrict costIn,
    __global ushort* restrict aggOut,
    const int width,
    const int height,
    const int numDisparities,
    const int P1,
    const int P2,
    const int dx,
    const int dy,
    const int isFirstAxis,
    const int doBackwardPass) {
  const int lane = get_local_id(0);
  const int sub = get_local_id(1);
  const int scanId = get_group_id(0) * WTA_ROWS_PER_WG + sub;

  int x0 = 0, y0 = 0, steps = 0;
  if (dy == 0) {
    if (scanId >= height) return;
    y0 = scanId; x0 = 0; steps = width;
  } else if (dx == 0) {
    if (scanId >= width) return;
    x0 = scanId; y0 = 0; steps = height;
  } else if (dx > 0 && dy > 0) {
    if (scanId >= width + height - 1) return;
    x0 = (scanId < height) ? 0 : (scanId - height + 1);
    y0 = (scanId < height) ? (height - 1 - scanId) : 0;
    steps = min(width - x0, height - y0);
  } else {
    // dx < 0 && dy > 0 (TR -> BL and reverse BL -> TR)
    if (scanId >= width + height - 1) return;
    x0 = (scanId < width) ? scanId : (width - 1);
    y0 = (scanId < width) ? 0 : (scanId - width + 1);
    steps = min(x0 + 1, height - y0);
  }

  __local int sharedLPrev[WTA_ROWS_PER_WG][NUM_DISPARITIES + 2];
  __local int sharedMin[WTA_ROWS_PER_WG][SGM_WG_SIZE];

  if (lane == 0) {
    *sharedLPrev[sub] = MAX_PATH_INT;
    sharedLPrev[sub][numDisparities + 1] = MAX_PATH_INT;
  }

  // ── Forward Sweep: (x0, y0) + s * (dx, dy) ──
  int lPrev[D_PER_LANE];
  for (int k = 0; k < D_PER_LANE; ++k) lPrev[k] = MAX_PATH_INT;
  bool prevValid = false;

  int x = x0, y = y0;
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
        sharedLPrev[sub][d + 1] = lPrev[k];
        laneMinPrev = min(laneMinPrev, lPrev[k]);
      }
    }

    sharedMin[sub][lane] = laneMinPrev;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) sharedMin[sub][lane] = min(sharedMin[sub][lane], sharedMin[sub][lane + stride]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int minPrev = *sharedMin[sub];

    sharedMin[sub][lane] = hasAnyValid ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) sharedMin[sub][lane] |= sharedMin[sub][lane + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const bool currHasValid = (*sharedMin[sub] != 0);

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d >= numDisparities) break;
      const int c = cVal[k];
      int lCurr = MAX_PATH_INT;
      if (c != INVALID_COST) {
        if (!prevValid || minPrev >= MAX_PATH_INT) {
          lCurr = c;
        } else {
          const int best = min(
              min(sharedLPrev[sub][d + 1], sharedLPrev[sub][d] + P1),
              min(sharedLPrev[sub][d + 2] + P1, minPrev + P2));
          lCurr = c + best - minPrev;
        }
      }
      lPrev[k] = lCurr;
      const int addVal = (c != INVALID_COST) ? lCurr : 0;
      if (isFirstAxis) {
        aggOut[baseIdx + (size_t)d] = convert_ushort_sat(addVal);
      } else {
        aggOut[baseIdx + (size_t)d] = convert_ushort_sat((int)aggOut[baseIdx + (size_t)d] + addVal);
      }
    }
    prevValid = currHasValid;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (!doBackwardPass) return;

  // ── Backward Sweep: reverse from end pixel back to (x0, y0) ──
  for (int k = 0; k < D_PER_LANE; ++k) lPrev[k] = MAX_PATH_INT;
  prevValid = false;

  x -= dx;
  y -= dy;
  for (int s = 0; s < steps; ++s, x -= dx, y -= dy) {
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
        sharedLPrev[sub][d + 1] = lPrev[k];
        laneMinPrev = min(laneMinPrev, lPrev[k]);
      }
    }

    sharedMin[sub][lane] = laneMinPrev;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) sharedMin[sub][lane] = min(sharedMin[sub][lane], sharedMin[sub][lane + stride]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int minPrev = *sharedMin[sub];

    sharedMin[sub][lane] = hasAnyValid ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = (SGM_WG_SIZE >> 1); stride > 0; stride >>= 1) {
      if (lane < stride) sharedMin[sub][lane] |= sharedMin[sub][lane + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const bool currHasValid = (*sharedMin[sub] != 0);

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d >= numDisparities) break;
      const int c = cVal[k];
      int lCurr = MAX_PATH_INT;
      if (c != INVALID_COST) {
        if (!prevValid || minPrev >= MAX_PATH_INT) {
          lCurr = c;
        } else {
          const int best = min(
              min(sharedLPrev[sub][d + 1], sharedLPrev[sub][d] + P1),
              min(sharedLPrev[sub][d + 2] + P1, minPrev + P2));
          lCurr = c + best - minPrev;
        }
      }
      lPrev[k] = lCurr;
      if (c != INVALID_COST) {
        aggOut[baseIdx + (size_t)d] = convert_ushort_sat((int)aggOut[baseIdx + (size_t)d] + lCurr);
      }
    }
    prevValid = currHasValid;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}


// ─── Helper A: Tiled Transpose [y][x][d] -> [y][d][x] (~0.5 ms) ───────────────
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void k_transpose_xd_to_dx(
    __global const short* restrict srcXD,
    __global short* restrict dstDX,
    const int width,
    const int height,
    const int numDisparities) {
  const int lx = get_local_id(0);  // 0..31
  const int ly = get_local_id(1);  // 0..7
  const int xTile = get_group_id(0) * 32;
  const int dTile = get_group_id(1) * 32;
  const int y = get_group_id(2);

  if (y >= height) return;

  __local short tile[32][33];  // +1 padding eliminates LDS bank conflicts

  const size_t yOff = (size_t)y * (size_t)width * (size_t)numDisparities;

  // Load 32x32 block from [y][x][d] (lx = d, ly+row = x) -> 100% coalesced!
  for (int row = ly; row < 32; row += 8) {
    const int gx = xTile + row;
    const int gd = dTile + lx;
    if (gx < width && gd < numDisparities) {
      tile[row][lx] = srcXD[yOff + (size_t)gx * (size_t)numDisparities + (size_t)gd];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store 32x32 block to [y][d][x] (lx = x, ly+row = d) -> 100% coalesced!
  for (int row = ly; row < 32; row += 8) {
    const int gd = dTile + row;
    const int gx = xTile + lx;
    if (gd < numDisparities && gx < width) {
      dstDX[yOff + (size_t)gd * (size_t)width + (size_t)gx] = tile[lx][row];
    }
  }
}

// ─── Helper B: Tiled Transpose-Add [y][d][x] -> aggOut[y][x][d] (~0.6 ms) ────
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void k_add_dx_to_xd(
    __global const ushort* restrict srcDX,
    __global ushort* restrict aggOutXD,
    const int width,
    const int height,
    const int numDisparities) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int xTile = get_group_id(0) * 32;
  const int dTile = get_group_id(1) * 32;
  const int y = get_group_id(2);

  if (y >= height) return;

  __local ushort tile[32][33];
  const size_t yOff = (size_t)y * (size_t)width * (size_t)numDisparities;

  // Coalesced load from [y][d][x] (lx = x, ly+row = d)
  for (int row = ly; row < 32; row += 8) {
    const int gd = dTile + row;
    const int gx = xTile + lx;
    if (gd < numDisparities && gx < width) {
      tile[row][lx] = srcDX[yOff + (size_t)gd * (size_t)width + (size_t)gx];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Coalesced read-modify-write into aggOut[y][x][d] (lx = d, ly+row = x)
  for (int row = ly; row < 32; row += 8) {
    const int gx = xTile + row;
    const int gd = dTile + lx;
    if (gx < width && gd < numDisparities) {
      const size_t idx = yOff + (size_t)gx * (size_t)numDisparities + (size_t)gd;
      aggOutXD[idx] = convert_ushort_sat((int)aggOutXD[idx] + (int)tile[lx][row]);
    }
  }
}

// ─── Kernel 3B (Zero-VGPR-Spill LDS State): Column-Coalesced Vertical/Diag SGM ─
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void k_sgm_col_vert_diag(
    __global const short* restrict costDX,
    __global ushort* restrict aggDX,
    const int width,
    const int height,
    const int numDisparities,
    const int P1,
    const int P2,
    const int dx,
    const int isFirstDX) {
  const int lane = get_local_id(0);  // 0..31 -> column offset
  const int tileIdx = get_group_id(0);

  const int xStartBase = (dx == 0) ? (tileIdx * 32) : (-height + tileIdx * 32);
  const int x0 = xStartBase + lane;

  // Store lPrev in LDS (16.5 KB) instead of 128 VGPRs -> ZERO register spills to DRAM!
  __local int lPrevLDS[32][NUM_DISPARITIES + 1];

  for (int d = 0; d < numDisparities; ++d) {
    lPrevLDS[lane][d] = MAX_PATH_INT;
  }
  bool prevValid = false;

  const size_t planeStride = (size_t)numDisparities * (size_t)width;

  for (int y = 0; y < height; ++y) {
    const int x = x0 + y * dx;
    if (x < 0 || x >= width) {
      for (int d = 0; d < numDisparities; ++d) lPrevLDS[lane][d] = MAX_PATH_INT;
      prevValid = false;
      continue;
    }

    const size_t yBase = (size_t)y * planeStride + (size_t)x;

    int minPrev = MAX_PATH_INT;
    if (prevValid) {
      for (int d = 0; d < numDisparities; ++d) {
        minPrev = min(minPrev, lPrevLDS[lane][d]);
      }
    }

    int prevDVal = MAX_PATH_INT;
    bool currHasValid = false;

    for (int d = 0; d < numDisparities; ++d) {
      const size_t idx = yBase + (size_t)d * (size_t)width;
      const short cShort = costDX[idx];
      const int curDVal = lPrevLDS[lane][d];
      const int nextDVal = (d + 1 < numDisparities) ? lPrevLDS[lane][d + 1] : MAX_PATH_INT;

      int lCurr = MAX_PATH_INT;
      if (cShort != INVALID_COST) {
        currHasValid = true;
        const int c = (int)cShort;
        if (!prevValid || minPrev >= MAX_PATH_INT) {
          lCurr = c;
        } else {
          const int best = min(
              min(curDVal, prevDVal + P1),
              min(nextDVal + P1, minPrev + P2));
          lCurr = c + best - minPrev;
        }
      }

      prevDVal = curDVal;
      lPrevLDS[lane][d] = lCurr;

      const int addVal = (cShort != INVALID_COST) ? lCurr : 0;
      if (isFirstDX) {
        aggDX[idx] = convert_ushort_sat(addVal);
      } else {
        aggDX[idx] = convert_ushort_sat((int)aggDX[idx] + addVal);
      }
    }

    prevValid = currHasValid;
  }
}

