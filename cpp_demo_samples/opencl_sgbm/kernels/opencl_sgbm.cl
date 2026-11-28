// sgbm_prefilter_gfx1102.cl
// Tuned for AMD Radeon 890M (gfx1102): Wave32, 256-thread WG (32x8), 32-bank LDS, native short2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// sgbm_kernels.cl — Hardware-Agnostic OpenCL 2.0 Kernels (Parameterized via Host JIT -D flags)
#define APRON 2
#define MAX_PATH_INT (1 << 28)

// ─── Kernel 1: Tiled Sobel-X Prefilter + Birchfield-Tomasi [min, max] Bounds ─
__attribute__((reqd_work_group_size(PF_WG_W, PF_WG_H, 1)))
__kernel void k_prefilter_and_bt_bounds(
    __global const uchar* restrict rawImg,
    __global short* restrict sobelOut,
    __global short2* restrict sobelBoundsOut,
    __global short2* restrict rawBoundsOut,
    const int width,
    const int height,
    const int preFilterCap) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int wgThreads = PF_WG_W * PF_WG_H;
  const int tid = ly * PF_WG_W + lx;

  const int groupX0 = get_group_id(0) * PF_WG_W;
  const int groupY0 = get_group_id(1) * PF_WG_H;
  const int gx = groupX0 + lx;
  const int gy = groupY0 + ly;

  __local uchar tileRaw[PF_TILE_H][PF_LDS_STRIDE];
  __local short tileSobel[PF_TILE_H][PF_LDS_STRIDE];

  const int totalTile = PF_TILE_W * PF_TILE_H;
  for (int i = tid; i < totalTile; i += wgThreads) {
    const int ty = i / PF_TILE_W;
    const int tx = i % PF_TILE_W;
    const int sx = groupX0 + tx - APRON;
    int sy = groupY0 + ty - APRON;

    if (sy < 0) sy = min(1, height - 1);
    if (sy >= height) sy = max(0, height - 2);
    const int clampedX = clamp(sx, 0, width - 1);

    tileRaw[ty][tx] = rawImg[sy * width + clampedX];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const int sobelW = PF_TILE_W - 2;
  const int sobelH = PF_TILE_H - 2;
  const int totalSobel = sobelW * sobelH;
  for (int i = tid; i < totalSobel; i += wgThreads) {
    const int ty = (i / sobelW) + 1;
    const int tx = (i % sobelW) + 1;
    const int globalX = groupX0 + tx - APRON;

    if (globalX <= 0 || globalX >= width - 1 || preFilterCap <= 0) {
      tileSobel[ty][tx] = (preFilterCap <= 0) ? (short)tileRaw[ty][tx] : (short)0;
    } else {
      int s = ((int)tileRaw[ty - 1][tx + 1] - (int)tileRaw[ty - 1][tx - 1])
            + 2 * ((int)tileRaw[ty][tx + 1] - (int)tileRaw[ty][tx - 1])
            + ((int)tileRaw[ty + 1][tx + 1] - (int)tileRaw[ty + 1][tx - 1]);
      tileSobel[ty][tx] = (short)clamp(s, -preFilterCap, preFilterCap);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (gx >= width || gy >= height) return;

  const int idx = gy * width + gx;
  const int tx = lx + APRON;
  const int ty = ly + APRON;

  const short sVal = tileSobel[ty][tx];
  sobelOut[idx] = sVal;

  if (gx == 0 || gx == width - 1) {
    sobelBoundsOut[idx] = (short2)(sVal, sVal);
    const short rVal = (short)tileRaw[ty][tx];
    rawBoundsOut[idx] = (short2)(rVal, rVal);
  } else {
    const int sPrev = ((int)sVal + (int)tileSobel[ty][tx - 1]) >> 1;
    const int sNext = ((int)sVal + (int)tileSobel[ty][tx + 1]) >> 1;
    sobelBoundsOut[idx] = (short2)(
        (short)min((int)sVal, min(sPrev, sNext)),
        (short)max((int)sVal, max(sPrev, sNext)));

    const int rVal = (int)tileRaw[ty][tx];
    const int rPrev = (rVal + (int)tileRaw[ty][tx - 1]) >> 1;
    const int rNext = (rVal + (int)tileRaw[ty][tx + 1]) >> 1;
    rawBoundsOut[idx] = (short2)(
        (short)min(rVal, min(rPrev, rNext)),
        (short)max(rVal, max(rPrev, rNext)));
  }
}
// ─── Helper Kernel: Fast Compute Clear (Replaces SDMA clEnqueueFillBuffer) ────
__kernel void k_clear_uint(__global uint* restrict buf, const uint val, const uint count) {
  const uint idx = get_global_id(0);
  if (idx < count) buf[idx] = val;
}
// ─── Kernel 2A (ushort2 Vectorized): 1x1 Dual Birchfield-Tomasi Pixel Cost ────
// Processes 2 disparities per 32-bit dword store (128 bytes/Wave32, zero sub-word RMW)
__attribute__((reqd_work_group_size(COST_WG_W, COST_WG_H, 1)))
__kernel void k_bt_pixel_cost(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global ushort2* restrict pixelCostOut2,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities) {
  const int lane = get_local_id(0);
  const int x = get_group_id(0) * COST_WG_H + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const int numPairs = numDisparities >> 1;
  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numPairs;

  if (x < 1 || x >= width - 1) {
    const ushort2 inv2 = (ushort2)((ushort)INVALID_COST, (ushort)INVALID_COST);
    for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
      pixelCostOut2[baseOut + (size_t)pIdx] = inv2;
    }
    return;
  }

  const int idxL = y * width + x;
  const int sL = (int)sobelL[idxL];
  const short2 sBL = sobelBoundsL[idxL];
  const int rL = (int)rawL[idxL];
  const short2 rBL = rawBoundsL[idxL];

  for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
    const int d0 = minDisparity + (pIdx << 1);
    const int d1 = d0 + 1;
    const int xr0 = x - d0;
    const int xr1 = x - d1;

    ushort c0 = (ushort)INVALID_COST;
    if (xr0 >= 1 && xr0 < width - 1) {
      const int idxR0 = y * width + xr0;
      const int sR0 = (int)sobelR[idxR0];
      const short2 sBR0 = sobelBoundsR[idxR0];
      const int btSobel0 = min(
          max(0, max(sL - (int)sBR0.y, (int)sBR0.x - sL)),
          max(0, max(sR0 - (int)sBL.y, (int)sBL.x - sR0)));

      const int rR0 = (int)rawR[idxR0];
      const short2 rBR0 = rawBoundsR[idxR0];
      const int btRaw0 = min(
          max(0, max(rL - (int)rBR0.y, (int)rBR0.x - rL)),
          max(0, max(rR0 - (int)rBL.y, (int)rBL.x - rR0)));

      c0 = (ushort)(btSobel0 + btRaw0);
    }

    ushort c1 = (ushort)INVALID_COST;
    if (xr1 >= 1 && xr1 < width - 1) {
      const int idxR1 = y * width + xr1;
      const int sR1 = (int)sobelR[idxR1];
      const short2 sBR1 = sobelBoundsR[idxR1];
      const int btSobel1 = min(
          max(0, max(sL - (int)sBR1.y, (int)sBR1.x - sL)),
          max(0, max(sR1 - (int)sBL.y, (int)sBL.x - sR1)));

      const int rR1 = (int)rawR[idxR1];
      const short2 rBR1 = rawBoundsR[idxR1];
      const int btRaw1 = min(
          max(0, max(rL - (int)rBR1.y, (int)rBR1.x - rL)),
          max(0, max(rR1 - (int)rBL.y, (int)rBL.x - rR1)));

      c1 = (ushort)(btSobel1 + btRaw1);
    }

    pixelCostOut2[baseOut + (size_t)pIdx] = (ushort2)(c0, c1);
  }
}

// ─── Kernel 2B (LDS-Tiled ushort2 -> short2): BxB Box Sum ────────────────────
// Cooperatively caches (COST_WG_H + 2*MAX_B2) horizontal columns in LDS per row by,
// cutting global VRAM reads from 25 taps down to ~7.5 taps (3.3x bandwidth reduction).
__attribute__((reqd_work_group_size(COST_WG_W, COST_WG_H, 1)))
__kernel void k_box_filter_cost(
    __global const ushort2* restrict pixelCostIn2,
    __global short2* restrict costOut2,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);        // 0 .. 31 (disparity pairs)
  const int sub = get_local_id(1);         // 0 .. 7  (x offset within WG)
  const int xBase = get_group_id(0) * COST_WG_H;
  const int x = xBase + sub;
  const int y = get_group_id(1);

  if (y >= height) return;

  const int B2 = blockSize >> 1;
  const int tileCols = COST_WG_H + 2 * B2;  // 12 columns for WG_H=8, blockSize=5
  const int numPairs = numDisparities >> 1;
  const short2 inv2 = (short2)((short)INVALID_COST, (short)INVALID_COST);

  __local ushort2 tileH[COST_WG_H + 2 * MAX_B2][(NUM_DISPARITIES >> 1) + 1];

  const bool yValid = (y >= B2 && y < height - B2);
  if (!yValid) {
    if (x < width) {
      const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numPairs;
      for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
        costOut2[baseOut + (size_t)pIdx] = inv2;
      }
    }
    return;
  }

  int2 blockSum[D_PER_LANE];
  for (int i = 0; i < D_PER_LANE; ++i) blockSum[i] = (int2)(0, 0);

  for (int by = -B2; by <= B2; ++by) {
    const size_t rowOffset = (size_t)(y + by) * (size_t)width;

    // 8 sub-rows cooperatively load tileCols (12) columns into LDS
    for (int col = sub; col < tileCols; col += COST_WG_H) {
      const int gx = xBase - B2 + col;
      if (gx >= 1 && gx < width - 1) {
        const size_t srcBase = (rowOffset + (size_t)gx) * (size_t)numPairs;
        for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
          tileH[col][pIdx] = pixelCostIn2[srcBase + (size_t)pIdx];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Accumulate 5 horizontal taps from LDS (zero global memory reads!)
    if (x >= B2 + 1 && x < width - B2 - 1) {
      int iter = 0;
      for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W, ++iter) {
        int2 rowSum = (int2)(0, 0);
        for (int k = 0; k <= 2 * B2; ++k) {
          rowSum += convert_int2(tileH[sub + k][pIdx]);
        }
        blockSum[iter] += rowSum;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (x >= width) return;

  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numPairs;
  if (x < B2 + 1 || x >= width - B2 - 1) {
    for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
      costOut2[baseOut + (size_t)pIdx] = inv2;
    }
    return;
  }

  int iter = 0;
  for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W, ++iter) {
    const int d0 = minDisparity + (pIdx << 1);
    const int d1 = d0 + 1;
    const int xr0 = x - d0;
    const int xr1 = x - d1;

    const bool valid0 = (xr0 - B2 >= 1 && xr0 + B2 < width - 1);
    const bool valid1 = (xr1 - B2 >= 1 && xr1 + B2 < width - 1);

    const short out0 = valid0 ? (short)min(blockSum[iter].x, INVALID_COST - 1) : (short)INVALID_COST;
    const short out1 = valid1 ? (short)min(blockSum[iter].y, INVALID_COST - 1) : (short)INVALID_COST;
    costOut2[baseOut + (size_t)pIdx] = (short2)(out0, out1);
  }
}


// ─── Kernel 2 (Fused OpenCL 1.2): Dual Birchfield-Tomasi + BxB Box Sum in LDS ──
// Eliminates bufPixelCost_ (119.5 MB VRAM & 239 MB DRAM roundtrip) across all GPUs!
__attribute__((reqd_work_group_size(COST_WG_W, COST_WG_H, 1)))
__kernel void k_bt_box_fused(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global short2* restrict costOut2,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);        // 0 .. COST_WG_W-1 (disparity pairs)
  const int sub = get_local_id(1);         // 0 .. COST_WG_H-1 (x offset in WG)
  const int xBase = get_group_id(0) * COST_WG_H;
  const int x = xBase + sub;
  const int y = get_group_id(1);

  if (y >= height) return;

  const int B2 = blockSize >> 1;
  const int tileCols = COST_WG_H + 2 * B2;
  const int numPairs = numDisparities >> 1;
  const short2 inv2 = (short2)((short)INVALID_COST, (short)INVALID_COST);

  __local ushort2 tileH[COST_WG_H + 2 * MAX_B2][(NUM_DISPARITIES >> 1) + 1];

  if (y < B2 || y >= height - B2) {
    if (x < width) {
      const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numPairs;
      for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
        costOut2[baseOut + (size_t)pIdx] = inv2;
      }
    }
    return;
  }

  int2 blockSum[D_PER_LANE];
  for (int i = 0; i < D_PER_LANE; ++i) blockSum[i] = (int2)(0, 0);

  for (int by = -B2; by <= B2; ++by) {
    const int gy = y + by;
    const size_t rowImgOff = (size_t)gy * (size_t)width;

    // 1. Cooperatively compute 1x1 Dual-BT for tileCols (12) columns directly into LDS
    for (int col = sub; col < tileCols; col += COST_WG_H) {
      const int gx = xBase - B2 + col;
      if (gx >= 1 && gx < width - 1) {
        const size_t idxL = rowImgOff + (size_t)gx;
        const int sL = (int)sobelL[idxL];
        const short2 sBL = sobelBoundsL[idxL];
        const int rL = (int)rawL[idxL];
        const short2 rBL = rawBoundsL[idxL];

        for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
          const int d0 = minDisparity + (pIdx << 1);
          const int d1 = d0 + 1;
          const int xr0 = gx - d0;
          const int xr1 = gx - d1;

          ushort c0 = (ushort)INVALID_COST;
          if (xr0 >= 1 && xr0 < width - 1) {
            const size_t idxR0 = rowImgOff + (size_t)xr0;
            const int sR0 = (int)sobelR[idxR0];
            const short2 sBR0 = sobelBoundsR[idxR0];
            const int bt0 = min(
                max(0, max(sL - (int)sBR0.y, (int)sBR0.x - sL)),
                max(0, max(sR0 - (int)sBL.y, (int)sBL.x - sR0)));
            const int rR0 = (int)rawR[idxR0];
            const short2 rBR0 = rawBoundsR[idxR0];
            const int btr0 = min(
                max(0, max(rL - (int)rBR0.y, (int)rBR0.x - rL)),
                max(0, max(rR0 - (int)rBL.y, (int)rBL.x - rR0)));
            c0 = (ushort)(bt0 + btr0);
          }

          ushort c1 = (ushort)INVALID_COST;
          if (xr1 >= 1 && xr1 < width - 1) {
            const size_t idxR1 = rowImgOff + (size_t)xr1;
            const int sR1 = (int)sobelR[idxR1];
            const short2 sBR1 = sobelBoundsR[idxR1];
            const int bt1 = min(
                max(0, max(sL - (int)sBR1.y, (int)sBR1.x - sL)),
                max(0, max(sR1 - (int)sBL.y, (int)sBL.x - sR1)));
            const int rR1 = (int)rawR[idxR1];
            const short2 rBR1 = rawBoundsR[idxR1];
            const int btr1 = min(
                max(0, max(rL - (int)rBR1.y, (int)rBR1.x - rL)),
                max(0, max(rR1 - (int)rBL.y, (int)rBL.x - rR1)));
            c1 = (ushort)(bt1 + btr1);
          }

          tileH[col][pIdx] = (ushort2)(c0, c1);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Accumulate 5 horizontal taps directly from LDS (zero global memory traffic!)
    if (x >= B2 + 1 && x < width - B2 - 1) {
      int iter = 0;
      for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W, ++iter) {
        int2 rowSum = (int2)(0, 0);
        for (int k = 0; k <= 2 * B2; ++k) {
          rowSum += convert_int2(tileH[sub + k][pIdx]);
        }
        blockSum[iter] += rowSum;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (x >= width) return;

  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numPairs;
  if (x < B2 + 1 || x >= width - B2 - 1) {
    for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W) {
      costOut2[baseOut + (size_t)pIdx] = inv2;
    }
    return;
  }

  int iter = 0;
  for (int pIdx = lane; pIdx < numPairs; pIdx += COST_WG_W, ++iter) {
    const int d0 = minDisparity + (pIdx << 1);
    const int d1 = d0 + 1;
    const int xr0 = x - d0;
    const int xr1 = x - d1;

    const bool valid0 = (xr0 - B2 >= 1 && xr0 + B2 < width - 1);
    const bool valid1 = (xr1 - B2 >= 1 && xr1 + B2 < width - 1);

    const short out0 = valid0 ? (short)min(blockSum[iter].x, INVALID_COST - 1) : (short)INVALID_COST;
    const short out1 = valid1 ? (short)min(blockSum[iter].y, INVALID_COST - 1) : (short)INVALID_COST;
    costOut2[baseOut + (size_t)pIdx] = (short2)(out0, out1);
  }
}

#pragma OPENCL EXTENSION cl_khr_subgroups : enable


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

// ─── Kernel 3 (Subgroup-Accelerated Bidirectional SGM): 4-Pass Axis Sweep ────
// Replaces 10 LDS barriers per pixel step with Wave32 sub_group_reduce_min & sub_group_any!
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
    if (scanId >= width + height - 1) return;
    x0 = (scanId < width) ? scanId : (width - 1);
    y0 = (scanId < width) ? 0 : (scanId - width + 1);
    steps = min(x0 + 1, height - y0);
  }

  __local int sharedLPrev[WTA_ROWS_PER_WG][NUM_DISPARITIES + 2];

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
    int hasAnyValid = 0;

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities) {
        const short c = costIn[baseIdx + (size_t)d];
        cVal[k] = (int)c;
        if (c != INVALID_COST) hasAnyValid = 1;
        sharedLPrev[sub][d + 1] = lPrev[k];
        laneMinPrev = min(laneMinPrev, lPrev[k]);
      }
    }

    // Hardware Wave32 reductions (zero LDS buffers, zero reduction barriers!)
    const int minPrev = sub_group_reduce_min(laneMinPrev);
    const bool currHasValid = (sub_group_any(hasAnyValid) != 0);

    // Single barrier so neighbor lanes see updated sharedLPrev[sub][d +/- 1]
    barrier(CLK_LOCAL_MEM_FENCE);

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
    int hasAnyValid = 0;

    for (int k = 0; k < D_PER_LANE; ++k) {
      const int d = k * SGM_WG_SIZE + lane;
      if (d < numDisparities) {
        const short c = costIn[baseIdx + (size_t)d];
        cVal[k] = (int)c;
        if (c != INVALID_COST) hasAnyValid = 1;
        sharedLPrev[sub][d + 1] = lPrev[k];
        laneMinPrev = min(laneMinPrev, lPrev[k]);
      }
    }

    const int minPrev = sub_group_reduce_min(laneMinPrev);
    const bool currHasValid = (sub_group_any(hasAnyValid) != 0);
    barrier(CLK_LOCAL_MEM_FENCE);

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

// ─── Optional 2x Downsample / Edge-Aware Upsample Kernels (~0.2 ms) ───────────
__attribute__((reqd_work_group_size(PF_WG_W, PF_WG_H, 1)))
__kernel void k_downsample_2x(
    __global const uchar* restrict srcFull,
    __global uchar* restrict dstHalf,
    const int fullW,
    const int fullH,
    const int halfW,
    const int halfH) {
  const int hx = get_global_id(0);
  const int hy = get_global_id(1);
  if (hx >= halfW || hy >= halfH) return;

  const int x0 = hx << 1;
  const int y0 = hy << 1;
  const int x1 = min(x0 + 1, fullW - 1);
  const int y1 = min(y0 + 1, fullH - 1);

  const uint sum =
      (uint)srcFull[(size_t)y0 * (size_t)fullW + (size_t)x0] +
      (uint)srcFull[(size_t)y0 * (size_t)fullW + (size_t)x1] +
      (uint)srcFull[(size_t)y1 * (size_t)fullW + (size_t)x0] +
      (uint)srcFull[(size_t)y1 * (size_t)fullW + (size_t)x1];

  dstHalf[(size_t)hy * (size_t)halfW + (size_t)hx] = (uchar)((sum + 2u) >> 2);
}

__attribute__((reqd_work_group_size(PF_WG_W, PF_WG_H, 1)))
__kernel void k_upsample_disp_2x(
    __global const float* restrict dispHalf,
    __global float* restrict dispFull,
    const int fullW,
    const int fullH,
    const int halfW,
    const int halfH,
    const float invalidDisparity) {
  const int fx = get_global_id(0);
  const int fy = get_global_id(1);
  if (fx >= fullW || fy >= fullH) return;

  // Map full-res pixel center to half-res coordinate space
  const float u = ((float)fx + 0.5f) * 0.5f - 0.5f;
  const float v = ((float)fy + 0.5f) * 0.5f - 0.5f;

  const int x0 = clamp((int)floor(u), 0, halfW - 1);
  const int y0 = clamp((int)floor(v), 0, halfH - 1);
  const int x1 = min(x0 + 1, halfW - 1);
  const int y1 = min(y0 + 1, halfH - 1);

  const float wx = clamp(u - (float)x0, 0.0f, 1.0f);
  const float wy = clamp(v - (float)y0, 0.0f, 1.0f);

  const float d00 = dispHalf[(size_t)y0 * (size_t)halfW + (size_t)x0];
  const float d10 = dispHalf[(size_t)y0 * (size_t)halfW + (size_t)x1];
  const float d01 = dispHalf[(size_t)y1 * (size_t)halfW + (size_t)x0];
  const float d11 = dispHalf[(size_t)y1 * (size_t)halfW + (size_t)x1];

  const bool v00 = (d00 != invalidDisparity) && isfinite(d00);
  const bool v10 = (d10 != invalidDisparity) && isfinite(d10);
  const bool v01 = (d01 != invalidDisparity) && isfinite(d01);
  const bool v11 = (d11 != invalidDisparity) && isfinite(d11);

  // Guard against depth discontinuity smearing (> 1.5 px in half-res = > 3.0 px full-res)
  float dRef = invalidDisparity;
  if (v00) dRef = d00;
  else if (v10) dRef = d10;
  else if (v01) dRef = d01;
  else if (v11) dRef = d11;

  if (dRef == invalidDisparity) {
    dispFull[(size_t)fy * (size_t)fullW + (size_t)fx] = invalidDisparity;
    return;
  }

  float wSum = 0.0f;
  float valSum = 0.0f;
  const float w00 = (1.0f - wx) * (1.0f - wy);
  const float w10 = wx * (1.0f - wy);
  const float w01 = (1.0f - wx) * wy;
  const float w11 = wx * wy;

  if (v00 && fabs(d00 - dRef) <= 1.5f) { wSum += w00; valSum += w00 * d00; }
  if (v10 && fabs(d10 - dRef) <= 1.5f) { wSum += w10; valSum += w10 * d10; }
  if (v01 && fabs(d01 - dRef) <= 1.5f) { wSum += w01; valSum += w01 * d01; }
  if (v11 && fabs(d11 - dRef) <= 1.5f) { wSum += w11; valSum += w11 * d11; }

  dispFull[(size_t)fy * (size_t)fullW + (size_t)fx] =
      (wSum > 0.0f) ? (2.0f * (valSum / wSum)) : invalidDisparity;
}
