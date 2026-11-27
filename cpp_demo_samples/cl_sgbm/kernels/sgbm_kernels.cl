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
