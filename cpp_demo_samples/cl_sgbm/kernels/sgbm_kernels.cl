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
