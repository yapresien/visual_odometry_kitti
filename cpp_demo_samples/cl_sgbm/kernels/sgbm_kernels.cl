// ─── Kernel 2: Dual Birchfield-Tomasi + BxB Box Cost Volume (gfx1102) ────────
// Computes cost[(y * width + x) * D + dIdx] directly in L1 cache + registers.
// Dispatch: localSize = {32, 8, 1}, globalSize = {ceil(W/32)*32, ceil(H/8)*8, D}
#define BT_WG_W 32
#define BT_WG_H 8
#define INVALID_COST 32767
#define MAX_PATH_COST 65535

// ─── Kernel 2 (Optimized): LDS-Tiled Dual-BT + BxB Box Cost Volume (gfx1102) ─
// Loads Left & Right tiles into 32-bank conflict-free LDS once per (tile, d),
// cutting global memory traffic by ~15x while staying 100% bit-exact.
#define MAX_B2 4                      // Supports up to blockSize = 9
#define BT_TILE_W (BT_WG_W + 2 * MAX_B2)  // 40
#define BT_TILE_H (BT_WG_H + 2 * MAX_B2)  // 16
#define BT_LDS_STRIDE 41              // Odd stride eliminates 32-bank LDS conflicts


// ─── Kernel 2A: 1x1 Dual Birchfield-Tomasi Pixel Cost (256-thread WG on gfx1102) ─
// lx (0..31) = disparity lane (100% coalesced Wave32 global memory access)
// ly (0..7)  = 8 consecutive x pixels per work-group (8 Wave32s = 100% CU occupancy)
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void k_bt_pixel_cost(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global ushort* restrict pixelCostOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities) {
  const int lane = get_local_id(0);  // 0..31 (disparity lane)
  const int x = get_group_id(0) * 8 + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  // Outside 1-pixel horizontal margin needed for valid Sobel/BT bounds
  if (x < 1 || x >= width - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
      pixelCostOut[baseOut + (size_t)dIdx] = (ushort)INVALID_COST;
    }
    return;
  }

  const int idxL = y * width + x;
  const int sL = (int)sobelL[idxL];
  const short2 sBL = sobelBoundsL[idxL];
  const int rL = (int)rawL[idxL];
  const short2 rBL = rawBoundsL[idxL];

  for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
    const int d = minDisparity + dIdx;
    const int xr = x - d;

    if (xr < 1 || xr >= width - 1) {
      pixelCostOut[baseOut + (size_t)dIdx] = (ushort)INVALID_COST;
      continue;
    }

    const int idxR = y * width + xr;

    // 1. Sobel-X symmetric BT distance
    const int sR = (int)sobelR[idxR];
    const short2 sBR = sobelBoundsR[idxR];
    const int btSobel = min(
        max(0, max(sL - (int)sBR.y, (int)sBR.x - sL)),
        max(0, max(sR - (int)sBL.y, (int)sBL.x - sR)));

    // 2. Raw intensity symmetric BT distance
    const int rR = (int)rawR[idxR];
    const short2 rBR = rawBoundsR[idxR];
    const int btRaw = min(
        max(0, max(rL - (int)rBR.y, (int)rBR.x - rL)),
        max(0, max(rR - (int)rBL.y, (int)rBL.x - rR)));

    pixelCostOut[baseOut + (size_t)dIdx] = (ushort)(btSobel + btRaw);
  }
}

// ─── Kernel 2B: BxB Box Sum over Pixel Cost Volume (256-thread WG on gfx1102) ──
// Sums BxB taps from pixelCostIn with 100% coalesced Wave32 reads and writes.
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void k_box_filter_cost(
    __global const ushort* restrict pixelCostIn,
    __global short* restrict costOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);  // 0..31 (disparity lane)
  const int x = get_group_id(0) * 8 + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const int B2 = blockSize >> 1;
  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  if (y < B2 || y >= height - B2 || x < B2 + 1 || x >= width - B2 - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
      costOut[baseOut + (size_t)dIdx] = INVALID_COST;
    }
    return;
  }

  for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
    const int d = minDisparity + dIdx;
    const int xr = x - d;

    if (xr - B2 < 1 || xr + B2 >= width - 1) {
      costOut[baseOut + (size_t)dIdx] = INVALID_COST;
      continue;
    }

    int blockSum = 0;
    for (int by = -B2; by <= B2; ++by) {
      const size_t rowBase = ((size_t)(y + by) * (size_t)width + (size_t)x) * (size_t)numDisparities + (size_t)dIdx;
      for (int bx = -B2; bx <= B2; ++bx) {
        blockSum += (int)pixelCostIn[rowBase + (size_t)(bx * numDisparities)];
      }
    }

    costOut[baseOut + (size_t)dIdx] = (short)min(blockSum, INVALID_COST - 1);
  }
}


// ─── Kernel 2 (Wave32 Coalesced over D): Dual-BT + BxB Box Cost Volume ──────
// Maps 1 Wave32 (32 threads) per (x, y) pixel, iterating across disparities d in
// chunks of 32 so global writes cost[(y*W + x)*D + d] are 100% contiguous!
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void k_bt_box_cost_40(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global short* restrict costOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);   // 0..31 (1 Wave32)
  const int x = get_group_id(0);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const int B2 = blockSize >> 1;
  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  // Border pixels: 100% coalesced Wave32 store of INVALID_COST
  if (y < B2 || y >= height - B2 || x < B2 + 1 || x >= width - B2 - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
      costOut[baseOut + (size_t)dIdx] = INVALID_COST;
    }
    return;
  }

  // Cache the BxB Left window in LDS once for all 128 disparities (max 9x9 = 81 elements)
  __local short lSobel[81];
  __local short2 lSobelB[81];
  __local uchar lRaw[81];
  __local short2 lRawB[81];

  const int winArea = blockSize * blockSize;
  for (int i = lane; i < winArea; i += 32) {
    const int by = (i / blockSize) - B2;
    const int bx = (i % blockSize) - B2;
    const int idxL = (y + by) * width + (x + bx);
    lSobel[i] = sobelL[idxL];
    lSobelB[i] = sobelBoundsL[idxL];
    lRaw[i] = rawL[idxL];
    lRawB[i] = rawBoundsL[idxL];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Each lane evaluates contiguous disparities dIdx = lane, lane+32, lane+64, lane+96
  for (int dIdx = lane; dIdx < numDisparities; dIdx += 32) {
    const int d = minDisparity + dIdx;
    const int xr = x - d;

    if (xr - B2 < 1 || xr + B2 >= width - 1) {
      costOut[baseOut + (size_t)dIdx] = INVALID_COST;
      continue;
    }

    int blockSum = 0;
    int tap = 0;
    for (int by = -B2; by <= B2; ++by) {
      const int rowOffR = (y + by) * width + xr;
      for (int bx = -B2; bx <= B2; ++bx, ++tap) {
        const int idxR = rowOffR + bx;

        // 1. Sobel-X BT distance (Left from LDS, Right from L1 cache)
        const int sL = (int)lSobel[tap];
        const short2 sBL = lSobelB[tap];
        const int sR = (int)sobelR[idxR];
        const short2 sBR = sobelBoundsR[idxR];
        const int btSobel = min(
            max(0, max(sL - (int)sBR.y, (int)sBR.x - sL)),
            max(0, max(sR - (int)sBL.y, (int)sBL.x - sR)));

        // 2. Raw intensity BT distance
        const int rL = (int)lRaw[tap];
        const short2 rBL = lRawB[tap];
        const int rR = (int)rawR[idxR];
        const short2 rBR = rawBoundsR[idxR];
        const int btRaw = min(
            max(0, max(rL - (int)rBR.y, (int)rBR.x - rL)),
            max(0, max(rR - (int)rBL.y, (int)rBL.x - rR)));

        blockSum += (btSobel + btRaw);
      }
    }

    // 100% coalesced 64-byte Wave32 store!
    costOut[baseOut + (size_t)dIdx] = (short)min(blockSum, INVALID_COST - 1);
  }
}


__attribute__((reqd_work_group_size(BT_WG_W, BT_WG_H, 1)))
__kernel void k_bt_box_cost_155(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global short* restrict costOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * BT_WG_W + lx;

  const int groupX0 = get_group_id(0) * BT_WG_W;
  const int groupY0 = get_group_id(1) * BT_WG_H;
  const int x = groupX0 + lx;
  const int y = groupY0 + ly;
  const int dIdx = get_global_id(2);

  if (dIdx >= numDisparities) return;

  const int B2 = blockSize >> 1;
  const int d = minDisparity + dIdx;
  const int tileW = BT_WG_W + 2 * B2;
  const int tileH = BT_WG_H + 2 * B2;

  // LDS cache for per-pixel (btSobel + btRaw) across the (32 + 2*B2) x (8 + 2*B2) tile
  __local ushort tilePixelCost[BT_TILE_H][BT_LDS_STRIDE];

  // 1. Cooperatively compute per-pixel BT cost once per tile element into LDS
  const int totalTileElems = tileW * tileH;
  for (int i = tid; i < totalTileElems; i += (BT_WG_W * BT_WG_H)) {
    const int ty = i / tileW;
    const int tx = i % tileW;
    const int sx = groupX0 + tx - B2;
    const int sy = groupY0 + ty - B2;
    const int sxr = sx - d;

    if (sy < 0 || sy >= height || sx < 1 || sx >= width - 1 || sxr < 1 || sxr >= width - 1) {
      tilePixelCost[ty][tx] = (ushort)INVALID_COST;
    } else {
      const int idxL = sy * width + sx;
      const int idxR = sy * width + sxr;

      const int sL = (int)sobelL[idxL];
      const int sR = (int)sobelR[idxR];
      const short2 sBL = sobelBoundsL[idxL];
      const short2 sBR = sobelBoundsR[idxR];
      const int btSobel = min(
          max(0, max(sL - (int)sBR.y, (int)sBR.x - sL)),
          max(0, max(sR - (int)sBL.y, (int)sBL.x - sR)));

      const int rL = (int)rawL[idxL];
      const int rR = (int)rawR[idxR];
      const short2 rBL = rawBoundsL[idxL];
      const short2 rBR = rawBoundsR[idxR];
      const int btRaw = min(
          max(0, max(rL - (int)rBR.y, (int)rBR.x - rL)),
          max(0, max(rR - (int)rBL.y, (int)rBL.x - rR)));

      tilePixelCost[ty][tx] = (ushort)(btSobel + btRaw);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (x >= width || y >= height) return;

  const size_t outIdx = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities + (size_t)dIdx;
  const int xr = x - d;

  if (y < B2 || y >= height - B2 ||
      x < B2 + 1 || x >= width - B2 - 1 ||
      xr - B2 < 1 || xr + B2 >= width - 1) {
    costOut[outIdx] = INVALID_COST;
    return;
  }

  // 2. Sum BxB block entirely from bank-conflict-free LDS (zero global reads!)
  int blockSum = 0;
  const int cx = lx + B2;
  const int cy = ly + B2;
  for (int by = -B2; by <= B2; ++by) {
    for (int bx = -B2; bx <= B2; ++bx) {
      blockSum += (int)tilePixelCost[cy + by][cx + bx];
    }
  }

  costOut[outIdx] = (short)min(blockSum, INVALID_COST - 1);
}


__attribute__((reqd_work_group_size(BT_WG_W, BT_WG_H, 1)))
__kernel void k_bt_box_cost_166(
    __global const uchar* restrict rawL,
    __global const uchar* restrict rawR,
    __global const short* restrict sobelL,
    __global const short* restrict sobelR,
    __global const short2* restrict sobelBoundsL,
    __global const short2* restrict sobelBoundsR,
    __global const short2* restrict rawBoundsL,
    __global const short2* restrict rawBoundsR,
    __global short* restrict costOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int dIdx = get_global_id(2);

  if (x >= width || y >= height || dIdx >= numDisparities) return;

  const int B2 = blockSize >> 1;
  const int d = minDisparity + dIdx;
  const int xr = x - d;
  const size_t outIdx = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities + (size_t)dIdx;

  // Match SgbmCpu::computeCost exact border validity bounds
  if (y < B2 || y >= height - B2 ||
      x < B2 + 1 || x >= width - B2 - 1 ||
      xr - B2 < 1 || xr + B2 >= width - 1) {
    costOut[outIdx] = INVALID_COST;
    return;
  }

  int blockSum = 0;

  // 5x5 (or BxB) window: adjacent rows/cols hit gfx1102's 32KB L1 cache across Wave32
  for (int by = -B2; by <= B2; ++by) {
    const int rowOff = (y + by) * width;
    for (int bx = -B2; bx <= B2; ++bx) {
      const int idxL = rowOff + (x + bx);
      const int idxR = rowOff + (xr + bx);

      // 1. Sobel-X symmetric Birchfield-Tomasi distance
      const int sL = (int)sobelL[idxL];
      const int sR = (int)sobelR[idxR];
      const short2 sBL = sobelBoundsL[idxL];
      const short2 sBR = sobelBoundsR[idxR];

      const int sDiffLR = max(0, max(sL - (int)sBR.y, (int)sBR.x - sL));
      const int sDiffRL = max(0, max(sR - (int)sBL.y, (int)sBL.x - sR));
      const int btSobel = min(sDiffLR, sDiffRL);

      // 2. Raw intensity symmetric Birchfield-Tomasi distance
      const int rL = (int)rawL[idxL];
      const int rR = (int)rawR[idxR];
      const short2 rBL = rawBoundsL[idxL];
      const short2 rBR = rawBoundsR[idxR];

      const int rDiffLR = max(0, max(rL - (int)rBR.y, (int)rBR.x - rL));
      const int rDiffRL = max(0, max(rR - (int)rBL.y, (int)rBL.x - rR));
      const int btRaw = min(rDiffLR, rDiffRL);

      blockSum += (btSobel + btRaw);
    }
  }

  costOut[outIdx] = (short)min(blockSum, INVALID_COST - 1);
}
