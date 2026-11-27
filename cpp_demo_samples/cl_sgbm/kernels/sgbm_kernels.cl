// ─── Helper Kernel: Fast Compute Clear (Replaces SDMA clEnqueueFillBuffer) ────
__kernel void k_clear_uint(__global uint* restrict buf, const uint val, const uint count) {
  const uint idx = get_global_id(0);
  if (idx < count) buf[idx] = val;
}

// ─── Kernel 2A (16-bit ushort): 1x1 Dual Birchfield-Tomasi Pixel Cost ─────────
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
    __global ushort* restrict pixelCostOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities) {
  const int lane = get_local_id(0);
  const int x = get_group_id(0) * COST_WG_H + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  if (x < 1 || x >= width - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
      pixelCostOut[baseOut + (size_t)dIdx] = (ushort)INVALID_COST;
    }
    return;
  }

  const int idxL = y * width + x;
  const int sL = (int)sobelL[idxL];
  const short2 sBL = sobelBoundsL[idxL];
  const int rL = (int)rawL[idxL];
  const short2 rBL = rawBoundsL[idxL];

  for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
    const int d = minDisparity + dIdx;
    const int xr = x - d;

    if (xr < 1 || xr >= width - 1) {
      pixelCostOut[baseOut + (size_t)dIdx] = (ushort)INVALID_COST;
      continue;
    }

    const int idxR = y * width + xr;
    const int sR = (int)sobelR[idxR];
    const short2 sBR = sobelBoundsR[idxR];
    const int btSobel = min(
        max(0, max(sL - (int)sBR.y, (int)sBR.x - sL)),
        max(0, max(sR - (int)sBL.y, (int)sBL.x - sR)));

    const int rR = (int)rawR[idxR];
    const short2 rBR = rawBoundsR[idxR];
    const int btRaw = min(
        max(0, max(rL - (int)rBR.y, (int)rBR.x - rL)),
        max(0, max(rR - (int)rBL.y, (int)rBL.x - rR)));

    pixelCostOut[baseOut + (size_t)dIdx] = (ushort)(btSobel + btRaw);
  }
}

// ─── Kernel 2B-1: Horizontal Bx1 Box Sum (100% Coalesced Wave32 over D) ───────
__attribute__((reqd_work_group_size(COST_WG_W, COST_WG_H, 1)))
__kernel void k_box_filter_h(
    __global const ushort* restrict pixelCostIn,
    __global ushort* restrict horizCostOut,
    const int width,
    const int height,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);
  const int x = get_group_id(0) * COST_WG_H + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const int B2 = blockSize >> 1;
  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  if (x < B2 + 1 || x >= width - B2 - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
      horizCostOut[baseOut + (size_t)dIdx] = (ushort)INVALID_COST;
    }
    return;
  }

  for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
    int sumH = 0;
    for (int bx = -B2; bx <= B2; ++bx) {
      sumH += (int)pixelCostIn[baseOut + (size_t)(bx * numDisparities) + (size_t)dIdx];
    }
    horizCostOut[baseOut + (size_t)dIdx] = (ushort)min(sumH, INVALID_COST - 1);
  }
}

// ─── Kernel 2B-2: Vertical 1xB Box Sum + Border Mask (100% Coalesced over D) ──
__attribute__((reqd_work_group_size(COST_WG_W, COST_WG_H, 1)))
__kernel void k_box_filter_v(
    __global const ushort* restrict horizCostIn,
    __global short* restrict costOut,
    const int width,
    const int height,
    const int minDisparity,
    const int numDisparities,
    const int blockSize) {
  const int lane = get_local_id(0);
  const int x = get_group_id(0) * COST_WG_H + get_local_id(1);
  const int y = get_group_id(1);

  if (x >= width || y >= height) return;

  const int B2 = blockSize >> 1;
  const size_t baseOut = ((size_t)y * (size_t)width + (size_t)x) * (size_t)numDisparities;

  if (y < B2 || y >= height - B2 || x < B2 + 1 || x >= width - B2 - 1) {
    for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
      costOut[baseOut + (size_t)dIdx] = (short)INVALID_COST;
    }
    return;
  }

  for (int dIdx = lane; dIdx < numDisparities; dIdx += COST_WG_W) {
    const int d = minDisparity + dIdx;
    const int xr = x - d;

    if (xr - B2 < 1 || xr + B2 >= width - 1) {
      costOut[baseOut + (size_t)dIdx] = (short)INVALID_COST;
      continue;
    }

    int blockSum = 0;
    const size_t rowStride = (size_t)width * (size_t)numDisparities;
    for (int by = -B2; by <= B2; ++by) {
      blockSum += (int)horizCostIn[baseOut + (size_t)(by * (int)rowStride) + (size_t)dIdx];
    }

    costOut[baseOut + (size_t)dIdx] = (short)min(blockSum, INVALID_COST - 1);
  }
}
