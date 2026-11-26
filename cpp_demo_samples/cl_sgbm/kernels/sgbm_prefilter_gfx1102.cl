// sgbm_prefilter_gfx1102.cl
// Tuned for AMD Radeon 890M (gfx1102): Wave32, 256-thread WG (32x8), 32-bank LDS, native short2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_W 32
#define WG_H 8
#define APRON 2
#define TILE_W (WG_W + 2 * APRON)   // 36
#define TILE_H (WG_H + 2 * APRON)   // 12
#define LDS_STRIDE 37               // Odd stride avoids 32-bank LDS conflicts on vertical reads

__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void k_prefilter_and_bt_bounds(
    __global const uchar* restrict rawImg,
    __global short* restrict sobelOut,
    __global short2* restrict sobelBoundsOut,  // .x = min, .y = max
    __global short2* restrict rawBoundsOut,    // .x = min, .y = max
    const int width,
    const int height,
    const int preFilterCap) {
  const int lx = get_local_id(0);   // 0..31 (1 Wave32 per row!)
  const int ly = get_local_id(1);   // 0..7
  const int tid = ly * WG_W + lx;   // 0..255

  const int groupX0 = get_group_id(0) * WG_W;
  const int groupY0 = get_group_id(1) * WG_H;
  const int gx = groupX0 + lx;
  const int gy = groupY0 + ly;

  // Bank-conflict-free LDS buffers (12 * 37 * (1 + 2) = 1,332 bytes per WG -> 100% CU occupancy)
  __local uchar tileRaw[TILE_H][LDS_STRIDE];
  __local short tileSobel[TILE_H][LDS_STRIDE];

  // 1. Cooperative coalesced load of 36x12 = 432 pixels across 256 work-items (<= 2 loads/thread)
  for (int i = tid; i < TILE_W * TILE_H; i += (WG_W * WG_H)) {
    const int ty = i / TILE_W;
    const int tx = i % TILE_W;
    const int sx = groupX0 + tx - APRON;
    int sy = groupY0 + ty - APRON;

    // OpenCV BORDER_REFLECT_101 on Y, clamp on X
    if (sy < 0) sy = min(1, height - 1);
    if (sy >= height) sy = max(0, height - 2);
    const int clampedX = clamp(sx, 0, width - 1);

    tileRaw[ty][tx] = rawImg[sy * width + clampedX];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2. Compute Sobel-X for interior [1..10] rows x [1..34] cols = 340 elements across 256 threads
  const int sobelW = TILE_W - 2;  // 34
  const int sobelH = TILE_H - 2;  // 10
  for (int i = tid; i < sobelW * sobelH; i += (WG_W * WG_H)) {
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

  // 3. Write Sobel-X and packed short2 BT [min, max] bounds (100% coalesced Wave32 stores)
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
    const short sMin = (short)min((int)sVal, min(sPrev, sNext));
    const short sMax = (short)max((int)sVal, max(sPrev, sNext));
    sobelBoundsOut[idx] = (short2)(sMin, sMax);

    const int rVal = (int)tileRaw[ty][tx];
    const int rPrev = (rVal + (int)tileRaw[ty][tx - 1]) >> 1;
    const int rNext = (rVal + (int)tileRaw[ty][tx + 1]) >> 1;
    const short rMin = (short)min(rVal, min(rPrev, rNext));
    const short rMax = (short)max(rVal, max(rPrev, rNext));
    rawBoundsOut[idx] = (short2)(rMin, rMax);
  }
}
