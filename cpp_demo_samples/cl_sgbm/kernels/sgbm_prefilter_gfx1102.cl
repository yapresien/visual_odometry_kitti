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
