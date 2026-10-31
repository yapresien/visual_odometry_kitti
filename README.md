# visual_odometry_kitti

Useful for using KITTI dataset nvidia gpu based visual odometry trap feature movements

# pykitti vvo

This package provides a minimal set of scripts for working with the KITTI dataset [[1]](#references) in Python.

## Installation

### Using pip

You can install pykitti via pip using

```
pip install pykitti
```

### From source

To install the package from source, simply clone or download the repository to your machine

```
git clone https://github.com/yapresien/visual_odometry_kitti.git
```

and run the provided setup tool

```
cd visual_odometry_kitti
python setup.py install
```

## KITTI dataset

```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
```


# KITTI Stereo Visual Odometry — Sequence 00

Classical stereo VO using CUDA SGBM + Lucas-Kanade flow + SQPNP RANSAC.
Achieves **2.3 % RPE@100 m** and **1.1 % endpoint error** on KITTI 00
without loop closure.

## Results

| Metric | Value | Reference (ORB-SLAM3 w/ loop closure) |
|---|---|---|
| ATE (mean) | **40.45 m** | 6.77 m |
| RPE@100 m trans | **2.26 m (2.3 %)** | ~1.1 % |
| RPE@100 m rot | **1.24°** | — |
| RPE@800 m trans | 14.82 m (1.9 %) | — |
| Path scale (VO / GT) | **0.98** | 1.00 |
| Path length | 3649 m / 3724 m | — |
| Runtime | ~20 ms / frame (~50 fps) | — |

**Interpretation** — RPE within 2× of state-of-the-art. ATE gap is entirely
loop closure; even ORB-SLAM3 degrades to ~189 m ATE without it.

## Dataset

- Sequence: **KITTI Odometry 00** — 4541 frames, 3724 m loop, urban Karlsruhe
- Images: `image_0/` (left grayscale), `image_1/` (right grayscale), 1242×375
- Calibration: `calib.txt` (P0, P1, Tr)
- GT: `poses/00.txt` — 12-col KITTI format (`r11 r12 r13 tx r21 … tz`)
- Stereo baseline: **0.537 m** (`P1[0,3] / fx`)

## Pipeline

```
Frame t−1 (L,R) ──► CUDA SGBM ──► disparity ──► Z = fx·b/d
    │
    ▼
Features (LKF, 1500 corners)
    │
    ▼
CUDA LK optical flow (t−1 → t) ──► tracked 2D
    │
    ▼
For each tracked feature:
    sample SGBM disparity @ (u,v)
    Z ∈ [3.0, 200] m filter
    3D = (X, Y, Z)
    │
    ▼
solvePnPRansac(SQPNP, iter=300, reproj=3.5, conf=0.99)
    │
    ▼
R_global ← R_global · R_delta
t_global ← t_global + R_global · t_delta
```

## The 8 Fixes That Took ATE From 161 m → 40 m

### 1. `P2=400` (not 800) — the single biggest fix

```python
gpu_sgbm = cv2.cuda.createStereoSGM(
    minDisparity=0, numDisparities=128,
    P1=200, P2=400,           # ← NOT 800
    uniquenessRatio=3,
)
```

SGBM smoothness penalty. At `P2=800` the mid-field (15–40 m) is over-smoothed:
SGBM rejects far features, only near-road pixels survive (`valid=18 %`), and PnP
runs on a coplanar point cloud. At `P2=400` the mid-field recovers
(`valid=65–75 %`), giving PnP a genuinely 3D set.

**Impact:** `valid 18 → 65 %`, `kept 40 → 400`, ATE **161 → 40 m**.

### 2. `uniquenessRatio=3` (not 10)

```python
uniquenessRatio=3,            # ← was 10
```

Filters matches by 2nd-best-cost margin. Ratio=10 is too strict for low-texture
scenes (sky, distant road). Ratio=3 accepts ~2× more mid-field matches.

### 3. `Z > 3.0` depth cutoff (not 0.5)

```python
if Z <= 3.0 or Z > 200.0:     # ← was 0.5
    continue
```

Pixels under 3 m are the road directly under the car — highest depth noise,
contribute nothing to motion, and dominate the planar set that causes rotation
flips in RANSAC.

### 4. `SQPNP` + `iter=300` + `reproj=3.5` (not `ITERATIVE` + `1000` + `2.0`)

```python
cv2.solvePnPRansac(
    points_3d, curr_2d_valid, K, distCoeffs=None,
    iterationsCount=300,
    reprojectionError=3.5,
    confidence=0.99,
    flags=cv2.SOLVEPNP_SQPNP,
)
```

SQPNP's 4-point minimal solver is more robust to planar degeneracy than
ITERATIVE's 6-point DLT. `reproj=3.5` accommodates KITTI far-feature depth noise
(SGBM ±5 % at 50 m → ±3–4 px reprojection). `iter=300` with early termination
runs 3× faster than 1000.

**Impact:** rotation RPE **62° → 1.24°** (rotation flips eliminated).

### 5. `maxLevel=4` on LK (not 3)

```python
gpu_lk_motion = cv2.cuda.SparsePyrLKOpticalFlow_create(
    winSize=(21, 21), maxLevel=4,   # ← was 3
    iters=40, useInitialFlow=False,
)
```

Road-region flow is 30–70 px/frame. `maxLevel=3` search radius ≈ 70 px is right
at LK's reliability edge — half-matched flow produces silent 10–15 %
under-tracking and compressed motion.

### 6. `valid_idx` remapping for RANSAC inliers

```python
# In the PnP loop:
valid_idx.append(i)                                # remember original index

# After solvePnPRansac:
ransac_status[valid_idx[inliers.ravel()]] = 1
```

`inliers` are indices into the *filtered* 3D array (~150), not the original
feature array (~800). Without remapping, `ransac_status` marks the wrong
positions and `kept` count is meaningless.

### 7. Save format — interleaved, not concatenated

```python
traj_log.append(np.hstack([R_global, t_global]).ravel())
# NOT: np.hstack([R_global.ravel(), t_global.ravel()])
```

KITTI format is interleaved: `[r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]`.
The concatenated form looks plausible but `reshape(3,4)` reads rotation values
into the translation column. Frame 0 becomes three identical `[1,0,0,0]` rows →
singular 4×4 matrix → `np.linalg.inv` crashes on every eval.

### 8. `(1, N, 2)` GpuMat for LK

```python
pts_clean = pts_cpu.reshape(1, -1, 2).astype(np.float32)
```

CUDA `SparsePyrLKOpticalFlow.calc()` asserts `prevPts.rows == 1` and
`type == CV_32FC2`. `(N, 1, 2)` fails the assertion; `(1, N, 2)` passes.

## What Didn't Work (Hypothesis Log)

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| Scale compression from planar PnP | Compare path length vs GT | VO/GT = **0.98** | ❌ Wrong — scale was never the issue |
| Y-drift (pitch bias) dominates ATE | Y-clamp, 500 frames | 5.03 → 5.05 m ATE (0.4 %) | ❌ Wrong — Umeyama already corrects Y |
| `numDisparities=192` gives more far features | CUDA SGBM init | `Unsupported num of disparities` | ❌ CUDA build supports only 64/128/256 |
| `P2=800` produces working trajectory | Full run | valid=18 %, ATE=161 m | ❌ Broken — must stay at 400 |
| Post-hoc `translate.py` can fix corrupted file | Load / reshape / save | Singular matrix on frame 0 | ❌ Compounded damage; unrecoverable |
| LiDAR-injected depth would halve ATE | KISS-ICP comparison | LO Y-RMS (14.5 m) worse than VO (10.8 m) | ⚠️ Not attempted — VO scale already 0.98 |

## Remaining Error — Per-Axis Breakdown

| Axis | VO RMS | LO (KISS-ICP) RMS | Endpoint error |
|---|---|---|---|
| X (lateral) | **15.95 m** | 3.65 m | 4.93 m |
| Y (vertical) | 10.84 m | 14.51 m | 13.00 m |
| Z (forward) | **14.93 m** | 2.36 m | 3.38 m |

X and Z show **large mid-path wander but small endpoint error** — the signature
of accumulated random-walk pose noise. Per-frame X/Z noise of ~0.24 m accumulates
to `0.24 × √4541 ≈ 16 m` RMS over 3.7 km. Endpoint error stays small because
random walk averages back toward zero statistically.

**Not fixable by per-frame tuning.** Requires longer-baseline solves
(keyframe PnP) or global optimization (loop closure).

## Known Limitations

1. **No loop closure** — ATE ~40 m over a 3.7 km loop.
2. **Frame-to-frame PnP only** — no keyframe / bundle adjustment → X/Z wander.
3. **No pose graph** — drift is unbounded by design.
4. **Assumes static scene** — moving vehicles cause outliers (RANSAC handles
   most, no explicit dynamic-object rejection).
5. **Single-stream GPU** — SGBM and LK serialized; multi-stream would give ~1.5×.

## Reproduce

### Environment

- OpenCV 4.10.0 (CUDA + contrib)
- NumPy
- CUDA-capable GPU (~2 GB VRAM)
- Dataset root: `/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00/`

### Run

```bash
python demo_samples/08_feature_vo_gpu_poses.py    # ~4 min for 4541 frames
python demo_samples/eval_vo.py                    # ATE / RPE report
```

### Config toggles (top of `08_feature_vo_gpu_poses.py`)

```python
USE_Y_CLAMP = False     # MUST stay False (tested and rejected)
MAX_FRAMES  = None      # int → partial run
SHOW_DISP   = True      # per-frame [DISP] stats
SHOW_POOL   = True      # per-frame [POOL] stats
```

### Critical values — do not change without re-testing

```python
NUM_DISP         = 128              # 192 crashes on this CUDA build
P1               = 200              # 8 * 1 * 5 * 5
P2               = 400              # 32 * 1 * 5 * 5 / 2   ← MUST be 400
uniquenessRatio  = 3
maxLevel         = 4                # LK pyramid
Z filter         : [3.0, 200.0] m
iterationsCount  = 300
reprojectionError = 3.5
confidence       = 0.99
flags            = cv2.SOLVEPNP_SQPNP
```

## File Manifest

| File | Purpose |
|---|---|
| `demo_samples/08_feature_vo_gpu_poses.py` | Main VO pipeline |
| `demo_samples/eval_vo.py` | ATE / RPE evaluator (Umeyama + KITTI RPE) |
| `demo_samples/diag_3way.py` | GT vs VO vs LO comparison, per-axis drift |
| `est_traj.txt` | Live output, KITTI 12-col format |
| `est_traj_baseline_40m.txt` | **Backup of the working 40 m / 2.3 % run** |
| `est_traj_yclamped_34m_ate_5p3rpe.txt` | Y-clamped variant (not recommended) |
| `velodyne_poses_kitti.txt` | KISS-ICP LiDAR odometry (diagnostics only) |

## If You Want to Push Further

| Goal | Effort | Expected improvement |
|---|---|---|
| Keyframe PnP (5-frame baseline) | 30–60 min | ATE 40 → 25–30 m, RPE 2.3 % → 1.8–2.0 % |
| Loop closure + pose graph | 2–4 h | ATE 40 → 5–10 m, RPE 2.3 % → 1.0–1.3 % |
| LiDAR-injected depth | 1–2 h | ATE 40 → 30–35 m (marginal; scale already 0.98) |
| Replace SGBM with RAFT-Stereo | 2–3 h + training data | Depth noise 5 % → 1 %, ATE 40 → 30 m |

**Recommendation** — unless you need sub-10 m ATE, the current pipeline is at
the practical ceiling for frame-to-frame classical stereo VO. Stop here, or
commit to full SLAM (loop closure).

## References

- **KITTI** — Geiger et al., *Vision meets Robotics: The KITTI Dataset*, IJRR 2013
- **ORB-SLAM3** — Campos et al., *ORB-SLAM3: An Accurate Open-Source Library
  for Visual, Visual-Inertial and Multi-Map SLAM*, T-RO 2021
- **SQPNP** — Terzakis & Lourakis, *A Consistently Fast and Globally Optimal
  Solution to the Perspective-n-Point Problem*, ECCV 2020
- **KISS-ICP** — Vizzo et al., *KISS-ICP: In Defense of Point-to-Point ICP*,
  RA-L 2023


  ## GPU vs CPU A/B Benchmark (4540 frames, KITTI 00)

| Configuration | Throughput | Runtime | ATE | RPE@100m |
|---|---|---|---|---|
| All CPU | 11.7 fps | 387 s | 41.9 m | 2.3 % |
| All GPU + async I/O | **41.0 fps** | **111 s** | 41.9 m | 2.3 % |
| **Speedup** | **3.5×** | **3.5×** | — | — |

Per-stage:
- CUDA SGBM: 38.4 → 5.1 ms (7.5×) ← dominant win
- Async I/O prefetch: 20.4 → 4.3 ms (4.8×)
- CUDA LK: 3.8 → 1.3 ms (2.9×)
- solvePnPRansac: 4.8 → 7.0 ms (slower — GPU SGBM feeds it 2× more valid 3D points)
- Incremental GT draw: 9.6 → 3.5 ms (2.8×)