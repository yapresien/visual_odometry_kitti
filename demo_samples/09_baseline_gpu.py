import os, glob, cv2, numpy as np, time
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════════════
# GPU vs CPU FLAGS
# ═══════════════════════════════════════════════════════════════════════════
USE_GPU_DETECTOR = True
USE_GPU_SGBM     = True
USE_GPU_LK       = True
ANY_GPU = USE_GPU_DETECTOR or USE_GPU_SGBM or USE_GPU_LK


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FLAGS
# ═══════════════════════════════════════════════════════════════════════════
MAX_FRAMES     = None
SHOW_TIMING    = True
SHOW_DISP      = False
SHOW_POOL      = False
USE_Y_CLAMP    = False
RENDER_EVERY_N = 1        # 1 = every frame; 5 = skip 4 (faster, fewer visuals)


# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
NUM_DISP        = 128
P1_SGBM         = 200
P2_SGBM         = 400
UNIQUENESS      = 3
BLOCK_SIZE      = 5
LK_WIN          = (21, 21)
LK_LEVELS       = 4
LK_ITERS        = 40
LK_EPS          = 0.01
Z_MIN           = 3.0
Z_MAX           = 200.0
RANSAC_ITERS    = 300
RANSAC_REPROJ   = 3.5
RANSAC_CONF     = 0.99
RANSAC_FLAGS    = cv2.SOLVEPNP_SQPNP
MERGE_THRESHOLD = 15.0
POOL_MIN_POINTS = 100


# ═══════════════════════════════════════════════════════════════════════════
# PATHS + CALIB
# ═══════════════════════════════════════════════════════════════════════════
BASEDIR       = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00'
CALIB_FILE    = os.path.join(BASEDIR, 'calib.txt')
GT_POSES_FILE = '/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/00.txt'

import threading, queue

# ─── ASYNC FRAME LOADER ────────────────────────────────────────────────────
def frame_loader(left_paths, right_paths, out_q, stop_ev):
    """Background thread: reads (left, right) pairs, pushes to queue."""
    for l_path, r_path in zip(left_paths, right_paths):
        if stop_ev.is_set():
            return
        l_img = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
        r_img = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        # Putt with timeout loop so we exit cleanly if the main loop breaks
        while not stop_ev.is_set():
            try:
                out_q.put((l_img, r_img), timeout=0.1)
                break
            except queue.Full:
                continue


def load_kitti_calib(path):
    with open(path) as f:
        lines = f.readlines()
    P0 = P1 = None
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        if parts[0].startswith('P0'):
            P0 = np.array([float(x) for x in parts[1:13]], dtype=np.float32).reshape(3, 4)
        elif parts[0].startswith('P1'):
            P1 = np.array([float(x) for x in parts[1:13]], dtype=np.float32).reshape(3, 4)
    if P0 is None or P1 is None: raise ValueError("P0/P1 missing")
    return P0[:, :3], abs(P1[0, 3] - P0[0, 3]) / P0[0, 0]


K, baseline = load_kitti_calib(CALIB_FILE)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

img0_files = sorted(glob.glob(os.path.join(BASEDIR, 'image_0', '*.png')))
img1_files = sorted(glob.glob(os.path.join(BASEDIR, 'image_1', '*.png')))
assert img0_files and img1_files

gt_poses = np.loadtxt(GT_POSES_FILE).reshape(-1, 3, 4)
gt_xyz   = gt_poses[:, :, 3]

print(f"[INIT] baseline={baseline:.4f}m  "
      f"det={'GPU' if USE_GPU_DETECTOR else 'CPU'}  "
      f"sgbm={'GPU' if USE_GPU_SGBM else 'CPU'}  "
      f"lk={'GPU' if USE_GPU_LK else 'CPU'}")

# ═══ THE TWO WINDOWS ═══════════════════════════════════════════════════════
cv2.namedWindow("KITTI Stereo Tracking", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Stereo Trajectory Map", cv2.WINDOW_AUTOSIZE)


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE INIT
# ═══════════════════════════════════════════════════════════════════════════
gpu_detector = gpu_sgbm = gpu_lk = cpu_sgbm = lk_criteria = None

if USE_GPU_DETECTOR:
    gpu_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
        srcType=cv2.CV_8UC1, maxCorners=1500, qualityLevel=0.01, minDistance=10)

if USE_GPU_SGBM:
    gpu_sgbm = cv2.cuda.createStereoSGM(
        minDisparity=0, numDisparities=NUM_DISP,
        P1=P1_SGBM, P2=P2_SGBM, uniquenessRatio=UNIQUENESS)
else:
    cpu_sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK_SIZE,
        P1=P1_SGBM, P2=P2_SGBM, disp12MaxDiff=1, uniquenessRatio=UNIQUENESS,
        speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

if USE_GPU_LK:
    gpu_lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
        winSize=LK_WIN, maxLevel=LK_LEVELS, iters=LK_ITERS, useInitialFlow=False)
else:
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, LK_ITERS, LK_EPS)


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED STAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def detect(gray_cpu, gpu_gray):
    if USE_GPU_DETECTOR and gpu_gray is not None:
        g = gpu_detector.detect(gpu_gray)
        return None if g.empty() else g.download().reshape(-1, 2).astype(np.float32)
    p = cv2.goodFeaturesToTrack(gray_cpu, maxCorners=1500, qualityLevel=0.01, minDistance=10)
    return None if p is None else p.reshape(-1, 2).astype(np.float32)


def stereo(prev_l_cpu, prev_r_cpu, gpu_prev_l, gpu_prev_r):
    if USE_GPU_SGBM:
        return gpu_sgbm.compute(gpu_prev_l, gpu_prev_r).download().astype(np.float32) / 16.0
    return cpu_sgbm.compute(prev_l_cpu, prev_r_cpu).astype(np.float32) / 16.0


def lk(prev_l_cpu, curr_l_cpu, prev_pts, gpu_prev_l, gpu_curr_l):
    if USE_GPU_LK:
        n = len(prev_pts)
        gp = cv2.cuda_GpuMat(); gp.upload(prev_pts.reshape(1, -1, 2).astype(np.float32))
        gn = cv2.cuda_GpuMat(); gn.upload(np.zeros((1, n, 2), np.float32))
        gs = cv2.cuda_GpuMat(); gs.upload(np.zeros((1, n), np.uint8))
        ge = cv2.cuda_GpuMat()
        gpu_lk.calc(gpu_prev_l, gpu_curr_l, gp, gn, gs, ge)
        return gn.download().reshape(-1, 2).astype(np.float32), gs.download().ravel() == 1
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_l_cpu, curr_l_cpu, prev_pts.reshape(-1, 1, 2).astype(np.float32),
        None, winSize=LK_WIN, maxLevel=LK_LEVELS, criteria=lk_criteria)
    return nxt.reshape(-1, 2).astype(np.float32), st.ravel() == 1


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════
prev_l_cpu = cv2.imread(img0_files[0], cv2.IMREAD_GRAYSCALE)
prev_r_cpu = cv2.imread(img1_files[0], cv2.IMREAD_GRAYSCALE)
h, w = prev_l_cpu.shape

gpu_prev_l = gpu_prev_r = None
if ANY_GPU:
    gpu_prev_l = cv2.cuda_GpuMat(); gpu_prev_l.upload(prev_l_cpu)
    gpu_prev_r = cv2.cuda_GpuMat(); gpu_prev_r.upload(prev_r_cpu)

pts_prev = detect(prev_l_cpu, gpu_prev_l)



# Before loop

FRAME_BUFFER = 16                        # ~16 MB of grayscale images — negligible
frame_q  = queue.Queue(maxsize=FRAME_BUFFER)
stop_ev  = threading.Event()

# Skip frame 0 (already loaded above the loop)
loader_thread = threading.Thread(
    target=frame_loader,
    args=(img0_files[1:], img1_files[1:], frame_q, stop_ev),
    daemon=True,
)
loader_thread.start()
print(f"[LOADER] background prefetch started, buffer={FRAME_BUFFER} frames")

gt_drawn_upto = 0

# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════
R_global = np.eye(3); t_global = np.zeros((3, 1))
traj_log = [np.hstack([R_global, t_global]).ravel()]

# ═══ THE TRAJECTORY MAP CANVAS ═════════════════════════════════════════════
map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)

STAGES = ['read', 'upload', 'detect', 'stereo', 'lk', 'pnp', 'merge', 'render']
stage_times = {k: [] for k in STAGES}
total_times = []

try:
    for frame_idx in range(1, len(img0_files)):
        t_frame = time.perf_counter()

        t = time.perf_counter()
        try:
            curr_l_cpu, curr_r_cpu = frame_q.get(timeout=5.0)
        except queue.Empty:
            print(f"[WARN] frame {frame_idx}: loader timeout, falling back to sync read")
            curr_l_cpu = cv2.imread(img0_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            curr_r_cpu = cv2.imread(img1_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        stage_times['read'].append(time.perf_counter() - t)
        if curr_l_cpu is None or curr_r_cpu is None:
            continue

        t = time.perf_counter()
        gpu_curr_l = gpu_curr_r = None
        if ANY_GPU:
            gpu_curr_l = cv2.cuda_GpuMat(); gpu_curr_l.upload(curr_l_cpu)
            gpu_curr_r = cv2.cuda_GpuMat(); gpu_curr_r.upload(curr_r_cpu)
        stage_times['upload'].append(time.perf_counter() - t)

        t = time.perf_counter()
        if pts_prev is None or len(pts_prev) < POOL_MIN_POINTS:
            pts_prev = detect(prev_l_cpu, gpu_prev_l)
            if pts_prev is None:
                prev_l_cpu, prev_r_cpu = curr_l_cpu, curr_r_cpu
                gpu_prev_l, gpu_prev_r = gpu_curr_l, gpu_curr_r
                stage_times['detect'].append(time.perf_counter() - t)
                continue
        stage_times['detect'].append(time.perf_counter() - t)

        t = time.perf_counter()
        disp_float = stereo(prev_l_cpu, prev_r_cpu, gpu_prev_l, gpu_prev_r)
        stage_times['stereo'].append(time.perf_counter() - t)

        if SHOW_DISP:
            v = disp_float[(disp_float > 0.5) & (disp_float < NUM_DISP - 1)]
            if v.size: print(f"[DISP] mean={v.mean():.1f} valid={100*v.size/disp_float.size:.0f}%")

        t = time.perf_counter()
        next_pts, st = lk(prev_l_cpu, curr_l_cpu, pts_prev, gpu_prev_l, gpu_curr_l)
        stage_times['lk'].append(time.perf_counter() - t)

        n = min(len(pts_prev), len(next_pts), len(st))
        good_prev = pts_prev[:n][st[:n]]
        good_curr = next_pts[:n][st[:n]]

        t = time.perf_counter()
        inlier_mask = np.zeros(len(good_prev), dtype=bool)
        valid_idx = np.array([], dtype=np.int32)
        if len(good_prev) > 15:
            P3, P2, idx = [], [], []
            for i in range(len(good_prev)):
                x, y = int(round(good_prev[i, 0])), int(round(good_prev[i, 1]))
                if not (0 <= x < w and 0 <= y < h): continue
                d = float(disp_float[y, x])
                if d <= 0.5 or d >= NUM_DISP - 1: continue
                Z = fx * baseline / d
                if Z <= Z_MIN or Z > Z_MAX: continue
                P3.append([(x - cx) * Z / fx, (y - cy) * Z / fy, Z])
                P2.append(good_curr[i]); idx.append(i)
            if len(P3) > 6:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    np.asarray(P3, np.float32), np.asarray(P2, np.float32), K,
                    distCoeffs=None, iterationsCount=RANSAC_ITERS,
                    reprojectionError=RANSAC_REPROJ, confidence=RANSAC_CONF,
                    flags=RANSAC_FLAGS)
                if ok and inl is not None:
                    idx_arr = np.asarray(idx, np.int32)
                    valid_idx = idx_arr[inl.ravel()]
                    inlier_mask[valid_idx] = True
                    Rm, _ = cv2.Rodrigues(rvec); Rd = Rm.T; td = -Rm.T.dot(tvec)
                    t_global = t_global + R_global.dot(td); R_global = R_global.dot(Rd)
                    if USE_Y_CLAMP and abs(t_global[1, 0]) > 3.0: t_global[1, 0] *= 0.98
                    traj_log.append(np.hstack([R_global, t_global]).ravel())
        stage_times['pnp'].append(time.perf_counter() - t)

        t = time.perf_counter()
        if inlier_mask.sum() > 10:   kept = good_curr[inlier_mask].astype(np.float32)
        elif len(valid_idx) > 0:     kept = good_curr[valid_idx].astype(np.float32)
        else:                        kept = good_curr.astype(np.float32)

        fresh = detect(curr_l_cpu, gpu_curr_l)
        if kept.size and fresh is not None and len(fresh) > 0:
            d, _ = cKDTree(kept).query(fresh, k=1)
            merged = np.vstack([kept, fresh[d > MERGE_THRESHOLD]])
        elif kept.size:      merged = kept
        elif fresh is not None and len(fresh) > 0:  merged = fresh
        else:                merged = np.zeros((0, 2), np.float32)
        if len(merged) < POOL_MIN_POINTS:
            fb = detect(curr_l_cpu, gpu_curr_l)
            if fb is not None: merged = fb
        stage_times['merge'].append(time.perf_counter() - t)

        pts_prev = merged
        prev_l_cpu, prev_r_cpu = curr_l_cpu, curr_r_cpu
        gpu_prev_l, gpu_prev_r = gpu_curr_l, gpu_curr_r
        if SHOW_POOL: print(f"[POOL] kept={len(kept)} merged={len(merged)}")

        # ═══ RENDER (both windows) ═════════════════════════════════════════
        t = time.perf_counter()
        if frame_idx % RENDER_EVERY_N == 0:
            # ── Stereo tracking view ──
            side = np.hstack((curr_l_cpu, curr_r_cpu))
            img = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
            for pt in good_curr:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            cv2.imshow("KITTI Stereo Tracking", img)

            # ── Trajectory map ──
            t_flat = t_global.ravel()
            x_coord, z_coord = float(t_flat[0]), float(t_flat[2])
            cv2.circle(map_canvas,
                       (int(-x_coord * 0.5) + 300, int(z_coord * 0.5) + 100),
                       2, (0, 165, 255), -1)   # orange VO dot
            for g in range(max(1, gt_drawn_upto + 1), min(frame_idx + 1, len(gt_xyz))):
                cv2.line(map_canvas,
                    (int(-gt_xyz[g-1, 0] * 0.5) + 300, int(gt_xyz[g-1, 2] * 0.5) + 100),
                    (int(-gt_xyz[g,   0] * 0.5) + 300, int(gt_xyz[g,   2] * 0.5) + 100),
                    (0, 0, 255), 1)              # red GT line
            cv2.imshow("Stereo Trajectory Map", map_canvas)
            gt_drawn_upto = frame_idx

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[EXIT] 'q'")
                break
        stage_times['render'].append(time.perf_counter() - t)
        total_times.append(time.perf_counter() - t_frame)

        if MAX_FRAMES and frame_idx >= MAX_FRAMES: break

finally:
    stop_ev.set()                    # ← NEW: tell loader to exit
    if traj_log:
        np.savetxt("est_traj.txt", np.array(traj_log))
        print(f"[SAVE] {len(traj_log)} rows → est_traj.txt")
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# PROFILING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
if SHOW_TIMING and total_times:
    mode = (f"det={'GPU' if USE_GPU_DETECTOR else 'CPU'}"
            f" | sgbm={'GPU' if USE_GPU_SGBM else 'CPU'}"
            f" | lk={'GPU' if USE_GPU_LK else 'CPU'}")
    print()
    print("═" * 72)
    print(f"PROFILE  {mode}  —  {len(total_times)} frames")
    print("─" * 72)
    print(f"  {'stage':<9}  {'mean':>7}  {'min':>7}  {'max':>7}  {'total':>8}")
    for name in STAGES:
        arr = np.array(stage_times[name]) * 1000.0
        if arr.size == 0: continue
        print(f"  {name:<9}  {arr.mean():6.2f}  {arr.min():6.2f}  {arr.max():6.2f}  {arr.sum()/1000:6.1f}s")
    tot = np.array(total_times) * 1000.0
    print("─" * 72)
    print(f"  {'TOTAL':<9}  {tot.mean():6.2f}  {tot.min():6.2f}  {tot.max():6.2f}  {tot.sum()/1000:6.1f}s")
    print(f"  Throughput: {1000/tot.mean():.1f} fps")
    print("═" * 72)