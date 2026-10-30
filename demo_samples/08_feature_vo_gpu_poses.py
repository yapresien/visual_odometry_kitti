import os
import glob
import cv2
import numpy as np
import time

# =========================================================================
# 1. PARSING & STEREO UTILITIES
# =========================================================================
def load_kitti_stereo_calib(calib_file_path):
    #print(f"[DEBUG CALIB] Attempting to open: {calib_file_path}")
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"Calibration file not found at: {calib_file_path}")
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
    P0, P1 = None, None
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0: continue
        if parts[0].startswith('P0'):
            P0 = np.array([float(x) for x in parts[1:13]], dtype=np.float32).reshape(3, 4)
        elif parts[0].startswith('P1'):
            P1 = np.array([float(x) for x in parts[1:13]], dtype=np.float32).reshape(3, 4)
    if P0 is None or P1 is None:
        raise ValueError("Could not find projection matrices P0 and P1 in calib file.")
    K = P0[:, :3]
    focal_length = P0[0, 0]
    baseline = abs(P1[0, 3] - P0[0, 3]) / focal_length
    print(f"[DEBUG CALIB] Baseline calculated: {baseline:.4f} meters")
    return K, baseline

# =========================================================================
# 2. PATHS & INITIALIZATION SETUP
# =========================================================================
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00'
image0_dir = os.path.join(basedir, 'image_0')
image1_dir = os.path.join(basedir, 'image_1')
calib_file = os.path.join(basedir, 'calib.txt')

print("[DEBUG INIT] Scanning image directories...")
img0_files = sorted(glob.glob(os.path.join(image0_dir, '*.png')))
img1_files = sorted(glob.glob(os.path.join(image1_dir, '*.png')))
#print(f"[DEBUG INIT] Found {len(img0_files)} frames in image_0, {len(img1_files)} frames in image_1")

if len(img0_files) == 0 or len(img1_files) == 0:
    raise FileNotFoundError("Could not find sequential frames inside image_0 or image_1.")

K, baseline = load_kitti_stereo_calib(calib_file)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print("[DEBUG INIT] Initializing Windows and GPU Device Engine instances...")
cv2.namedWindow("KITTI Stereo Tracking Stream", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Stereo Trajectory Map", cv2.WINDOW_AUTOSIZE)

gt_poses_file = '/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/00.txt'
gt_poses = np.loadtxt(gt_poses_file).reshape(-1, 3, 4)
gt_xyz = gt_poses[:, :, 3]
print(f"[GT] loaded {len(gt_poses)} poses, "
      f"step0={np.linalg.norm(gt_xyz[1]-gt_xyz[0]):.3f}m, "
      f"path={np.linalg.norm(gt_xyz[-1]-gt_xyz[0]):.1f}m")

# =========================================================================
# 2. INITIALIZATION SETUP (UPDATED TRACKER FLAGS)
# =========================================================================
gpu_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
    srcType=cv2.CV_8UC1, maxCorners=1500, qualityLevel=0.01, minDistance=10
)

# FIX: We must use winSize=(21,21), maxLevel=3, and explicitly define 30 iterations.
NUM_DISP = 128  # must be divisible by 16; KITTI 00 max disparity is ~120

gpu_sgbm = cv2.cuda.createStereoSGM(
    minDisparity=0,
    numDisparities=NUM_DISP,
    P1=8 * 1 * 5 * 5,       # 600
    P2=400,      # 2400
    uniquenessRatio=3,
)

gpu_lk_motion = cv2.cuda.SparsePyrLKOpticalFlow_create(
    winSize=(21, 21), maxLevel=4, iters=40, useInitialFlow=False
)


R_global = np.eye(3, dtype=np.float64)
t_global = np.zeros((3, 1), dtype=np.float64)
map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)

def count_points(gpu_pts):
    if gpu_pts is None or gpu_pts.empty():
        return 0
    return int(np.prod(gpu_pts.size()))  # works for both (1,N,2) and (N,1,2)

def count_pts(g):
    return 0 if g is None or g.empty() else int(np.prod(g.size()))
# -------------------------------------------------------------------------
# DEFENSIVE HELPER WITH VERBOSE DEBUG METRICS
# -------------------------------------------------------------------------
def get_clean_gpu_pts(gpu_img):
    raw_gpu_pts = gpu_detector.detect(gpu_img)
    if raw_gpu_pts.empty():
        return None
    pts_cpu = raw_gpu_pts.download()
    if pts_cpu is None or len(pts_cpu) == 0:
        return None
    
    # --- KEY FIX: Reshape to a single row (1, N, 2) and ensure CV_32FC2 type ---
    pts_clean = pts_cpu.reshape(1, -1, 2).astype(np.float32)
    
    gpu_pts = cv2.cuda_GpuMat()
    gpu_pts.upload(pts_clean)
    return gpu_pts

# Bootstrapping Frame 0
print("[DEBUG INIT] Loading Frame 0 files into system memory...")
prev_l_cpu = cv2.imread(img0_files[0], cv2.IMREAD_GRAYSCALE)
prev_r_cpu = cv2.imread(img1_files[0], cv2.IMREAD_GRAYSCALE)
h, w = prev_l_cpu.shape

print("[DEBUG INIT] Uploading Frame 0 matrices to VRAM...")
gpu_prev_l = cv2.cuda_GpuMat()
gpu_prev_r = cv2.cuda_GpuMat()
gpu_prev_l.upload(prev_l_cpu)
gpu_prev_r.upload(prev_r_cpu)

print("[DEBUG INIT] Extracting first features wave...")
gpu_pts_prev_l = get_clean_gpu_pts(gpu_prev_l)

print("\n[START] Entering processing tracking loop. Press 'q' to stop.")


# before the for loop
traj_log = []

# after R_global = R_global.dot(R_delta) inside the pose block
traj_log.append(np.hstack([R_global, t_global]).ravel())

# =========================================================================
# 3. MAIN CUDA LOOP WITH RUNTIME TIMERS
# =========================================================================
for frame_idx in range(1, len(img0_files)):
    #print(f"\n⚡ [FRAME {frame_idx:04d} / {len(img0_files)}] starting cycle...")
    
    t_frame_start = time.perf_counter()
    
    # 1. IO Read checks
    curr_l_cpu = cv2.imread(img0_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    curr_r_cpu = cv2.imread(img1_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    if curr_l_cpu is None or curr_r_cpu is None:
        #print(f" -> ERROR: Failed to load disk files for index {frame_idx}. Skipping.")
        continue

    # 2. VRAM Upload checks
    t_step = time.perf_counter()
    gpu_curr_l = cv2.cuda_GpuMat()
    gpu_curr_r = cv2.cuda_GpuMat()
    gpu_curr_l.upload(curr_l_cpu)
    gpu_curr_r.upload(curr_r_cpu)
    #print(f" -> [STEP 1] VRAM Upload finished in {(time.perf_counter() - t_step)*1000:.2f}ms")

    # 3. Features Capacity Pool check
    if count_pts(gpu_pts_prev_l) < 100:      # only as a hard failsafe now
        gpu_pts_prev_l = get_clean_gpu_pts(gpu_prev_l)
        if gpu_pts_prev_l is None:
            gpu_prev_l, gpu_prev_r = gpu_curr_l, gpu_curr_r
            continue

    # =========================================================================
    # 4. CUDA Optical Flow Calculations
    # =========================================================================
    t_step = time.perf_counter()

    # STEP 2: SGBM dense disparity (CPU→GPU→CPU). Output is CV_16S, Q4 fixed point.
    gpu_disparity = gpu_sgbm.compute(gpu_prev_l, gpu_prev_r)

    # STEP 3: temporal LK, unchanged
    #print(f" -> [STEP 3] Launching CUDA Temporal Motion Flow...")
    #print(f"    prevPts shape before motion calc: {gpu_pts_prev_l.size()}, type: {gpu_pts_prev_l.type()}")
    gpu_pts_curr_l = cv2.cuda_GpuMat()
    gpu_pts_curr_l.upload(np.zeros_like(gpu_pts_prev_l.download()))   # pre-alloc (1,N,2)
    gpu_status_motion = cv2.cuda_GpuMat()
    gpu_status_motion.upload(np.zeros((1, gpu_pts_prev_l.download().shape[1]), dtype=np.uint8))
    gpu_err_motion = cv2.cuda_GpuMat()
    gpu_lk_motion.calc(gpu_prev_l, gpu_curr_l, gpu_pts_prev_l,
                    gpu_pts_curr_l, gpu_status_motion, gpu_err_motion)
    #print(f" -> [FLOW COMPLETE] GPU calculations finished in {(time.perf_counter() - t_step):.4f}s")

    # 5. Pipeline Downloads
    t_step = time.perf_counter()
    pts_prev_l_raw = gpu_pts_prev_l.download()
    pts_curr_l_raw = gpu_pts_curr_l.download()
    status_motion_raw = gpu_status_motion.download()
    disp_raw = gpu_disparity.download()          # (H, W) CV_16S
    disp_float = disp_raw.astype(np.float32) / 16.0  # real disparity in pixels
    #print(f" -> [STEP 4] Matrix memory block downloads finished in {(time.perf_counter() - t_step)*1000:.2f}ms")
    _v = disp_float[(disp_float > 0.5) & (disp_float < NUM_DISP - 1)]
    if _v.size > 0:
        print(f"[DISP] mean={_v.mean():.1f}px  p50={np.median(_v):.1f}px  "
            f"max={_v.max():.1f}px  valid={100*_v.size/disp_float.size:.0f}%")
            
    # Per depth band valid %
    # for lo, hi in [(0,5),(5,15),(15,40),(40,100),(100,300)]:
    #     mask = (disp_float > 0.5) & (disp_float < NUM_DISP-1)
    #     Z_map = fx * baseline / np.maximum(disp_float, 1e-3)
    #     band = mask & (Z_map >= lo) & (Z_map < hi)
    #     frac = 100 * band.sum() / (disp_float.size)
    #     print(f"  Z∈[{lo:3d},{hi:3d})m: {frac:5.1f}%")
                
    # -------------------------------------------------------------------------
    # 6. Null Matrix validations (DEFENSIVELY DEFENDED SHAPE CHECK)
    # -------------------------------------------------------------------------
    if (pts_prev_l_raw is None or pts_curr_l_raw is None or
        status_motion_raw is None or disp_float is None or
        pts_prev_l_raw.size == 0 or pts_curr_l_raw.size == 0):
        print(" -> [VALIDATION] Empty tracking arrays dropped out. Recovering...")
        side_by_side = np.hstack((curr_l_cpu, curr_r_cpu))
        display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_frame, "SEARCHING FOR SCENE FEATURES...", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("KITTI Stereo Tracking Stream", display_frame)
        cv2.imshow("Stereo Trajectory Map", map_canvas)
        cv2.waitKey(1)
        gpu_pts_prev_l = None
        gpu_prev_l = gpu_curr_l
        gpu_prev_r = gpu_curr_r
        continue

    # -------------------------------------------------------------------------
    # 7. Reshaping arrays safely (Strict Multi-dimensional extraction protection)
    # -------------------------------------------------------------------------
    # Flatten multi-row tensors securely down into flat 2D pixel coordinates (N, 2)
    pts_prev_l = pts_prev_l_raw.reshape(-1, 2)
    pts_curr_l = pts_curr_l_raw.reshape(-1, 2)
    mask_motion = status_motion_raw.ravel() == 1

    min_len = min(len(pts_prev_l), len(pts_curr_l), len(mask_motion))
    valid_mask = mask_motion[:min_len]

    good_prev_l = pts_prev_l[:min_len][valid_mask]
    good_curr_l = pts_curr_l[:min_len][valid_mask]

    #print(f" -> [TRACKS] Motion-flow matched {len(good_curr_l)} / {min_len} points.")


    # 8. Essential 3D Geometry mapping and PnP Pose recovery
    ransac_status = np.zeros(len(good_prev_l))
    if len(good_prev_l) > 15:
        t_step = time.perf_counter()
        points_3d, curr_2d_valid, valid_idx = [], [], []
        
        for i in range(len(good_prev_l)):
            x, y = int(round(good_prev_l[i, 0])), int(round(good_prev_l[i, 1]))
            if not (0 <= x < w and 0 <= y < h):
                continue
            disparity = float(disp_float[y, x])
            if disparity <= 0.5 or disparity >= NUM_DISP - 1:
                continue
            Z = (fx * baseline) / disparity
            if Z <= 3.0 or Z > 200.0:   # drop outlier depths (sky, mismatches)
                continue
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points_3d.append([X, Y, Z])
            curr_2d_valid.append(good_curr_l[i])
            valid_idx.append(i)
        if len(points_3d) > 6:
            points_3d = np.asarray(points_3d, dtype=np.float32)
            curr_2d_valid = np.asarray(curr_2d_valid, dtype=np.float32)
            valid_idx = np.asarray(valid_idx, dtype=np.int32)
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, curr_2d_valid, K, distCoeffs=None,
                iterationsCount=300,        # was 100 — 10× more draws
                reprojectionError=3.5,       # was 2.0 — slightly more lenient
                confidence=0.99,            # 99.9% confidence target
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if success and inliers is not None:
                ransac_status[valid_idx[inliers.ravel()]] = 1
                R_matrix, _ = cv2.Rodrigues(rvec)
                R_delta = R_matrix.T
                t_delta = -R_matrix.T.dot(tvec)
                t_global = t_global + R_global.dot(t_delta)
                R_global = R_global.dot(R_delta)
                traj_log.append(np.hstack([R_global, t_global]).ravel())             # KITTI-interleaved
                
        #print(f" -> [POSE] PnP RANSAC on {len(points_3d)} valid 3D points in {(time.perf_counter()-t_step)*1000:.2f}ms")

    # 9. UI Rendering updates
    t_step = time.perf_counter()
    side_by_side = np.hstack((curr_l_cpu, curr_r_cpu))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
    
    for pt in good_curr_l:
        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        
    cv2.putText(display_frame, "CUDA-Accelerated Stereo Stream", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("KITTI Stereo Tracking Stream", display_frame)
    
    t_flat = t_global.ravel()
    x_coord, z_coord = float(t_flat[0]), float(t_flat[2])
    draw_x = int(-x_coord * 0.5) + 300
    draw_y = int(z_coord * 0.5) + 100
    
    cv2.circle(map_canvas, (draw_x, draw_y), 2, (0, 165, 255), -1)
    
    # GT path (red), drawn up to the current frame
    for g in range(1, min(frame_idx + 1, len(gt_xyz))):
        cv2.line(map_canvas,
                (int(-gt_xyz[g-1, 0] * 0.5) + 300, int(gt_xyz[g-1, 2] * 0.5) + 100),
                (int(-gt_xyz[g,   0] * 0.5) + 300, int(gt_xyz[g,   2] * 0.5) + 100),
                (0, 0, 255), 1)

    
    info_canvas = map_canvas.copy()
    cv2.putText(info_canvas, f"CUDA Global X: {x_coord:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(info_canvas, f"CUDA Global Z: {z_coord:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Stereo Trajectory Map", info_canvas)
    #print(f" -> [RENDER] UI window pixel paints handled in {(time.perf_counter() - t_step)*1000:.2f}ms")
    
    # 10. Historic State Updates
    gpu_prev_l = gpu_curr_l
    gpu_prev_r = gpu_curr_r

    # --- Detect + merge ---
    # 1. Keep RANSAC inliers if available, else all motion-tracked points
    if ransac_status.sum() > 10:
        kept_pts = good_curr_l[ransac_status[:len(good_curr_l)] == 1].astype(np.float32)
    else:
        if 'valid_idx' in locals() and isinstance(valid_idx, np.ndarray) and len(valid_idx) > 0:
            kept_pts = good_curr_l[valid_idx].astype(np.float32)
        else:
            kept_pts = good_curr_l.astype(np.float32)
            
    # 2. Detect fresh corners on the CURRENT left frame
    fresh_gpu = get_clean_gpu_pts(gpu_curr_l)
    fresh_pts = None
    if fresh_gpu is not None:
        fresh_pts = fresh_gpu.download().reshape(-1, 2).astype(np.float32)

    # 3. Merge: keep fresh points farther than 15 px from any kept point
    if kept_pts.size and fresh_pts is not None and len(fresh_pts) > 0:
        diff = fresh_pts[:, None, :] - kept_pts[None, :, :]        # (M, K, 2)
        min_dist = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)    # (M,)
        novel = fresh_pts[min_dist > 15.0]
        merged = np.vstack([kept_pts, novel])
    elif kept_pts.size:
        merged = kept_pts
    elif fresh_pts is not None and len(fresh_pts) > 0:
        merged = fresh_pts
    else:
        merged = np.zeros((0, 2), dtype=np.float32)

    # 4. Failsafe: if the merge collapsed somehow, re-detect
    if len(merged) < 100:
        fallback = get_clean_gpu_pts(gpu_curr_l)
        if fallback is not None:
            merged = fallback.download().reshape(-1, 2).astype(np.float32)

    gpu_pts_prev_l = cv2.cuda_GpuMat()
    gpu_pts_prev_l.upload(merged.reshape(1, -1, 2).astype(np.float32))
    print(f"[POOL] next frame features: kept={len(kept_pts)} fresh={0 if fresh_pts is None else len(fresh_pts)} merged={len(merged)}")
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("[EXIT] 'q' key pressed. Stopping playback.")
        break

cv2.destroyAllWindows()

np.savetxt("est_traj.txt", np.array(traj_log))