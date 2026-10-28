import os
import glob
import cv2
import numpy as np
import time

# =========================================================================
# 1. PARSING & STEREO UTILITIES
# =========================================================================
def load_kitti_stereo_calib(calib_file_path):
    print(f"[DEBUG CALIB] Attempting to open: {calib_file_path}")
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
print(f"[DEBUG INIT] Found {len(img0_files)} frames in image_0, {len(img1_files)} frames in image_1")

if len(img0_files) == 0 or len(img1_files) == 0:
    raise FileNotFoundError("Could not find sequential frames inside image_0 or image_1.")

K, baseline = load_kitti_stereo_calib(calib_file)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print("[DEBUG INIT] Initializing Windows and GPU Device Engine instances...")
cv2.namedWindow("KITTI Stereo Tracking Stream", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Stereo Trajectory Map", cv2.WINDOW_AUTOSIZE)

# =========================================================================
# 2. INITIALIZATION SETUP (UPDATED TRACKER FLAGS)
# =========================================================================
gpu_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
    srcType=cv2.CV_8UC1, maxCorners=1500, qualityLevel=0.01, minDistance=10
)

# FIX: We must use winSize=(21,21), maxLevel=3, and explicitly define 30 iterations.
# Crucially, we do NOT use an initial flow guess array for the TEMPORAL loop,
# so we create TWO separate trackers with distinct hardware behaviors.
gpu_lk_stereo = cv2.cuda.SparsePyrLKOpticalFlow_create(
    winSize=(21, 21), maxLevel=3, iters=30, useInitialFlow=True
)

gpu_lk_motion = cv2.cuda.SparsePyrLKOpticalFlow_create(
    winSize=(21, 21), maxLevel=3, iters=30, useInitialFlow=False
)


R_global = np.eye(3, dtype=np.float64)
t_global = np.zeros((3, 1), dtype=np.float64)
map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)

def count_points(gpu_pts):
    if gpu_pts is None or gpu_pts.empty():
        return 0
    return int(np.prod(gpu_pts.size()))  # works for both (1,N,2) and (N,1,2)

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

# =========================================================================
# 3. MAIN CUDA LOOP WITH RUNTIME TIMERS
# =========================================================================
for frame_idx in range(1, len(img0_files)):
    print(f"\n⚡ [FRAME {frame_idx:04d} / {len(img0_files)}] starting cycle...")
    
    t_frame_start = time.perf_counter()
    
    # 1. IO Read checks
    curr_l_cpu = cv2.imread(img0_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    curr_r_cpu = cv2.imread(img1_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    if curr_l_cpu is None or curr_r_cpu is None:
        print(f" -> ERROR: Failed to load disk files for index {frame_idx}. Skipping.")
        continue

    # 2. VRAM Upload checks
    t_step = time.perf_counter()
    gpu_curr_l = cv2.cuda_GpuMat()
    gpu_curr_r = cv2.cuda_GpuMat()
    gpu_curr_l.upload(curr_l_cpu)
    gpu_curr_r.upload(curr_r_cpu)
    print(f" -> [STEP 1] VRAM Upload finished in {(time.perf_counter() - t_step)*1000:.2f}ms")

    # 3. Features Capacity Pool check
    if count_points(gpu_pts_prev_l) < 500:
        current_count = count_points(gpu_pts_prev_l)
        print(f" -> [POOL CHECK] Target count low ({current_count} corners). Replenishing reservoir...")
        gpu_pts_prev_l = get_clean_gpu_pts(gpu_prev_l)
        if gpu_pts_prev_l is None:
            print(" -> [POOL CHECK] Cannot extract fresh features. Skipping frame.")
            gpu_prev_l = gpu_curr_l
            gpu_prev_r = gpu_curr_r
            continue

    # =========================================================================
    # 4. CUDA Optical Flow Calculations
    # =========================================================================
    t_step = time.perf_counter()
    
    # Download points to CPU momentarily to provide a smart baseline guess for the right camera
    pts_cpu_guess = gpu_pts_prev_l.download()          # (1, N, 2)
    n_pts = pts_cpu_guess.shape[1]

    # Stereo tracker: initial guess is the shifted points (useInitialFlow=True)
    pts_right_guess = pts_cpu_guess.copy()
    pts_right_guess[0, :, 0] -= 30.0
    gpu_pts_prev_r = cv2.cuda_GpuMat()
    gpu_pts_prev_r.upload(pts_right_guess)

    # Pre-allocate outputs — this is the key fix for the motion tracker
    gpu_pts_curr_l = cv2.cuda_GpuMat()
    gpu_pts_curr_l.upload(np.zeros_like(pts_cpu_guess))      # same (1, N, 2) shape

    gpu_status_stereo = cv2.cuda_GpuMat()
    gpu_status_stereo.upload(np.zeros((1, n_pts), dtype=np.uint8))

    gpu_status_motion = cv2.cuda_GpuMat()
    gpu_status_motion.upload(np.zeros((1, n_pts), dtype=np.uint8))

    gpu_err_stereo = cv2.cuda_GpuMat()
    gpu_err_motion = cv2.cuda_GpuMat()

    # Stereo: use initial flow (large disparity needs the guess)
    gpu_lk_stereo.calc(gpu_prev_l, gpu_prev_r, gpu_pts_prev_l,
                    gpu_pts_prev_r, gpu_status_stereo, gpu_err_stereo)

    print(f"prevPts shape before motion calc: {gpu_pts_prev_l.size()}, type: {gpu_pts_prev_l.type()}")

    # Motion: no initial flow, but nextPts is pre-allocated so calc populates it
    gpu_lk_motion.calc(gpu_prev_l, gpu_curr_l, gpu_pts_prev_l,
                    gpu_pts_curr_l, gpu_status_motion, gpu_err_motion)
    # Before gpu_lk_motion.calc(...)
    # Expected output: prevPts shape before motion calc: (1, 633), type: 13
    print(f" -> [FLOW COMPLETE] GPU calculations finished in {(time.perf_counter() - t_step):.4f}s")


        # 5. Pipeline Downloads
    t_step = time.perf_counter()
    pts_prev_l_raw = gpu_pts_prev_l.download()
    pts_prev_r_raw = gpu_pts_prev_r.download()
    pts_curr_l_raw = gpu_pts_curr_l.download()
    
    status_stereo_raw = gpu_status_stereo.download()
    status_motion_raw = gpu_status_motion.download()
    print(f" -> [STEP 4] Matrix memory block downloads finished in {(time.perf_counter() - t_step)*1000:.2f}ms")

    # -------------------------------------------------------------------------
    # 6. Null Matrix validations (DEFENSIVELY DEFENDED SHAPE CHECK)
    # -------------------------------------------------------------------------
    if (pts_prev_l_raw is None or pts_prev_r_raw is None or pts_curr_l_raw is None or
        status_stereo_raw is None or status_motion_raw is None or
        pts_prev_l_raw.size == 0 or pts_prev_r_raw.size == 0 or pts_curr_l_raw.size == 0):
        
        print(" -> [VALIDATION] Empty tracking arrays dropped out. Recovering hardware contexts...")
        
        # Display rendering baseline update step so your windows stay active
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
    pts_prev_r = pts_prev_r_raw.reshape(-1, 2)
    pts_curr_l = pts_curr_l_raw.reshape(-1, 2)
    
    # Extract tracking masks safely 
    mask_stereo = status_stereo_raw.ravel() == 1
    mask_motion = status_motion_raw.ravel() == 1
    
    # Verify tracking array array lengths match up completely
    min_len = min(len(pts_prev_l), len(pts_prev_r), len(pts_curr_l), len(mask_stereo), len(mask_motion))
    
    valid_mask = mask_stereo[:min_len] & mask_motion[:min_len]
    good_prev_l = pts_prev_l[:min_len][valid_mask]
    good_prev_r = pts_prev_r[:min_len][valid_mask]
    good_curr_l = pts_curr_l[:min_len][valid_mask]

    print(f" -> [TRACKS] Successfully matched {len(good_curr_l)} tracking vectors across time and sensors.")


    # 8. Essential 3D Geometry mapping and PnP Pose recovery
    ransac_status = np.zeros(len(good_prev_l))
    if len(good_prev_l) > 15:
        t_step = time.perf_counter()
        points_3d = []
        valid_indices = []
        
        for i in range(len(good_prev_l)):
            disparity = good_prev_l[i, 0] - good_prev_r[i, 0]
            if disparity > 0.5:
                Z = (fx * baseline) / disparity
                X = (good_prev_l[i, 0] - cx) * Z / fx
                Y = (good_prev_l[i, 1] - cy) * Z / fy
                points_3d.append([X, Y, Z])
                valid_indices.append(i)

        points_3d = np.array(points_3d, dtype=np.float32)
        curr_2d_tracked = good_curr_l[valid_indices]

        if len(points_3d) > 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, curr_2d_tracked, K, distCoeffs=None, 
                iterationsCount=100, reprojectionError=2.0, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success and inliers is not None:
                ransac_status[inliers.ravel()] = 1
                R_matrix, _ = cv2.Rodrigues(rvec)
                R_delta = R_matrix.T
                t_delta = -R_matrix.T.dot(tvec)

                t_global = t_global + R_global.dot(t_delta)
                R_global = R_global.dot(R_delta)
        print(f" -> [POSE] PnP RANSAC geometric estimation completed in {(time.perf_counter() - t_step)*1000:.2f}ms")

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
    info_canvas = map_canvas.copy()
    cv2.putText(info_canvas, f"CUDA Global X: {x_coord:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(info_canvas, f"CUDA Global Z: {z_coord:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Stereo Trajectory Map", info_canvas)
    print(f" -> [RENDER] UI window pixel paints handled in {(time.perf_counter() - t_step)*1000:.2f}ms")
    
    # 10. Historic State Updates
    gpu_prev_l = gpu_curr_l
    gpu_prev_r = gpu_curr_r
    
    if ransac_status.sum() > 10:
        next_pts_cpu = good_curr_l[ransac_status == 1].astype(np.float32).reshape(1, -1, 2)
    else:
        next_pts_cpu = good_curr_l.astype(np.float32).reshape(1, -1, 2)
        
    gpu_pts_prev_l = cv2.cuda_GpuMat()
    gpu_pts_prev_l.upload(next_pts_cpu)
    print(f" -> [CYCLE COMPLETE] Total frame lifecycle: {(time.perf_counter() - t_frame_start)*1000:.2f}ms")
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("[EXIT] 'q' key pressed. Stopping playback.")
        break

cv2.destroyAllWindows()
