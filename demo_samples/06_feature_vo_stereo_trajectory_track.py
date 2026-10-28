import os
import glob
import cv2
import numpy as np


# =========================================================================
# 1. PARSING & STEREO UTILITIES
# =========================================================================
def load_kitti_stereo_calib(calib_file_path):
    """
    Parses calib.txt to extract the Intrinsic Matrix K and the baseline distance (b).
    For KITTI, baseline is derived from the horizontal shift in projection matrix P1.
    """
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
    # Baseline (b) in meters = absolute horizontal translation shift / focal length
    focal_length = P0[0, 0]
    baseline = abs(P1[0, 3] - P0[0, 3]) / focal_length
    
    return K, baseline


# =========================================================================
# 2. FILE PATH AND INITIALIZATION SETUP
# =========================================================================
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00'
image0_dir = os.path.join(basedir, 'image_0')
image1_dir = os.path.join(basedir, 'image_1')
calib_file = os.path.join(basedir, 'calib.txt')

img0_files = sorted(glob.glob(os.path.join(image0_dir, '*.png')))
img1_files = sorted(glob.glob(os.path.join(image1_dir, '*.png')))

if len(img0_files) == 0 or len(img1_files) == 0:
    raise FileNotFoundError("Could not find sequential frames inside image_0 or image_1.")

# Load calibration
K, baseline = load_kitti_stereo_calib(calib_file)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"Successfully loaded Stereo Config.\nBaseline: {baseline:.4f} meters\nIntrinsic Matrix K:\n{K}")

# Tracking parameters
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
detector = cv2.GFTTDetector_create(maxCorners=1500, qualityLevel=0.01, minDistance=10)

# Global states initialization
R_global = np.eye(3, dtype=np.float64)
t_global = np.zeros((3, 1), dtype=np.float64)
map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)

# Bootstrapping with initial frame 0
prev_l = cv2.imread(img0_files[0], cv2.IMREAD_GRAYSCALE)
prev_r = cv2.imread(img1_files[0], cv2.IMREAD_GRAYSCALE)
h, w = prev_l.shape

# Initial tracking features on Left Camera
kp_prev_l = detector.detect(prev_l, None)
pts_prev_l = np.array([kp.pt for kp in kp_prev_l], dtype=np.float32).reshape(-1, 1, 2)

print("\nProcessing Stereo Pipelines. Press 'q' to stop.")

# =========================================================================
# 3. MAIN LOOP
# =========================================================================
for frame_idx in range(1, len(img0_files)):
    curr_l = cv2.imread(img0_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    curr_r = cv2.imread(img1_files[frame_idx], cv2.IMREAD_GRAYSCALE)
    if curr_l is None or curr_r is None: continue

    # 1. Feature Replenishment target check
    if len(pts_prev_l) < 800:
        new_kp = detector.detect(prev_l, None)
        if new_kp:
            pts_prev_l = np.array([kp.pt for kp in new_kp], dtype=np.float32).reshape(-1, 1, 2)

    # 2. Track spatially (Left frame to Right frame) to find disparity depth targets
    pts_prev_r, status_stereo, _ = cv2.calcOpticalFlowPyrLK(prev_l, prev_r, pts_prev_l, None, **lk_params)
    
    # 3. Track temporally (Past Left frame to Current Left frame) to find motion matching
    pts_curr_l, status_motion, _ = cv2.calcOpticalFlowPyrLK(prev_l, curr_l, pts_prev_l, None, **lk_params)

    # Filter out features that failed either stereo alignment or forward motion tracking
    valid_mask = ((status_stereo == 1) & (status_motion == 1)).reshape(-1)
    
    good_prev_l = pts_prev_l[valid_mask].reshape(-1, 2)
    good_prev_r = pts_prev_r[valid_mask].reshape(-1, 2)
    good_curr_l = pts_curr_l[valid_mask].reshape(-1, 2)

    if len(good_prev_l) > 15:
        # 4. Triangulate 3D point cloud landmarks relative to the previous frame position
        points_3d = []
        valid_indices = []
        
        for i in range(len(good_prev_l)):
            disparity = good_prev_l[i, 0] - good_prev_r[i, 0]
            # Safety checks to prevent divide-by-zero on negative or tiny disparities
            if disparity > 0.5: 
                Z = (fx * baseline) / disparity
                X = (good_prev_l[i, 0] - cx) * Z / fx
                Y = (good_prev_l[i, 1] - cy) * Z / fy
                points_3d.append([X, Y, Z])
                valid_indices.append(i)

        points_3d = np.array(points_3d, dtype=np.float32)
        curr_2d_tracked = good_curr_l[valid_indices]

        # 5. Solve Perspective-n-Point (PnP) using RANSAC to get absolute physical scale pose step
        if len(points_3d) > 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, curr_2d_tracked, K, distCoeffs=None, 
                iterationsCount=100, reprojectionError=2.0, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success and inliers is not None:
                # convert rotation vector to 3x3 matrix layout format
                R_matrix, _ = cv2.Rodrigues(rvec)
                
                # Invert transformations back to transform from camera coordinate tracking maps
                R_delta = R_matrix.T
                t_delta = -R_matrix.T.dot(tvec)

                # Accumulate actual scale metrics natively!
                t_global = t_global + R_global.dot(t_delta)
                R_global = R_global.dot(R_delta)

    # --- RENDER DUAL LIVE CAMERA STREAMS (Left Cam & Right Cam) ---
    side_by_side = np.hstack((curr_l, curr_r))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
    
    # Draw simple visual dots over active tracking targets on left screen window view
    for pt in good_curr_l:
        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

    cv2.putText(display_frame, "LEFT CAMERA (image_0)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, "RIGHT CAMERA (image_1)", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("KITTI Stereo Tracking Stream", display_frame)

    # --- RENDER SCALE-ACCURATE POSITION MAP ---
    # --- RENDER SCALE-ACCURATE POSITION MAP ---
    # .ravel() completely flattens the array into a clean 1D list [X, Y, Z]
    t_flat = t_global.ravel()
    x_coord = float(t_flat[0])
    z_coord = float(t_flat[2])

    # FIX: Add a negative sign (-0.5) to flip the mirrored horizontal trajectory
    draw_x = int(-x_coord * 0.5) + 300
    draw_y = int(z_coord * 0.5) + 100
    
    cv2.circle(map_canvas, (draw_x, draw_y), 2, (0, 165, 255), -1) # Orange path
    info_canvas = map_canvas.copy()
    
    # Use the clean scalar variables here
    cv2.putText(info_canvas, f"Stereo Global X: {x_coord:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(info_canvas, f"Stereo Global Z: {z_coord:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Stereo Trajectory Map", info_canvas)


    # Handle update history frames states
    prev_l = curr_l.copy()
    prev_r = curr_r.copy()
    pts_prev_l = good_curr_l.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
