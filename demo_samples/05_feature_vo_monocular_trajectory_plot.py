import os
import glob
import cv2
import numpy as np


# =========================================================================
# 1. PARSING HELPER FUNCTIONS
# =========================================================================
def load_kitti_calib(calib_file_path):
    """Parses the KITTI calib.txt file and extracts the 3x3 Intrinsic Matrix (K)."""
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"Calibration file not found at: {calib_file_path}")
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].startswith('P0'):
                p0_values = np.array([float(x) for x in parts[1:13]], dtype=np.float32)
                return p0_values.reshape(3, 4)[:, :3]
    raise ValueError(f"Could not extract P0 from {calib_file_path}")


def load_kitti_ground_truth(poses_file_path):
    """Loads ground truth trajectories to extract true physical movement scales."""
    if not os.path.exists(poses_file_path):
        print(f"Warning: Ground truth poses not found at {poses_file_path}. Using flat scale factor.")
        return None
    with open(poses_file_path, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        pose = np.fromstring(line, sep=' ').reshape(3, 4)
        poses.append(pose)
    return poses


# =========================================================================
# 2. PATHS & INITIALIZATION SETUP
# =========================================================================
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/01'
poses_file = '/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/01.txt' # Change path if needed

image_dir = os.path.join(basedir, 'image_0')
calib_file = os.path.join(basedir, 'calib.txt')

image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
if len(image_files) == 0:
    raise FileNotFoundError(f"No images found in {image_dir}.")

# Load calibration and optional ground truth poses
K = load_kitti_calib(calib_file)
gt_poses = load_kitti_ground_truth(poses_file)

# Hyperparameters
min_features_thresh = 1500  
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
detector = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.01, minDistance=10)

# Initialize tracking states using Frame 0
prev_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
prev_pts = detector.detect(prev_img, None)
prev_pts = np.array([kp.pt for kp in prev_pts], dtype=np.float32).reshape(-1, 1, 2)
h, w = prev_img.shape

# --- GLOBAL ODOMETRY MATRIX POSE ---
# The camera starts at world origin coordinates [0, 0, 0] looking straight ahead
R_global = np.eye(3, dtype=np.float64)
t_global = np.zeros((3, 1), dtype=np.float64)

# Create an empty map canvas for our birds-eye-view route tracking
map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)

print("\nTracking trajectory trajectory path. Press 'q' to stop.")

# =========================================================================
# 3. MAIN LOOP
# =========================================================================
for frame_idx, img_path in enumerate(image_files[1:], start=1):
    curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if curr_img is None: continue

    if len(prev_pts) < min_features_thresh:
        new_pts = detector.detect(prev_img, None)
        if new_pts: prev_pts = np.array([kp.pt for kp in new_pts], dtype=np.float32).reshape(-1, 1, 2)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)
    valid_mask = (status == 1).reshape(-1)
    good_prev = prev_pts[valid_mask].reshape(-1, 2)
    good_curr = curr_pts[valid_mask].reshape(-1, 2)

    ransac_status = np.zeros(len(good_prev))
    
    if len(good_prev) > 10:
        pts1 = np.ascontiguousarray(good_prev, dtype=np.float32)
        pts2 = np.ascontiguousarray(good_curr, dtype=np.float32)

        E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        
        if mask is not None and len(mask) > 0:
            ransac_status = mask.reshape(-1)
            inlier_count = int(ransac_status.sum())
            
            if inlier_count > 5:
                _, R, t, pinhole_mask = cv2.recoverPose(E, pts2, pts1, K, mask=mask)
                
                # --- CALCULATE TRUE PHYSICAL SCALE FACTOR ---
                scale = 1.0  # Fallback guess if no ground truth file exists
                if gt_poses is not None and frame_idx < len(gt_poses):
                    # Distance between previous frame and current frame in real space
                    prev_gt_t = gt_poses[frame_idx - 1][:, 3]
                    curr_gt_t = gt_poses[frame_idx][:, 3]
                    scale = np.linalg.norm(curr_gt_t - prev_gt_t)

                # --- ACCUMULATE ABSOLUTE COORDINATES ---
                # Only adjust positions if motion is clearly forward-facing (Z vector dominance)
                if t[2] > t[0] and t[2] > t[1]:
                    # Update absolute translation using absolute orientation matrix
                    t_global = t_global + scale * R_global.dot(t)
                    # Update global rotation step direction matrix
                    R_global = R_global.dot(R)

    # Drawing side-by-side video frame tracking matching view
    side_by_side = np.hstack((prev_img, curr_img))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
    for i, (pt_prev, pt_curr) in enumerate(zip(good_prev, good_curr)):
        x_prev, y_prev = map(int, pt_prev)
        x_curr_shifted = int(pt_curr[0]) + w
        y_curr = int(pt_curr[1])
        color = (0, 255, 0) if ransac_status[i] == 1 else (0, 0, 255)
        cv2.line(display_frame, (x_prev, y_prev), (x_curr_shifted, y_curr), color, 1)

    cv2.imshow("KITTI Calibrated Pose Tracking", display_frame)

    # --- RENDER BIRDS-EYE-VIEW POSITION MAP ---
    # Extract absolute X coordinates and absolute Z coordinate (depth map perspective)
    x_coord = float(t_global[0][0])
    z_coord = float(t_global[2][0])

    # Convert coordinates to pixel space coordinates on canvas (Offset and scale)
    # Adding offsets centers the path, multiplying scales meters to visible screen pixels
    draw_x = int(x_coord * 0.5) + 300
    draw_y = int(z_coord * 0.5) + 100

    # Draw the vehicle path onto the map stream window as a bright blue dot
    cv2.circle(map_canvas, (draw_x, draw_y), 2, (255, 120, 0), -1)
    
    # Render Text data over background map tracking window canvas
    info_canvas = map_canvas.copy()
    cv2.putText(info_canvas, f"Global X: {x_coord:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(info_canvas, f"Global Z: {z_coord:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Trajectory Map", info_canvas)

    # Prepare historical state for next cycle loop window iteration
    if ransac_status.sum() > 10:
        prev_pts = good_curr[ransac_status == 1].reshape(-1, 1, 2)
    else:
        prev_pts = good_curr.reshape(-1, 1, 2)
    prev_img = curr_img.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
