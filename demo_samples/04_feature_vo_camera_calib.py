import os
import glob
import cv2
import numpy as np


# =========================================================================
# 1. FUNCTION DEFINITION
# =========================================================================
def load_kitti_calib(calib_file_path):
    """
    Safely parses the KITTI calib.txt file and extracts the 3x3 Intrinsic Matrix (K).
    """
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"Calibration file not found at: {calib_file_path}")

    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
            
        if parts[0].startswith('P0'):
            try:
                p0_values = np.array([float(x) for x in parts[1:13]], dtype=np.float32)
                if len(p0_values) == 12:
                    P0 = p0_values.reshape(3, 4)
                    K = P0[:, :3]  # Extract the leftmost 3x3 matrix
                    return K
            except ValueError:
                continue
            
    raise ValueError(f"Could not extract 12 valid projection matrix values from {calib_file_path}")


# =========================================================================
# 2. FILE PATH AND INITIALIZATION SETUP
# =========================================================================
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00'
image_dir = os.path.join(basedir, 'image_0')
calib_file = os.path.join(basedir, 'calib.txt')

image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
if len(image_files) == 0:
    raise FileNotFoundError(f"No images found in {image_dir}.")

try:
    K = load_kitti_calib(calib_file)
    print("Successfully loaded KITTI Calibration Matrix K:\n", K)
except Exception as e:
    print("Calib file parsing failed. Using hardcoded KITTI baseline matrix.")
    K = np.array([[718.856,   0.0,   607.1928],
                  [  0.0,   718.856, 185.2157],
                  [  0.0,     0.0,     1.0]], dtype=np.float32)

min_features_thresh = 1500  
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

detector = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.01, minDistance=10)

prev_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
prev_pts = detector.detect(prev_img, None)
prev_pts = np.array([kp.pt for kp in prev_pts], dtype=np.float32).reshape(-1, 1, 2)
h, w = prev_img.shape

print("\nProcessing with Calibration. Press 'q' to stop.")

# =========================================================================
# 3. MAIN TRACKING LOOP
# =========================================================================
for img_path in image_files[1:]:
    curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if curr_img is None:
        continue

    if len(prev_pts) < min_features_thresh:
        new_pts = detector.detect(prev_img, None)
        if new_pts:
            prev_pts = np.array([kp.pt for kp in new_pts], dtype=np.float32).reshape(-1, 1, 2)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)

    valid_mask = (status == 1).reshape(-1)
    good_prev = prev_pts[valid_mask].reshape(-1, 2)
    good_curr = curr_pts[valid_mask].reshape(-1, 2)

    ransac_status = np.zeros(len(good_prev))
    t = None  # Reset tracking direction variable per frame loop iteration
    
    if len(good_prev) > 10:
        pts1 = np.ascontiguousarray(good_prev, dtype=np.float32)
        pts2 = np.ascontiguousarray(good_curr, dtype=np.float32)

        E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        
        if mask is not None and len(mask) > 0:
            ransac_status = mask.reshape(-1)
            
            if ransac_status.sum() > 5:
                _, R, t, pinhole_mask = cv2.recoverPose(E, pts2, pts1, K, mask=mask)

    # -------------------------------------------------------------------------
    # CRITICAL FIX: Calculate the count instantly from the mask array structure
    # -------------------------------------------------------------------------
    inlier_count = int(ransac_status.sum())

    side_by_side = np.hstack((prev_img, curr_img))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)

    for i, (pt_prev, pt_curr) in enumerate(zip(good_prev, good_curr)):
        x_prev, y_prev = map(int, pt_prev)
        x_curr, y_curr = map(int, pt_curr)
        x_curr_shifted = x_curr + w

        color = (0, 255, 0) if ransac_status[i] == 1 else (0, 0, 255)
        cv2.line(display_frame, (x_prev, y_prev), (x_curr_shifted, y_curr), color, 1)

    # UI Metrics Panel Drawing
    cv2.putText(display_frame, f"Calibrated Inliers: {inlier_count}", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Check that we have a valid translation vector and clean targets to print
    if inlier_count > 10 and t is not None:
        t_flat = t.ravel()
        motion_str = f"Direction Vec: X:{t_flat[0]:.2f} Y:{t_flat[1]:.2f} Z:{t_flat[2]:.2f}"
        cv2.putText(display_frame, motion_str, (w + 20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("KITTI Calibrated Pose Tracking", display_frame)

    # Propagate ONLY verified features forward to stop tracking skew errors
    if inlier_count > 10:
        prev_pts = good_curr[ransac_status == 1].reshape(-1, 1, 2)
    else:
        prev_pts = good_curr.reshape(-1, 1, 2)
        
    prev_img = curr_img.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
