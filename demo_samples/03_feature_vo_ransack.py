import os
import glob
import cv2
import numpy as np

# 1. Setup paths
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti'
sequence = '00'
seq_dir = os.path.join(basedir, 'sequences', sequence)
sequence_dir = os.path.join(seq_dir, 'image_0')

image_files = sorted(glob.glob(os.path.join(sequence_dir, '*.png')))

if len(image_files) == 0:
    raise FileNotFoundError(f"No images found in {sequence_dir}.")

# 2. Tracking Setup
min_features_thresh = 1500  
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

detector = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.01, minDistance=10)

# Initialize frame 0
prev_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
prev_pts = detector.detect(prev_img, None)
prev_pts = np.array([kp.pt for kp in prev_pts], dtype=np.float32).reshape(-1, 1, 2)

h, w = prev_img.shape

print("Playing back with RANSAC filtering. Green = Inlier, Red = Rejected. Press 'q' to exit.")

# 3. Main Loop
for img_path in image_files[1:]:
    curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if curr_img is None:
        continue

    if len(prev_pts) < min_features_thresh:
        new_pts = detector.detect(prev_img, None)
        if new_pts:
            prev_pts = np.array([kp.pt for kp in new_pts], dtype=np.float32).reshape(-1, 1, 2)

    # Track pixels moving from prev_img to curr_img
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)

    # Filter for valid optical flow tracking status
    valid_mask = (status == 1).reshape(-1)
    good_prev = prev_pts[valid_mask].reshape(-1, 2)
    good_curr = curr_pts[valid_mask].reshape(-1, 2)

    # --- APPLY RANSAC FILTERING ---
    # We use findHomography or findEssentialMat as a geometric filter. 
    # Points matching the background 'consensus' get marked with a 1, outliers marked with a 0.
    ransac_status = np.zeros(len(good_prev))
    if len(good_prev) > 10:
        _, mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
        if mask is not None:
            ransac_status = mask.reshape(-1)

    # Reconstruct Side-by-Side View
    side_by_side = np.hstack((prev_img, curr_img))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)

    inlier_count = 0
    # Draw pairing matches based on RANSAC validation
    for i, (pt_prev, pt_curr) in enumerate(zip(good_prev, good_curr)):
        x_prev, y_prev = map(int, pt_prev)
        x_curr = int(pt_curr[0]) + w
        y_curr = int(pt_curr[1])

        if ransac_status[i] == 1:
            # Valid point (Inlier) -> Green
            color = (0, 255, 0)
            inlier_count += 1
            cv2.circle(display_frame, (x_prev, y_prev), 2, color, -1)
            cv2.circle(display_frame, (x_curr, y_curr), 2, color, -1)
            cv2.line(display_frame, (x_prev, y_prev), (x_curr, y_curr), color, 1)
        else:
            # Bad tracking or dynamic object (Outlier) -> Red
            color = (0, 0, 255)
            # We draw outliers slightly lighter/thinner so they don't clutter the screen
            cv2.line(display_frame, (x_prev, y_prev), (x_curr, y_curr), color, 1)

    # Text overlay
    cv2.putText(display_frame, f"Total Tracked: {len(good_curr)}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"RANSAC Inliers (Valid): {inlier_count}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Rejected Outliers: {len(good_curr) - inlier_count}", (w + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("KITTI RANSAC Filter View", display_frame)

    # Only pass the valid INLIER points forward to the next frame to prevent error propagation
    # If RANSAC fails entirely for a frame, fall back to tracking all points
    if inlier_count > 10:
        prev_pts = good_curr[ransac_status == 1].reshape(-1, 1, 2)
    else:
        prev_pts = good_curr.reshape(-1, 1, 2)
        
    prev_img = curr_img.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
