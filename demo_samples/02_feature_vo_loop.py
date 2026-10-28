import os
import glob
import cv2
import numpy as np

# 1. Setup paths (Change this to your actual KITTI path)
basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti'

# Specify the dataset to load
sequence = '00'

# --- Detect what is actually available on disk -----------------------------
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

# Use GFTT/FAST for rapid, dense corner placement
detector = cv2.GFTTDetector_create(maxCorners=1, qualityLevel=0.01, minDistance=10)

# Initialize with frame 0
prev_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
prev_pts = detector.detect(prev_img, None)
prev_pts = np.array([kp.pt for kp in prev_pts], dtype=np.float32).reshape(-1, 1, 2)

h, w = prev_img.shape

print("Playing back side-by-side tracking stream. Press 'q' to exit.")

# 3. Main Loop
for img_path in image_files[1:]:
    curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if curr_img is None:
        continue

    # Adaptive Re-detection if feature pool dries up
    if len(prev_pts) < min_features_thresh:
        new_pts = detector.detect(prev_img, None)
        if new_pts:
            prev_pts = np.array([kp.pt for kp in new_pts], dtype=np.float32).reshape(-1, 1, 2)

    # Track pixels moving from prev_img to curr_img
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)

    # Filter for successfully tracked features
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]

    # --- RECONSTRUCT SIDE-BY-SIDE VIEW ---
    # Stitch the two grayscale images horizontally, then convert to BGR for color drawing
    side_by_side = np.hstack((prev_img, curr_img))
    display_frame = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)

    # Draw pairing matches spanning across the split screen
    for pt_prev, pt_curr in zip(good_prev, good_curr):
        x_prev, y_prev = map(int, pt_prev)
        x_curr, y_curr = map(int, pt_curr)
        
        # Shift the current frame's X coordinate to the right by the width of the first image
        x_curr_shifted = x_curr + w

        # Draw point markers on both frames
        cv2.circle(display_frame, (x_prev, y_prev), 3, (0, 0, 255), -1)          # Red on Frame T-1
        cv2.circle(display_frame, (x_curr_shifted, y_curr), 3, (0, 255, 0), -1)  # Green on Frame T
        
        # Draw a line joining the matching features across the center boundary
        cv2.line(display_frame, (x_prev, y_prev), (x_curr_shifted, y_curr), (255, 100, 0), 1)

    # Add telemetry annotations
    cv2.putText(display_frame, "PREVIOUS FRAME (T-1)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, "CURRENT FRAME (T)", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Matches: {len(good_curr)}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Render window
    cv2.imshow("KITTI Side-by-Side Optical Flow", display_frame)

    # Update states for next loop iteration
    prev_img = curr_img.copy()
    prev_pts = good_curr.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
