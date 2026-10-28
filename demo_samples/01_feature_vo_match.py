import cv2
import numpy as np
import os

# 1. Initialize the ORB Detector
min_features_needed = 1000
detector = cv2.ORB_create(nfeatures=3000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

basedir = '/media/master/2tbkup/MySpace/projects/dataset_kiti'

# Specify the dataset to load
sequence = '00'

# --- Detect what is actually available on disk -----------------------------
seq_dir = os.path.join(basedir, 'sequences', sequence)

# Load consecutive grayscale driving frames from KITTI sequence
img1 = cv2.imread(os.path.join(seq_dir, 'image_0', '000000.png'), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(seq_dir, 'image_0', '000001.png'), cv2.IMREAD_GRAYSCALE)

# 2. Detect and Compute Features
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# 3. Robust Feature Matching
matches = bf_matcher.match(des1, des2)
# Sort them in the order of their distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# 4. Outlier Rejection Framework (Crucial step before Essential Matrix / PnP calculation)
# Retain only reliable top-tier matches to minimize trajectory drift
good_matches = matches[:500] 

# Draw the detected, matched features across the road frames
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("KITTI Feature Tracks", matched_img)
cv2.waitKey(0)
