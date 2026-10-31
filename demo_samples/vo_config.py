# demo_samples/vo_config.py
"""Single source of truth. Every value here was debugged — see README.md."""

import os
import cv2

NUM_DISP        = 128              # CUDA StereoSGM: 128 max (192 crashes)
P1_SGBM         = 200
P2_SGBM         = 400              # ← CRITICAL: 800 rejects mid-field (160m ATE)
UNIQUENESS      = 3                # ← CRITICAL: 10 rejects mid-field
BLOCK_SIZE      = 5
LK_WIN          = (21, 21)
LK_LEVELS       = 4                # ← 3 under-tracks 30-70px road flow
LK_ITERS        = 40
LK_EPS          = 0.01
Z_MIN           = 3.0              # ← 0.5 includes planar road trap
Z_MAX           = 200.0
RANSAC_ITERS    = 300
RANSAC_REPROJ   = 3.5              # ← 2.0 too tight for SGBM noise
RANSAC_CONF     = 0.99
RANSAC_FLAGS    = cv2.SOLVEPNP_SQPNP          # ← ITERATIVE + planar → 62° rot flips
MERGE_THRESHOLD = 15.0
POOL_MIN_POINTS = 100
USE_Y_CLAMP     = False            # ← True worsens RPE 2.3% → 5.3%

# Paths
BASEDIR       = '/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00'
CALIB_FILE    = os.path.join(BASEDIR, 'calib.txt')
GT_POSES_FILE = '/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/00.txt'
