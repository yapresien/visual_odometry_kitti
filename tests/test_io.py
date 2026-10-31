import os
import numpy as np
import pytest
import vo_config as cfg
from vo_utils import load_poses


def test_calib_baseline_is_0_537():
    """KITTI 00 baseline. If this changes, calibration file is wrong."""
    if not os.path.exists(cfg.BASEDIR + "/calib.txt"):
        pytest.skip("dataset not present")
    with open(cfg.BASEDIR + "/calib.txt") as f:
        lines = [l.strip().split() for l in f if l.strip()]
    P0 = np.array([float(x) for x in lines[0][1:13]]).reshape(3, 4)
    P1 = np.array([float(x) for x in lines[1][1:13]]).reshape(3, 4)
    b = abs(P1[0, 3] - P0[0, 3]) / P0[0, 0]
    assert 0.53 < b < 0.55, f"baseline {b} outside expected 0.537 ± 0.01"


def test_gt_poses_format():
    if not os.path.exists(cfg.GT_POSES_FILE):
        pytest.skip("dataset not present")
    poses = load_poses(cfg.GT_POSES_FILE)
    assert poses.shape == (4541, 3, 4)
    assert np.allclose(poses[0], np.hstack([np.eye(3), np.zeros((3, 1))]))
    step0 = np.linalg.norm(poses[1, :, 3])
    assert 0.7 < step0 < 1.0, f"frame 0→1 step {step0}m outside KITTI 00 range"