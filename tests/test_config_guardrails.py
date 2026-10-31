"""Each test here corresponds to a bug that cost hours of debugging.
   If any of these fail, README.md's 'What Didn't Work' section is your reference."""
import numpy as np
import vo_config as cfg
import cv2

def test_sgbm_p2_is_400():
    """P2=800 → 160m ATE. P2=400 → 40m ATE. Do not change."""
    assert cfg.P2_SGBM == 400

def test_sgbm_p1_is_200():
    assert cfg.P1_SGBM == 200

def test_sgbm_uniqueness_is_3():
    """Ratio=10 rejects mid-field. Ratio=3 unlocks them."""
    assert cfg.UNIQUENESS == 3

def test_num_disp_is_128():
    """192 crashes CUDA StereoSGM on this build ('Unsupported num of disparities')."""
    assert cfg.NUM_DISP == 128

def test_z_min_is_3():
    """Z_MIN=0.5 keeps near-road planar trap → 62° rotation flips."""
    assert cfg.Z_MIN == 3.0

def test_ransac_flags_is_sqpnp():
    """ITERATIVE + planar cloud → rotation flips. SQPNP fixes it."""
    assert cfg.RANSAC_FLAGS == cv2.SOLVEPNP_SQPNP

def test_ransac_iters_is_300():
    """1000 wastes compute at high inlier rate; 300 with early-stop is optimal."""
    assert cfg.RANSAC_ITERS == 300

def test_ransac_reproj_is_3p5():
    """2.0 is too tight for SGBM far-field noise."""
    assert cfg.RANSAC_REPROJ == 3.5

def test_lk_levels_is_4():
    """maxLevel=3 under-tracks 30-70px road flow."""
    assert cfg.LK_LEVELS == 4

def test_y_clamp_disabled():
    """Y-clamp improves ATE 40→34m but worsens RPE 2.3%→5.3%. Not worth it."""
    assert cfg.USE_Y_CLAMP is False

def test_save_format_is_interleaved():
    """Regression: [R.ravel() | t.ravel()] reshapes to singular 4×4 on frame 0."""
    R = np.eye(3)
    t = np.array([[1.0], [2.0], [3.0]])
    correct = np.hstack([R, t]).ravel()           # interleaved: [r11 r12 r13 tx ...]
    assert list(correct[:4]) == [1, 0, 0, 1]      # tx lands in col 3
    wrong = np.hstack([R.ravel(), t.ravel()])      # concatenated
    assert list(wrong[:4]) == [1, 0, 0, 0]         # tx lands in col 9 → invalid SE3
    m = correct.reshape(3, 4)
    assert np.allclose(m[:, :3], R) and np.allclose(m[:, 3], [1, 2, 3])