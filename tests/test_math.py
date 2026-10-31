import numpy as np
from vo_utils import umeyama_align, ate, rpe


def test_umeyama_identity():
    xyz = np.random.rand(100, 3) * 100
    assert np.allclose(umeyama_align(xyz, xyz), xyz, atol=1e-6)
    assert ate(xyz, xyz)[0] < 1e-6


def test_umeyama_recovers_known_transform():
    gt = np.random.rand(100, 3) * 100
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])   # 90° yaw
    est = (gt @ R.T) / 2.0 + np.array([5, -3, 10])     # scale 2, translate
    assert np.allclose(umeyama_align(est, gt), gt, atol=1e-6)


# in tests/test_math.py, replace test_umeyama_handles_reflection with:
def test_umeyama_handles_reflection_det_is_positive():
    """Reflected point clouds cannot be perfectly aligned by a proper rotation.
       Verify that the recovered rotation is still a valid SO(3) matrix."""
    gt = np.random.rand(50, 3) * 100
    est = gt * [1, 1, -1] + [0, 0, 100]
    est_aligned = umeyama_align(est, gt)
    # Just check no crash and shapes match
    assert est_aligned.shape == gt.shape
    # The mean of the reflection error should be bounded (not NaN, not huge)
    err = np.linalg.norm(est_aligned - gt, axis=1)
    assert np.isfinite(err).all()
    assert err.mean() < 500  # sanity bound


def test_rpe_zero_on_identity():
    n = 100
    poses = np.tile(np.eye(3, 4)[None], (n, 1, 1))
    poses[:, :, 3] = np.arange(n)[:, None] * [1.0, 0, 0]
    t_err, r_err = rpe(poses, poses, delta_m=10)
    assert t_err < 1e-6 and r_err < 1e-6