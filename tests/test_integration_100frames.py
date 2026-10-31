"""Catches: config regressions, format regressions, GPU vs CPU drift."""
import os
import sys
import subprocess
import tempfile
import numpy as np
import pytest
import vo_config as cfg
from vo_utils import load_poses, ate

SCRIPT = "demo_samples/09_feature_vo_gpu_vs_cpu.py"
GOLDEN = "tests/data/est_traj_100frames_golden.txt"


def _run(env_extra=None, timeout=180):
    """Run 100 frames, return tempdir with est_traj.txt."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "est_traj.txt")
        env = {**os.environ, "VO_MAX_FRAMES": "100", "VO_OUTPUT": out}
        if env_extra: env.update(env_extra)
        r = subprocess.run([sys.executable, SCRIPT], env=env,
                           capture_output=True, text=True, timeout=timeout)
        assert r.returncode == 0, f"pipeline crashed:\n{r.stderr[-2000:]}"
        assert "Traceback" not in r.stderr
        return load_poses(out)


def test_100_frames_complete():
    """Baseline: 100 frames, no crash, no NaN, valid SE3 rows."""
    est = _run()
    assert len(est) == 100
    assert np.isfinite(est).all(), "trajectory contains NaN/Inf"
    # Every row must be a valid SE3 (R^T R = I, det R = +1)
    for i, P in enumerate(est):
        R = P[:, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-3), f"frame {i} R not orthogonal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-3, f"frame {i} det(R) != 1"


def test_100_frames_ate_under_10m():
    """Baseline ATE on 100 frames is ~2-3m. >10m = config regression."""
    if not os.path.exists(cfg.GT_POSES_FILE):
        pytest.skip("dataset not present")
    est = _run()
    gt = load_poses(cfg.GT_POSES_FILE)[:len(est)]
    m, rmse, mx = ate(est[:, :, 3], gt[:, :, 3])
    assert m < 10.0, f"100-frame ATE {m:.2f}m — check P2, Z_MIN, SQPNP flags"
    assert m > 0.5,  f"100-frame ATE {m:.2f}m — too good, suspicious (sanity floor)"


@pytest.mark.integration
def test_100_frames_matches_golden():
    """If you change ANY pipeline code, verify against the pinned snapshot."""
    if not os.path.exists(GOLDEN):
        pytest.skip(f"generate golden first: see README")
    est = _run()
    golden = load_poses(GOLDEN)
    assert len(est) == len(golden)
    max_diff = np.linalg.norm(est[:, :, 3] - golden[:, :, 3], axis=1).max()
    # 0.5m tolerance covers numerical noise only — real bug shows >5m
    assert max_diff < 0.5, (
        f"trajectory diverged {max_diff:.2f}m from golden. "
        f"Likely a config regression — run test_config_guardrails.py first."
    )


@pytest.mark.integration
def test_gpu_and_cpu_produce_same_trajectory():
    """CPU and GPU pipeline should match within 1m on 100 frames.
       Catches 'I broke the CPU fallback' regressions."""
    if not os.path.exists(cfg.GT_POSES_FILE):
        pytest.skip("dataset not present")
    gpu = _run()
    cpu = _run(env_extra={
        "VO_FORCE_CPU": "1",   # add this env check in script — see below
    })
    assert len(gpu) == len(cpu)
    max_diff = np.linalg.norm(gpu[:, :, 3] - cpu[:, :, 3], axis=1).max()
    # CPU and GPU SGBM differ in Q4 rounding → a few cm is normal
    assert max_diff < 1.0, f"GPU/CPU trajectories differ by {max_diff:.2f}m"