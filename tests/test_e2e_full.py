import os, sys, subprocess, tempfile
import numpy as np
import pytest
import vo_config as cfg
from vo_utils import load_poses, ate, rpe


@pytest.mark.e2e
def test_full_sequence_matches_baseline():
    """Full 4540-frame run. Baseline: ATE=40.45m, RPE@100m=2.26m (2.3%)."""
    if not os.path.exists(cfg.GT_POSES_FILE):
        pytest.skip("dataset not present")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "est_traj.txt")
        env = {**os.environ, "VO_OUTPUT": out}        # no MAX_FRAMES → full run
        r = subprocess.run(
            [sys.executable, "demo_samples/09_feature_vo_gpu_vs_cpu.py"],
            env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]

        est = load_poses(out)
        gt  = load_poses(cfg.GT_POSES_FILE)
        m, rmse, mx = ate(est[:, :, 3], gt[:, :, 3])
        t_err, r_err = rpe(est, gt, delta_m=100)

        # Baseline measured: ATE=40.45, RPE=2.26, rot=1.24
        assert  35 < m < 45,   f"ATE {m:.1f}m outside [35,45] — regression"
        assert t_err < 3.0,    f"RPE@100m {t_err:.2f}m > 3.0m — regression"
        assert r_err < 2.0,    f"RPE rot {r_err:.2f}° > 2.0° — regression"