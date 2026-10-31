# demo_samples/vo_utils.py
import numpy as np

def load_poses(path):
    return np.loadtxt(path).reshape(-1, 3, 4)

def umeyama_align(est, gt):
    """Align est → gt via Umeyama (rigid + uniform scale).
       Reference: Umeyama 1991, eq. 32-34.
       Returns points in the GT frame."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    e, g = est - mu_e, gt - mu_g

    # Sigma_xy in Umeyama notation: (1/N) * sum (y - mu_y)(x - mu_x)^T
    # Here x = est, y = gt, so this is g.T @ e / N
    W = g.T @ e / len(est)

    U, S, Vt = np.linalg.svd(W)

    # S_fix = diag(1, 1, ..., 1, sgn) with sgn = +1 if det(U)det(V) > 0, else -1
    sgn = 1.0 if np.linalg.det(U) * np.linalg.det(Vt) > 0 else -1.0
    S_fix = np.ones(3)
    S_fix[-1] = sgn
    R = U @ np.diag(S_fix) @ Vt

    # Scale: trace(D @ S_fix) / sigma_x^2
    scale = (S * S_fix).sum() / (e ** 2).sum() * len(est)

    t = mu_g - scale * R @ mu_e
    return (scale * R @ est.T).T + t

def ate(est_xyz, gt_xyz):
    err = np.linalg.norm(umeyama_align(est_xyz, gt_xyz) - gt_xyz, axis=1)
    return err.mean(), np.sqrt((err**2).mean()), err.max()

def rpe(est_poses, gt_poses, delta_m=100):
    gt_xyz = gt_poses[:, :, 3]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))])
    t_err, r_err = [], []
    for i in range(len(gt_poses) - 1):
        j = np.searchsorted(cum, cum[i] + delta_m)
        if j >= len(gt_poses): break
        def dT(P):
            return np.linalg.inv(np.vstack([P[i], [0,0,0,1]])) @ np.vstack([P[j], [0,0,0,1]])
        dE = np.linalg.inv(dT(gt_poses)) @ dT(est_poses)
        t_err.append(np.linalg.norm(dE[:3, 3]))
        ca = (np.trace(dE[:3, :3]) - 1) / 2
        r_err.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
    return np.mean(t_err), np.mean(r_err)