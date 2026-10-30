import numpy as np

def load_poses(path):
    p = np.loadtxt(path)
    assert p.shape[1] == 12, f"unexpected width: {p.shape}"
    return p.reshape(-1, 3, 4)

def umeyama_align(est, gt):
    mu_e, mu_g = est.mean(0), gt.mean(0)
    e, g = est - mu_e, gt - mu_g
    W = e.T @ g / len(est)
    U, S, Vt = np.linalg.svd(W)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = S.sum() / (e ** 2).sum() * len(est)
    t = mu_g - scale * R @ mu_e
    return (scale * R @ est.T).T + t

def ate(est_xyz, gt_xyz):
    est_al = umeyama_align(est_xyz, gt_xyz)
    err = np.linalg.norm(est_al - gt_xyz, axis=1)
    return err.mean(), np.sqrt((err ** 2).mean()), err.max()

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

GT  = "/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/00.txt"
EST = "/media/master/2tbkup/MySpace/projects/visual_odometry_kitti/est_traj.txt"

gt = load_poses(GT); 
est = load_poses(EST)
n = min(len(gt), len(est)); gt, est = gt[:n], est[:n]
print(f"comparing {n} frames")

m, rmse, mx = ate(est[:, :, 3], gt[:, :, 3])
print(f"ATE   mean={m:.2f}  rmse={rmse:.2f}  max={mx:.2f}  (m)")

for d in [100, 200, 400, 800]:
    t, r = rpe(est, gt, d)
    print(f"RPE@{d:3d}m  trans={t:.2f}m ({100*t/d:.1f}%)  rot={r:.2f}deg")