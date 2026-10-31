import numpy as np
import matplotlib.pyplot as plt

GT_PATH = "/media/master/2tbkup/MySpace/projects/dataset_kiti/poses/00.txt"
VO_PATH = "/media/master/2tbkup/MySpace/projects/visual_odometry_kitti/est_traj.txt"
LO_PATH = ("/media/master/2tbkup/MySpace/projects/visual_odometry_kitti/"
           "results/2026-10-30_20-16-42/velodyne_poses_kitti.txt")

def load(p): return np.loadtxt(p).reshape(-1, 3, 4)

gt, vo, lo = load(GT_PATH), load(VO_PATH), load(LO_PATH)
gt_xyz, vo_xyz, lo_xyz = gt[:,:,3], vo[:,:,3], lo[:,:,3]

# LO is in LiDAR frame; rotate to camera frame for direct comparison.
# KITTI convention (Tr_velo_to_cam):
#   cam_x = -velo_y   (right)
#   cam_y = -velo_z   (down)  ← this axis is where VO's Y drift lives
#   cam_z =  velo_x   (forward)
lo_cam = np.column_stack([-lo_xyz[:,1], -lo_xyz[:,2], lo_xyz[:,0]])

# ── Endpoint comparison ────────────────────────────────────────────
print("Endpoint drift from origin (camera frame):")
for name, xyz in [('GT', gt_xyz), ('VO', vo_xyz), ('LO', lo_cam)]:
    print(f"  {name:3s}  end = [{xyz[-1,0]:7.2f}, {xyz[-1,1]:7.2f}, {xyz[-1,2]:7.2f}]"
          f"   |end| = {np.linalg.norm(xyz[-1]):6.2f} m")

# ── Per-frame axis drift vs GT (RMS across full sequence) ─────────
print("\nPer-axis drift (VO vs GT and LO vs GT, RMS over all frames):")
for i, ax in enumerate(['X (lateral)', 'Y (vertical)', 'Z (forward)']):
    vo_err = np.sqrt(np.mean((vo_xyz[:,i] - gt_xyz[:,i])**2))
    lo_err = np.sqrt(np.mean((lo_cam[:,i] - gt_xyz[:,i])**2))
    print(f"  {ax:12s}   VO={vo_err:6.2f} m   LO={lo_err:6.2f} m   "
          f"{'← VO BUG' if vo_err > 3*lo_err else ''}")

# ── Three-row plot: X, Y, Z over time ─────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i, (name, ax) in enumerate(zip(['X (lateral)', 'Y (vertical)', 'Z (forward)'], axes)):
    ax.plot(gt_xyz[:,i], 'r-',  lw=1.5, label='GT')
    ax.plot(lo_cam[:,i], 'g-',  lw=1.2, label='LO (KISS-ICP)')
    ax.plot(vo_xyz[:,i], '-',   lw=1.2, color='orange', label='VO (yours)')
    ax.set_ylabel(name); ax.legend(loc='best'); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('frame index')
axes[0].set_title('Per-axis trajectory components — the Y row reveals VO pitch bias')
plt.tight_layout()
plt.savefig("drift_axes.png", dpi=150)
print("\nsaved drift_axes.png — open it and look at the MIDDLE row (Y)")