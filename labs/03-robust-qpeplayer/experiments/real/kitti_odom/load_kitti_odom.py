#!/usr/bin/env python3
"""KITTI *odometry* sequences 00/05/07 loader: exact GT poses + calib + LiDAR.

Adapted from the parent raw-drive loader (``../kitti/load_kitti.py``).  The
geometry code (voxel downsample, GT-mapped nearest-neighbour correspondences,
``tau`` gate) is unchanged; only the pose source differs:

* KITTI **raw** drives (0001/0002, parent) have no per-frame ground truth;
  the ego motion is integrated from the OXTS GPS/IMU stream and mapped to the
  Velodyne frame with ``calib_imu_to_velo``.  That introduces OXTS/calibration
  error, so the "noise floor" of the clean fit is ~0.02-0.03 deg / 1-3 cm.
* KITTI **odometry** sequences (00/05/07, here) ship exact per-frame poses
  ``T_w_cam0`` (``dataset/poses/NN.txt``, one 3x4 row per frame) from the
  benchmark's laser-based SLAM ground truth, plus a per-sequence
  ``calib.txt`` with the Velodyne->cam0 ``Tr:`` row.

Relative Velodyne motion:

    T_rel_cam  = inv(T_w_cam_j) @ T_w_cam_i
    T_cam_velo = Tr                 (calib "Tr:")
    T_rel_velo = inv(T_cam_velo) @ T_rel_cam @ T_cam_velo

The direction is verified empirically (``verify_kitti_odom.py``): mapping
frame i into frame j with the correct convention makes LiDAR surfaces
coincide, so its nearest-neighbour median distance is much smaller than the
inverse's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DATA = Path("/mnt/storage1/syk/data/kitti_odom")
SEQS = ["00", "05", "07"]


def read_calib_tr(seq):
    """Per-sequence Velodyne->cam0 transform from the ``Tr:`` row of calib.txt."""
    path = DATA / "calib" / f"{seq}.txt"
    for line in path.read_text().splitlines():
        if line.startswith("Tr:"):
            v = np.array([float(x) for x in line.split()[1:]])
            M = np.eye(4)
            M[:3, :] = v.reshape(3, 4)
            return M
    raise ValueError(f"no Tr: in {path}")


def load_poses(seq, n=None):
    """Exact GT world poses T_w_cam0, sorted by frame index."""
    rows = [l for l in (DATA / "poses" / f"{seq}.txt").read_text().splitlines() if l.strip()]
    out = []
    for line in rows[:n] if n else rows:
        v = np.array([float(x) for x in line.split()])
        M = np.eye(4)
        M[:3, :] = v.reshape(3, 4)
        out.append(M)
    return out


def n_gt_frames(seq):
    return len([l for l in (DATA / "poses" / f"{seq}.txt").read_text().splitlines() if l.strip()])


def read_velodyne(seq, idx, max_range=30.0):
    path = DATA / "sequences" / seq / "velodyne" / f"{idx:06d}.bin"
    arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = arr[:, :3].astype(np.float64)
    if max_range is not None:
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > 1.0) & (r < max_range)]
    return xyz


def voxel_downsample(points, voxel):
    if len(points) == 0:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=points[:, k]) / cnt
    return out


def load_clouds(seq, idxs, voxel=0.15, max_range=30.0, cache_dir=None):
    """Cached voxel-downsampled clouds; cache namespace is per sequence."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DATA / "_cache" / seq
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{seq}_velo_v{voxel:.2f}_r{max_range:.0f}.npz"
    if cache.exists():
        z = np.load(cache)
        if all(str(i) in z.files for i in idxs):
            return {int(i): z[str(i)] for i in idxs}
    out = {}
    if cache.exists():
        z = np.load(cache)
        out = {int(k): z[k] for k in z.files}
    for i in idxs:
        if int(i) not in out:
            out[int(i)] = voxel_downsample(read_velodyne(seq, i, max_range=max_range), voxel)
    np.savez_compressed(cache, **{str(k): v for k, v in out.items()})
    return {int(i): out[int(i)] for i in idxs}


def relative_velo_pose(poses, i, j, T_cam_velo):
    """GT relative transform mapping Velodyne frame i into frame j."""
    T_w_i, T_w_j = poses[i], poses[j]
    T_cam = np.linalg.inv(T_w_j) @ T_w_i          # i -> j in cam0 frame
    T_velo_cam = np.linalg.inv(T_cam_velo)
    return T_velo_cam @ T_cam @ T_cam_velo


def pose_motion(poses, i, j, T_cam_velo):
    """(|t|, rotation deg) of the relative Velodyne motion i->j."""
    T = relative_velo_pose(poses, i, j, T_cam_velo)
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return float(np.linalg.norm(T[:3, 3])), float(ang)


def gt_nn_correspondences(cloud_i, cloud_j, T, n=300, rng=None, tau=0.45, oversample=6):
    """Sample real LiDAR correspondences with GT mapping and NN target."""
    rng = np.random.default_rng(rng)
    if len(cloud_i) == 0 or len(cloud_j) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    n_probe = min(len(cloud_i), max(n, oversample * n))
    idx = rng.choice(len(cloud_i), n_probe, replace=False) if n_probe < len(cloud_i) else np.arange(len(cloud_i))
    x = cloud_i[idx]
    xj = (T[:3, :3] @ x.T).T + T[:3, 3]
    d, j = cKDTree(cloud_j).query(xj, k=1, workers=-1)
    keep = d < tau
    if keep.sum() < 3:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    x, y, d = x[keep], cloud_j[j[keep]], d[keep]
    if len(x) > n:
        take = rng.choice(len(x), n, replace=False)
        x, y, d = x[take], y[take], d[take]
    return x, y, d
