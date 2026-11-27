#!/usr/bin/env python3
"""KITTI raw (2011_09_26 drive 0002) loader: OXTS poses, calib, LiDAR.

Second-drive cross-check copy of the parent ``real/kitti/load_kitti.py``
(drive 0001).  Only the drive path/cache namespace and the module name
changed, so the two drives are processed by identical geometry code.

KITTI raw does not ship per-frame ego poses for the *sync* drive; the OXTS
GPS/IMU stream does.  We convert lat/lon/alt + roll/pitch/yaw to a local
Mercator world pose following the KITTI devkit, compute the relative
IMU-frame motion between two frames, and map it into the Velodyne frame
with the provided extrinsics:

    T_rel_velo = T_velo_imu @ T_rel_imu @ inv(T_velo_imu)

where ``calib_imu_to_velo.txt`` is the IMU->Velodyne transform (KITTI's
``Tr_imu_to_velo``).  The direction is verified empirically on a
consecutive frame pair: the correct transform should make LiDAR surfaces
coincide, so its nearest-neighbour median distance is much smaller than
the inverse.

Ground-truth nearest-neighbour correspondences are then built exactly as
for 3DMatch: sample source points, map with GT, take the nearest target
point within ``tau``.  This gives real outdoor LiDAR geometry (sparse,
occlusions, dynamic objects) with controlled injected outliers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

KITTI = Path("/mnt/storage1/syk/data/kitti_raw/2011_09_26")
DRIVE = KITTI / "2011_09_26_drive_0002_sync"


def read_calib_rigid(path):
    """Read a KITTI calib file with ``R:`` and ``T:`` rows into a 4x4."""
    R = T = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("R:"):
            R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        elif line.startswith("T:"):
            T = np.array([float(x) for x in line.split()[1:]]).reshape(3)
    if R is None or T is None:
        raise ValueError(f"no R/T in {path}")
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def oxts_to_pose(values, scale=1.0):
    """Convert one OXTS row to a 4x4 world pose (KITTI devkit convention)."""
    lat, lon, alt = values[0], values[1], values[2]
    roll, pitch, yaw = values[3], values[4], values[5]
    er = 6378137.0
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    R = rotz(yaw) @ roty(pitch) @ rotx(roll)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, alt]
    return M


def load_oxts_poses():
    files = sorted((DRIVE / "oxts" / "data").glob("*.txt"))
    poses = [oxts_to_pose(np.loadtxt(f)) for f in files]
    return poses


def read_velodyne(idx, max_range=30.0):
    path = DRIVE / "velodyne_points" / "data" / f"{idx:010d}.bin"
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


def load_clouds(idxs, voxel=0.15, max_range=30.0, cache_dir=None):
    cache_dir = Path(cache_dir) if cache_dir is not None else KITTI / "_cache" / DRIVE.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{DRIVE.name}_velo_v{voxel:.2f}_r{max_range:.0f}.npz"
    if cache.exists():
        z = np.load(cache)
        if all(str(i) in z.files for i in idxs):
            return {int(i): z[str(i)] for i in idxs}
    out = {}
    for i in idxs:
        out[int(i)] = voxel_downsample(read_velodyne(i, max_range=max_range), voxel)
    np.savez_compressed(cache, **{str(k): v for k, v in out.items()})
    return out


def relative_velo_pose(poses, i, j, T_velo_imu):
    """GT relative transform mapping Velodyne frame i into frame j."""
    T_w_i = poses[i]
    T_w_j = poses[j]
    T_imu = np.linalg.inv(T_w_j) @ T_w_i  # i -> j in IMU/world frame
    T_velo = T_velo_imu @ T_imu @ np.linalg.inv(T_velo_imu)
    return T_velo


def gt_nn_correspondences(cloud_i, cloud_j, T, n=300, rng=None, tau=0.45, oversample=6):
    """Sample real LiDAR correspondences with GT mapping and NN target."""
    rng = np.random.default_rng(rng)
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
