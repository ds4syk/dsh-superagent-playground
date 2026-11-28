#!/usr/bin/env python3
"""Verify KITTI OXTS->Velodyne relative motion and register a clean pair."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_kitti as K  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def nn_stats(src, dst, T, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(src), min(n, len(src)), replace=False)
    p = src[idx]
    q = (T[:3, :3] @ p.T).T + T[:3, 3]
    d, _ = cKDTree(dst).query(q, k=1, workers=-1)
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                frac_lt_tau=float(np.mean(d < 0.45)), n=len(d))


def main():
    T_velo_imu = K.read_calib_rigid(K.KITTI / "calib_imu_to_velo.txt")
    poses = K.load_oxts_poses()
    idxs = list(range(0, 60))
    clouds = K.load_clouds(idxs, voxel=0.15, max_range=30.0)
    rows = []
    for i, j in [(0, 1), (1, 2), (2, 3), (10, 11), (30, 31), (50, 51)]:
        T = K.relative_velo_pose(poses, i, j, T_velo_imu)
        fwd = nn_stats(clouds[i], clouds[j], T)
        inv = nn_stats(clouds[i], clouds[j], np.linalg.inv(T))
        rows.append(dict(i=i, j=j, T=T.tolist(),
                         translation=float(np.linalg.norm(T[:3, 3])),
                         rotation_deg=float(np.degrees(np.arccos(
                             np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                         forward=fwd, inverse=inv,
                         direction="i->j" if fwd["median"] < inv["median"] else "j->i"))
        print(f"{i}->{j}: |t|={rows[-1]['translation']:.3f} m, "
              f"rot={rows[-1]['rotation_deg']:.2f} deg, fwd NN={fwd['median']:.3f} m, "
              f"inv NN={inv['median']:.3f} m, dir={rows[-1]['direction']}", flush=True)
    (OUT / "verify_kitti.json").write_text(json.dumps(dict(rows=rows), indent=2))
    print("wrote", OUT / "verify_kitti.json")


if __name__ == "__main__":
    main()
