#!/usr/bin/env python3
"""Direction check for the odometry loader (same test as the raw-drive parent).

For a handful of consecutive pairs per sequence, map frame i into frame j with
the GT transform and with its inverse, and compare the nearest-neighbour median
distance on real LiDAR points.  The correct convention must be much tighter.

Run: /home/syk/miniconda3/envs/WM/bin/python verify_kitti_odom.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_kitti_odom as K  # noqa: E402

OUT = HERE / "results"
NPTS = 3000


def nn_median(cloud_i, cloud_j, T, rng):
    idx = rng.choice(len(cloud_i), min(NPTS, len(cloud_i)), replace=False)
    x = cloud_i[idx]
    xj = (T[:3, :3] @ x.T).T + T[:3, 3]
    d, _ = cKDTree(cloud_j).query(xj, k=1, workers=-1)
    return float(np.median(d))


def available_pairs(seq, want=6):
    """Consecutive pairs whose two frames were actually streamed from the mirror."""
    man = json.loads((K.DATA / "manifest.json").read_text())
    frames = set(man[seq]["frames"])
    pairs = [(i, i + 1) for i in sorted(frames) if i + 1 in frames]
    if not pairs:
        pairs = [(a, b) for a, b in man[seq]["pairs"]]
    if len(pairs) <= want:
        return pairs
    pick = np.linspace(0, len(pairs) - 1, want).round().astype(int)
    return [pairs[k] for k in sorted(set(pick.tolist()))]


def main():
    t0 = time.time()
    rows, verdicts = [], []
    for seq in K.SEQS:
        poses = K.load_poses(seq)
        T_cam_velo = K.read_calib_tr(seq)
        pairs = available_pairs(seq)
        clouds = K.load_clouds(seq, sorted({i for p in pairs for i in p}), voxel=0.15)
        for k, (i, j) in enumerate(pairs):
            rng = np.random.default_rng(np.random.SeedSequence([20261127, int(seq), i]))
            T = K.relative_velo_pose(poses, i, j, T_cam_velo)
            m_fwd = nn_median(clouds[i], clouds[j], T, rng)
            m_inv = nn_median(clouds[i], clouds[j], np.linalg.inv(T), rng)
            tmag, rot = K.pose_motion(poses, i, j, T_cam_velo)
            ok = m_fwd < m_inv
            verdicts.append(ok)
            rows.append(dict(seq=seq, i=i, j=j, t_m=tmag, rot_deg=rot,
                             nn_fwd=m_fwd, nn_inv=m_inv,
                             direction="i->j" if ok else "j->i", pass_=bool(ok)))
            print(f"seq {seq} {i}->{j}: |t|={tmag:.2f} m rot={rot:.3f} deg  "
                  f"fwd={m_fwd:.3f} inv={m_inv:.3f}  {'PASS' if ok else 'FAIL'}", flush=True)
    summary = dict(n_pairs=len(rows), n_pass=int(sum(verdicts)),
                   all_pass=bool(all(verdicts)),
                   fwd_range=[min(r["nn_fwd"] for r in rows), max(r["nn_fwd"] for r in rows)],
                   inv_range=[min(r["nn_inv"] for r in rows), max(r["nn_inv"] for r in rows)],
                   elapsed_s=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify_kitti_odom.json").write_text(
        json.dumps(dict(summary=summary, pairs=rows), indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
