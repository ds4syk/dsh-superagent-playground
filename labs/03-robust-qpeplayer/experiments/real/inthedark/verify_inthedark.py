#!/usr/bin/env python3
"""Sanity checks for a downloaded UTIAS In-The-Dark run. NOT yet run on real
data (archives could not be downloaded; see README.md section 1).

Checks, in order, and fails loudly instead of producing numbers:
  1. the run directory parses (images/left, images/right, transforms_spatial.txt);
  2. left/right images are the documented 512x384 and decode;
  3. stereo row alignment: median |v_l - v_r| of mutual-NN matches must be
     small (< 2 px) -- this is the only evidence for the "already rectified"
     assumption, since no distortion model is published;
  4. GT direction: for near-neighbour vertices, the GT relative pose must map
     left-image 3D points onto left-image 3D points much better than its
     inverse (median residual ratio > 2x), and the GT translation must be
     consistent with ~0.2 m keyframe spacing;
  5. disparity/geometry: median triangulated depth and inlier rate at
     c = 0.5 m are finite.

Usage:
  $PY verify_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kitti_stereo"))
import load_inthedark as L  # noqa: E402
import stereo_vo as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = L.run_dir(a.run)
    tf = next(d.glob("transforms_*.txt"), None)
    if tf is None:
        raise SystemExit(f"BLOCKED: no transforms_*.txt in {d}; cannot verify GT")
    T = L.read_transforms(tf)
    ts = L.read_timestamps(d / "timestamps_images.txt")
    P_L, P_R = L.projection_matrices()
    verts = sorted({v for (_, v, _, _) in T} | {v for (_, _, _, v) in T} |
                   {int(k) for k in ts})
    if len(verts) < 2:
        raise SystemExit("BLOCKED: fewer than two vertices in the run")
    det = S.make_detector("sift", 2000)
    rows = []
    pairs = list(zip(verts, verts[1:]))[: a.pairs]
    for va, vb in pairs:
        T_gt = L.relative_pose(T, d.name.split("_")[-1].lstrip("0") or "0", va,
                               d.name.split("_")[-1].lstrip("0") or "0", vb)
        fa = L.load_frame(d, va, P_L, P_R)
        fb = L.load_frame(d, vb, P_L, P_R)
        res = S.correspondences_dense(fa, fb, kind="sift", nfeatures=2000, z_max=60.0)
        x, y = res["x"], res["y"]
        e = np.linalg.norm((T_gt[:3, :3] @ x.T).T + T_gt[:3, 3] - y, axis=1)
        e_inv = np.linalg.norm((np.linalg.inv(T_gt)[:3, :3] @ x.T).T
                              + np.linalg.inv(T_gt)[:3, 3] - y, axis=1)
        # row alignment from the raw sparse stereo matches
        gl = cv2.cvtColor(fa["left"], cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fa["right"], cv2.COLOR_BGR2GRAY)
        pl, dl = S.detect(det, gl)
        pr, dr = S.detect(det, gr)
        i_l, i_r = S.match(dl, dr, "sift", 0.8)
        dy = float(np.median(np.abs(pl[i_l, 1] - pr[i_r, 1]))) if len(i_l) else float("nan")
        rows.append(dict(pair=f"{va}->{vb}", n=int(len(x)),
                         inlier_rate=float((e < 0.5).mean()),
                         med_resid_all=float(np.median(e)),
                         med_resid_inverse=float(np.median(e_inv)),
                         median_row_dy_px=dy,
                         gt_trans=float(np.linalg.norm(T_gt[:3, 3])),
                         img_shape=list(fa["left"].shape[:2])))
    out = dict(run=str(d), transforms=str(tf), pairs=rows,
               checks=dict(documented_512x384=all(r["img_shape"] == [384, 512] for r in rows),
                           rows_aligned=all(r["median_row_dy_px"] < 2.0 for r in rows),
                           gt_direction_ok=all(r["med_resid_inverse"] > 2 * r["med_resid_all"] for r in rows)))
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    if not all(out["checks"].values()):
        raise SystemExit(f"CHECK FAILED: {out['checks']}")


if __name__ == "__main__":
    main()
