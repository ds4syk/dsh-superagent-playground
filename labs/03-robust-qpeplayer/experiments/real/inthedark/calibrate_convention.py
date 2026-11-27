#!/usr/bin/env python3
"""Decide the In-The-Dark transform row convention from stereo geometry alone.

For a repeat-run pair (va, vb) we have four candidate "relative poses":

  ba       compose_pairs(convention='ba')      -- p_b = T_ba p_a
  ab       compose_pairs(convention='ab')      -- transpose reading
  inv(ba)  direction-reversed ba
  inv(ab)  direction-reversed ab

The test needs no GT: SIFT-match left(va)<->left(vb), triangulate the va points
from the va stereo pair (known baseline 0.239965 m), push them through the
candidate pose, project into the vb left image and measure the pixel residual
against the matched vb pixels.  Only the geometry decides.

Usage:
  $PY calibrate_convention.py --run .../run_000002 --teach .../teach_meta \
      --start 0 --pairs 12 --out results/convention_calib.json
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
import compose_inthedark as C  # noqa: E402
import load_inthedark as L  # noqa: E402
import stereo_vo as S  # noqa: E402


def reproj_residual(fa, fb, T, det, nfeatures=4000, z_max=60.0):
    """Median |uv_b - pi(T X_a)| in px over matched landmarks."""
    cache = {}
    for k, f in (("a", fa), ("b", fb)):
        gl = cv2.cvtColor(f["left"], cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(f["right"], cv2.COLOR_BGR2GRAY)
        pl, dl = S.detect(det, gl)
        pr, dr = S.detect(det, gr)
        i_l, i_r = S.match(dl, dr, "sift", 0.8)
        if len(i_l) < 8:
            return None, 0, 0
        X, kept, b = S.stereo_triangulate(pl[i_l], pr[i_r], f["P_L"], f["P_R"],
                                          z_max=z_max)
        cache[k] = dict(uv=pl[i_l[kept]], desc=dl, st=i_l[kept], X=X)
    A, B = cache["a"], cache["b"]
    ia, ib = S.match(A["desc"], B["desc"], "sift", 0.8)
    a2s = {int(s): i for i, s in enumerate(A["st"])}
    b2s = {int(s): i for i, s in enumerate(B["st"])}
    rows = []
    for i, j in zip(ia, ib):
        ka, kb = a2s.get(int(i)), b2s.get(int(j))
        if ka is None or kb is None:
            continue
        rows.append((A["X"][ka], B["uv"][kb]))
    if len(rows) < 8:
        return None, len(rows), len(A["X"])
    X = np.stack([r[0] for r in rows])
    uv_b = np.stack([r[1] for r in rows])
    P = fa["P_L"]
    Xc = (T[:3, :3] @ X.T).T + T[:3, 3]
    z = Xc[:, 2]
    ok = z > 0.3
    if ok.sum() < 8:
        return None, len(rows), int(ok.sum())
    u = P[0, 0] * Xc[ok, 0] / z[ok] + P[0, 2]
    v = P[1, 1] * Xc[ok, 1] / z[ok] + P[1, 2]
    e = np.hypot(u - uv_b[ok, 0], v - uv_b[ok, 1])
    return float(np.median(e)), len(rows), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teach", required=True)
    ap.add_argument("--data-root", default="/mnt/storage1/syk/data/utias_in_the_dark")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--z-max", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = L.run_dir(a.run)
    comp = {c: C.compose_pairs(d, a.teach, c) for c in ("ba", "ab")}
    E = C.load_camera_vehicle(a.data_root)
    cand = {
        # hypothesis space: row reading (ba/ab) x camera conjugation x direction
        "cam_ba": lambda T: E @ T @ np.linalg.inv(E),
        "cam_ab": lambda T: E @ T @ np.linalg.inv(E),
        "cam_ba_inv": lambda T: np.linalg.inv(E @ T @ np.linalg.inv(E)),
        "cam_ab_inv": lambda T: np.linalg.inv(E @ T @ np.linalg.inv(E)),
        "veh_ba": lambda T: T,
        "identity": lambda T: np.eye(4),
    }
    order = sorted(comp["ba"]["pairs"])
    sel = order[a.start:a.start + a.pairs]
    P_L, P_R = L.projection_matrices()
    det = S.make_detector("sift", 4000)
    rows = []
    for va, vb in sel:
        fa = L.load_frame(d, va, P_L, P_R)
        fb = L.load_frame(d, vb, P_L, P_R)
        T_ba, T_ab = comp["ba"]["pairs"][(va, vb)], comp["ab"]["pairs"][(va, vb)]
        src = {"cam_ba": T_ba, "cam_ab": T_ab, "cam_ba_inv": T_ba,
               "cam_ab_inv": T_ab, "veh_ba": T_ba, "identity": T_ba}
        rec = dict(pair=f"{va}->{vb}", gt_trans=float(np.linalg.norm(T_ba[:3, 3])))
        for name, fn in cand.items():
            e, n_corr, n_proj = reproj_residual(fa, fb, fn(src[name]), det,
                                                z_max=a.z_max)
            rec[name] = e
            rec[f"n_{name}"] = n_corr
        rows.append(rec)
        print(f"{va}->{vb} " + " ".join(
            f"{k}={rec[k]:.2f}px" if rec[k] is not None else f"{k}=NA"
            for k in cand), flush=True)
    summary = {}
    for name in cand:
        v = [r[name] for r in rows if r[name] is not None]
        summary[name] = dict(median_px=float(np.median(v)) if v else None,
                             n=len(v))
    best = min((s["median_px"], k) for k, s in summary.items()
               if s["median_px"] is not None)[1]
    out = dict(run=d.name, teach=str(a.teach), start=a.start, n_pairs=len(rows),
               rows=rows, summary=summary, winner=best,
               ratio_veh_over_cam=(summary["veh_ba"]["median_px"] /
                                   summary["cam_ba"]["median_px"]
                                   if summary["cam_ba"]["median_px"] else None))
    print(json.dumps(dict(summary=summary, winner=best), indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
