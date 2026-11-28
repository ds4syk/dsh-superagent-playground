#!/usr/bin/env python3
"""OpenCV SQPNP batch solver for the LMO comparison (runs in the sdpr env).

Why a separate process
----------------------
`/home/syk/miniconda3/envs/WM/bin/python` (the env that owns the P1 estimator
code) has no `cv2`, while `/home/syk/miniconda3/envs/sdpr/bin/python` has
OpenCV 5.0.0 with `cv2.SOLVEPNP_SQPNP`.  Rather than install anything or change
envs, `run_two_mode_lmo_full.py` writes the *same* 3-D model points / 2-D image
points it used for the two-mode cost into an `.npz`, and this helper solves them
in the sdpr env and writes the solutions back.

Input npz
---------
  K                 : (3, 3) float64            camera matrix (same for all frames)
  meta              : 0-d string array, JSON list of {"key": str, "set": str}
  "<key>|<set>|P"   : (N, 3) float64             3-D points in the model frame
  "<key>|<set>|uv"  : (N, 2) float64             pixel coordinates

`set` is "plane" (the RANSAC plane inliers, i.e. exactly the points the P1
estimator uses) or "all" (all 900 depth-model correspondences).

Output npz
----------
  meta : 0-d string array, JSON dict  key -> set -> {
      "status": "ok" | "fail",
      "n_points": int,
      "n_solutions": int,
      "solutions": [{"rvec": [...], "tvec": [...], "reproj_px": float}, ...],
      "chosen": int,       # index into solutions, lowest mean reprojection error
      "error": str | None,
  }

No GT is read here: the solver picks the solution with the lowest mean
reprojection error, exactly what a classical pipeline would do.  Choosing the
GT-closest solution is done *downstream* only as a labelled oracle diagnostic.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np


def solve_one(K: np.ndarray, P: np.ndarray, uv: np.ndarray) -> dict:
    P = np.ascontiguousarray(P, dtype=np.float64)
    uv = np.ascontiguousarray(uv, dtype=np.float64)
    try:
        out = cv2.solvePnPGeneric(P, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
        return dict(status="fail", n_points=int(len(P)), n_solutions=0,
                    solutions=[], chosen=None, error=str(exc).splitlines()[-1][:300])
    # OpenCV 4.x returns (ok, rvecs, tvecs, reproj_errs);
    # OpenCV 5.0 here returns (n_solutions, rvecs, tvecs, reproj_errs).
    if len(out) == 5:
        _ok, rvecs, tvecs, _inl, _errs = out
    else:
        _ok, rvecs, tvecs, _errs = out
    rvecs = np.asarray(rvecs, dtype=float).reshape(-1, 3)
    tvecs = np.asarray(tvecs, dtype=float).reshape(-1, 3)
    sols = []
    for rv, tv in zip(rvecs, tvecs):
        Re = cv2.Rodrigues(rv.reshape(3, 1))[0]
        X = (Re @ P.T).T + tv
        z = X[:, 2]
        uvp = (K @ X.T).T
        uvp = uvp[:, :2] / uvp[:, 2:3]
        reproj = float(np.mean(np.linalg.norm(uvp - uv, axis=1)))
        sols.append(dict(rvec=[float(x) for x in rv],
                         tvec=[float(x) for x in tv],
                         reproj_px=reproj,
                         n_in_front=int(np.count_nonzero(z > 0))))
    if not sols:
        return dict(status="fail", n_points=int(len(P)), n_solutions=0,
                    solutions=[], chosen=None, error="no solution returned")
    # data-driven selection: lowest mean reprojection error; tie -> prefer the
    # solution with more points in front of the camera, then the first.
    best = min(range(len(sols)),
               key=lambda i: (sols[i]["reproj_px"], -sols[i]["n_in_front"], i))
    return dict(status="ok", n_points=int(len(P)), n_solutions=len(sols),
                solutions=sols, chosen=int(best), error=None)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sqpnp_lmo_helper.py IN.npz OUT.npz", file=sys.stderr)
        return 2
    zin = np.load(sys.argv[1], allow_pickle=False)
    K = np.asarray(zin["K"], dtype=float)
    meta = json.loads(str(zin["meta"][()]))
    out = {}
    for rec in meta:
        key, st = rec["key"], rec["set"]
        P = np.asarray(zin[f"{key}|{st}|P"], dtype=float)
        uv = np.asarray(zin[f"{key}|{st}|uv"], dtype=float)
        out.setdefault(key, {})[st] = solve_one(K, P, uv)
    np.savez(sys.argv[2], meta=np.array(json.dumps(out)))
    print(f"sqpnp: solved {len(meta)} problems for {len(out)} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
