#!/usr/bin/env python3
"""Loader/adapter for the UTIAS In-The-Dark run format, exercised on the real
run_000002 / run_000012 archives (see README.md for the GT protocol).

Format implemented here (official docs + utiasASRL/vtr-dataset-tools):

    run_XXXXXX/
      images/left/000000.png, 000001.png, ...
      images/right/000000.png, ...
      timestamps_images.txt      # "vertex, unix_ns" per row
      transforms_spatial.txt     # repeat runs: "iA_run,iA_vertex,iB_run,iB_vertex,
                                 #   T_ba row-major 16 values", p_b = T_ba p_a
      transforms_temporal.txt    # teach run, same column layout
      gps.txt, timestamps_gps.txt  (optional, 23/39 runs)

Per-pair GT is NOT in the run directory: ``compose_inthedark.py`` composes it
from this run's ``transforms_spatial.txt`` and the teach run's
``transforms_temporal.txt``, then conjugates it into the camera frame with the
official ``transform_camera_vehicle.txt``.

The adapter returns exactly the frame dict that ``stereo_vo`` consumes
(``left``/``right`` BGR images, ``P_L``/``P_R`` 3x4 projection matrices), so
the KITTI stereo front-end and the robust layer run unchanged.

Camera (official ``camera_parameters.txt``): 512x384, f=387.777, cx=257.446,
cy=197.718, baseline=0.239965 m, no documented distortion model -> the images
are assumed rectified.  ``verify_inthedark.py`` checks that assumption on real
data (median left/right row difference 0.56-0.58 px on runs 000002/000012).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CAM = dict(width=512, height=384, fx=387.777, fy=387.777, cx=257.446,
           cy=197.718, baseline=0.239965)


def projection_matrices(fx=None, fy=None, cx=None, cy=None, b=None):
    """P_L, P_R for a rectified stereo pair (b>0 = baseline in metres)."""
    fx = CAM["fx"] if fx is None else fx
    fy = CAM["fy"] if fy is None else fy
    cx = CAM["cx"] if cx is None else cx
    cy = CAM["cy"] if cy is None else cy
    b = CAM["baseline"] if b is None else b
    P_L = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], float)
    P_R = np.array([[fx, 0, cx, -fx * b], [0, fy, cy, 0], [0, 0, 1, 0]], float)
    return P_L, P_R


def read_transforms(path):
    """Parse transforms_{spatial,temporal}.txt -> {(i,j): 4x4} with T_ba: p_b = T_ba p_a."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        v = [float(x) for x in line.replace(";", ",").split(",") if x.strip()]
        if len(v) < 20:
            raise ValueError(f"bad transform row ({len(v)} values): {line[:80]}")
        ia, ja, ib, jb = (int(round(v[0])), int(round(v[1])),
                          int(round(v[2])), int(round(v[3])))
        T = np.array(v[4:20], float).reshape(4, 4)
        out[(ia, ja, ib, jb)] = T
    return out


def read_timestamps(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        out[int(float(parts[0]))] = int(float(parts[1]))
    return out


def relative_pose(transforms, run_a, va, run_b, vb):
    """GT T_{b<-a} (p_b = T p_a) for vertices (run_a,va) -> (run_b,vb)."""
    direct = (run_a, va, run_b, vb)
    if direct in transforms:
        return transforms[direct]
    inv = (run_b, vb, run_a, va)
    if inv in transforms:
        return np.linalg.inv(transforms[inv])
    raise KeyError(f"no transform for {(run_a, va)} -> {(run_b, vb)}")


def run_dir(root):
    root = Path(root)
    return root if (root / "images").is_dir() else next(root.glob("run_*"))


def load_frame(root, vertex, P_L, P_R, max_side=None):
    d = run_dir(root)
    left = cv2.imread(str(d / "images" / "left" / f"{vertex:06d}.png"))
    right = cv2.imread(str(d / "images" / "right" / f"{vertex:06d}.png"))
    if left is None or right is None:
        raise FileNotFoundError(f"missing stereo image for vertex {vertex} in {d}")
    if max_side is not None:
        s = max_side / max(left.shape[:2])
        if s < 1.0:
            left = cv2.resize(left, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            P_L, P_R = P_L.copy(), P_R.copy()
            P_L[:2] *= s
            P_R[:2] *= s
    return {"left": left, "right": right, "P_L": P_L, "P_R": P_R}
