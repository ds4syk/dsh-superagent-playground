#!/usr/bin/env python3
"""Minimal real stereo visual front-end (rectified pinhole) -> 3D-3D pairs.

This is a *hand-crafted* stereo feature front-end (ORB or SIFT, mutual-NN
matching, sub-pixel corner refinement, analytic rectified triangulation).
It is written for KITTI raw rectified cameras (image_02 / image_03), which
share the same intrinsics and differ only in the ``P[0,3]`` term, so the
disparity -> depth map is a closed form:

    Z = fx * b / d,   b = (P_L[0,3] - P_R[0,3]) / fx
    X = (u_L - cx) * Z / fx,   Y = (v_L - cy) * Z / fy

It deliberately contains **no ground-truth use** in matching/triangulation;
the GT relative pose is only applied afterwards, to *label* the 3D-3D
correspondences.

The same module also serves any rectified stereo pair (e.g. the UTIAS
In-The-Dark Bumblebee XB3 images, once/if downloaded: 512x384,
f=387.777 px, b=0.239965 m, no distortion model documented -> identity
rectification), so the front-end is dataset-agnostic.
"""
from __future__ import annotations

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def make_detector(kind: str, nfeatures: int = 3000):
    """Create a detector; ``nlevels=1`` keeps keypoints at native scale so the
    sub-pixel refinement below is meaningful."""
    if kind.lower() == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=1,
                              edgeThreshold=19, fastThreshold=15)
    if kind.lower() == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01)
    raise ValueError(f"unknown detector {kind}")


def detect(det, gray, mask=None):
    kps, desc = det.detectAndCompute(gray, mask)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), None
    pts = np.array([kp.pt for kp in kps], np.float32)
    return pts, desc


def subpixel(gray, pts, win: int = 5):
    """Refine keypoint locations to sub-pixel accuracy (both images, same call)."""
    if len(pts) == 0:
        return pts
    p = pts.reshape(-1, 1, 2).astype(np.float32).copy()
    cv2.cornerSubPix(gray, p, (win, win), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return p.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match(desc_a, desc_b, kind: str, ratio: float = 0.8):
    """Ratio-test + mutual-nearest-neighbour match. Returns (idx_a, idx_b)."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    norm = cv2.NORM_HAMMING if kind.lower() == "orb" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    fwd = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for m in fwd:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance:
            good.append((a.queryIdx, a.trainIdx))
    if not good:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    bwd = bf.knnMatch(desc_b, desc_a, k=1)
    # right->left best match; a true mutual NN has back[right] == left
    back = {b[0].queryIdx: b[0].trainIdx for b in bwd if len(b)}
    ia = np.array([g[0] for g in good], np.int64)
    ib = np.array([g[1] for g in good], np.int64)
    keep = np.array([back.get(int(j), -1) == int(i) for i, j in zip(ia, ib)])
    return ia[keep], ib[keep]


# ---------------------------------------------------------------------------
# Rectified stereo triangulation
# ---------------------------------------------------------------------------
def intr_from_P(P):
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    b = (P[0, 3] - P[2, 3] * 0) / fx  # caller passes P_L and P_R; b from diff
    return fx, fy, cx, cy


def stereo_triangulate(uv_l, uv_r, P_L, P_R, min_disp=1.0, max_disp=140.0,
                       dy_max=2.0, z_max=60.0):
    """Triangulate rectified stereo matches.

    ``uv_l`` (N,2) keypoints in the left image, ``uv_r`` (N,2) matches in the
    right image (same row after rectification).  Returns X (M,3) in the left
    camera frame and the kept indices into ``uv_l``.
    """
    fx, fy, cx, cy = P_L[0, 0], P_L[1, 1], P_L[0, 2], P_L[1, 2]
    b = (P_L[0, 3] - P_R[0, 3]) / fx
    d = uv_l[:, 0] - uv_r[:, 0]
    dy = np.abs(uv_l[:, 1] - uv_r[:, 1])
    ok = (d > min_disp) & (d < max_disp) & (dy < dy_max)
    d = np.where(np.abs(d) < 1e-6, 1e-6, d)
    Z = fx * b / d
    ok &= (Z > 0.5) & (Z < z_max)
    X = np.stack([(uv_l[:, 0] - cx) * Z / fx,
                  (uv_l[:, 1] - cy) * Z / fy,
                  Z], axis=1)
    return X[ok], np.where(ok)[0], float(b)


def correspondences(frame_a, frame_b, kind="orb", nfeatures=3000,
                    stereo_ratio=0.8, temporal_ratio=0.8, z_max=60.0,
                    min_disp=1.0, dy_max=2.0):
    """Build 3D-3D correspondences between two rectified stereo frames.

    ``frame_a``/``frame_b`` are dicts with keys ``left``/``right`` (BGR
    images) and ``P_L``/``P_R`` (3x4 projection matrices).  The returned
    ``x`` are 3D points in frame A's left camera, ``y`` the same matched
    landmarks triangulated independently in frame B's left camera.
    """
    det = make_detector(kind, nfeatures)
    out = {"n_kp_L": [], "n_stereo": [], "n_temporal": [], "n_corr": 0}
    cache = {}
    for key, f in (("a", frame_a), ("b", frame_b)):
        gl = cv2.cvtColor(f["left"], cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(f["right"], cv2.COLOR_BGR2GRAY)
        pl, dl = detect(det, gl)
        pr, dr = detect(det, gr)
        i_l, i_r = match(dl, dr, kind, ratio=stereo_ratio)
        pl_r = subpixel(gl, pl[i_l]) if len(i_l) else pl[:0]
        pr_r = subpixel(gr, pr[i_r]) if len(i_r) else pr[:0]
        X, kept, b = stereo_triangulate(pl_r, pr_r, f["P_L"], f["P_R"],
                                        min_disp=min_disp, dy_max=dy_max, z_max=z_max)
        cache[key] = {"pts": pl, "desc": dl, "stereo_idx": i_l[kept], "X": X,
                      "n_stereo": len(i_l), "baseline": b}
        out["n_kp_L"].append(len(pl))
        out["n_stereo"].append(len(i_l))

    A, B = cache["a"], cache["b"]
    ia, ib = match(A["desc"], B["desc"], kind, ratio=temporal_ratio)
    out["n_temporal"] = len(ia)
    l2s_a = {int(s): k for k, s in enumerate(A["stereo_idx"])}
    l2s_b = {int(s): k for k, s in enumerate(B["stereo_idx"])}
    rows = []
    for i, j in zip(ia, ib):
        ka = l2s_a.get(int(i))
        kb = l2s_b.get(int(j))
        if ka is None or kb is None:
            continue
        rows.append((A["X"][ka], B["X"][kb]))
    if not rows:
        out.update(x=np.zeros((0, 3)), y=np.zeros((0, 3)))
        return out
    x = np.stack([r[0] for r in rows])
    y = np.stack([r[1] for r in rows])
    out.update(x=x, y=y, n_corr=len(x))
    return out


# ---------------------------------------------------------------------------
# Dense-disparity variant (StereoSGBM): far more correspondences than sparse
# left-right ORB/SIFT matching, and a lower depth bias at mid range.
# ---------------------------------------------------------------------------
def make_sgbm(num_disparities: int = 128, block_size: int = 5) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def frame_dense(frame, det, sgbm, z_min=3.0, z_max=60.0, min_disp=1.0):
    """Keypoints + descriptors + per-keypoint 3D via dense SGBM disparity."""
    gl = cv2.cvtColor(frame["left"], cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(frame["right"], cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    pts, desc = detect(det, gl)
    if len(pts) == 0:
        return pts, desc, np.zeros((0, 3)), np.zeros(0, bool)
    pts = subpixel(gl, pts)
    P_L, P_R = frame["P_L"], frame["P_R"]
    fx, fy, cx, cy = P_L[0, 0], P_L[1, 1], P_L[0, 2], P_L[1, 2]
    b = (P_L[0, 3] - P_R[0, 3]) / fx
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, disp.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, disp.shape[0] - 1)
    d = disp[v, u]
    Z = np.full(len(pts), np.nan)
    ok = d > min_disp
    Z[ok] = fx * b / d[ok]
    ok &= (Z > z_min) & (Z < z_max)
    X = np.full((len(pts), 3), np.nan)
    X[ok, 0] = (pts[ok, 0] - cx) * Z[ok] / fx
    X[ok, 1] = (pts[ok, 1] - cy) * Z[ok] / fy
    X[ok, 2] = Z[ok]
    return pts, desc, X, ok


def correspondences_dense(frame_a, frame_b, kind="sift", nfeatures=4000,
                          temporal_ratio=0.8, z_max=60.0, min_disp=1.0):
    """Same contract as :func:`correspondences`, using dense SGBM disparity."""
    det = make_detector(kind, nfeatures)
    sgbm = make_sgbm()
    out = {"n_kp_L": [], "n_stereo": [], "n_temporal": [], "n_corr": 0}
    cache = {}
    for key, f in (("a", frame_a), ("b", frame_b)):
        pts, desc, X, ok = frame_dense(f, det, sgbm, z_max=z_max, min_disp=min_disp)
        cache[key] = {"pts": pts, "desc": desc, "X": X, "ok": ok}
        out["n_kp_L"].append(len(pts))
        out["n_stereo"].append(int(ok.sum()))
    A, B = cache["a"], cache["b"]
    ia, ib = match(A["desc"], B["desc"], kind, ratio=temporal_ratio)
    out["n_temporal"] = len(ia)
    keep = [(a, b) for a, b in zip(ia, ib) if A["ok"][a] and B["ok"][b]]
    if not keep:
        out.update(x=np.zeros((0, 3)), y=np.zeros((0, 3)))
        return out
    x = np.stack([A["X"][a] for a, _ in keep])
    y = np.stack([B["X"][b] for _, b in keep])
    out.update(x=x, y=y, n_corr=len(x))
    return out
