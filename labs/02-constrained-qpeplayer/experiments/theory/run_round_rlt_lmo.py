#!/usr/bin/env python3
"""round-rlt on a small real BOP LINEMOD-Occlusion (LMO) subset.

Loader pattern copied from `experiments/real/bop/run_bop.py` (not edited):
real mesh vertices, real camera intrinsics, RANSAC table plane from the
background depth, 2D-3D correspondences by projecting the model at the GT pose
plus pixel noise + downward pixel bias.  Only the *certificate* comparison is
new: base Shor vs bound-RLT cuts, both with the rigorous dual repair.

This is a layer-level diagnostic (small N), not a full benchmark.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402
from certificate import shor_bound  # noqa: E402

ROOT = Path("/mnt/storage1/syk/data/bop/lmo")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

N = 30
P_CON = 16
NOISE_PX = 1.0
BIAS_PX = [20.0, 30.0, 40.0]
MAX_IMG = 2
FRAMES = sorted([d.name for d in (ROOT / "test").iterdir() if d.is_dir()])[:6]
N_MAX = int(8)


def load_model(obj_id):
    p = ROOT / "models" / f"obj_{obj_id:06d}.ply"
    v = PlyData.read(str(p))["vertex"]
    return np.stack([np.asarray(v[a], dtype=float) for a in "xyz"], axis=1) * 0.001


def backproject(depth, K):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(us - cx) / fx * depth, (vs - cy) / fy * depth, depth], axis=-1)


def fit_table_plane(depth, mask, bbox, K, rng):
    x, y, w, h = bbox
    bg = (~mask) & (depth > 0)
    reg = np.zeros_like(bg)
    x0, y0 = max(0, x - 40), max(0, y - 40)
    x1, y1 = min(bg.shape[1], x + w + 40), min(bg.shape[0], y + h + 40)
    reg[y0:y1, x0:x1] = True
    ys, xs = np.where(bg & reg)
    if len(xs) < 100:
        return None
    idx = rng.choice(len(xs), min(4000, len(xs)), replace=False)
    X = backproject(depth, K)[ys[idx], xs[idx]]
    X = X[np.isfinite(X).all(axis=1)]
    best = None
    for _ in range(300):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        d = -n @ p[0]
        inl = np.abs(X @ n + d) < 0.005
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), n, d, X[inl])
    if best is None:
        return None
    _, n, d, inl = best
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[-1]
    return n, -n @ c


def project(K, X):
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:3]


def main():
    rng = np.random.default_rng(20261127)
    from collections import Counter
    cnt = Counter()
    for frame in FRAMES:
        gt_all = json.load(open(ROOT / "test" / frame / "scene_gt.json"))
        for gt in gt_all.values():
            for g in gt:
                cnt[g["obj_id"]] += 1
    obj_ids = [oid for oid, _ in cnt.most_common(3)]
    models = {oid: load_model(oid) for oid in obj_ids}
    print("frames", FRAMES, "objects", obj_ids, flush=True)
    rows = []
    t0 = time.time()
    for frame in FRAMES:
        if len(rows) >= N_MAX:
            break
        fdir = ROOT / "test" / frame
        gt_all = json.load(open(fdir / "scene_gt.json"))
        info_all = json.load(open(fdir / "scene_gt_info.json"))
        cam_all = json.load(open(fdir / "scene_camera.json"))
        n_img = 0
        for img, gt in gt_all.items():
            if n_img >= MAX_IMG or len(rows) >= N_MAX:
                break
            n_img += 1
            info = info_all[img]
            cam = cam_all[img]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            imgz = f"{int(img):06d}"
            depth = np.asarray(Image.open(fdir / "depth" / f"{imgz}.png"), dtype=float) * 0.001
            for k, g in enumerate(gt):
                if len(rows) >= N_MAX:
                    break
                oid = g["obj_id"]
                if oid not in models or info[k]["visib_fract"] < 0.4:
                    continue
                mask = np.asarray(Image.open(fdir / "mask" / f"{imgz}_{k:06d}.png")) > 0
                plane = fit_table_plane(depth, mask, info[k]["bbox_visib"], K, rng)
                if plane is None:
                    continue
                n, d = plane
                R = np.array(g["cam_R_m2c"]).reshape(3, 3)
                t = np.array(g["cam_t_m2c"]) * 0.001
                V = models[oid]
                x_gt = (R @ V.T).T + t
                if n @ x_gt.mean(0) + d < 0:
                    n, d = -n, -d
                gmin = (x_gt @ n + d).min()
                d = d - gmin + 1e-4
                for bias_px in BIAS_PX:
                    idx = rng.choice(len(V), N, replace=False)
                    P = V[idx]
                    x = (R @ P.T).T + t
                    uv = project(K, x)
                    uv = uv + rng.normal(scale=NOISE_PX, size=uv.shape) + np.array([0.0, bias_px])
                    U = np.linalg.inv(K) @ np.vstack([uv.T, np.ones(len(uv))])
                    U = U.T
                    U = U / np.linalg.norm(U, axis=1, keepdims=True)
                    M0, D0, G0 = core.build_objective(P, U)
                    order = np.argsort(x_gt @ n + d)
                    A, B = [], []
                    for j in order[:P_CON]:
                        a, b = core.c_plane(V[j], n, d)
                        A.append(a); B.append(b)
                    A = np.array(A); B = np.array(B)
                    Ju, q_u, t_u = core.solve_qp_global(M0, D0, G0, n_starts=32, rng=np.random.default_rng(1))
                    alm = core.alm_qpep(M0, D0, G0, A, B, q_u, t_u, rho0=10.0, n_starts=6, rng=rng)
                    J = core.obj_value(alm["q"], alm["t"], M0, D0, G0)
                    feas = core.max_viol(A, B, alm["q"], alm["t"]) <= 1e-6
                    for vname, vkw in [("base", {}), ("bound_rlt", dict(add_bound_rlt=True))]:
                        r = shor_bound(M0, D0, G0, A, B, solver="SCS", eps=1e-9,
                                       max_iters=50000, repair_dual=True, **vkw)
                        rows.append(dict(frame=frame, img=img, obj=oid, bias_px=bias_px,
                                         variant=vname, J=float(J), alm_feasible=bool(feas),
                                         value=float(r["value"]), rigorous=float(r["rigorous"]),
                                         status=r["status"], ratio=float(r["ratio"]),
                                         n_ineq=r["n_ineq"], time_s=float(r["time_s"]),
                                         gap=float(J - r["value"]),
                                         rigorous_gap=float(J - r["rigorous"]),
                                         cert_naive=bool(feas and J - r["value"] <= 1e-6),
                                         cert_rigorous=bool(feas and J - r["rigorous"] <= 1e-6),
                                         dual_lam_min=float(r["dual_lam_min"])))
                    print(rows[-1], flush=True)
                    if len(rows) >= N_MAX * len(BIAS_PX):
                        break
    json.dump(rows, open(OUT / "round_rlt_lmo.json", "w"), indent=2, default=float)
    print("wrote", OUT / "round_rlt_lmo.json", "rows", len(rows), "elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
