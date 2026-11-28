#!/usr/bin/env python3
"""P1 real-data symmetry experiment on BOP LINEMOD (eggbox / glue).

*** DEPRECATED / NEGATIVE ARTIFACT (kept reproducible on purpose, 2026-11-27) ***
This script is the original failed attempt.  It is NOT a valid negative result
about symmetric objects, because it contains three bugs:

  (S1) `sym_of()` uses the models_info 4x4 symmetry translation in millimetres
       as metres, offsetting the second hypothesis point cloud by ~0.69 m.
  (S2) `gap_at()` sorts J before taking J[1]-J[0], so the "gap" it feeds to the
       bracket search is >= 0 by construction; the sign-change test can never
       fire.  The empty table is therefore a code bug, not evidence that the
       data has no crossing.
  (S3) FRAMES omits 000011, and most of the listed frames contain objects other
       than obj 10/11 and are silently skipped; the original run effectively
       covers very few eggbox/glue frames.

  Even after fixing S1/S3, mixing two FULL bearing costs and then eliminating
  ONE shared translation pulls both wells to a compromise pose, so the cost is
  unimodal at the crossing (S4).  The corrected constructions are in
  `run_two_mode.py` (real planar-face two-mode cost + controlled transverse
  crossing with per-hypothesis translation elimination) and are documented in
  `README.md`.

The original docstring follows.

For a symmetric object, the same image is consistent with two poses A and
A*S.  We build two bearing-cost matrices from the same real model and image
projections: (P, U) for pose A and (S P, U) for pose A*S.  Their convex
combination M(theta) has two minima that tie at theta=0.5.  We then compare
hard argmin (undefined at the crossing), multi-modal Laplace, and mode-level
MC, using the real object mesh and camera intrinsics.

This is a controlled ambiguity experiment, not a detector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

HERE = Path(__file__).resolve().parent
P1CODE = HERE.parents[1] / "code"
sys.path.insert(0, str(P1CODE))
import qpep_core as p1  # noqa: E402
import instance as inst_mod  # noqa: E402

SQRT2 = np.sqrt(2.0)


def _m_rows(p):
    px, py, pz = p
    m = np.zeros((3, 10))
    m[0] = [px, px, -px, -px, 0.0, SQRT2 * pz, -SQRT2 * py, SQRT2 * py, SQRT2 * pz, 0.0]
    m[1] = [py, -py, py, -py, -SQRT2 * pz, 0.0, SQRT2 * px, SQRT2 * px, 0.0, SQRT2 * pz]
    m[2] = [pz, -pz, -pz, pz, SQRT2 * py, -SQRT2 * px, 0.0, 0.0, SQRT2 * px, SQRT2 * py]
    return m


def build_objective(P, U):
    N = P.shape[0]
    M0 = np.zeros((10, 10)); D0 = np.zeros((3, 10)); G0 = np.zeros((3, 3))
    for i in range(N):
        u = U[i]
        Pm = np.eye(3) - np.outer(u, u) / (u @ u)
        Mm = _m_rows(P[i])
        M0 += 0.5 * (Mm.T @ Pm @ Mm + Mm.T @ Pm.T @ Mm)
        D0 += Pm @ Mm
        G0 += Pm
    return M0, D0, G0

ROOT = Path("/mnt/storage1/syk/data/bop/lm")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

OBJ = {10: "eggbox", 11: "glue"}
N = 30
TAU = 0.01
H = 1e-3
FRAMES = ["000001", "000002", "000003", "000005", "000006", "000008", "000010", "000013", "000015"]


def load_model(obj_id):
    v = PlyData.read(str(ROOT / "models" / f"obj_{obj_id:06d}.ply"))["vertex"]
    return np.stack([np.asarray(v[a], dtype=float) for a in "xyz"], axis=1) * 0.001


def sym_of(obj_id):
    info = json.load(open(ROOT / "models" / "models_info.json"))
    S = np.array(info[str(obj_id)]["symmetries_discrete"][0], float).reshape(4, 4)
    return S[:3, :3], S[:3, 3]


def project(K, X):
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:3]


def mix(M0a, D0a, G0a, M0b, D0b, G0b, th):
    M0 = (1 - th) * M0a + th * M0b
    D0 = (1 - th) * D0a + th * D0b
    G0 = (1 - th) * G0a + th * G0b
    Mred = M0 - D0.T @ np.linalg.solve(G0, D0)
    GinvD = -np.linalg.solve(G0, D0)
    return Mred, GinvD, M0, D0, G0


def main():
    rng = np.random.default_rng(20261127)
    models = {oid: load_model(oid) for oid in OBJ}
    syms = {oid: sym_of(oid) for oid in OBJ}
    I = inst_mod.Instance(seed=20261127)
    rows = []
    for frame in FRAMES:
        fdir = ROOT / "test" / frame
        gt_all = json.load(open(fdir / "scene_gt.json"))
        cam_all = json.load(open(fdir / "scene_camera.json"))
        n_img = 0
        for img, gt in gt_all.items():
            if n_img >= 2:
                break
            n_img += 1
            imgz = f"{int(img):06d}"
            cam = cam_all[img]
            K = np.array(cam["cam_K"]).reshape(3, 3)
            for g in gt:
                oid = g["obj_id"]
                if oid not in models:
                    continue
                R = np.array(g["cam_R_m2c"]).reshape(3, 3)
                t = np.array(g["cam_t_m2c"]) * 0.001
                V = models[oid]
                SR, St = syms[oid]
                idx = rng.choice(len(V), N, replace=False)
                P = V[idx]
                x = (R @ P.T).T + t
                uv = project(K, x)
                uv = uv + rng.normal(scale=1.0, size=uv.shape)
                U = np.linalg.inv(K) @ np.vstack([uv.T, np.ones(len(uv))])
                U = U.T; U = U / np.linalg.norm(U, axis=1, keepdims=True)
                P_B = (SR @ P.T).T + St
                M0a, D0a, G0a = build_objective(P, U)
                M0b, D0b, G0b = build_objective(P_B, U)
                # locate the crossing theta* in [0.4,0.6] where the two
                # lowest-J modes tie (real object symmetry + noise gives a
                # small offset from 0.5)
                def gap_at(th):
                    Mr, Gi, *_ = mix(M0a, D0a, G0a, M0b, D0b, G0b, th)
                    mm = I.find_modes(Mr, n_starts=128)
                    if len(mm["J"]) < 2:
                        return None, mm
                    Js = np.sort(np.asarray(mm["J"]))
                    return float(Js[1] - Js[0]), mm
                grid = np.linspace(0.1, 0.9, 9)
                vals = [gap_at(t)[0] for t in grid]
                bracket = None
                for i in range(len(grid) - 1):
                    if vals[i] is None or vals[i + 1] is None:
                        continue
                    if vals[i] * vals[i + 1] <= 0:
                        bracket = (grid[i], grid[i + 1], vals[i], vals[i + 1])
                        break
                if bracket is None:
                    continue
                lo, hi, glo, ghi = bracket
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    gm, _ = gap_at(mid)
                    if gm is None:
                        break
                    if glo * gm <= 0:
                        hi = mid; ghi = gm
                    else:
                        lo = mid; glo = gm
                th_star = 0.5 * (lo + hi)
                Mred, GinvD, *_ = mix(M0a, D0a, G0a, M0b, D0b, G0b, th_star)
                m = I.find_modes(Mred, n_starts=128)
                if len(m["q"]) < 2:
                    continue
                # gradient vs FD of the Laplace pose at theta=0.5
                def lap(th):
                    Mr, Gi, *_ = mix(M0a, D0a, G0a, M0b, D0b, G0b, th)
                    mm = I.find_modes(Mr, n_starts=128)
                    q = mm["q"]; J = np.array(mm["J"]); ld = np.array(mm["logdetH"])
                    w = np.exp(-J / TAU - 0.5 * ld); w = w / w.sum()
                    poses = np.array([p1.pose_vector(qq, Gi) for qq in q])
                    return w @ poses, w, q, J, ld, Mr, Gi
                pose0, w, q, J, ld, Mr, Gi = lap(th_star)
                fd = (lap(th_star + H)[0] - lap(th_star - H)[0]) / (2 * H)
                # analytic: dq, dpose, dJ, dlogdet
                Mred_p, GinvD_p, *_ = mix(M0a, D0a, G0a, M0b, D0b, G0b, th_star + 1e-6)
                Mred_m, GinvD_m, *_ = mix(M0a, D0a, G0a, M0b, D0b, G0b, th_star - 1e-6)
                dM = (Mred_p - Mred_m) / 2e-6
                dGi = (GinvD_p - GinvD_m) / 2e-6
                dq = np.array([p1.ift_dq(qq, Mr, dM) for qq in q])
                dposes = []
                for k, qq in enumerate(q):
                    dp = p1.dpose_dq(qq, Gi) @ dq[k]
                    dp[9:] += dGi @ p1.v2(qq)
                    dposes.append(dp)
                dposes = np.array(dposes)
                dJ = np.array([p1.dJ_dtheta(qq, Mr, dM) for qq in q])
                dld = np.array([p1.logdetH(qq, Mr) for qq in q])  # placeholder; numeric below
                # numeric dlogdet (tangent + explicit theta)
                dld = np.zeros(len(q))
                for k, qq in enumerate(q):
                    Eq = p1.E(qq); xi = Eq.T @ dq[k]
                    gq = np.zeros(3)
                    for b in range(3):
                        e = np.zeros(3); e[b] = 1e-6
                        qp = qq + Eq @ e; qp /= np.linalg.norm(qp)
                        qm = qq - Eq @ e; qm /= np.linalg.norm(qm)
                        gq[b] = (p1.logdetH(qp, Mr) - p1.logdetH(qm, Mr)) / 2e-6
                    ex = (p1.logdetH(qq, Mr + 1e-6 * dM) - p1.logdetH(qq, Mr - 1e-6 * dM)) / 2e-6
                    dld[k] = ex + gq @ xi
                g = -J / TAU - 0.5 * ld
                gd = -dJ / TAU - 0.5 * dld
                dw = w * (gd - w @ gd)
                an = w @ dposes + (dw[:, None] * np.array([p1.pose_vector(qq, Gi) for qq in q])).sum(0)
                rel = np.linalg.norm(an - fd) / max(np.linalg.norm(fd), 1e-12)
                # hard argmin gradient: undefined if the two lowest J are tied
                order = np.argsort(J)
                hard_defined = (J[order[1]] - J[order[0]]) > 1e-8
                rows.append(dict(frame=frame, img=img, obj=oid, K=len(q), theta=float(th_star),
                                 min_gap=float(J[order[1]] - J[order[0]]),
                                 laplace_rel=float(rel), hard_defined=bool(hard_defined),
                                 w=np.asarray(w).tolist()))
                print(frame, img, oid, "K", len(q), "theta*", th_star, "gap", rows[-1]["min_gap"], "rel", rel, flush=True)
    json.dump(rows, open(OUT / "symmetry.json", "w"), indent=2)
    good = [r for r in rows if r["laplace_rel"] < 1e-2]
    lines = ["# P1 real-data symmetry ambiguity (BOP LINEMOD eggbox/glue)\n",
             f"tau={TAU}, FD h={H}; N={N}; frames={len(FRAMES)}\n",
             "| frame | obj | K | theta* | min gap | Laplace grad rel | hard defined |",
             "|-|-|-|-|-|-|-|"]
    for r in rows:
        lines.append(f"| {r['frame']} | {r['obj']} | {r['K']} | {r['theta']:.4f} | {r['min_gap']:.3e} | {r['laplace_rel']:.2e} | {r['hard_defined']} |")
    (OUT / "symmetry.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "symmetry.md", "valid", len(good), "/", len(rows))


if __name__ == "__main__":
    main()
