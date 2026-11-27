#!/usr/bin/env python3
"""KITTI odometry: layer-level constrained PnP with a roll/pitch cone + depth.

Data actually used (see ``meta`` in the JSON for the exact ids)
--------------------------------------------------------------
``/mnt/storage1/syk/data/kitti_odom`` ships *LiDAR only* for sequences
00 / 05 / 07 (``sequences/<seq>/velodyne/``): no ``image_0``/``image_1``.
The 3D-2D correspondences are therefore built by *GT projection*:

1. take the Velodyne scan of frame ``i``, map it to rectified camera 0 with the
   calibration extrinsic ``Tr`` (``calib/<seq>.txt``),
2. move it to frame ``j`` with the GT relative motion from ``poses/<seq>.txt``,
3. project with ``P0`` and keep the points that land inside the 1241x376 image
   with 2 m < depth < 60 m,
4. perturb the pixel with Gaussian noise (1 px) plus an optional downward bias,
   and optionally corrupt a fraction of the correspondences (wrong match).

That is a *layer* benchmark: no detector, no descriptor, no RANSAC.  It
isolates "given this 3D-2D layer, what does the constrained solver do".

Constraints
-----------
* ``depth``      : every selected 3D point must end up in front of the camera,
  ``(R p + t)_z >= 0`` (P_CON points closest to the camera are used; for the
  other points this is checked as a *violation*, not enforced).
* ``depth+tilt`` : additionally a 10 deg roll/pitch cone
  ``a . (R a) >= cos(10 deg)`` where ``a`` is the world-up vector expressed in
  the source camera frame -- for a ground vehicle this prior comes from the
  previous frame's attitude (IMU / prior odometry); here it is read off the GT
  pose of frame ``i`` (documented oracle ingredient), and the GT's own tilt is
  reported so the reader can see whether the cone is trivially satisfied.

Usage
-----
    python run_kitti_layer.py --seq 00
    KITTI_ROOT=... python run_kitti_layer.py --seq 05 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import layerlib as L  # noqa: E402

core = L.core
ROOT = Path(os.environ.get("KITTI_ROOT", "/mnt/storage1/syk/data/kitti_odom"))
H, W = 376, 1241
ZMIN, ZMAX = 2.0, 60.0
ALPHA = 10.0                     # roll/pitch cone half-angle (deg)
NOISE_PX = 1.0
SCALE = float(os.environ.get("KITTI_SCALE", "0.1"))   # m -> 0.1 m for conditioning
G_WORLD = np.array([0.0, -1.0, 0.0])   # KITTI world frame: y points *down*
SETTINGS = [
    dict(name="clean", bias_px=0.0, outlier_frac=0.0),
    dict(name="bias6", bias_px=6.0, outlier_frac=0.0),
    dict(name="bias6+out15", bias_px=6.0, outlier_frac=0.15),
    dict(name="bias6+out50", bias_px=6.0, outlier_frac=0.50),
]
CONFIGS = ["unc", "depth", "depth+tilt"]
METHODS = ["Unconstrained", "Soft penalty", "Projection", "ALM-QPEP"]


def load_calib(seq):
    d = {}
    for line in (ROOT / "calib" / f"{seq}.txt").read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = np.array([float(x) for x in v.split()])
    P0 = d["P0"].reshape(3, 4)
    Tr = np.vstack([d["Tr"].reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    return P0[:3, :3], Tr


def load_poses(seq):
    a = np.loadtxt(ROOT / "poses" / f"{seq}.txt")
    out = []
    for row in np.atleast_2d(a):
        p = row.reshape(3, 4)
        out.append((p[:, :3].copy(), p[:, 3].copy()))
    return out


def load_velo(seq, i):
    p = ROOT / "sequences" / seq / "velodyne" / f"{i:06d}.bin"
    return np.fromfile(str(p), dtype=np.float32).reshape(-1, 4)[:, :3].astype(float)


def pairs_of(seq, manifest):
    m = manifest[seq]
    out, seen = [], set()
    for i, j in m["pairs"] + m["sweep"]:
        if (i, j) not in seen:
            seen.add((i, j))
            out.append((int(i), int(j)))
    return out


def build_instance(seq, i, j, K, Tr, poses, rng):
    """3D points (source-cam coords, metres) + GT 2D projection in frame j."""
    Pw = load_velo(seq, i)
    if Pw.size == 0:
        return None
    keep = rng.choice(len(Pw), size=min(len(Pw), 30000), replace=False)
    Pw = Pw[keep]
    Xi = Pw @ Tr[:3, :3].T + Tr[:3, 3]
    R_gt, t_gt = L.rel_motion(poses[i], poses[j])
    Xj = Xi @ R_gt.T + t_gt
    uv, z = L.project(K, Xj)
    ok = ((z > ZMIN) & (z < ZMAX) & (uv[:, 0] > 6) & (uv[:, 0] < W - 6)
          & (uv[:, 1] > 6) & (uv[:, 1] < H - 6))
    idx = np.where(ok)[0]
    if idx.size < 40:
        return None
    sel = rng.choice(idx, size=min(NPTS, idx.size), replace=False)
    P = Xi[sel]                      # 3D in source camera frame
    uv_gt = uv[sel]
    depth_gt = z[sel]
    a_up = poses[i][0].T @ G_WORLD  # world-up expressed in camera-i coords
    a_up = a_up / np.linalg.norm(a_up)
    return dict(P=P, uv_gt=uv_gt, depth_gt=depth_gt, R_gt=R_gt, t_gt=t_gt,
                a_up=a_up, n_visible=int(idx.size))


def make_constraints(P, a_up, depth_gt):
    """depth-only and depth+tilt (A, B); depth on all selected points."""
    cs_depth = [core.c_depth(P[k], 0.0) for k in range(len(P))]
    A_d, B_d = L.stack(cs_depth)
    A_t, B_t = L.stack(cs_depth + [L.c_tilt_axis(a_up, ALPHA)])
    return (A_d, B_d), (A_t, B_t)


def solve_config(cfg, M0, D0, G0, A, B, q_u, t_u, rng):
    """Return list of (method, q, t, sec)."""
    if cfg == "unc":
        t0 = time.perf_counter()
        _, q, t = core.solve_qp_global(M0, D0, G0, n_starts=32, rng=rng)
        return [("Unconstrained", q, t, time.perf_counter() - t0)]
    out = []
    t0 = time.perf_counter()
    _, q_u, t_u = core.solve_qp_global(M0, D0, G0, n_starts=32, rng=rng)
    out.append(("Unconstrained", q_u, t_u, time.perf_counter() - t0))
    t0 = time.perf_counter()
    try:
        r = core.soft_penalty(M0, D0, G0, A, B, 1e3, q_u, t_u, n_starts=8, rng=rng)
        out.append(("Soft penalty", r["q"], r["t"], time.perf_counter() - t0))
    except Exception:
        out.append(("Soft penalty", q_u, t_u, time.perf_counter() - t0))
    t0 = time.perf_counter()
    try:
        r = core.project_feasible(M0, D0, G0, A, B, q_u, t_u, n_starts=4, rng=rng)
        out.append(("Projection", r["q"], r["t"], time.perf_counter() - t0))
    except Exception:
        out.append(("Projection", q_u, t_u, time.perf_counter() - t0))
    t0 = time.perf_counter()
    r = core.alm_qpep(M0, D0, G0, A, B, q_u, t_u, rho0=10.0, n_starts=6, rng=rng)
    out.append(("ALM-QPEP", r["q"], r["t"], time.perf_counter() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()
    global NPTS, P_CON
    NPTS = int(os.environ.get("KITTI_NPTS", "80"))
    P_CON = int(os.environ.get("KITTI_PCON", "24"))
    out_dir = Path(args.out or (HERE / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(ROOT / "manifest.json"))
    seq = args.seq
    K, Tr = load_calib(seq)
    poses = load_poses(seq)
    pairs = pairs_of(seq, manifest)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"seq {seq}: {len(pairs)} frame pairs, {len(poses)} poses, K="
          f"fx={K[0,0]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}", flush=True)
    w0 = time.perf_counter()
    rows, meta_pairs = [], []
    # GT tilt feasibility diagnostic
    gt_tilts = []
    for (i, j) in pairs:
        a_up = poses[i][0].T @ G_WORLD
        R_gt, _ = L.rel_motion(poses[i], poses[j])
        gt_tilts.append(L.tilt_deg(a_up, R_gt))
    for setting in SETTINGS:
        rng = np.random.default_rng(9000 + 37 * int(seq) + SETTINGS.index(setting))
        for (i, j) in pairs:
            inst = build_instance(seq, i, j, K, Tr, poses, rng)
            if inst is None:
                print(f"  skip {seq} {i}->{j} (too few visible points)", flush=True)
                continue
            meta_pairs.append((i, j))
            P = inst["P"]
            n = len(P)
            # noisy 2D observations
            uv = inst["uv_gt"].copy()
            uv += rng.normal(scale=NOISE_PX, size=uv.shape)
            uv += np.array([0.0, setting["bias_px"]])
            n_out = int(round(setting["outlier_frac"] * n))
            if n_out:
                bad = rng.choice(n, size=n_out, replace=False)
                perm = rng.permutation(len(bad))
                uv[bad] = uv[bad][perm]     # grossly wrong 2D observations
            U = L.bearings(K, uv)
            M0, D0, G0 = core.build_objective(P * SCALE, U)
            (A_d, B_d), (A_t, B_t) = make_constraints(P * SCALE, inst["a_up"],
                                                       inst["depth_gt"])
            # baseline unconstrained solution (shared warm start)
            _, q_u, t_u = core.solve_qp_global(M0, D0, G0, n_starts=32,
                                              rng=np.random.default_rng(1))
            for cfg, (A, B) in [("unc", (None, None)), ("depth", (A_d, B_d)),
                                ("depth+tilt", (A_t, B_t))]:
                if cfg == "unc":
                    sols = solve_config("unc", M0, D0, G0, None, None, q_u, t_u, rng)
                else:
                    sols = solve_config(cfg, M0, D0, G0, A, B, q_u, t_u, rng)
                if cfg == "unc":
                    cert = None
                    cert_rlt = None
                else:
                    t0 = time.perf_counter()
                    try:
                        sres = L.shor_bound(M0, D0, G0, A, B)
                        cert = sres["value"]
                    except Exception as ex:
                        print("cert fail", ex, flush=True)
                        cert = float("nan")
                    t_cert = time.perf_counter() - t0
                for (name, q, t, sec) in sols:
                    R_e = core.R_of(q)
                    t_e = t / SCALE
                    zmin_est = float(np.min((P @ R_e.T + t_e)[:, 2]))
                    viol = core.max_viol(A, B, q, t) if A is not None else 0.0
                    row = dict(
                        seq=seq, setting=setting["name"], i=int(i), j=int(j),
                        cfg=cfg, method=name, n_pts=int(n),
                        n_out=int(n_out), rot_deg=core.rot_deg(q, core.q_from_R(inst["R_gt"])),
                        trans_mm=core.trans_mm(t_e, inst["t_gt"]),
                        tilt_deg=L.tilt_deg(inst["a_up"], R_e),
                        min_depth_m=zmin_est,
                        viol_lin=float(viol), feas=bool(viol <= 1e-6),
                        time_ms=1000.0 * sec,
                        J=float(core.obj_value(q, t, M0, D0, G0)),
                    )
                    if cert is not None:
                        row["shor"] = float(cert)
                        row["cert"] = bool(np.isfinite(cert) and (row["J"] - cert <= 1e-6)
                                           and row["feas"])
                        row["cert_two_sided"] = bool(np.isfinite(cert)
                                                     and abs(row["J"] - cert) <= 1e-6
                                                     and row["feas"])
                        row["time_cert_s"] = t_cert
                    rows.append(row)
        print(f"setting {setting['name']} done, rows {len(rows)}, "
              f"{time.perf_counter() - w0:.1f}s", flush=True)
    meta = dict(
        dataset="kitti_odom", seq=seq, n_pairs=len(meta_pairs),
        pairs=[[int(i), int(j)] for (i, j) in meta_pairs],
        settings=SETTINGS, alpha_deg=ALPHA, noise_px=NOISE_PX,
        scale=SCALE, n_pts=NPTS, p_con=P_CON,
        image=(W, H), z_range=[ZMIN, ZMAX],
        gt_tilt_deg=dict(median=float(np.median(gt_tilts)),
                         max=float(np.max(gt_tilts)),
                         frac_over_alpha=float(np.mean(np.array(gt_tilts) > ALPHA)),
                         n=len(gt_tilts)),
        K=K.tolist(), Tr=Tr.tolist(),
        wall_s=time.perf_counter() - w0,
        correspondence="GT-projected LiDAR (no images available)",
        tilt_prior="world-up in camera-i coords from GT pose of frame i")
    L.dump(out_dir / f"real_kitti_{seq}.json", dict(meta=meta, rows=rows))
    print(f"seq {seq}: {len(rows)} rows, wall {time.perf_counter() - w0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
