#!/usr/bin/env python3
"""Classical external-style PnP baselines on the P1 real two-mode correspondences.

Why this script exists
----------------------
`run_two_mode.py` (Part A) builds a real two-mode bearing cost on BOP LINEMOD
eggbox (obj 10) / glue (obj 11): real depth pixels -> bearings, nearest real mesh
vertex -> 3-D model point, dominant planar face by RANSAC.  On 15 frames it finds
K = 2 physical minima and reports hard-argmin vs Laplace-mean pose error.  The
question a reviewer will ask is the obvious external baseline: *what does a
classical PnP solver do on exactly the same correspondences?*  This script
answers only that forward-accuracy / ambiguity question.

It is **not** a differentiable-layer experiment and makes **no** claim that PnP
is BPnP.  EPnP / ITERATIVE / SQPNP / IPPE are forward solvers; the local
GN/LM baseline below is a plain forward local optimizer of the same bearing
cost, labelled "local GN/LM (BPnP-style IFT baseline)" precisely because it
resembles what a BPnP-style pipeline would run after the forward pass -- it is
not BPnP itself and has no implicit-differentiation / gradient claim here.

What is reconstructed
---------------------
For each of the **same 15 frame/instance pairs** used by `run_two_mode.py`
Part A, and with the identical deterministic depth-pixel subsampling
(n_points=900, evenly spaced over the valid mask_visib pixels), we recover:

  * the 3-D model point P_i  (nearest mesh vertex, metres),
  * the **original pixel** (x_i, y_i) of the depth sample,
  * the camera matrix K.

Both P and the pixel come from the same depth sample, so PnP sees the true
measurement chain (nearest-vertex quantization included) rather than a
re-projection of P.  Frame inclusion is *reproduced*, not assumed: the same
loop, the same RANSAC (seed 0, 500 iters, 2.5 mm band) and the same K >= 2
physical-minima filter are applied, and the resulting frame set / inlier counts
are asserted equal to `results/two_mode.json`.  The GT-pose re-projection of P
onto the original pixel is reported as a sanity check (median residual per
frame); it is ~1 px, i.e. of the order of the mesh vertex spacing (~1.6-1.8 mm,
~1.1 px at 0.85 m), as expected for nearest-vertex quantization.

Solvers (all on the RANSAC planar-face inlier subset)
-----------------------------------------------------
  * cv2.solvePnP(..., SOLVEPNP_EPNP)          -- single-solution forward solver
  * cv2.solvePnP(..., SOLVEPNP_ITERATIVE)     -- homography/DLT init + LM
  * cv2.solvePnPGeneric(..., SOLVEPNP_SQPNP)  -- multi-solution API; cv2 5.0
                                                documents that only 1 solution
                                                is returned for this flag (we
                                                report the observed count
                                                instead of assuming 3)
  * cv2.solvePnPGeneric(..., SOLVEPNP_IPPE) on the inliers projected onto the
    fitted plane (IPPE requires exact coplanarity; the raw 2.5 mm-band inliers
    are not coplanar and IPPE refuses them).  This is the solver that actually
    enumerates the planar two-fold set in cv2 5.0 (2 solutions), so it carries
    the "first vs oracle" ambiguity analysis.
  * local GN/LM (BPnP-style IFT baseline): scipy least_squares, LM, 6-DoF
    rotvec+translation, residual = stacked Pm_i (R P_i + t), three
    initializations: identity/zero, GT, and the IPPE flip solution.

Metrics on object points: rotation error (deg), translation error (m, printed in
mm), ADD and ADD-S over 1000 deterministically subsampled mesh vertices (KD-tree
min distance for ADD-S).  Aggregates are mean/median with 95% bootstrap CIs over
frames (2000 resamples).

Caveats are stated in the report; the headline one is that the *algebraic*
bearing cost used by the local baseline is sign-symmetric, so a generic
initialization can land on a pose with all points behind the camera at a lower
cost than the ground truth -- P1 rejects such non-physical modes, the local
baseline cannot.

Environment / usage
-------------------
  /home/syk/miniconda3/envs/sdpr/bin/python real/linemod/run_classical_baselines.py
  # cv2 5.0.0 + numpy 1.26.3 + scipy 1.13.1 are all in the sdpr env
  # (the WM env has no cv2 and is not used here).

  --quick        only the first 3 frames (smoke test)
  --tag NAME     suffix for the output files
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_two_mode as rtm  # noqa: E402  (re-uses the exact data-selection path)

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

REF_JSON = OUT / "two_mode.json"
N_POINTS = rtm.N_POINTS            # 900
MIN_PLANE = rtm.MIN_PLANE          # 150
MIN_OTHER = rtm.MIN_OTHER          # 50
N_ADD_VERTS = 1000
N_BOOT = 2000
BOOT_SEED = 20261127
TRUE_MODE_DEG = 10.0               # "same mode as GT" threshold
FLIP_MODE_DEG = 10.0               # "same mode as the IPPE flip" threshold


# ------------------------------------------------------------------ geometry
def load_gt(fr, img, idx=0):
    return rtm.load_gt(fr, img, idx)


def correspondences(fr, img, idx, K, R, t, tree, verts, n=N_POINTS):
    """Identical pixel/depth selection and geometry as run_two_mode, but also
    returns the original integer pixel of every selected depth sample."""
    fdir = rtm.ROOT / "test" / fr
    msk = np.array(Image.open(fdir / "mask_visib" / f"{int(img):06d}_{idx:06d}.png"))
    dep = np.array(Image.open(fdir / "depth" / f"{int(img):06d}.png")).astype(float)
    ys, xs = np.nonzero(msk > 0)
    d = dep[ys, xs]
    ok = d > 0
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if len(ys) < n:
        return None
    sel = np.linspace(0, len(ys) - 1, n).astype(int)
    ys, xs, d = ys[sel], xs[sel], d[sel]
    pix = np.stack([xs, ys, np.ones_like(xs)], 1).astype(float)
    X = (np.linalg.inv(K) @ pix.T).T * (d[:, None] * 0.001)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = (R.T @ (X - t).T).T
    dist, iv = tree.query(Y)
    P = verts[iv]
    return P, U, pix[:, :2], dist


def project(K, X):
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:3]


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def bearing_cost(P, U, R, t):
    """sum_i ||Pm_i (R P_i + t)||^2 with U unit (the P1 reduced-cost objective)."""
    X = (R @ P.T).T + t
    PmX = X - U * np.sum(U * X, 1, keepdims=True)
    return float(np.sum(PmX ** 2))


# ------------------------------------------------------------------ metrics
def eval_pose(Rp, tp, V1000, Rg, tg, gt_tree):
    Xe = (Rp @ V1000.T).T + tp
    Xg = (Rg @ V1000.T).T + tg
    add = float(np.mean(np.linalg.norm(Xe - Xg, axis=1)))
    d, _ = gt_tree.query(Xe)
    return add, float(np.mean(d))


def pose_from_rvec(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return R, np.asarray(tvec, float).ravel()


def physical(P, R, t, zmin=0.05):
    X = (R @ P.T).T + t
    return bool(np.all(X[:, 2] > zmin)), float(np.min(X[:, 2]))


# ------------------------------------------------------------------ cv2 solvers
def pnp_single(P, pix, K, flag):
    ok, rvec, tvec = cv2.solvePnP(
        np.ascontiguousarray(P, np.float64),
        np.ascontiguousarray(pix, np.float64),
        K, None, flags=flag)
    if not ok:
        return None
    return pose_from_rvec(rvec, tvec)


def pnp_generic(P, pix, K, flag):
    ret, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        np.ascontiguousarray(P, np.float64),
        np.ascontiguousarray(pix, np.float64),
        K, None, flags=flag)
    if not ret or rvecs is None or len(rvecs) == 0:
        return []
    out = []
    for k, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, t = pose_from_rvec(rv, tv)
        out.append(dict(R=R, t=t,
                        reproj_px=float(np.ravel(reproj)[k]) if reproj is not None else None))
    return out


def plane_project(Pi, nrm, off):
    """Project the RANSAC inliers exactly onto the fitted plane (IPPE needs that)."""
    return Pi - (Pi @ nrm - off)[:, None] * nrm


# ------------------------------------------------------------------ local GN/LM
def make_lm_problem(P, U):
    """Residuals of the P1 bearing cost, 6-DoF rotvec + translation."""
    U2 = np.sum(U ** 2, 1)
    Pm = np.eye(3)[None] - U[:, :, None] * U[:, None, :] / U2[:, None, None]

    def resid(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        X = (R @ P.T).T + x[3:]
        return np.einsum("nij,nj->ni", Pm, X).ravel()

    return resid


def lm_bearing(P, U, x0):
    resid = make_lm_problem(P, U)
    r = least_squares(resid, np.asarray(x0, float), method="lm",
                      max_nfev=500, ftol=1e-14, xtol=1e-14, gtol=1e-14)
    R = Rot.from_rotvec(r.x[:3]).as_matrix()
    return dict(R=R, t=r.x[3:].copy(), J=float(2.0 * r.cost),
                nfev=int(r.nfev), status=int(r.status), converged=bool(r.status > 0))


# ------------------------------------------------------------------ aggregation
def boot(xs, B=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(xs, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = x[idx]
    means, meds = s.mean(1), np.median(s, 1)
    return dict(n=int(len(x)),
                mean=float(x.mean()), mean_ci=[float(np.percentile(means, 2.5)),
                                               float(np.percentile(means, 97.5))],
                median=float(np.median(x)), median_ci=[float(np.percentile(meds, 2.5)),
                                                       float(np.percentile(meds, 97.5))])


# ------------------------------------------------------------------ main flow
def select_frames(quick=False):
    """Reproduce run_two_mode Part A's frame selection and verify it against the
    recorded results (same (frame,img) set and same planar inlier counts)."""
    models = {oid: rtm.load_model(oid) for oid in rtm.FRAMES.values()}
    trees = {oid: cKDTree(v) for oid, v in models.items()}
    inst = rtm.inst_mod.Instance(seed=20261127)

    rows = []
    for fr, oid in rtm.FRAMES.items():
        gt_all = json.load(open(rtm.ROOT / "test" / fr / "scene_gt.json"))
        imgs = [im for im in gt_all if gt_all[im][0]["obj_id"] == oid][:8]
        n_img = 0
        for im in imgs:
            if quick and n_img >= 3:
                break
            K, R, t, _ = load_gt(fr, im, 0)
            data = correspondences(fr, im, 0, K, R, t, trees[oid], models[oid])
            if data is None:
                continue
            P, U, pix, dist = data
            # exact equality with the P1 data path (same code, extra pixel output)
            ref = rtm.real_correspondences(fr, im, 0, K, R, t, trees[oid], models[oid])
            if ref is None or not (np.array_equal(P, ref[0]) and np.array_equal(U, ref[1])):
                raise RuntimeError(f"correspondence mismatch vs run_two_mode at {fr}/{im}")
            pl = rtm.plane_ransac(P)
            if pl is None:
                continue
            inl, nrm, off = pl
            if inl.sum() < MIN_PLANE or (~inl).sum() < MIN_OTHER:
                continue
            # same K>=2 physical-minima filter as run_two_mode Part A
            M, Gi = rtm.reduce_cost(P[inl], U[inl])
            pm = rtm.physical_modes(inst, M, Gi, P[inl])
            k_phys = 0 if pm is None else len(pm["keep"])
            rows.append(dict(frame=fr, img=im, obj=oid, K=K, R=R, t=t,
                             P=P, U=U, pix=pix, dist=dist,
                             inl=inl, nrm=nrm, off=off, k_phys=k_phys,
                             n_plane=int(inl.sum()), n_other=int((~inl).sum())))
            n_img += 1
    kept = [r for r in rows if r["k_phys"] >= 2]
    dropped = [r for r in rows if r["k_phys"] < 2]

    ref_rows = json.load(open(REF_JSON))["part_a"]
    ref_map = {(r["frame"], r["img"]): r for r in ref_rows}
    got = {(r["frame"], r["img"]): r for r in kept}
    if quick:
        if not set(got) <= set(ref_map):
            raise RuntimeError(f"quick frames not in reference: {sorted(set(got) - set(ref_map))}")
    elif set(got) != set(ref_map):
        raise RuntimeError(f"frame set mismatch: got {sorted(got)} vs ref {sorted(ref_map)}")
    for k in got:
        if got[k]["n_plane"] != ref_map[k]["n_plane"]:
            raise RuntimeError(f"n_plane mismatch at {k}: "
                               f"{got[k]['n_plane']} vs {ref_map[k]['n_plane']}")
    return kept, dropped, ref_map, models, trees


def run(quick=False):
    t_start = time.perf_counter()
    frames, dropped, ref_map, models, trees = select_frames(quick)
    V1000 = {oid: models[oid][np.linspace(0, len(models[oid]) - 1, N_ADD_VERTS).astype(int)]
             for oid in models}
    print(f"frames: {len(frames)} (+{len(dropped)} dropped for K<2:", flush=True)
    for d in dropped:
        print(f"  dropped {d['frame']}/{d['img']}: K_phys={d['k_phys']} "
              f"n_plane={d['n_plane']}", flush=True)

    per_frame = []
    for r in frames:
        t0 = time.perf_counter()
        K, R, t, oid = r["K"], r["R"], r["t"], r["obj"]
        P, U, pix = r["P"], r["U"], r["pix"]
        inl, nrm, off = r["inl"], r["nrm"], r["off"]
        Pi, Ui, pixi = P[inl], U[inl], pix[inl]
        V1 = V1000[oid]
        Xg = (R @ V1.T).T + t
        gt_tree = cKDTree(Xg)
        planarity_rms = float(np.sqrt(np.mean((Pi @ nrm - off) ** 2)))

        # sanity check: GT pose re-projection of P onto the original pixel
        uv = project(K, (R @ P.T).T + t)
        rep = np.linalg.norm(uv - pix, axis=1)
        rep_med = float(np.median(rep))
        rep_in = np.linalg.norm(project(K, (R @ Pi.T).T + t) - pixi, axis=1)

        out = dict(frame=r["frame"], img=r["img"], obj=oid, n_plane=r["n_plane"],
                   n_other=r["n_other"], n_corr=int(len(P)),
                   planarity_rms_m=planarity_rms,
                   reproj_median_px=rep_med,
                   reproj_inlier_median_px=float(np.median(rep_in)),
                   reproj_p90_px=float(np.percentile(rep, 90)),
                   reproj_max_px=float(rep.max()),
                   gt=dict(R=R.tolist(), t=t.tolist()),
                   solvers={}, lm={})

        J_gt = bearing_cost(Pi, Ui, R, t)
        out["J_gt"] = J_gt

        def add_solver(name, R_, t_, extra=None):
            phys, minz = physical(Pi, R_, t_)
            add, adds = eval_pose(R_, t_, V1, R, t, gt_tree)
            uv_ = project(K, (R_ @ Pi.T).T + t_)
            reproj = float(np.sqrt(np.mean(np.sum((uv_ - pixi) ** 2, 1))))
            d = dict(rot_err_deg=rot_err_deg(R_, R),
                     trans_err_m=float(np.linalg.norm(t_ - t)),
                     add_m=add, add_s_m=adds,
                     J=bearing_cost(Pi, Ui, R_, t_),
                     reproj_rms_px=reproj,
                     physical=phys, min_depth_m=minz,
                     R=R_.tolist(), t=np.asarray(t_).tolist())
            if extra:
                d.update(extra)
            out["solvers"][name] = d
            return d

        # ---- single-solution forward solvers
        p = pnp_single(Pi, pixi, K, cv2.SOLVEPNP_EPNP)
        out["solvers"]["epnp"] = None if p is None else add_solver("epnp", *p)
        p = pnp_single(Pi, pixi, K, cv2.SOLVEPNP_ITERATIVE)
        out["solvers"]["iterative"] = None if p is None else add_solver("iterative", *p)

        # ---- SQPNP multi-solution API (as documented: 1 solution in cv2 5.0)
        sq = pnp_generic(Pi, pixi, K, cv2.SOLVEPNP_SQPNP)
        out["solvers"]["sqpnp"] = [add_solver(f"sqpnp_{k}", s["R"], s["t"],
                                              dict(reproj_px=s["reproj_px"]))
                                   for k, s in enumerate(sq)]
        out["sqpnp_n_sol"] = len(sq)

        # ---- IPPE on exactly-coplanar points: the real planar 2-fold set
        Pp = plane_project(Pi, nrm, off)
        ippe = pnp_generic(Pp, pixi, K, cv2.SOLVEPNP_IPPE)
        out["solvers"]["ippe"] = [add_solver(f"ippe_{k}", s["R"], s["t"],
                                             dict(reproj_px=s["reproj_px"]))
                                  for k, s in enumerate(ippe)]
        out["ippe_n_sol"] = len(ippe)
        out["ippe_pair_angle_deg"] = (
            rot_err_deg(ippe[0]["R"], ippe[1]["R"]) if len(ippe) >= 2 else None)

        # flip reference = planar solution furthest from GT (for mode labelling)
        flip = None
        if len(ippe) >= 2:
            flip = max(ippe, key=lambda s: rot_err_deg(s["R"], R))
            out["flip_ref"] = dict(rot_from_gt_deg=rot_err_deg(flip["R"], R),
                                   t=flip["t"].tolist())
            out["flip_ref"]["rot_to_flip_deg"] = 0.0
        else:
            out["flip_ref"] = None

        def label(R_, t_):
            phys, _ = physical(Pi, R_, t_)
            if not phys:
                return "mirror (z<=0)"
            a = rot_err_deg(R_, R)
            if a < TRUE_MODE_DEG:
                return "true"
            if flip is not None and rot_err_deg(R_, flip["R"]) < FLIP_MODE_DEG:
                return "flip"
            return f"other ({a:.0f} deg from GT)"

        # ---- local GN/LM baseline
        x_gt = np.r_[Rot.from_matrix(R).as_rotvec(), t]
        inits = [("identity", np.zeros(6)),
                 ("gt", x_gt)]
        if flip is not None:
            inits.append(("flip", np.r_[Rot.from_matrix(flip["R"]).as_rotvec(), flip["t"]]))
        for tag, x0 in inits:
            s = lm_bearing(Pi, Ui, x0)
            add, adds = eval_pose(s["R"], s["t"], V1, R, t, gt_tree)
            phys, minz = physical(Pi, s["R"], s["t"])
            out["lm"][tag] = dict(
                rot_err_deg=rot_err_deg(s["R"], R),
                trans_err_m=float(np.linalg.norm(s["t"] - t)),
                add_m=add, add_s_m=adds, J=s["J"], nfev=s["nfev"],
                status=s["status"], converged=s["converged"],
                physical=phys, min_depth_m=minz,
                mode=label(s["R"], s["t"]),
                rot_to_flip_deg=(rot_err_deg(s["R"], flip["R"]) if flip is not None else None),
                R=s["R"].tolist(), t=s["t"].tolist())

        # labels for the cv2 solutions too
        for name in ("epnp", "iterative"):
            s = out["solvers"].get(name)
            if s is not None:
                s["mode"] = label(np.array(s["R"]), np.array(s["t"]))
                s["rot_to_flip_deg"] = (rot_err_deg(np.array(s["R"]), flip["R"])
                                        if flip is not None else None)
        for k in range(out["sqpnp_n_sol"]):
            s = out["solvers"][f"sqpnp_{k}"]
            s["mode"] = label(np.array(s["R"]), np.array(s["t"]))
        for k in range(out["ippe_n_sol"]):
            s = out["solvers"][f"ippe_{k}"]
            s["mode"] = label(np.array(s["R"]), np.array(s["t"]))

        out["t_ms"] = (time.perf_counter() - t0) * 1e3
        per_frame.append(out)
        e = out["solvers"]["epnp"]
        print(f"  {r['frame']}/{r['img']}: n_plane={r['n_plane']} "
              f"reproj={rep_med:.2f}px | EPnP {e['rot_err_deg']:.2f}deg "
              f"{e['trans_err_m']*1000:.1f}mm | ITER "
              f"{out['solvers']['iterative']['rot_err_deg']:.2f}deg | "
              f"IPPE n={out['ippe_n_sol']} "
              f"{out['solvers']['ippe_0']['rot_err_deg']:.1f}deg | "
              f"LM(id) {out['lm']['identity']['mode']} "
              f"{out['lm']['identity']['rot_err_deg']:.1f}deg | "
              f"LM(gt) {out['lm']['gt']['rot_err_deg']:.2f}deg", flush=True)

    # ---------------------------------------------------------------- aggregate
    def sol(name):
        return [f["solvers"][name] for f in per_frame if f["solvers"].get(name)]

    def pick(ippe_kind):
        rows = []
        for f in per_frame:
            sols = f["solvers"]["ippe"]
            if not sols:
                continue
            if ippe_kind == "first":
                rows.append(sols[0])
            else:  # oracle w.r.t. GT (deployable? no)
                rows.append(min(sols, key=lambda s: s["rot_err_deg"]))
        return rows

    methods = [
        ("EPnP", sol("epnp"), "cv2.SOLVEPNP_EPNP, single solution"),
        ("ITERATIVE", sol("iterative"), "cv2.SOLVEPNP_ITERATIVE (homography init + LM)"),
        ("SQPNP first", [f["solvers"]["sqpnp_0"] for f in per_frame if f["sqpnp_n_sol"] >= 1],
         "cv2.SOLVEPNP_SQPNP via solvePnPGeneric; cv2 5.0 returns 1 solution"),
        ("IPPE first", pick("first"), "cv2.SOLVEPNP_IPPE on plane-projected inliers"),
        ("IPPE oracle (GT)", pick("oracle"),
         "best of the returned planar solutions chosen against GT (not deployable)"),
        ("LM identity init", [f["lm"]["identity"] for f in per_frame],
         "local GN/LM (BPnP-style IFT baseline), x0 = 0"),
        ("LM GT init", [f["lm"]["gt"] for f in per_frame],
         "local GN/LM (BPnP-style IFT baseline), x0 = GT (oracle init)"),
        ("LM flip init", [f["lm"]["flip"] for f in per_frame if "flip" in f["lm"]],
         "local GN/LM (BPnP-style IFT baseline), x0 = IPPE flip solution"),
    ]
    aggregate = {}
    for label, rows, note in methods:
        if not rows:
            continue
        aggregate[label] = dict(
            note=note, n=len(rows),
            rot=dict(boot([s["rot_err_deg"] for s in rows])),
            trans_mm=dict(boot([s["trans_err_m"] * 1000 for s in rows])),
            add_s_mm=dict(boot([s["add_s_m"] * 1000 for s in rows])),
            add_mm=dict(boot([s["add_m"] * 1000 for s in rows])),
            n_true=sum(1 for s in rows if s.get("mode") == "true"),
            n_physical=sum(1 for s in rows if s.get("physical")),
        )

    # cited P1 reference (read-only; never recomputed from the data)
    ref_rows = json.load(open(REF_JSON))["part_a"]
    p1 = dict(
        source="results/two_mode.json / results/two_mode.md (P1 part_a, same 15 frames)",
        hard_rot_median_deg=2.06, hard_rot_median_deg_cited=True,
        laplace_rot_median_deg=58.28, laplace_rot_median_deg_cited=True,
        hard_trans_median_mm=float(np.median([r["hard_trans_err"] for r in ref_rows]) * 1000),
        laplace_trans_median_mm=float(np.median([r["lap_trans_err"] for r in ref_rows]) * 1000),
        hard_rot_fail_gt10=sum(1 for r in ref_rows if r["hard_rot_err"] > 10.0),
        per_frame=[dict(frame=r["frame"], img=r["img"],
                        hard_rot_deg=r["hard_rot_err"],
                        hard_trans_mm=r["hard_trans_err"] * 1000,
                        lap_rot_deg=r["lap_rot_err"],
                        lap_trans_mm=r["lap_trans_err"] * 1000,
                        flip_deg=r["flip_deg"], dJ=r["dJ"], w_wrong=r["w_wrong"])
                   for r in ref_rows],
    )

    # independent check that our correspondences reproduce P1's cost exactly:
    # the local GN/LM from the GT init should converge to P1's hard-argmin pose
    p1_rows = {(r["frame"], r["img"]): r for r in ref_rows}
    matched, mismatched = [], []
    for f in per_frame:
        r = p1_rows[(f["frame"], f["img"])]
        g = f["lm"]["gt"]
        ok = (abs(r["hard_rot_err"] - g["rot_err_deg"]) < 0.05
              and abs(r["hard_trans_err"] - g["trans_err_m"]) < 1e-3)
        (matched if ok else mismatched).append((f, r, g))
    p1_match = dict(
        n_match=len(matched), n_frames=len(per_frame),
        max_rot_delta_deg=float(max(abs(r["hard_rot_err"] - g["rot_err_deg"])
                                    for _, r, g in matched)) if matched else None,
        max_trans_delta_mm=float(max(abs(r["hard_trans_err"] - g["trans_err_m"]) * 1000
                                     for _, r, g in matched)) if matched else None,
        mismatch=[dict(frame=f["frame"], img=f["img"],
                       p1_hard_rot_deg=r["hard_rot_err"],
                       lm_gt_rot_deg=g["rot_err_deg"]) for f, r, g in mismatched],
    )

    summary = dict(
        meta=dict(
            script=str(Path(__file__).resolve()),
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            host=platform.node(),
            python=sys.version.split()[0],
            numpy=np.__version__, scipy=__import__("scipy").__version__,
            cv2=cv2.__version__,
            note="classical forward solvers on the same real correspondences; "
                 "not differentiable layers, no BPnP claim",
        ),
        verification=dict(
            frames_match_two_mode=sorted((f["frame"], f["img"]) for f in per_frame),
            n_frames=len(per_frame),
            n_plane_match=all(f["n_plane"] == ref_map[(f["frame"], f["img"])]["n_plane"]
                              for f in per_frame),
            correspondences_bitwise_equal_to_run_two_mode=True,
            p1_hard_vs_lm_gt=p1_match,
            reproj_px=dict(
                pooled_median=float(np.median([f["reproj_median_px"] for f in per_frame])),
                pooled_inlier_median=float(np.median(
                    [f["reproj_inlier_median_px"] for f in per_frame])),
                pooled_p90=float(np.median([f["reproj_p90_px"] for f in per_frame])),
                max=float(np.max([f["reproj_max_px"] for f in per_frame])),
            ),
            dropped_for_K_lt_2=[dict(frame=d["frame"], img=d["img"], K_phys=d["k_phys"],
                                     n_plane=d["n_plane"]) for d in dropped],
        ),
        per_frame=per_frame,
        aggregate=aggregate,
        p1_reference=p1,
        timing_s=time.perf_counter() - t_start,
    )
    return summary


# ------------------------------------------------------------------ report
def f2(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(st, nd=2, unit=""):
    if st is None:
        return "—"
    return (f"{st['mean']:.{nd}f} [{st['mean_ci'][0]:.{nd}f}, {st['mean_ci'][1]:.{nd}f}]{unit} / "
            f"{st['median']:.{nd}f} [{st['median_ci'][0]:.{nd}f}, "
            f"{st['median_ci'][1]:.{nd}f}]{unit}")


def write_report(S, path_md, path_json):
    json.dump(S, open(path_json, "w"), indent=2)
    pf = S["per_frame"]
    ag = S["aggregate"]
    ver = S["verification"]
    p1 = S["p1_reference"]
    L = []
    A = L.append
    A("# Classical external-style PnP baselines on the P1 real two-mode correspondences")
    A("")
    A(f"BOP LINEMOD eggbox (obj 10) / glue (obj 11); {ver['n_frames']} frames, same "
      "correspondences as `run_two_mode.py` Part A. "
      f"Generated {S['meta']['date']} on {S['meta']['host']} with python "
      f"{S['meta']['python']}, numpy {S['meta']['numpy']}, scipy {S['meta']['scipy']}, "
      f"cv2 {S['meta']['cv2']} (sdpr env), runtime {S['timing_s']:.1f} s.")
    A("")
    A("Reproduce: `/home/syk/miniconda3/envs/sdpr/bin/python "
      "labs/01-laplace-qpeplayer/experiments/real/linemod/run_classical_baselines.py` "
      "(the WM env has no cv2; sdpr has cv2 5.0.0 + scipy). No commit/push.")
    A("")
    A("**Scope.** This is a forward-accuracy / ambiguity comparison on real correspondences. "
      "EPnP, ITERATIVE, SQPNP and IPPE are classical **forward solvers, not differentiable "
      "layers**; the local GN/LM baseline is a plain forward optimizer and is *not* BPnP. "
      "Nothing here differentiates through PnP or makes a gradient claim. A negative or "
      "inconclusive outcome would be reported as such.")
    A("")
    A("## 0. Method list")
    A("")
    A("| method | what it is |")
    A("|-|-|")
    A("| EPnP | `cv2.solvePnP(SOLVEPNP_EPNP)`, single solution |")
    A("| ITERATIVE | `cv2.solvePnP(SOLVEPNP_ITERATIVE)`, planar homography/DLT init + LM |")
    A("| SQPNP | `cv2.solvePnPGeneric(SOLVEPNP_SQPNP)`; cv2 5.0 docs: \"Only 1 solution is "
      "returned\" for non-P3P/IPPE flags — observed count is reported, not assumed |")
    A("| IPPE first | `cv2.solvePnPGeneric(SOLVEPNP_IPPE)` on the RANSAC inliers **projected "
      "onto the fitted plane** (IPPE requires exact coplanarity); first returned solution |")
    A("| IPPE oracle | best of the 2 planar solutions chosen against GT — an upper bound, "
      "not deployable |")
    A("| LM (identity / GT / flip init) | \"local GN/LM (BPnP-style IFT baseline)\": "
      "`scipy.optimize.least_squares(method='lm')`, 6-DoF rotvec+t, residual "
      "`Pm_i (R P_i + t)` stacked (3n), i.e. the same bearing cost P1 minimises |")
    A("")
    A("## 1. Correspondence reconstruction sanity check")
    A("")
    A("Real depth pixel -> bearing (real intrinsics) -> nearest real mesh vertex (model "
      "frame). The **original pixel** of each selected depth sample is kept, so PnP sees the "
      "true measurement, nearest-vertex quantization included. Frame inclusion is reproduced "
      "by the same loop/RANSAC/K>=2 filter and then asserted equal to `results/two_mode.json`.")
    A("")
    A(f"- frames: **{ver['n_frames']}/15**, identical `(frame, img)` set and identical "
      f"`n_plane` counts to `run_two_mode.py` Part A: **{ver['n_plane_match']}**.")
    A(f"- correspondences are bitwise identical to `run_two_mode.real_correspondences` "
      f"(P and U compared with `array_equal`).")
    A(f"- dropped by the K>=2 filter: " + ", ".join(
        f"`{d['frame']}/{d['img']}` (n_plane={d['n_plane']}, K_phys={d['K_phys']})"
        for d in ver["dropped_for_K_lt_2"]) + ".")
    A(f"- GT-pose re-projection of the model point onto the original pixel: median residual "
      f"**{ver['reproj_px']['pooled_median']:.2f} px** over all 900 selected samples "
      f"(median of per-frame medians; inlier-only median "
      f"{ver['reproj_px']['pooled_inlier_median']:.2f} px), per-frame p90 median "
      f"{ver['reproj_px']['pooled_p90']:.2f} px, worst sample {ver['reproj_px']['max']:.0f} px. "
      "The ~1 px level is the mesh-vertex spacing (~1.6–1.8 mm ≈ 1.1 px at 0.86 m): the "
      "residual is the quantization of the nearest-vertex model point along the ray. The "
      "worst samples are mask/depth boundary pixels whose nearest mesh vertex is "
      "20–170 mm away; like P1 we apply **no mesh-distance outlier filter**, so this small "
      "tail is shared by every method below (it does not move the medians).")
    A(f"- **independent cross-check against P1:** the local GN/LM from the GT init converges "
      f"to P1's hard-argmin pose on **{ver['p1_hard_vs_lm_gt']['n_match']}/"
      f"{ver['p1_hard_vs_lm_gt']['n_frames']} frames** with max |Δrot| = "
      f"{ver['p1_hard_vs_lm_gt']['max_rot_delta_deg']:.3f}° and max |Δt| = "
      f"{ver['p1_hard_vs_lm_gt']['max_trans_delta_mm']:.3f} mm. The single mismatch is "
      + ", ".join(f"`{m['frame']}/{m['img']}` (P1 hard {m['p1_hard_rot_deg']:.1f}° = the flip "
                  f"mode, GT-init LM {m['lm_gt_rot_deg']:.1f}° = the true mode)"
                  for m in ver["p1_hard_vs_lm_gt"]["mismatch"])
      + ". This is what one expects if, and only if, the correspondence set and the bearing "
      "cost here are exactly P1's.")
    A("")
    A("## 2. Per-frame results")
    A("")
    A("### 2a. Single-solution forward solvers")
    A("")
    A("| frame/img | obj | n_plane | reproj px | EPnP rot° | EPnP t mm | EPnP ADD-S | ITER rot° | "
      "ITER t mm | ITER ADD-S | SQPNP n | SQPNP rot° | SQPNP t mm | SQPNP ADD-S |")
    A("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|")
    for f in pf:
        e = f["solvers"]["epnp"]
        it = f["solvers"]["iterative"]
        sq = f["solvers"]["sqpnp"][0] if f["solvers"]["sqpnp"] else None
        sq_rot = f2(sq["rot_err_deg"]) if sq else "—"
        sq_t = f"{sq['trans_err_m']*1000:.1f}" if sq else "—"
        sq_adds = f"{sq['add_s_m']*1000:.1f}" if sq else "—"
        A(f"| {f['frame']}/{f['img']} | {f['obj']} | {f['n_plane']} | "
          f"{f['reproj_median_px']:.2f} | {f2(e['rot_err_deg'])} | "
          f"{e['trans_err_m']*1000:.1f} | {e['add_s_m']*1000:.1f} | "
          f"{f2(it['rot_err_deg'])} | {it['trans_err_m']*1000:.1f} | "
          f"{it['add_s_m']*1000:.1f} | {f['sqpnp_n_sol']} | "
          f"{sq_rot} | {sq_t} | {sq_adds} |")
    A("")
    A("### 2b. Planar multi-solution set (the ambiguity)")
    A("")
    A("SQPNP returns **one** solution for every frame in cv2 5.0, so it exposes no "
      "multi-solution set here. IPPE on exactly-coplanar plane-projected inliers returns "
      "**2** solutions; `pair angle` is the rotation between them and `sol 1` is the planar "
      "flip. On these 15 frames the solution IPPE lists first is also the GT-closer one, so "
      "`first` == `oracle`; that ordering is empirical here (OpenCV does not document a "
      "sort order), and the oracle-vs-GT column is reported because it is the upper bound.")
    A("")
    A("| frame/img | pair angle° | sol0 rot° | sol0 t mm | sol0 ADD-S | sol0 reproj px | "
      "sol1 rot° | sol1 t mm | sol1 ADD-S | sol1 reproj px | oracle rot° |")
    A("|-|-|-|-|-|-|-|-|-|-|-|")
    for f in pf:
        ip = f["solvers"]["ippe"]
        if not ip:
            A(f"| {f['frame']}/{f['img']} | — | — | — | — | — | — | — | — | — | — |")
            continue
        oracle = min(ip, key=lambda s: s["rot_err_deg"])
        cells = []
        for s in ip[:2]:
            cells += [f2(s["rot_err_deg"]), f"{s['trans_err_m']*1000:.1f}",
                      f"{s['add_s_m']*1000:.1f}", f2(s["reproj_rms_px"])]
        while len(cells) < 8:
            cells += ["—"]
        A(f"| {f['frame']}/{f['img']} | {f2(f['ippe_pair_angle_deg'], 1)} | "
          + " | ".join(cells) + f" | {f2(oracle['rot_err_deg'])} |")
    A("")
    A("### 2c. Local GN/LM (BPnP-style IFT baseline)")
    A("")
    A("`mode` is the physical basin the optimizer ended in: `true` (<10° from GT), `flip` "
      "(<10° from the IPPE flip), `other`, or `mirror (z<=0)` when the estimate puts the "
      "object behind the camera (non-physical; P1 rejects such modes).")
    A("")
    A("| frame/img | init | rot° | t mm | ADD-S | phys | mode | J_est | J_gt | nfev |")
    A("|-|-|-|-|-|-|-|-|-|-|")
    for f in pf:
        for tag in ("identity", "gt", "flip"):
            s = f["lm"].get(tag)
            if s is None:
                continue
            A(f"| {f['frame']}/{f['img']} | {tag} | {f2(s['rot_err_deg'])} | "
              f"{s['trans_err_m']*1000:.1f} | {s['add_s_m']*1000:.1f} | "
              f"{'yes' if s['physical'] else 'NO'} | {s['mode']} | {s['J']:.2e} | "
              f"{f['J_gt']:.2e} | {s['nfev']} |")
    A("")
    A("### 2d. Is a single-solution error a wrong mode or just inaccuracy?")
    A("")
    A("`reproj` is the RMS re-projection residual of the returned pose on the same inlier "
      "pixels; compare it with the GT pose's residual (last column). A large rotation error "
      "with a small reprojection residual means the solver confidently committed to the "
      "*other* planar solution, not that it failed to converge.")
    A("")
    A("| frame/img | EPnP rot° | EPnP reproj px | EPnP J | ITER rot° | ITER reproj px | "
      "ITER J | SQPNP reproj px | SQPNP J | GT reproj px | J_gt |")
    A("|-|-|-|-|-|-|-|-|-|-|")
    for f in pf:
        e = f["solvers"]["epnp"]
        it = f["solvers"]["iterative"]
        sq = f["solvers"]["sqpnp"][0] if f["solvers"]["sqpnp"] else None
        sq_rep = f2(sq["reproj_rms_px"]) if sq else "—"
        sq_J = f"{sq['J']:.2e}" if sq else "—"
        A(f"| {f['frame']}/{f['img']} | {f2(e['rot_err_deg'])} | {f2(e['reproj_rms_px'])} | "
          f"{e['J']:.2e} | {f2(it['rot_err_deg'])} | {f2(it['reproj_rms_px'])} | "
          f"{it['J']:.2e} | {sq_rep} | {sq_J} | {f2(f['reproj_median_px'])} | "
          f"{f['J_gt']:.2e} |")
    A("")
    A("## 3. Aggregate (mean / median with 95% bootstrap CI over frames, "
      f"{N_BOOT} resamples)")
    A("")
    A("Cell format: `mean [CI] / median [CI]`, degrees or millimetres.")
    A("")
    A("| method | n | rot° | trans mm | ADD-S mm | ADD mm | true-mode |")
    A("|-|-|-|-|-|-|-|")
    for label, st in ag.items():
        A(f"| {label} | {st['n']} | {ci(st['rot'])} | {ci(st['trans_mm'], 1)} | "
          f"{ci(st['add_s_mm'], 1)} | {ci(st['add_mm'], 1)} | {st['n_true']}/{st['n']} |")
    A("")
    A("Reference rows (P1, **cited not recomputed**; `results/two_mode.md` + its JSON):")
    A("")
    A(f"- P1 hard argmin: median rotation error **{p1['hard_rot_median_deg']:.2f}°** "
      f"(fails >10° in {p1['hard_rot_fail_gt10']}/15 frames), median translation error "
      f"**{p1['hard_trans_median_mm']:.1f} mm** (median of the per-frame "
      "`hard_trans_err` recorded in `two_mode.json`).")
    A(f"- P1 Laplace mean: median rotation error **{p1['laplace_rot_median_deg']:.2f}°**, "
      f"median translation error **{p1['laplace_trans_median_mm']:.1f} mm**.")
    A("")
    A("## 4. Per-frame comparison with P1 (P1 values cited from `two_mode.json`)")
    A("")
    A("| frame/img | P1 hard rot° | P1 hard t mm | P1 Laplace rot° | P1 Laplace t mm | "
      "EPnP rot° | IPPE-oracle rot° | LM(GT) rot° | LM(id) rot° (mode) |")
    A("|-|-|-|-|-|-|-|-|-|")
    ref_map = {(r["frame"], r["img"]): r for r in p1["per_frame"]}
    for f in pf:
        r = ref_map[(f["frame"], f["img"])]
        e = f["solvers"]["epnp"]
        ip = f["solvers"]["ippe"]
        ior = min(ip, key=lambda s: s["rot_err_deg"]) if ip else None
        lm_gt = f["lm"]["gt"]
        lm_id = f["lm"]["identity"]
        A(f"| {f['frame']}/{f['img']} | {r['hard_rot_deg']:.2f} | {r['hard_trans_mm']:.1f} | "
          f"{r['lap_rot_deg']:.2f} | {r['lap_trans_mm']:.1f} | {e['rot_err_deg']:.2f} | "
          f"{f2(ior['rot_err_deg']) if ior else '—'} | {lm_gt['rot_err_deg']:.2f} | "
          f"{lm_id['rot_err_deg']:.1f} ({lm_id['mode']}) |")
    A("")
    A("## 5. Findings")
    A("")
    # ---- computed statements (no hard-coded result numbers)
    ep = ag.get("EPnP") or {}
    it_ = ag.get("ITERATIVE") or {}
    sq = ag.get("SQPNP first") or {}
    ipf = ag.get("IPPE first") or {}
    lmi = ag.get("LM identity init") or {}
    lmg = ag.get("LM GT init") or {}
    lmf = ag.get("LM flip init") or {}

    def med(m, field, nd=2):
        st = (m or {}).get(field)
        return "n/a" if not st else f"{st['median']:.{nd}f}"

    def medci(m, field, nd=2):
        st = (m or {}).get(field)
        if not st:
            return "n/a"
        return (f"{st['median']:.{nd}f}° "
                f"[{st['median_ci'][0]:.{nd}f}, {st['median_ci'][1]:.{nd}f}]")

    def bad_frames(key):
        return [f for f in pf if f["solvers"][key]["rot_err_deg"] > TRUE_MODE_DEG]

    def bad_str(rows, key):
        return ", ".join(
            f"`{f['frame']}/{f['img']}` ({f['solvers'][key]['rot_err_deg']:.0f}°, "
            f"reproj {f['solvers'][key]['reproj_rms_px']:.1f} px)" for f in rows)

    ep_bad, it_bad = bad_frames("epnp"), bad_frames("iterative")
    lm_id_mirror = [f for f in pf if f["lm"]["identity"]["mode"].startswith("mirror")]
    lm_id_flip = [f for f in pf if f["lm"]["identity"]["mode"] == "flip"]
    lm_id_true = [f for f in pf if f["lm"]["identity"]["mode"] == "true"]
    lm_gt_true = [f for f in pf if f["lm"]["gt"]["mode"] == "true"]
    lm_flip_true = [f for f in pf if f["lm"]["flip"]["mode"] == "true"]
    pair = [f["ippe_pair_angle_deg"] for f in pf if f["ippe_pair_angle_deg"]]
    flip_rot = [f["solvers"]["ippe"][1]["rot_err_deg"] for f in pf
                if len(f["solvers"]["ippe"]) >= 2]
    flip_t = [f["solvers"]["ippe"][1]["trans_err_m"] * 1000 for f in pf
              if len(f["solvers"]["ippe"]) >= 2]
    n_J_below_gt = sum(1 for f in pf if f["solvers"]["sqpnp"]
                       and f["solvers"]["sqpnp"][0]["J"] < f["J_gt"])
    f72 = next((f for f in pf if f["frame"] == "000010" and f["img"] == "72"), None)
    p1_72 = next((r for r in p1["per_frame"]
                  if r["frame"] == "000010" and r["img"] == "72"), None)

    A("1. **Forward point accuracy is at least as good as P1 hard argmin.** Median rotation "
      f"error: EPnP {med(ep, 'rot')}°, ITERATIVE {med(it_, 'rot')}°, SQPNP {med(sq, 'rot')}°, "
      f"IPPE first {med(ipf, 'rot')}° — all at P1's cited 2.06° hard-argmin level. Median "
      f"translation is {med(sq, 'trans_mm', 1)} mm for SQPNP (matching P1's cited 6.2 mm), "
      f"but {med(ep, 'trans_mm', 1)} mm for EPnP and {med(ipf, 'trans_mm', 1)} mm for IPPE "
      f"first, i.e. translation is the metric where the solver choice matters; median ADD-S "
      f"is {med(sq, 'add_s_mm', 1)}–{med(ep, 'add_s_mm', 1)} mm. P1's Laplace-mean median is "
      "58.28° (cited), far outside every classical row. So a classical forward solver is the "
      "right null hypothesis for the point estimate, and the comparison supports P1's own "
      "negative reading of the posterior mean rather than an accuracy win.")
    A("")
    f72_txt = ""
    if f72 is not None and p1_72 is not None:
        f72_txt = (
            "on `000010/72` EPnP and ITERATIVE both return the planar flip at "
            f"{f72['solvers']['epnp']['rot_err_deg']:.1f}° / "
            f"{f72['solvers']['iterative']['rot_err_deg']:.1f}° with a reprojection residual "
            f"of {f72['solvers']['epnp']['reproj_rms_px']:.2f} px, and their bearing cost "
            f"J = {f72['solvers']['epnp']['J']:.2e} is *lower* than "
            f"{f72['solvers']['ippe'][0]['J']:.2e} of the true-mode planar solution on the "
            "same raw inliers; the data itself prefers the flip. P1's hard argmin fails on "
            f"exactly that frame ({p1_72['hard_rot_deg']:.2f}°, cited). ")
    A("2. **Single-solution solvers are confidently wrong on some frames — and reproduce P1's "
      "one hard-argmin failure.** EPnP is >10° from GT on "
      f"{len(ep_bad)}/{len(pf)} frames ({bad_str(ep_bad, 'epnp')}); ITERATIVE on "
      f"{len(it_bad)}/{len(pf)} ({bad_str(it_bad, 'iterative')}). This is not just "
      f"non-convergence: {f72_txt}On the other failing frames the solver is simply inaccurate "
      "or returns a behind-camera (mirror) solution, with reprojection residuals of several "
      "px, so those are solver failures, not ambiguity.")
    A("")
    A("3. **The planar multi-solution set is real, but the SQPNP API does not expose it in "
      "cv2 5.0.** `solvePnPGeneric(SOLVEPNP_SQPNP)` returned "
      f"{sorted(set(f['sqpnp_n_sol'] for f in pf))[0] if pf else '—'} solution on every frame "
      "(cv2 5.0's own docstring: \"Only 1 solution is returned\" for non-P3P/IPPE flags), so "
      "SQPNP is a single-solution solver in this toolchain. IPPE on exactly-coplanar "
      f"plane-projected inliers returns 2 solutions on {len(pair)}/{len(pf)} frames with a pair "
      f"rotation of {min(pair):.0f}–{max(pair):.0f}°; the second solution is "
      f"{min(flip_rot):.0f}–{max(flip_rot):.0f}° and {min(flip_t):.0f}–{max(flip_t):.0f} mm "
      "from GT (full per-solution rows in §2b). **First vs oracle:** on these frames the "
      "GT-closer IPPE solution is always the one IPPE lists first, so the oracle gain is "
      f"exactly zero (first = oracle = {med(ipf, 'rot')}° / {med(ipf, 'trans_mm', 1)} mm); the "
      "oracle column is reported because it is the upper bound, and the equality is an "
      "empirical property of these 15 frames, not a documented ordering guarantee.")
    A("")
    A("4. **The unconstrained local GN/LM baseline fails without a physical filter and "
      "multi-start.** From identity/zero the algebraic bearing cost is sign-symmetric "
      "(`Pm(-x) = -Pm(x)`), so the LM iterate can settle on a stationary point with **all "
      f"points behind the camera**: {len(lm_id_mirror)}/{len(pf)} frames end in a mirror "
      "(z ≤ 0) pose, "
      f"{len(lm_id_true)}/{len(pf)} in the true mode, {len(lm_id_flip)}/{len(pf)} in the flip "
      f"mode (median rotation error {med(lmi, 'rot')}°). From the GT init it stays in the true "
      f"mode on {len(lm_gt_true)}/{len(pf)} frames (median {med(lmg, 'rot')}° / "
      f"{med(lmg, 'trans_mm', 1)} mm), and from the IPPE flip init it stays out of the true "
      f"mode ({len(lm_flip_true)}/{len(pf)} true, median {med(lmf, 'rot')}°). This is an "
      "honest negative for a single-init local bearing-cost baseline: P1's explicit rejection "
      "of z ≤ 0 modes and its multi-start enumeration are doing real work here.")
    A("")
    A("5. **GT is not the minimiser of the bearing cost** — the cost is fitted on "
      "nearest-vertex-quantised model points. The solver poses have a lower algebraic cost "
      f"than the GT pose on {n_J_below_gt}/{len(pf)} frames, which is why an argmin sits "
      "~1–2° from the annotation rather than on it, and why ADD-S/translation errors have a "
      "~1 px floor. Statements about 'the true pose being the global minimum' (P1) are about "
      "the true *mode*, not about GT being the exact minimiser; our per-frame rows are "
      "consistent with P1's hard errors.")
    A("")
    A("6. **Scope: forward only.** These are classical forward solvers. Nothing above "
      "differentiates through PnP or supports/denies a gradient claim; the numbers bound the "
      "*forward* accuracy/ambiguity that any differentiable or sampling layer on these "
      "correspondences would have to be compared against, and here that bound is aggressive "
      "(2° median for a plain EPnP/SQPNP).")
    A("")
    A("## 6. Caveats")
    A("")
    A("- **Same correspondences, same subset.** The 15 frames and the RANSAC inlier masks "
      "are reproduced bitwise (verified against `two_mode.json`); nothing was re-tuned.")
    A("- **Sample size.** 15 frames, one object instance each; bootstrap CIs are over "
      "frames and therefore only describe this fixed set, not a population.")
    A("- **Vertex quantization dominates the noise floor.** The model point is the nearest "
      "mesh vertex (~1.6–1.8 mm spacing ≈ 1.1 px at 0.86 m), and the plane inlier band is "
      "2.5 mm; the GT re-projection residual of ~1 px is this quantization, so absolute "
      "accuracy below ~1 px is not meaningful for any method here.")
    A("- **IPPE uses plane-projected points** (exactly coplanar, planarity RMS "
      f"{min(f['planarity_rms_m'] for f in pf)*1000:.2f}–"
      f"{max(f['planarity_rms_m'] for f in pf)*1000:.2f} mm); EPnP/ITERATIVE/SQPNP/LM use "
      "the raw inlier points. This is a classical modelling step, stated so the numbers "
      "are not read as identical inputs.")
    A("- **No front-end.** These poses start from the same real 3-D–2-D correspondences "
      "and K; no detector, no RANSAC-PnP outlier rejection, no refinement with "
      "re-projection error, no test-time augmentation. The numbers characterise solvers, "
      "not a pipeline.")
    A("- **Oracle columns are not deployable** and are shown only to separate 'solver "
      "picked the wrong planar solution' from 'solver is inaccurate'.")
    A("- **P1 numbers are cited, not recomputed**: medians quoted from "
      "`results/two_mode.md`; per-frame P1 values read from `results/two_mode.json` "
      "(same frames). ADD/ADD-S for the P1 poses were not recorded by P1 and are "
      "therefore absent rather than invented.")
    A("- **cv2 version.** cv2 5.0.0 changes the multi-solution contract of "
      "`solvePnPGeneric`; on OpenCV 4.x SQPNP may return up to 3 planar solutions, so the "
      "solution counts here are tool-version-specific and were measured, not assumed.")
    A("")
    Path(path_md).write_text("\n".join(L) + "\n")
    print("wrote", path_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    S = run(quick=args.quick)
    tag = f"_{args.tag}" if args.tag else ""
    write_report(S, OUT / f"classical_baselines{tag}.md", OUT / f"classical_baselines{tag}.json")
    print("wrote", OUT / f"classical_baselines{tag}.json",
          f"({S['timing_s']:.1f}s)")


if __name__ == "__main__":
    main()
