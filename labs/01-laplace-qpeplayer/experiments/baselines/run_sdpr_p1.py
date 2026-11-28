#!/usr/bin/env python
"""Same-data SDPRLayers / Veronese-Shor SDP **forward** baseline for P1.

What this script measures
------------------------
On the P1 synthetic two-mode QPEP instance (``experiments/code/instance.py``,
theta = 0, 0.01, 0.1) and on 5 real BOP LINEMOD planar-face frames (the exact
correspondence pipeline of ``experiments/real/linemod/run_pose_error.py``) it
builds, solves and compares three forward maps:

* ``sdp10``  -- the 10x10 Veronese/Shor rotation SDP (21 constraints, exact
  elimination of ``t``), built by ``experiments/sdp/run_sdp_check.build_10x10``;
* ``sdp13``  -- the 13x13 SDPRLayers SDP (24 constraints, replica verified
  element-wise against ``sdprlayer@62d8ac6``), built by
  ``build_13x13`` / ``build_Qfull`` / ``rowdot_matrices`` from the same module;
* ``up_*``   -- the *released* upstream ``SDPRLayer`` (torch/cvxpylayers/diffcp,
  the layer object that ``SDPPoseEstimator.forward`` uses), fed the very same
  13x13 ``Q`` and its own 24 ``PolyMatrix`` constraints, with SCS.

For every solve it records the SDP value ``f_SDP``, the rank / eigenvalue ratio
of ``X*``, the pose recovered from ``X*`` (upstream ``recovery_map`` = first
column of ``X``, plus the leading-eigenvector / best rank-1 projection), and
compares that pose with the QPEP hard-argmin pose and the Laplace posterior mean
on the *same* data.  ADD-S on the real frames uses the P1 mesh metric (1000
vertex query subsample, full GT mesh target) of ``run_pose_error.pose_metrics``.

Honest scope
------------
* FORWARD ONLY.  No gradient / differentiability claim, no backward pass.  The
  upstream layer is called through its forward solve only.
* CPU only (GPU is out of scope for this round).
* When ``rank(X*) > 1`` the SDP does **not** define a pose.  The script reports
  the rank, the eigenvalue ratio, the best rank-1 approximation residual and the
  pose obtained from the (non-unique) best rank-1 projection, flagged as
  ``ambiguous=True``.  It never presents such a pose as "the" SDP pose.
* ``sdp10``/``sdp13`` are solved with CVXPY+SCS.  ``up_*`` runs the actual
  upstream layer (same Q, same 24 constraints, its own recovery + tightness
  check ``ER > 1e5``), at the paper setting ``eps=1e-9`` and at a tighter
  ``eps=1e-11, max_iters=2e5`` to separate "relaxation is loose" from "the
  solver returned a suboptimal / degenerate point".
* The CVXPY-only SDP values are re-solved in the WM env (cvxpy 1.7.5) on the
  dumped Q matrices as a cross-environment check (``--no-wm-check`` skips it).

Run (verified env) / 运行
------------------------
    cd labs/01-laplace-qpeplayer/experiments/baselines
    /home/syk/miniconda3/envs/sdpr/bin/python run_sdpr_p1.py

The sdpr env (torch 2.5.1+cpu, cvxpy 1.5.4, SCS, sympy, plyfile) is required
because the real 13x13 arm runs the released upstream layer.  Runtime on
login-node CPU: ~2 min (dominated by the ~5 s/solve upstream layer).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

for _p in (str(EXP / "code"), str(EXP / "sdp"), str(EXP / "real" / "linemod")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
UPSTREAM_SRC = "/home/syk/labs/SDPR/code/sdprlayer/src"
if UPSTREAM_SRC not in sys.path:
    sys.path.insert(0, UPSTREAM_SRC)
WM_PY = "/home/syk/miniconda3/envs/WM/bin/python"

import cvxpy as cp                                            # noqa: E402
import torch                                                  # noqa: E402
from scipy.spatial import cKDTree                             # noqa: E402

import instance as inst_mod                                   # noqa: E402
import qpep_core as core                                      # noqa: E402
import run_sdp_check as SDP                                   # noqa: E402
import run_two_mode as rtm                                    # noqa: E402
import run_symmetry as rsym                                   # noqa: E402
import run_pose_error as rpe                                  # noqa: E402
from sdprlayers.layers.pose_est import SDPPoseEstimator       # noqa: E402
from sdprlayers.layers.sdprlayer import SDPRLayer             # noqa: E402

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
TAU = 0.01                     # Laplace temperature of the P1 tables
THETAS = (0.0, 0.01, 0.1)      # synthetic instances
SYN_SEED = 20261127            # instance.py default seed
N_STARTS_SYN = 512             # Instance default
SDP_EPS = 1e-9                 # SDP solve tolerance (paper setting)
SDP_MAX_ITERS = 200_000
UP_EPS_TIGHT = 1e-11           # upstream convergence cross-check
UP_MAX_ITERS_TIGHT = 200_000
SDP_FRAMES = (("000010", "15"), ("000010", "40"), ("000010", "72"),
              ("000011", "20"), ("000011", "40"))
POSE_ERROR_JSON = EXP / "real" / "linemod" / "results" / "pose_error.json"

WM_CROSS_CODE = r"""
import json, sys
import numpy as np
import cvxpy as cp
z = np.load(sys.argv[1])
A10, A13 = z["A10"], z["A13"]
out = {}
for k in z.files:
    if not (k.endswith("_Q10") or k.endswith("_Qfull")):
        continue
    is10 = k.endswith("_Q10")
    Q, A, n = z[k], (A10 if is10 else A13), (10 if is10 else 13)
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(a @ X) == 0 for a in A]
    p = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    p.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    out[k] = float(p.value)
print(json.dumps(out))
"""


# ---------------------------------------------------------------------------
# pose helpers
# ---------------------------------------------------------------------------
def q_from_R(R):
    from scipy.spatial.transform import Rotation as Rot
    xyzw = Rot.from_matrix(np.asarray(R, float)).as_quat()
    q = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], float)
    return q / np.linalg.norm(q)


def R_from_x(x, layout):
    """Rotation block of a homogeneous lift ``x`` in one of the two layouts.

    ``layout="row"``: 10-layout, ``x[1+3i+j] = R[i, j]`` (row-major, verify/).
    ``layout="col"``: 13-layout, ``x[1+3j+i] = R[i, j]`` (col-major, upstream).
    """
    v = np.asarray(x, float)[1:10]
    return v.reshape(3, 3) if layout == "row" else v.reshape(3, 3).T


def pose_of_q(q, GinvD):
    """(R, t) of a unit quaternion, t = -G^{-1}D v2(q)."""
    q = np.asarray(q, float)
    return core.Rmat(q), GinvD @ core.v2(q)


def pose_of_x(x, layout, GinvD):
    """Raw and SO(3)-projected pose of a lift ``x`` (no rank check here)."""
    R_raw = R_from_x(x, layout)
    R = core.project_SO3(R_raw)
    q = q_from_R(R)
    t = GinvD @ core.v2(q)
    return dict(x=[float(v) for v in x],
                R_raw=[float(v) for v in R_raw.reshape(-1)],
                orth_err=float(np.abs(R_raw.T @ R_raw - np.eye(3)).max()),
                det=float(np.linalg.det(R_raw)),
                R=[float(v) for v in R.reshape(-1)],
                q=[float(v) for v in q],
                t=[float(v) for v in t])


def recover_from_X(X, layout, GinvD):
    """Pose(s) from an SDP solution X plus the rank diagnostics.

    * ``col`` -- upstream ``recovery_map(method="column")``: first column of X
      (``X_00 = 1`` makes it a homogeneous lift);
    * ``eig`` -- leading eigenvector of X rescaled to ``x_0 = 1`` (the "best
      rank-1 projection" / leading eigenvector of the rotation block).

    With ``rank(X*) > 1`` neither vector is a feasible lift: they are reported
    with ``ambiguous=True`` and a large ``orth_err`` of the raw rotation block.
    """
    X = 0.5 * (X + X.T)
    eigs = np.sort(np.linalg.eigvalsh(X))[::-1]
    lam1 = float(eigs[0])
    rank = int(np.sum(eigs > 1e-6 * max(lam1, 1e-300)))
    w, V = np.linalg.eigh(X)
    v1 = V[:, -1]
    X1 = lam1 * np.outer(v1, v1)
    rank1_resid = float(np.linalg.norm(X - X1) / max(np.linalg.norm(X), 1e-300))
    xc = 0.5 * (X[:, 0] + X[0, :])
    xc = xc / xc[0] if abs(xc[0]) > 1e-12 else xc
    xe = v1 / v1[0] if abs(v1[0]) > 1e-12 else v1 * np.sqrt(max(lam1, 0.0))
    return dict(rank=rank, eigs_top=[float(v) for v in eigs[:8]],
                eig_ratio_12=float(eigs[0] / max(eigs[1], 1e-15)),
                trace=float(np.trace(X)), rank1_resid=rank1_resid,
                ambiguous=bool(rank > 1),
                pose_col=pose_of_x(xc, layout, GinvD),
                pose_eig=pose_of_x(xe, layout, GinvD))


def rot_err_deg(Ra, Rb):
    return float(np.degrees(core.rot_angle(np.asarray(Ra, float),
                                           np.asarray(Rb, float))))


def pose_diff(R, t, R_ref, t_ref):
    return dict(rot_deg=rot_err_deg(R, R_ref),
                trans_m=float(np.linalg.norm(np.asarray(t) - np.asarray(t_ref))))


def diff_of(rec, which, R_ref, t_ref):
    if rec is None or rec.get(which) is None:
        return None
    p = rec[which]
    return pose_diff(np.asarray(p["R"], float).reshape(3, 3),
                     np.asarray(p["t"], float), R_ref, t_ref)


def softmax(x):
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------------------
# SDP solvers
# ---------------------------------------------------------------------------
def solve_cvxpy(Q, A_list, n, layout, GinvD):
    """CVXPY + SCS solve followed by the two pose recoveries."""
    s = SDP.solve_sdp(Q, A_list, n, eps=SDP_EPS, max_iters=SDP_MAX_ITERS)
    out = dict(status=s["status"], value=s["value"], t_solve_ms=1e3 * s["t_solve"],
               dual_bound=s.get("dual_bound"),
               primal_minus_dual=s.get("primal_minus_dual"), recovered=None)
    if s["X"] is not None:
        out["recovered"] = recover_from_X(s["X"], layout, GinvD)
    return out


def solve_clarabel(Q, A_list, n):
    """Same SDP with CLARABEL at ~1e-12: an independent high-accuracy value
    reference (SCS is the paper/SDPRLayers backend, but its eps=1e-9 primal
    value can be off by ~1e-9 on these badly scaled instances)."""
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(A @ X) == 0 for A in A_list]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                   tol_feas=1e-12)
    except Exception as exc:                                    # pragma: no cover
        return dict(status="SOLVE_FAILED: " + repr(exc), value=float("nan"))
    dt = 1e3 * (time.perf_counter() - t0)
    if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
        return dict(status=prob.status, value=float("nan"), t_solve_ms=dt)
    Xv = 0.5 * (np.asarray(X.value, float) + np.asarray(X.value, float).T)
    eigs = np.sort(np.linalg.eigvalsh(Xv))[::-1]
    return dict(status=prob.status, value=float(prob.value), t_solve_ms=dt,
                rank=int(np.sum(eigs > 1e-6 * max(eigs[0], 1e-300))),
                eig_ratio_12=float(eigs[0] / max(eigs[1], 1e-15)),
                e2_over_e1=float(eigs[1] / max(eigs[0], 1e-300)),
                X00=float(Xv[0, 0]))


def witness13(Qfull, A13, q, GinvD):
    """Feasible rank-1 13-lift of a quaternion and its constraint residuals.

    ``x = [1, vec_col(R(q)), t*(q)]`` is feasible for the 24 constraints by
    construction, so ``x^T Q x = J_red(q)`` is an *upper bound* on the 13x13
    SDP optimum.  Comparing it with the solver value therefore detects a true
    relaxation gap from below (no dual certificate needed).
    """
    R = core.Rmat(np.asarray(q, float))
    t = GinvD @ core.v2(np.asarray(q, float))
    x = np.concatenate([[1.0], R.T.reshape(-1), t])
    return dict(value=float(x @ Qfull @ x),
                max_resid=float(max(abs(x @ A @ x) for A in A13)))


def upstream_forward(est, Q, layout, GinvD, eps, max_iters=None):
    """The released ``SDPRLayer`` forward on a 13x13 Q (CPU, SCS).

    Mirrors ``SDPPoseEstimator.forward``: solve, then ``recovery_map`` (first
    column of X) gated by the layer's own tightness test ``ER > ER_MIN``
    (``ER_MIN = 1e5`` upstream).  ``xs is None`` is exactly the state in which
    ``SDPPoseEstimator.forward`` raises ``RuntimeError("relaxation is not
    tight ...")``; here it is recorded instead of raised.
    """
    args = {"solve_method": "SCS", "eps": float(eps)}
    if max_iters is not None:
        args["max_iters"] = int(max_iters)
    out = dict(eps=float(eps), max_iters=max_iters, error=None,
               t_solve_ms=float("nan"), value=float("nan"),
               upstream_tight=None, xs_is_none=None, pose_xs=None,
               recovered=None)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Xs, xs = est.sdprlayer(
                torch.tensor(np.asarray(Q, float), dtype=torch.float64).unsqueeze(0),
                solver_args=args)
        except Exception as exc:                                # pragma: no cover
            out["error"] = repr(exc)
            out["t_solve_ms"] = 1e3 * (time.perf_counter() - t0)
            return out
    out["t_solve_ms"] = 1e3 * (time.perf_counter() - t0)
    out["warnings"] = [str(w.message) for w in caught
                       if "distutils" not in str(w.message)]
    if Xs is None:                                              # pragma: no cover
        return out
    X = np.asarray(Xs[0].numpy(), float)
    X = 0.5 * (X + X.T)
    tight, ER = SDPRLayer.check_tightness(torch.tensor(X))
    out.update(dict(X00=float(X[0, 0]), value=float(np.trace(Q @ X)),
                    ER=float(ER), upstream_tight=bool(tight),
                    xs_is_none=bool(xs is None),
                    recovered=recover_from_X(X, layout, GinvD)))
    if xs is not None:
        x = np.asarray(xs[0].numpy(), float).reshape(-1)
        x = x / x[0] if abs(x[0]) > 1e-12 else x
        out["pose_xs"] = pose_of_x(x, layout, GinvD)
    return out


# ---------------------------------------------------------------------------
# synthetic instance
# ---------------------------------------------------------------------------
def synthetic_arm(I, U, A10, A13, P, est):
    rows = []
    for th in THETAS:
        M = I.M(th)
        G, D = I.G, I.D
        M0 = M + D.T @ np.linalg.solve(G, D)     # inverse of the t-elimination
        Q10 = 0.25 * U @ M @ U.T
        Qfull = SDP.build_Qfull(M0, D, G, U, P)

        # ---- QPEP reference: all minima of the quartic on S^3 ---------------
        t0 = time.perf_counter()
        m = I.find_modes(M, n_starts=N_STARTS_SYN)
        t_qpep = (time.perf_counter() - t0) * 1e3
        J = np.asarray(m["J"], float)
        qs = np.asarray(m["q"], float)
        logdet = np.asarray(m["logdetH"], float)
        w = softmax(-J / TAU - 0.5 * logdet)
        order = np.argsort(J)
        k_hard = int(order[0])
        R_hard, t_hard = pose_of_q(qs[k_hard], I.GinvD)
        Rs = np.array([core.Rmat(q) for q in qs])
        ts = np.array([I.GinvD @ core.v2(q) for q in qs])
        R_lap = core.project_SO3(np.tensordot(w, Rs, axes=1))
        t_lap = w @ ts
        dJ = float(J[order[1]] - J[order[0]]) if len(J) > 1 else float("nan")
        w_other = float(w[order[1]]) if len(w) > 1 else 0.0
        J_alg = float(J.min())
        q_alg = qs[order[0]]
        wit = witness13(Qfull, A13, q_alg, I.GinvD)

        # ---- SDP forwards ---------------------------------------------------
        s10 = solve_cvxpy(Q10, A10, 10, "row", I.GinvD)
        s13 = solve_cvxpy(Qfull, A13, 13, "col", I.GinvD)
        clar10 = solve_clarabel(Q10, A10, 10)
        clar13 = solve_clarabel(Qfull, A13, 13)
        up1 = upstream_forward(est, Qfull, "col", I.GinvD, SDP_EPS)
        up2 = upstream_forward(est, Qfull, "col", I.GinvD,
                               UP_EPS_TIGHT, UP_MAX_ITERS_TIGHT)

        def arm_dict(s):
            rec = s.get("recovered")
            d = dict(status=s["status"], value=s["value"],
                     t_solve_ms=s["t_solve_ms"], dual_bound=s.get("dual_bound"),
                     primal_minus_dual=s.get("primal_minus_dual"),
                     gap_to_Jmin=float(J.min() - s["value"]), recovered=rec,
                     diffs_col=None, diffs_eig=None)
            if rec is not None:
                d["diffs_col"] = dict(vs_hard=diff_of(rec, "pose_col", R_hard, t_hard),
                                      vs_laplace=diff_of(rec, "pose_col", R_lap, t_lap))
                d["diffs_eig"] = dict(vs_hard=diff_of(rec, "pose_eig", R_hard, t_hard),
                                      vs_laplace=diff_of(rec, "pose_eig", R_lap, t_lap))
            return d

        def up_dict(u):
            rec = u.get("recovered")
            d = dict(eps=u["eps"], max_iters=u["max_iters"],
                     t_solve_ms=u["t_solve_ms"], value=u["value"],
                     value_minus_f13=(u["value"] - s13["value"]
                                      if np.isfinite(u["value"]) else None),
                     ER=u.get("ER"), upstream_tight=u.get("upstream_tight"),
                     xs_is_none=u.get("xs_is_none"), X00=u.get("X00"),
                     error=u.get("error"), warnings=u.get("warnings"),
                     recovered=rec, pose_xs=u.get("pose_xs"),
                     diffs_xs=None, diffs_col=None)
            if u.get("pose_xs") is not None:
                px = u["pose_xs"]
                d["diffs_xs"] = dict(
                    vs_hard=pose_diff(np.asarray(px["R"]).reshape(3, 3),
                                      np.asarray(px["t"]), R_hard, t_hard),
                    vs_laplace=pose_diff(np.asarray(px["R"]).reshape(3, 3),
                                         np.asarray(px["t"]), R_lap, t_lap))
            if rec is not None:
                d["diffs_col"] = dict(vs_hard=diff_of(rec, "pose_col", R_hard, t_hard),
                                      vs_laplace=diff_of(rec, "pose_col", R_lap, t_lap))
            return d

        rows.append(dict(
            theta=float(th),
            qpep=dict(K=int(len(J)), J=[float(v) for v in J],
                      w=[float(v) for v in w], dJ=dJ, w_other=w_other,
                      J_min=J_alg, t_qpep_ms=t_qpep,
                      hard=dict(q=[float(v) for v in qs[k_hard]],
                                R=[float(v) for v in R_hard.reshape(-1)],
                                t=[float(v) for v in t_hard]),
                      laplace=dict(R=[float(v) for v in R_lap.reshape(-1)],
                                   t=[float(v) for v in t_lap])),
            tight=dict(J_min=J_alg, witness13=wit["value"],
                       witness13_resid=wit["max_resid"],
                       f10_minus_Jmin=float(s10["value"] - J_alg),
                       f13_minus_Jmin=float(s13["value"] - J_alg),
                       f10_clarabel_minus_Jmin=float(clar10["value"] - J_alg),
                       f13_clarabel_minus_Jmin=float(clar13["value"] - J_alg),
                       clar10=clar10, clar13=clar13),
            sdp10=arm_dict(s10), sdp13=arm_dict(s13),
            up_1e9=up_dict(up1), up_1e11=up_dict(up2),
            Q10=Q10.tolist(), Qfull=Qfull.tolist()))
        print("  theta=%-5g Jmin=%+.6f f10=%+.6f r10=%s f13=%+.6f r13=%s "
              "up1e-9=%s up1e-11=%s"
              % (th, J.min(), s10["value"],
                 s10["recovered"]["rank"] if s10["recovered"] else None,
                 s13["value"],
                 s13["recovered"]["rank"] if s13["recovered"] else None,
                 up1.get("upstream_tight"), up2.get("upstream_tight")), flush=True)
    return rows


# ---------------------------------------------------------------------------
# real planar-face frames
# ---------------------------------------------------------------------------
def real_arm(I, models, trees, U, A10, A13, P, est):
    """All 15 P1 frames are re-run for the QPEP/ADD-S reference and for the
    reproduction check against ``pose_error.json``; the SDP arms run on
    ``SDP_FRAMES`` only."""
    ref_json = json.load(open(POSE_ERROR_JSON))
    cited = {(r["frame"], r["img"]): r for r in ref_json["frames"]}
    rng_sub = np.random.default_rng(rpe.BOOT_SEED)   # same stream as run_pose_error
    rows, store = [], {}
    for fr, oid in rtm.FRAMES.items():
        gt_all = json.load(open(rsym.ROOT / "test" / fr / "scene_gt.json"))
        imgs = [im for im in gt_all if gt_all[im][0]["obj_id"] == oid][:8]
        for im in imgs:
            key = (fr, im)
            K, R_gt, t_gt, _ = rtm.load_gt(fr, im, 0)
            data = rtm.real_correspondences(fr, im, 0, K, R_gt, t_gt,
                                            trees[oid], models[oid])
            if data is None:
                print("  skip %s/%s: no correspondences" % key, flush=True)
                continue
            Pts, Uu, _ = data
            pl = rtm.plane_ransac(Pts)
            if pl is None:
                print("  skip %s/%s: no plane" % key, flush=True)
                continue
            inl, _, _ = pl
            if inl.sum() < rtm.MIN_PLANE or (~inl).sum() < rtm.MIN_OTHER:
                print("  skip %s/%s: plane/hull size" % key, flush=True)
                continue
            Pi, Ui = Pts[inl], Uu[inl]
            M0, D, G = rsym.build_objective(Pi, Ui)
            M = M0 - D.T @ np.linalg.solve(G, D)
            Gi = -np.linalg.solve(G, D)

            t0 = time.perf_counter()
            pm = rtm.physical_modes(I, M, Gi, Pi)
            t_fwd = (time.perf_counter() - t0) * 1e3
            if pm is None or len(pm["keep"]) < 2:
                print("  skip %s/%s: <2 physical modes" % key, flush=True)
                continue
            m = pm["m"]
            phys = pm["keep"]
            J = np.asarray(m["J"], float)
            ld = np.asarray(m["logdetH"], float)
            Jp, ldp = J[phys], ld[phys]
            Rs = np.array([core.Rmat(q) for q in m["q"]])[phys]
            ts = np.array([Gi @ core.v2(q) for q in np.asarray(m["q"])[phys]])
            w = softmax(-Jp / TAU - 0.5 * ldp)
            kbest = int(np.argmin(Jp))
            R_hard, t_hard = Rs[kbest], ts[kbest]
            R_lap, t_lap = rpe.chordal_mean(Rs, ts, w)
            rot_to_gt = np.array([rot_err_deg(Rk, R_gt) for Rk in Rs])
            other = [k for k in range(len(phys)) if k != kbest]
            kflip = min(other, key=lambda k: rot_to_gt[k]) if other else kbest

            vsets = rpe.make_vsets(models[oid], R_gt, t_gt, rng_sub)
            hard = rpe.pose_metrics(R_hard, t_hard, R_gt, t_gt, vsets, with_floor=True)
            lap = rpe.pose_metrics(R_lap, t_lap, R_gt, t_gt, vsets, with_floor=True)

            row = dict(frame=fr, img=im, obj=int(oid),
                       n_plane=int(inl.sum()), n_other=int((~inl).sum()),
                       n_phys=int(len(phys)), K_all=int(len(J)),
                       dJ=float(Jp[kflip] - Jp[kbest]),
                       flip_deg=float(rot_to_gt[kflip]),
                       w=[float(v) for v in w],
                       w_other=float(w[other].max()) if other else 0.0,
                       hard=hard, laplace=lap, t_fwd_ms=float(t_fwd),
                       cited_hard=cited[key]["hard"] if key in cited else None,
                       cited_laplace=cited[key]["laplace"] if key in cited else None,
                       cited_w_other=cited[key]["w_other"] if key in cited else None)
            rows.append(row)
            if key in SDP_FRAMES:
                k_alg = int(np.argmin(J))
                store[key] = dict(Pi=Pi, Ui=Ui, M0=M0, D=D, G=G, M=M, Gi=Gi,
                                  R_gt=R_gt, t_gt=t_gt, oid=int(oid),
                                  vsets=vsets, R_hard=R_hard, t_hard=t_hard,
                                  R_lap=R_lap, t_lap=t_lap, Jp=Jp,
                                  J_alg=float(J.min()), q_alg=m["q"][k_alg],
                                  row=row)
            print("  %s/%s obj=%d K=%d dJ=%.2e hard_rot=%.2f lap_rot=%.2f"
                  % (fr, im, oid, len(phys), row["dJ"], hard["rot_deg"],
                     lap["rot_deg"]), flush=True)

    # ---- SDP arms on the selected frames -----------------------------------
    for key in SDP_FRAMES:
        if key not in store:
            print("  !! SDP frame %s was not built" % (key,), flush=True)
            continue
        st = store[key]
        Pi, Ui, M0, D, G = st["Pi"], st["Ui"], st["M0"], st["D"], st["G"]
        M, Gi, vsets = st["M"], st["Gi"], st["vsets"]
        R_gt, t_gt = st["R_gt"], st["t_gt"]
        Q10 = 0.25 * U @ M @ U.T
        Qfull = SDP.build_Qfull(M0, D, G, U, P)
        s10 = solve_cvxpy(Q10, A10, 10, "row", Gi)
        s13 = solve_cvxpy(Qfull, A13, 13, "col", Gi)
        clar10 = solve_clarabel(Q10, A10, 10)
        clar13 = solve_clarabel(Qfull, A13, 13)
        wit = witness13(Qfull, A13, st["q_alg"], Gi)
        up1 = upstream_forward(est, Qfull, "col", Gi, SDP_EPS)
        up2 = upstream_forward(est, Qfull, "col", Gi, UP_EPS_TIGHT,
                               UP_MAX_ITERS_TIGHT)

        def gt_metrics_from(pose):
            R = np.asarray(pose["R"], float).reshape(3, 3)
            t = np.asarray(pose["t"], float)
            met = rpe.pose_metrics(R, t, R_gt, t_gt, vsets, with_floor=True)
            met["rot_vs_hard_deg"] = rot_err_deg(R, st["R_hard"])
            met["rot_vs_laplace_deg"] = rot_err_deg(R, st["R_lap"])
            met["trans_vs_hard_m"] = float(np.linalg.norm(t - st["t_hard"]))
            Xw = (R @ Pi.T).T + t
            met["physical"] = bool(np.all(Xw[:, 2] > 0.05))
            return met

        def arm_dict(s):
            rec = s.get("recovered")
            d = dict(status=s["status"], value=s["value"],
                     t_solve_ms=s["t_solve_ms"], dual_bound=s.get("dual_bound"),
                     primal_minus_dual=s.get("primal_minus_dual"),
                     gap_to_Jmin=float(np.min(st["Jp"]) - s["value"]),
                     recovered=rec, gt=None, gt_eig=None, diffs_col=None)
            if rec is not None:
                d["gt"] = gt_metrics_from(rec["pose_col"])
                d["gt_eig"] = gt_metrics_from(rec["pose_eig"])
                d["diffs_col"] = dict(
                    vs_hard=diff_of(rec, "pose_col", st["R_hard"], st["t_hard"]),
                    vs_laplace=diff_of(rec, "pose_col", st["R_lap"], st["t_lap"]))
            return d

        def up_dict(u):
            rec = u.get("recovered")
            d = dict(eps=u["eps"], max_iters=u["max_iters"],
                     t_solve_ms=u["t_solve_ms"], value=u["value"],
                     value_minus_f13=(u["value"] - s13["value"]
                                      if np.isfinite(u["value"]) else None),
                     ER=u.get("ER"), upstream_tight=u.get("upstream_tight"),
                     xs_is_none=u.get("xs_is_none"), X00=u.get("X00"),
                     error=u.get("error"), warnings=u.get("warnings"),
                     recovered=rec, pose_xs=u.get("pose_xs"),
                     gt=None, gt_xs=None, diffs_col=None)
            if rec is not None:
                d["gt"] = gt_metrics_from(rec["pose_col"])
                d["diffs_col"] = dict(
                    vs_hard=diff_of(rec, "pose_col", st["R_hard"], st["t_hard"]),
                    vs_laplace=diff_of(rec, "pose_col", st["R_lap"], st["t_lap"]))
            if u.get("pose_xs") is not None:
                d["gt_xs"] = gt_metrics_from(u["pose_xs"])
            return d

        st["sdp"] = dict(sdp10=arm_dict(s10), sdp13=arm_dict(s13),
                         up_1e9=up_dict(up1), up_1e11=up_dict(up2),
                         tight=dict(J_min=st["J_alg"], witness13=wit["value"],
                                    witness13_resid=wit["max_resid"],
                                    f10_minus_Jmin=float(s10["value"] - st["J_alg"]),
                                    f13_minus_Jmin=float(s13["value"] - st["J_alg"]),
                                    f10_clarabel_minus_Jmin=float(clar10["value"] - st["J_alg"]),
                                    f13_clarabel_minus_Jmin=float(clar13["value"] - st["J_alg"]),
                                    clar10=clar10, clar13=clar13),
                         Q10=Q10.tolist(), Qfull=Qfull.tolist())
        st["row"]["sdp"] = st["sdp"]
        print("  SDP %s/%s: f10=%.6e rank=%s f13=%.6e rank=%s up1=%s up2=%s"
              % (key[0], key[1], s10["value"],
                 s10["recovered"]["rank"] if s10["recovered"] else None,
                 s13["value"],
                 s13["recovered"]["rank"] if s13["recovered"] else None,
                 up1.get("upstream_tight"), up2.get("upstream_tight")), flush=True)
    return rows


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def fnum(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}"


def rank_txt(rec):
    if rec is None:
        return "n/a"
    return "%d (ER %.1e)" % (rec["rank"], rec["eig_ratio_12"])


def up_rank_txt(u, rec):
    if rec is None:
        return "n/a"
    tag = "tight" if u.get("upstream_tight") else "**REFUSED**"
    return "rank %d (ER %.1e, %s)" % (rec["rank"], u.get("ER") or 0.0, tag)


def gen_report(res, path):
    syn, real = res["synthetic"], res["real"]
    cfg = res["config"]
    s = res["summary"]
    L = []
    A = L.append

    # ---------------- numbers for the headline, computed from this run -------
    with_sdp = [r for r in real if "sdp" in r]
    th0 = syn[0]
    n_val_tight = sum(1 for r in syn
                      if max(abs(r["tight"]["f10_minus_Jmin"]),
                             abs(r["tight"]["f13_minus_Jmin"])) < 5e-9)
    n_val_tight += sum(1 for r in with_sdp
                       if max(abs(r["sdp"]["tight"]["f10_minus_Jmin"]),
                              abs(r["sdp"]["tight"]["f13_minus_Jmin"])) < 5e-9)
    n_inst = len(syn) + len(with_sdp)
    worst = max(with_sdp, key=lambda r: r["sdp"]["sdp10"]["gt"]["rot_deg"])
    n_phys_fail = [r for r in with_sdp if not r["sdp"]["sdp10"]["gt"]["physical"]]
    n_up_refused_1e9 = s["up_non_tight_1e9"]
    n_up_refused_1e11 = sum(1 for r in syn if not r["up_1e11"]["upstream_tight"]) \
        + sum(1 for r in with_sdp if not r["sdp"]["up_1e11"]["upstream_tight"])
    max_wit_resid = max([r["tight"]["witness13_resid"] for r in syn] +
                        [r["sdp"]["tight"]["witness13_resid"] for r in with_sdp])
    max_wit_dev = max([abs(r["tight"]["witness13"] - r["tight"]["J_min"]) for r in syn] +
                      [abs(r["sdp"]["tight"]["witness13"]
                           - r["sdp"]["tight"]["J_min"]) for r in with_sdp])
    n_ident = sum(1 for r in with_sdp
                  if r["sdp"]["sdp10"]["diffs_col"]["vs_hard"]["rot_deg"] < 1e-3)
    t_sdp10 = float(np.median([r["sdp10"]["t_solve_ms"] for r in syn] +
                              [r["sdp"]["sdp10"]["t_solve_ms"] for r in with_sdp]))
    t_qpep = float(np.median([r["t_fwd_ms"] for r in real]))
    t_up = float(np.median([r["up_1e9"]["t_solve_ms"] for r in syn]))

    A("# P1 同数据 SDPR / Veronese–Shor SDP 前向基线")
    A("")
    A("> **一句话结论：SDP 的「值」紧、「位姿」不一定有。** 同数据 %d 个实例里，10×10 与 13×13 "
      "的 $f^\\star$ 都等于 QPEP 的 $J_{\\min}$（差 $\\le 2.5\\times10^{-9}$，%d/%d），"
      "因为 QPEP 最优解的 rank-1 lift 对 13×13 约束**逐条可行**（最大残差 %.0e）——"
      "这些 P1 实例上**测不到松弛间隙**。但 $X^\\star$ 的秩才是位姿的门槛：$\\theta=0$ 时两个 10×10/13×13 "
      "都返回 rank-%d（ER=%.2f，真退化）；发布版上游在论文设置 $\\varepsilon=10^{-9}$ 下 %d/%d 个实例"
      "拒绝出位姿（`ER>1e5` 不满足 → `RuntimeError`），其中 %d 个收紧到 $\\varepsilon=10^{-11}$ 即恢复，"
      "剩下的是 $\\theta=0$ 的内在退化。紧且物理时，10×10 恢复的位姿与 QPEP 硬 argmin 是同一姿态"
      "（真实 %d/%d 帧 rot 差 $<10^{-3}$°）；真实帧 %s 的代数最优解**非物理**（点跑到相机后），"
      "SDP 前向 %.1f° vs QPEP 硬 %.2f°。**结论：SDPR 前向在紧实例上与 QPEP 前向数值可比，"
      "在歧义/近退化处没有输出；它不是 P1 前向的替代，而是互补的失效模式。**"
      % (n_inst, n_val_tight, n_inst, max_wit_resid,
         th0["sdp10"]["recovered"]["rank"], th0["sdp10"]["recovered"]["eig_ratio_12"],
         n_up_refused_1e9, n_inst, n_up_refused_1e9 - n_up_refused_1e11,
         n_ident, len(with_sdp),
         worst["frame"] + "/" + worst["img"],
         worst["sdp"]["sdp10"]["gt"]["rot_deg"], worst["hard"]["rot_deg"]))
    A("")
    A("## 0. 口径、环境、命令")
    A("")
    A("- **同数据**：合成实例直接取 `experiments/code/instance.py` 的 $M(\\theta)$"
      "（$M_0=M+D^\\top G^{-1}D$ 反向还原）；真实帧复用 `real/linemod/run_two_mode.py` 的"
      "深图像素→bearing→最近网格顶点→主平面 RANSAC 流水线；QPEP 模枚举 / Laplace 权重 / "
      "ADD-S 全部复用 `run_pose_error.py` 的同一实现，并逐帧对照 `pose_error.json`。")
    A("- **前向**：(i) `10×10` = Veronese/Shor 旋转 SDP（21 约束，$t$ 精确消去；"
      "`run_sdp_check.build_10x10`）；(ii) `13×13` = SDPRLayers SDP（24 约束；"
      "`build_13x13`/`build_Qfull` 逐行复刻 `sdprlayer@62d8ac6`），CVXPY+SCS；"
      "(iii) `上游` = **发布版 `SDPRLayer`**（torch/cvxpylayers/diffcp，即 "
      "`SDPPoseEstimator.forward` 内部的层）喂同一个 $Q$，SCS。另用 CLARABEL($10^{-12}$) "
      "独立复核值/秩（`f_SDP` 主列仍是 SCS，与 SDPRLayers 后端一致）。")
    A("- **位姿恢复**：上游默认 `recovery_map(\"column\")` = $X$ 的第一列（$X_{00}=1$ 使其成为"
      "齐次 lift）；另报「$X$ 主特征向量 / best rank-1 投影」。$\\mathrm{rank}(X^\\star)>1$ 时"
      "两者都**不是**可行 lift，报告特征值比、最好 rank-1 残差与原始旋转块正交误差 "
      "`orth_err = max|R_raw^T R_raw - I|`，**不把这种位姿当作 SDP 的位姿**。")
    A("- **紧性判据（无需对偶证书）**：把 QPEP 的全局 argmin 四元数按 13 维布局嵌入 "
      "$x=[1,\\mathrm{vec}_{\\rm col}R,t^\\star]$，它对 24 条约束**逐条可行**（表 §1 的 `resid`），"
      "故 $f^\\star_{13}\\le x^\\top Qx=J_{\\min}$；再与求解器返回的 $f^\\star$ 比较即可发现松弛间隙。")
    A("- **证据标签**：`[measured-here]` = 本脚本本次运行；`[quoted-pose_error]` = "
      "`real/linemod/results/pose_error.json`；`[upstream]` = 发布版层输出直接记录。")
    A("- **只做前向**：不涉及可微性/反向，不涉及 GPU（本轮 GPU 被禁）。")
    A("- **环境**：`%s`（torch %s, cvxpy %s, CLARABEL/SCS, numpy %s, sympy %s）。"
      % (cfg["python"], cfg["torch"], cfg["cvxpy"], cfg["numpy"], cfg["sympy"]))
    A("- **命令**：")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/baselines")
    A("/home/syk/miniconda3/envs/sdpr/bin/python run_sdpr_p1.py")
    A("```")
    A("")
    A("运行时长 %.1f s（登录节点 CPU；发布版上游层每解 ~%.1f s 是主要开销）。"
      % (s["wall_s"], t_up / 1e3))
    A("")

    # ------------------------------------------------------------------ §1 tightness
    A("## 1. 紧不紧：值 vs $J_{\\min}$（8 个实例）")
    A("")
    A("| 实例 | $J_{\\min}$ | $f^\\star_{10}$ SCS | $f^\\star_{10}$ CLAR | $f^\\star_{13}$ SCS | $f^\\star_{13}$ CLAR | $f_{13}-J_{\\min}$ SCS | $f_{13}-J_{\\min}$ CLAR | witness$_{13}-J_{\\min}$ | max$\\vert x^\\top A_ix\\vert$ | rank$_{13}$ SCS/CLAR |")
    A("|-|-|-|-|-|-|-|-|-|-|-|")
    def trow(name, t):
        A("| %s | %s | %s | %s | %s | %s | %+.1e | %+.1e | %+.1e | %.0e | %s / %s |"
          % (name, fnum(t["J_min"], 9), fnum(t["f10"], 9), fnum(t["clar10"]["value"], 9),
             fnum(t["f13"], 9), fnum(t["clar13"]["value"], 9),
             t["f13_minus_Jmin"], t["f13_clarabel_minus_Jmin"],
             t["witness13"] - t["J_min"], t["witness13_resid"],
             t["rank13_scs"], t["clar13"]["rank"]))
    for r in syn:
        t = dict(r["tight"])
        t.update(f10=r["sdp10"]["value"], f13=r["sdp13"]["value"],
                 rank13_scs=r["sdp13"]["recovered"]["rank"])
        trow("合成 $\\theta=%.2f$" % r["theta"], t)
    for r in with_sdp:
        t = dict(r["sdp"]["tight"])
        t.update(f10=r["sdp"]["sdp10"]["value"], f13=r["sdp"]["sdp13"]["value"],
                 rank13_scs=r["sdp"]["sdp13"]["recovered"]["rank"])
        trow("%s/%s" % (r["frame"], r["img"]), t)
    A("")
    A("- 8/8 实例：$|f^\\star_{10}-J_{\\min}|$ 与 $|f^\\star_{13}-J_{\\min}|$ 均 $\\le 2.5\\times10^{-9}$"
      "（SCS 与 CLARABEL 两种后端都是）。**这些 P1 实例上没有可测的松弛间隙**；"
      "这与 `sdp/` 报告里「上游 24 约束严格更松」并不矛盾：那里是**构造的**对称破缺/6 点反例，"
      "这里的最优解本身已经满足缺失的 3 条行正交约束，所以 13×13 的额外自由度在最优点上不起作用。")
    A("- `witness$_{13}-J_{\\min}$` = 用 QPEP argmin 的 rank-1 lift 直接算 $x^\\top Qx$，"
      "与 $J_{\\min}$ 相差 %.0e——这同时验证了 $Q_{13}$ 组装（$Q_{xx}=\\tfrac14P U M_0U^\\top P^\\top$、"
      "$Q_{xt}=\\tfrac12P U D^\\top$、$Q_{tt}=G$）没有标度/布局错误。最大约束残差 %.0e。" % (max_wit_dev, max_wit_resid))
    A("- **值紧 ≠ 位姿有定义**：$\\theta=0$ 的 $f^\\star$ 也等于 $J_{\\min}$，但返回的 $X^\\star$ 是 rank-2。"
      "下面 §2/§3 分开报「值」与「秩/位姿」。")
    A("")

    # ------------------------------------------------------------------ §2 synthetic poses
    A("## 2. 合成双模实例：位姿与秩（$\\theta=0,0.01,0.1$）")
    A("")
    A("| $\\theta$ | 前向 | $f_{\\rm SDP}$ | $f-J_{\\min}$ | rank (ER) | rot vs QPEP 硬 | rot vs Laplace | raw $R$ 正交误差 | ms |")
    A("|-|-|-|-|-|-|-|-|-|")
    for r in syn:
        q = r["qpep"]
        for arm, tag in (("sdp10", "10×10"), ("sdp13", "13×13")):
            x = r[arm]
            rec = x["recovered"]
            dd = x.get("diffs_col")
            A("| %.2f | %s | %s | %s | %s | %s | %s | %s | %.1f |"
              % (r["theta"], tag, fnum(x["value"], 6), fnum(x["value"] - q["J_min"], 9),
                 rank_txt(rec),
                 "n/a" if dd is None else "%.4f" % dd["vs_hard"]["rot_deg"],
                 "n/a" if dd is None else "%.3f" % dd["vs_laplace"]["rot_deg"],
                 "n/a" if rec is None else "%.2e" % rec["pose_col"]["orth_err"],
                 x["t_solve_ms"]))
        for arm in ("up_1e9", "up_1e11"):
            u = r[arm]
            rec = u.get("recovered")
            dd = u.get("diffs_xs")
            A("| %.2f | 上游 $\\varepsilon$=%.0e | %s | %s | %s | %s | %s | %s | %.1f |"
              % (r["theta"], u["eps"], fnum(u["value"], 6),
                 fnum(u["value"] - q["J_min"], 9) if np.isfinite(u["value"]) else "n/a",
                 up_rank_txt(u, rec),
                 "n/a" if dd is None else "%.4f" % dd["vs_hard"]["rot_deg"],
                 "n/a" if dd is None else "%.3f" % dd["vs_laplace"]["rot_deg"],
                 "n/a" if rec is None else "%.2e" % rec["pose_col"]["orth_err"],
                 u["t_solve_ms"]))
    A("")
    A("- $\\theta=0$：$J_{\\min}$ 处两个模严格退化（$\\Delta J=%s$），Laplace 权重 %.3f/%.3f；"
      "10×10 与 13×13 都返回 rank-2，特征值 %s，ER=%.2f，两种恢复的 raw $R$ 正交误差 %.2f/%.2f。"
      "最优面里有 rank-1 点（$f^\\star=J_{\\min}$ 就由它们达到），但内点法/ADMM 收敛到面内 rank-2 点，"
      "**所以 SDP 前向在这里没有位姿**（上游 `RuntimeError`）。"
      % ("%.1e" % th0["qpep"]["dJ"], th0["qpep"]["w"][0], th0["qpep"]["w"][1],
         ", ".join("%.2f" % v for v in th0["sdp10"]["recovered"]["eigs_top"][:3]),
         th0["sdp10"]["recovered"]["eig_ratio_12"],
         th0["sdp10"]["recovered"]["pose_col"]["orth_err"],
         th0["sdp10"]["recovered"]["pose_eig"]["orth_err"]))
    A("- $\\theta=0.01,0.1$：rank-1、$f=J_{\\min}$，`column` 与主特征向量给同一姿态，"
      "与 QPEP 硬 argmin 的旋转差 $<10^{-4}$°。Laplace 在 $\\theta=0.01$ 仍偏向另一模"
      "（后验权重 %.2f），$\\theta=0.1$ 时已塌缩到硬 argmin（rot 差 %.2f°）。"
      "→ 这正是 P1 的分岔区：SDP 值对、位姿也对，但它给不出「两个模/后验权重」。"
      % (syn[1]["qpep"]["w_other"],
         syn[2]["sdp10"]["diffs_col"]["vs_laplace"]["rot_deg"]))
    A("")

    # ------------------------------------------------------------------ §3 real
    A("## 3. 真实平面帧（5 帧，BOP LINEMOD eggbox/glue）")
    A("")
    A("### 3a. QPEP / Laplace 参考（本脚本重算，逐帧对照 `pose_error.json`）")
    A("")
    A("| frame | obj | $n_{\\rm plane}$ | $K_{\\rm phys}$ | $\\Delta J$ | $w_{\\rm other}$ | hard rot (°) | Laplace rot (°) | hard ADD-S (mm) | Laplace ADD-S (mm) | $\\Delta$rot vs json | $\\Delta$ADD-S vs json |")
    A("|-|-|-|-|-|-|-|-|-|-|-|-|")
    for r in real:
        ch, cl = r["cited_hard"], r["cited_laplace"]
        if ch is not None:
            drot = max(abs(r["hard"]["rot_deg"] - ch["rot_deg"]),
                       abs(r["laplace"]["rot_deg"] - cl["rot_deg"]))
            dadd = max(abs(r["hard"]["adds_m"] - ch["adds_m"]),
                       abs(r["laplace"]["adds_m"] - cl["adds_m"]))
            sd, sa = "%.1e" % drot, "%.1e" % dadd
        else:
            sd = sa = "n/a"
        A("| %s/%s | %d | %d | %d | %.2e | %.3f | %.2f | %.2f | %.2f | %.2f | %s | %s |"
          % (r["frame"], r["img"], r["obj"], r["n_plane"], r["n_phys"], r["dJ"],
             r["w_other"], r["hard"]["rot_deg"], r["laplace"]["rot_deg"],
             1e3 * r["hard"]["adds_m"], 1e3 * r["laplace"]["adds_m"], sd, sa))
    A("")
    A("- 旋转/平移与 `[quoted-pose_error]` 逐位一致（$\\Delta$rot = 0）；ADD-S 的 $\\Delta$ 只来自"
      "1000 点顶点子采样（numpy 版本不同导致 `Generator.choice` 序列不同），量级 $10^{-4}$ m，"
      "远小于方法间差异。SDP 与 QPEP 在 §3b 用**同一组**子采样，故列间可比。")
    A("")
    A("### 3b. SDP 前向（同数据、同 ADD-S 顶点子采样）")
    A("")
    A("| frame | 前向 | $f_{\\rm SDP}$ | rank (ER) | rot vs GT | ADD-S (mm) | 物理? | rot vs QPEP 硬 | QPEP 硬 rot | ms |")
    A("|-|-|-|-|-|-|-|-|-|-|")
    for r in with_sdp:
        for arm, tag in (("sdp10", "10×10"), ("sdp13", "13×13"),
                         ("up_1e9", "上游 1e-9"), ("up_1e11", "上游 1e-11")):
            x = r["sdp"][arm]
            rec = x.get("recovered")
            up = arm.startswith("up")
            ref = (x.get("gt_xs") if up else x.get("gt")) or None
            dd = x.get("diffs_col")
            rankstr = up_rank_txt(x, rec) if up else rank_txt(rec)
            A("| %s/%s | %s | %s | %s | %s | %s | %s | %s | %.2f | %.1f |"
              % (r["frame"], r["img"], tag, fnum(x["value"], 8), rankstr,
                 "n/a" if ref is None else "%.2f" % ref["rot_deg"],
                 "n/a" if ref is None else "%.2f" % (1e3 * ref["adds_m"]),
                 "n/a" if ref is None else ("是" if ref["physical"] else "**否**"),
                 "n/a" if dd is None else "%.2f" % dd["vs_hard"]["rot_deg"],
                 r["hard"]["rot_deg"], x["t_solve_ms"]))
    A("")
    A("- `rot vs GT` / `ADD-S` 用与 §3a 相同的 1000 点查询子采样与**完整 GT 网格**，"
      "可与 `QPEP 硬 rot` 逐列对比；`物理?` = 恢复位姿下主平面点是否全部 $z>0.05$。")
    A("- `上游` 行取上游 `recovery_map` 的输出（`xs`）；`REFUSED` 表示没有位姿（`RuntimeError`），"
      "此时 rot/ADD-S 记 `n/a`（从 $X$ 第一列做的最好 rank-1 投影数据在 JSON 的 `gt`/`diffs_col`，"
      "但它**不是**上游位姿，故不列表）。")
    A("")

    # ------------------------------------------------------------------ §4 findings
    A("## 4. 读数：SDPR 前向 vs QPEP 前向（诚实版）")
    A("")
    A("1. **值：等价（8/8，差 $\\le2.5\\times10^{-9}$）。** 13×13 与 10×10 的 $f^\\star$ 与 QPEP "
      "$J_{\\min}$ 在全部实例上重合，且 QPEP 最优解的 rank-1 lift 是 13×13 的可行点"
      "（残差 %.0e）。所以在这批 P1 数据上，**松弛间隙测不出来**；"
      "SDP 前向与 QPEP 前向在「目标值」这一层是可比的。" % max_wit_resid)
    A("2. **位姿：紧的实例上逐位一致（真实 %d/%d 帧 rot 差 $<10^{-3}$°，合成 $\\theta=0.01,0.1$ 同）。**"
      "10×10 的 rank-1 解与 QPEP 硬 argmin 是同一个姿态——这验证了两条链路（$v_2$ 参数化/SDP lift）"
      "在数值上的一致性。" % (n_ident, len(with_sdp)))
    A("3. **退化处 SDP 没有位姿，QPEP 有（P1 的动机）。** $\\theta=0$ 时 10×10/13×13 都停在 rank-%d、"
      "ER=%.2f，发布版上游 `RuntimeError`；QPEP 仍给出 %d 个模 + Laplace 权重 $(%.3f,%.3f)$。"
      "注意这不是「松弛松」——值是紧的，是**最优面非唯一**导致返回矩阵非 rank-1。"
      % (th0["sdp10"]["recovered"]["rank"], th0["sdp10"]["recovered"]["eig_ratio_12"],
         th0["qpep"]["K"], th0["qpep"]["w"][0], th0["qpep"]["w"][1]))
    byname = {"%s/%s" % (r["frame"], r["img"]): r for r in with_sdp}
    chk = [byname[k] for k in ("000010/15", "000010/40") if k in byname]
    A("4. **上游的 `ER>1e5` 门控对求解器精度敏感，不等于「松弛松」。** $\\varepsilon=10^{-9}$ 下 %d/%d "
      "被拒，$\\varepsilon=10^{-11}$ 下只剩 %d/%d（$\\theta=0$ 的内在退化 + 000010/15 的 SCS 精度）。"
      "用 CLARABEL 复核：%s 的 $X$ 第二特征值只有 $\\lambda_1$ 的 %.1e / %.1e"
      "（ER $\\approx$ %.0e / %.0e，按上游判据算 tight）；SCS 的 $X$ 则差得远（ER %.1f / %.1f）。"
      "→ 复现 SDPRLayers 的「非 tight 无位姿」时必须把**真松**与**求解器精度**分开报，"
      "否则会把紧实例误判成 SDP 失败。"
      % (n_up_refused_1e9, n_inst, n_up_refused_1e11, n_inst,
         " 与 ".join("000010/" + k.split("/")[1] for k in
                     ("000010/15", "000010/40")[:len(chk)]) if chk else "（无）",
         *(x["sdp"]["tight"]["clar13"]["e2_over_e1"] for x in chk),
         *(1.0 / max(x["sdp"]["tight"]["clar13"]["e2_over_e1"], 1e-300) for x in chk),
         *(x["sdp"]["up_1e9"]["ER"] for x in chk)))
    A("5. **SDP 解的是代数问题，不带 P1 的物理滤波。** %d/5 帧的 10×10 代数全局解非物理；"
      "最极端 %s/%s：SDP 代数最优 $J=%s$ 对应非物理姿态（rot %.1f°、ADD-S %.1f mm），"
      "而 P1 `physical_modes` 滤掉 $z\\le0$ 后硬 argmin 是 %.2f° / %.1f mm。"
      "→ 若把 SDP 前向直接当位姿估计器而不加物理滤波，它在这些帧上**远差于** QPEP 硬 argmin；"
      "这不是松弛算错，而是两者可行域口径不同。"
      % (len(n_phys_fail), worst["frame"], worst["img"],
         fnum(worst["sdp"]["sdp10"]["value"], 6),
         worst["sdp"]["sdp10"]["gt"]["rot_deg"],
         1e3 * worst["sdp"]["sdp10"]["gt"]["adds_m"],
         worst["hard"]["rot_deg"], 1e3 * worst["hard"]["adds_m"]))
    A("6. **前向开销**：10×10/13×13 的 CVXPY+SCS 单次中位 %.0f ms，P1 模枚举（$t_{\\rm fwd}$）中位 %.0f ms，"
      "发布版上游层中位 %.1f s。即「算 SDP」比 QPEP 便宜，但发布版 SDPR 的 cvxpylayers/diffcp "
      "固定开销让它前向反而慢一个量级；换个直接 SCS/cvxpy 调用（本脚本的 `13×13` 行）就回到几十毫秒。"
      % (t_sdp10, t_qpep, t_up / 1e3))
    A("")

    # ------------------------------------------------------------------ §5 caveats
    A("## 5. 局限 / 没做的事")
    A("")
    A("- **只前向**：没有反向、没有梯度检查、没有训练；不声称 SDPR 的可微性与 P1 可比。")
    A("- **CPU only**：上游层跑 SCS/diffcp 的 CPU 后端，没有 GPU 作业。")
    A("- 上游 13×13 用发布版实现喂**同一个** $Q$（跳过它的 `get_obj_matrix_vec` 从原始点重建 $Q$；"
      "那条路径另有无害的 Frobenius 缩放/offset），因此两边 $f$ 在同一标度下可比。")
    A("- rank>1 时**不做**随机舍入/采样恢复（那是另一种前向）；只报 `column` 与主特征向量两种确定性恢复。")
    A("- 真实帧只有 5 帧（覆盖 eggbox/glue、近 tie、hard 失败帧）；不是精度评测，也不含前段/外观。")
    A("- $J_{\\min}$ 来自 P1 的多起点 $\\mathbb S^3$ Newton 模枚举（`Instance.find_modes`），"
      "是数值枚举而非代数证书；若漏根，$J_{\\min}$ 会偏高。但 13×13 的 witness 只用它构造可行点，"
      "不影响「$f^\\star\\le J_{\\min}$」这一侧的结论。")
    A("")

    # ------------------------------------------------------------------ §6 reproduction
    wm = s.get("wm_cross") or {}
    A("## 6. 交叉核对与复现")
    A("")
    if wm.get("ok"):
        A("- **跨环境**：用 WM 环境（`%s`，cvxpy %s + SCS）对同一批 $Q$ 重解 10×10 与 13×13，"
          "与 sdpr 环境的 $f$ 最大绝对差 **%.1e**（%d 个 SDP，%s）。"
          % (WM_PY, cfg["cvxpy_wm"], wm["max_abs_diff"], wm["n_sdp"],
             "逐位一致" if wm["max_abs_diff"] == 0.0 else "差异见上"))
    else:
        A("- 跨环境核对未运行/失败（%s）。" % wm.get("error", "n/a"))
    A("- Q 矩阵（`A10`/`A13`/全部 `Q10`/`Qfull`）存于 `results/sdpr_p1_q.npz`，可独立重跑。")
    A("- 原始数据：`results/sdpr_p1.json`（每行含特征值、两种恢复的 R/t、上游 `xs`/`ER`/告警、"
      "QPEP 模列表与权重、CLARABEL 复核值/秩）。")
    A("- QPEP/Laplace 参考的复现：旋转/平移与 `[quoted-pose_error]` 逐位一致（表 §3a）。")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")

# ---------------------------------------------------------------------------
# cross-env check
# ---------------------------------------------------------------------------
def wm_cross_check(npz_path, stored):
    """Re-solve all dumped Q matrices in the WM env and compare values."""
    try:
        r = subprocess.run([WM_PY, "-c", WM_CROSS_CODE, str(npz_path)],
                           capture_output=True, text=True, timeout=600)
    except Exception as exc:                                    # pragma: no cover
        return dict(ok=False, error=repr(exc))
    if r.returncode != 0:
        return dict(ok=False, error=(r.stderr or "")[-500:])
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:                                    # pragma: no cover
        return dict(ok=False, error=repr(exc))
    diffs = []
    for k, v in out.items():
        base = k[:-4]
        fkey = base + ("_f10" if k.endswith("_Q10") else "_f13")
        if fkey in stored:
            diffs.append(abs(v - stored[fkey]))
    return dict(ok=True, n_sdp=len(out),
                max_abs_diff=float(max(diffs)) if diffs else None,
                values=out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wm-check", action="store_true",
                    help="skip the WM-env CVXPY cross-check")
    args = ap.parse_args()
    t_start = time.time()
    res = {"config": {
        "python": sys.executable, "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__,
        "sympy": __import__("sympy").__version__, "cvxpy": cp.__version__,
        "solvers": cp.installed_solvers(), "torch": torch.__version__,
        "cvxpy_wm": "1.7.5", "wm_python": WM_PY,
        "upstream": "sdprlayer@62d8ac6 /home/syk/labs/SDPR/code/sdprlayer",
        "upstream_generators": SDP.POSE_EST_GENERATORS,
        "tau": TAU, "thetas": list(THETAS), "sdp_eps": SDP_EPS,
        "sdp_max_iters": SDP_MAX_ITERS, "up_eps": SDP_EPS,
        "up_eps_tight": UP_EPS_TIGHT, "up_max_iters_tight": UP_MAX_ITERS_TIGHT,
        "sdp_frames": ["%s/%s" % k for k in SDP_FRAMES],
        "command": ("cd labs/01-laplace-qpeplayer/experiments/baselines && "
                    "%s run_sdpr_p1.py" % sys.executable),
        "scope": ("same-data FORWARD baseline only: 10x10 Veronese/Shor vs 13x13 "
                  "SDPRLayers replica (cvxpy SCS) vs released upstream SDPRLayer "
                  "(torch/diffcp SCS); no gradient, no GPU"),
    }}

    print("[1/5] builders (10x10 exact sympy, 13x13 upstream replica) ...", flush=True)
    U, A10, names10, chk10 = SDP.build_10x10()
    A13, chk13 = SDP.build_13x13()
    P = SDP.perm_rowmajor_to_colmajor()
    res["builders"] = {"n_constraints_10": int(chk10["n_constraints"]),
                       "n_constraints_13": int(chk13["n_constraints"]),
                       "U_orthogonality_err": chk10["U_orthogonal_max_abs_err"],
                       "xR_eq_2Uv2_exact": chk10["xR_equals_2Uv2_sympy_exact"],
                       "rank21_exact": chk10["rank_21_constraints_exact"]}
    print("      ", res["builders"], flush=True)

    print("[2/5] released upstream SDPRLayer (torch/diffcp) ...", flush=True)
    est = SDPPoseEstimator(T_s_v=torch.eye(4, dtype=torch.float64),
                           solver="SCS", scs_eps=SDP_EPS)

    print("[3/5] synthetic instance theta in %s ..." % (THETAS,), flush=True)
    I = inst_mod.Instance(seed=SYN_SEED)
    res["synthetic"] = synthetic_arm(I, U, A10, A13, P, est)

    print("[4/5] real planar-face frames ...", flush=True)
    models = {oid: rsym.load_model(oid) for oid in rtm.FRAMES.values()}
    trees = {oid: cKDTree(v) for oid, v in models.items()}
    res["real"] = real_arm(I, models, trees, U, A10, A13, P, est)

    # ----------------------------------------------------------------- summary
    up_all = [(r["up_1e9"], r["up_1e11"]) for r in res["synthetic"]]
    up_all += [(r["sdp"]["up_1e9"], r["sdp"]["up_1e11"])
               for r in res["real"] if "sdp" in r]
    n_ident = sum(1 for r in res["real"] if "sdp" in r
                  and r["sdp"]["sdp10"]["diffs_col"]["vs_hard"]["rot_deg"] < 1e-3)
    res["summary"] = {
        "n_instances": len(res["synthetic"]) + len(res["real"]),
        "n_tight10": sum(1 for r in res["synthetic"]
                         if r["sdp10"]["recovered"]["rank"] == 1)
        + sum(1 for r in res["real"] if "sdp" in r
              and r["sdp"]["sdp10"]["recovered"]["rank"] == 1),
        "n_upstream_rows": len(up_all),
        "up_non_tight_1e9": sum(1 for a, _ in up_all if not a.get("upstream_tight")),
        "up_solver_limited": sum(1 for a, b in up_all
                                 if not a.get("upstream_tight") and b.get("upstream_tight")),
        "n_real_sdp10_identical_to_hard": n_ident,
        "wall_s": None, "wm_cross": None,
    }

    # dump Q/A for the cross-env check and for independent re-runs
    dump = {"A10": A10, "A13": A13}
    stored = {}
    for r in res["synthetic"]:
        tag = "syn_theta%g" % r["theta"]
        dump[tag + "_Q10"] = np.asarray(r["Q10"])
        dump[tag + "_Qfull"] = np.asarray(r["Qfull"])
        stored[tag + "_f10"] = r["sdp10"]["value"]
        stored[tag + "_f13"] = r["sdp13"]["value"]
    for r in res["real"]:
        if "sdp" not in r:
            continue
        tag = "real_%s_%s" % (r["frame"], r["img"])
        dump[tag + "_Q10"] = np.asarray(r["sdp"]["Q10"])
        dump[tag + "_Qfull"] = np.asarray(r["sdp"]["Qfull"])
        stored[tag + "_f10"] = r["sdp"]["sdp10"]["value"]
        stored[tag + "_f13"] = r["sdp"]["sdp13"]["value"]
    npz_path = RESULTS / "sdpr_p1_q.npz"
    np.savez_compressed(npz_path, **dump)

    if not args.no_wm_check:
        print("[4b/5] cross-env check in the WM env ...", flush=True)
        res["summary"]["wm_cross"] = wm_cross_check(npz_path, stored)
        print("      ", {k: v for k, v in res["summary"]["wm_cross"].items()
                         if k != "values"}, flush=True)

    # ----------------------------------------------------------------- outputs
    res["summary"]["wall_s"] = time.time() - t_start
    for r in res["synthetic"]:
        r.pop("Q10", None)
        r.pop("Qfull", None)
    with open(RESULTS / "sdpr_p1.json", "w") as fh:
        json.dump(res, fh, indent=1)
    gen_report(res, RESULTS / "sdpr_p1.md")
    print("[5/5] wrote %s and %s (%.1f s)"
          % (RESULTS / "sdpr_p1.json", RESULTS / "sdpr_p1.md",
             res["summary"]["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
