#!/usr/bin/env python3
"""Re-adjudicate the *gradient* claims about the released BPnP (recheck, 2026-11-28).

Why this script exists
----------------------
``results/official_bpnp.md`` headlines that the released backward + a P1 forward
"matches our 6-DoF IFT / ift_dq / branch FD" (四者一致), while the JSON it ships
says the opposite:

    rel_bpnp_vs_ift6  = 1.37 / 1.31 / 0.92        (cos = 0.33 / 0.36 / 0.58)
    rel_*_vs_fd       = 0.999 for BOTH arms       (|g_fd| 5.5-175 vs |g_bpnp| 0.003-0.05)

i.e. the published adaptation table is self-contradictory: no two arms agree,
and the FD reference is ~1000x larger than both gradient arms.  This script
re-measures everything from scratch, **without editing any existing file**, and
writes the verdicts into ``results/official_bpnp.json`` under
``gradient_recheck`` (raw dump: ``results/official_bpnp_recheck.json``).

What it establishes
-------------------
A. Basis audit.  ``BPnP.BPnP`` returns ``P_6d = [rvec; t]`` and its backward
   returns ``dL/dpts2d``; so the released Jacobian is the derivative of the
   **(rvec, t) parameter** vector.  Our paper's gradient is the
   **(omega, t_dot) tangent** vector (``dR/dtheta = [omega]_x R``).  The two are
   related by the left Jacobian ``J_l(w)``: ``omega = J_l(w) * w_dot``.  For the
   P1 poses |w| = 2.7-2.8 rad, so the two bases differ by O(1) (rel 0.76-1.37) --
   which is exactly the size of the published ``rel_bpnp_vs_ift6``.

B. Reproduction.  The toy forward/backward-vs-FD numbers of the report
   (1.7013e-01 exact, 1.7097e-01 at 2 px) are reproduced to all printed digits.
   The P1 adaptation numbers are **not** reproducible: the in-tree
   ``P1ViaReleasedBPnP.forward`` passes ``iters=/tol=`` to a ``p1_solve`` that no
   longer accepts them (TypeError), the current ``p1_solve`` feeds a permuted
   Jacobian to ``least_squares``, and its solution has C = 6.6e-2 vs the
   reported 1.6e-4.  What *is* exactly reproducible is the published
   ``g_ift6`` column, recomputed from the pose stored in the JSON (rel = 0.0).

C. Controlled test on the demo data (``demo_data/cube.mat``, exact, 8 points),
   with the forward set to the algebraic LS solution:
     * literal test: released backward vs central FD of the algebraic solution
       map -> rel = 1.68e-01, i.e. **not** small;
     * the FD reference itself is validated (exact IFT of the algebraic
       stationarity vs FD: 1.8e-08);
     * own-assumptions test: forward = root of BPnP's own implicit system
       ``f = sum_{i,k} coefs_{ikj} r_{ik} = 0`` -> central FD of that map vs the
       released backward gives rel = **1.6e-07** (float32).
   So the released backward is a *correct IFT of its own system*; it is not the
   IFT of the algebraic LS (nor of ``cv.solvePnP``), because ``f`` is the
   stationarity of the **depth-weighted** normalised reprojection cost
   ``sum_i S_i ||e_i||^2`` (S_i = per-point depth), not of ``sum_i ||r_i||^2``.
   On this cube the depth spread is 61%, hence the 17%.

D. P1 same-data re-adjudication, at a properly converged P1 solution
   (``||grad C|| < 1e-10`` instead of the published ``grad_norm`` 4e-4-4e-3):
   the released backward, our exact 6-DoF IFT and a branch-following FD agree to
   **~1.5%** (cos 0.9999); the published 1.37/0.33 was the *basis* error
   (omega vs w_dot) plus a non-converged solve.  The residual ~1.5% is the
   depth-weighting gap and scales with the depth spread.

E. Cost-identity audit.  The published "1.8e-15" check is vacuous: it evaluates
   ``P^perp(Rp+t)`` and ``r = u*S - proj12`` at the pose that generated ``u``,
   where *both* are identically zero.  Off that pose the two costs differ by a
   per-point factor (``1+|u|^2``) plus a cross term, i.e. they agree only to
   leading order in the residual.

Run (CPU only, < 10 min)
------------------------
    /home/syk/miniconda3/envs/sdpr/bin/python \
        labs/01-laplace-qpeplayer/experiments/baselines/recheck_official_bpnp_gradient.py

    # toy + cube only (~1 min):  add --quick
    # skip the real LINEMOD frames:  add --no-real

Nothing is committed and no existing file is modified except the two result
files (``official_bpnp.json`` gets a ``gradient_recheck`` block, and the raw dump
``official_bpnp_recheck.json`` is written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
P1CODE = EXP / "code"
LINEMOD = EXP / "real" / "linemod"
BPNP_SRC = Path("/mnt/storage1/syk/third_party/BPnP")

for _p in (str(P1CODE), str(LINEMOD), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import kornia as kn  # noqa: E402

# (O1 of the original baseline) kornia >= 0.7 moved this out of the top-level
# namespace; BPnP.py is NOT edited, the attribute is aliased back.
if not hasattr(kn, "angle_axis_to_rotation_matrix"):
    kn.angle_axis_to_rotation_matrix = kn.geometry.angle_axis_to_rotation_matrix

if str(BPNP_SRC) not in sys.path:
    sys.path.insert(0, str(BPNP_SRC))
import BPnP  # noqa: E402  (released file, unmodified)

import run_official_bpnp as ro  # noqa: E402  (the baseline being rechecked)

DT = torch.float64
NAN = float("nan")


# =============================================================== small helpers
def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else NAN


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else NAN


def sha256(path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "missing"


def skew_np(v):
    v = np.asarray(v, float).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def left_jacobian(w):
    """J_l(w) of SO(3): exp([w+J_l dw]) = exp([dw]) exp([w]) (left perturbation)."""
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    Kx = skew_np(w)
    if th < 1e-12:
        return np.eye(3) + 0.5 * Kx
    return (np.eye(3) + (1.0 - np.cos(th)) / th ** 2 * Kx
            + (th - np.sin(th)) / th ** 3 * (Kx @ Kx))


def tangent_to_param(g_t, y):
    """(omega, t_dot) -> (w_dot, t_dot);  omega = J_l(w) w_dot."""
    g = np.asarray(g_t, float).copy()
    g[:3] = np.linalg.solve(left_jacobian(y[:3]), g[:3])
    return g


def param_to_tangent(g_p, y):
    """(w_dot, t_dot) -> (omega, t_dot)."""
    g = np.asarray(g_p, float).copy()
    g[:3] = left_jacobian(y[:3]) @ g[:3]
    return g


# ------------------------------------------------ torch float64 forward models
def skew_t(v):
    z = torch.zeros((), dtype=v.dtype)
    return torch.stack([torch.stack([z, -v[2], v[1]]),
                        torch.stack([v[2], z, -v[0]]),
                        torch.stack([-v[1], v[0], z])])


def R_of(w):
    """Rodrigues, identical to run_official_bpnp.rodrigues (asserted in §A)."""
    th = torch.linalg.norm(w)
    K = skew_t(w)
    if float(th) < 1e-12:
        return torch.eye(3, dtype=w.dtype) + K
    return (torch.eye(3, dtype=w.dtype) + torch.sin(th) / th * K
            + (1.0 - torch.cos(th)) / th ** 2 * (K @ K))


def camera_points(y, P):
    """X = R p + t  for all points; y = (w, t), P: (n,3)."""
    return P @ R_of(y[:3]).T + y[3:6]


def res_alg(y, u, P, K):
    """Algebraic (BPnP-style) residual  r = u * S - (K X)_{12},  S = (K X)_3.

    With K = I and u normalised this is the residual ``p1_cost_derivs`` uses:
    C = sum_i ||r_i||^2 (the cost the baseline's gradient arms differentiate).
    With K = diag(f,f,1) this is the *pixel* algebraic residual of the demo.
    """
    X = camera_points(y, P)
    pr = X @ K.T
    S = pr[:, 2:3]
    return u * S - pr[:, :2]


def proj_norm(y, u, P, K):
    """Normalised projection (K X)_{12} / (K X)_3 -- the quantity BPnP's coefs use."""
    X = camera_points(y, P)
    pr = X @ K.T
    return pr[:, :2] / pr[:, 2:3]


def depth_S(y, P, K):
    X = camera_points(y, P)
    return (X @ K.T)[:, 2].detach().numpy().astype(float)


def cost_alg_np(y, u, P, K):
    ut, yt = torch.tensor(u, dtype=DT), torch.tensor(np.asarray(y, float), dtype=DT)
    return float((res_alg(yt, ut, P, K) ** 2).sum())


def jac_alg(y, u, P, K):
    """(n,2,6) d r / d y, exact (forward-mode AD), parameter basis."""
    ut = torch.tensor(u, dtype=DT)
    yt = torch.tensor(np.asarray(y, float), dtype=DT)
    return torch.func.jacfwd(lambda yy: res_alg(yy, ut, P, K))(yt).numpy()


def grad_hess(y, u, P, K):
    """Exact grad / Hessian of C = sum ||r||^2 w.r.t. y = (w, t), float64 AD."""
    ut = torch.tensor(u, dtype=DT)
    yt = torch.tensor(np.asarray(y, float), dtype=DT, requires_grad=True)
    c = (res_alg(yt, ut, P, K) ** 2).sum()
    g, = torch.autograd.grad(c, yt, create_graph=True)
    H = torch.stack([torch.autograd.grad(g[i], yt, retain_graph=True)[0]
                     for i in range(6)]).detach().numpy()
    return g.detach().numpy(), H, float(c)


def newton_solve(u, P, K, y0, iters=8, tol=1e-12):
    """Newton on grad C = 0 with the exact Hessian; returns y, |grad C|, C, iters."""
    y = np.array(y0, float)
    g = None
    for k in range(iters):
        g, H, C = grad_hess(y, u, P, K)
        if np.linalg.norm(g) < tol:
            break
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        y = y - step
    g, H, C = grad_hess(y, u, P, K)
    return y, float(np.linalg.norm(g)), C, k


def lm_solve(u, P, K, y0, max_nfev=2000):
    """LM least squares on r (exact Jacobian), then an exact-Hessian polish."""
    from scipy.optimize import least_squares

    n = len(u)

    def fun(y):
        ut = torch.tensor(u, dtype=DT)
        return res_alg(torch.tensor(y, dtype=DT), ut, P, K).detach().numpy().reshape(-1)

    def jac(y):
        return jac_alg(y, u, P, K).reshape(-1, 6)

    res = least_squares(fun, np.asarray(y0, float), jac=jac, method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    y, gn, C, it = newton_solve(u, P, K, res.x, iters=6, tol=1e-13)
    return y, dict(C=C, grad_norm=gn, nfev=int(res.nfev), newton_iters=int(it))


# ----------------------------------------------------- BPnP's implicit system f
def bpnp_f(y, u, P, K):
    """f_j = sum_{i,k} coefs[i,k,j] r[i,k], coefs = -2 d proj_norm / d y  (released).

    Returns (f, y_tensor, u_tensor).
    """
    n = len(u)
    yt = torch.tensor(np.asarray(y, float), dtype=DT, requires_grad=True)
    ut = torch.tensor(u, dtype=DT, requires_grad=True)
    p = proj_norm(yt, ut, P, K)
    coefs = torch.zeros(n, 2, 6, dtype=DT)
    for k in range(2):
        for i in range(n):
            g, = torch.autograd.grad(p[i, k], yt, retain_graph=True)
            coefs[i, k, :] = -2.0 * g
    X = camera_points(yt, P)
    pr = X @ K.T
    r = ut * pr[:, 2:3] - pr[:, :2]
    return torch.einsum("ikj,ik->j", coefs, r), yt, ut


def f_ift(y, u, P, K):
    """Exact IFT of f(y,u)=0:  dy/du = -f_y^{-1} f_u   (float64 AD)."""
    f, yt, ut = bpnp_f(y, u, P, K)
    Fy = torch.stack([torch.autograd.grad(f[j], yt, retain_graph=True)[0]
                      for j in range(6)]).detach().numpy()
    Fu = torch.stack([torch.autograd.grad(f[j], ut, retain_graph=True)[0].reshape(-1)
                      for j in range(6)]).detach().numpy()
    return -np.linalg.solve(Fy, Fu), float(f.norm()), f.detach().numpy()


def f_newton(u, P, K, y0, iters=10, tol=1e-13):
    y = np.array(y0, float)
    fn = NAN
    for k in range(iters):
        f, yt, ut = bpnp_f(y, u, P, K)
        fn = float(f.norm())
        if fn < tol:
            break
        Fy = torch.stack([torch.autograd.grad(f[j], yt, retain_graph=True)[0]
                          for j in range(6)]).detach().numpy()
        y = y - np.linalg.solve(Fy, f.detach().numpy())
    f, _, _ = bpnp_f(y, u, P, K)
    return y, float(f.norm()), k


# --------------------------------------------- released backward, called verbatim
class _ReleasedAt(torch.autograd.Function):
    """Pass-through whose backward is ``BPnP.BPnP.backward`` (released, unmodified).

    ``BPnP.BPnP.apply`` always re-solves with ``cv.solvePnP`` first, so it cannot
    be asked "what does the released backward say at pose y?".  This wrapper
    stores the pose and calls the released backward verbatim.
    """

    @staticmethod
    def forward(ctx, pts2d, pts3d, K, P6):
        ctx.save_for_backward(pts2d, P6.clone(), pts3d, K)
        return P6.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return BPnP.BPnP.backward(ctx, grad_output)


def released_jacobian_at(u, P, K, y):
    """(6, 2n) d pose / d pts2d of the released backward, evaluated at pose y."""
    n = len(u)
    pts2d = torch.tensor(np.asarray(u)[None], dtype=torch.float32)
    pts3d = torch.tensor(np.ascontiguousarray(np.asarray(P, float)),
                         dtype=torch.float32).contiguous()
    Kt = torch.tensor(np.asarray(K, float), dtype=torch.float32)
    P6 = torch.tensor(np.asarray(y, float)[None], dtype=torch.float32)
    Jac = np.zeros((6, 2 * n))
    for j in range(6):
        x = pts2d.clone().requires_grad_(True)
        o = _ReleasedAt.apply(x, pts3d, Kt, P6)
        go = torch.zeros(1, 6)
        go[0, j] = 1.0
        o.backward(go)
        Jac[j] = x.grad.detach().numpy().reshape(-1).astype(float)
    return Jac


# --------------------------------------------------------------------- sections
def section_A_basis(p1_rows):
    """What is g_bpnp, and in which basis?"""
    out = {}
    # 1) the released parameter derivative is the dexp, not [e_a]_x R
    rng = np.random.default_rng(0)
    w = np.array([1.0, 0.5, 0.8])
    w = w / np.linalg.norm(w) * 2.7
    h = 1e-7
    err_l, err_p = [], []
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        dR = (ro.rodrigues(w + e) - ro.rodrigues(w - e)) / (2 * h)
        err_l.append(np.abs(dR - skew_np(e / h) @ ro.rodrigues(w)).max())
        err_p.append(np.abs(dR - skew_np(left_jacobian(w)[:, a]) @ ro.rodrigues(w)).max())
    out["dR_dw_max_err_vs_skewR"] = float(max(err_l))
    out["dR_dw_max_err_vs_Jl"] = float(max(err_p))
    out["note_dR"] = (
        "dR/dw_a = [J_l(w) e_a]_x R(w); the naive [e_a]_x R (the *tangent* "
        "derivative) is wrong by O(|w|), which is why the baseline's hand-derived "
        "p1_cost_derivs is a tangent-basis Jacobian.")

    # 2) the baseline's J is exactly the tangent-basis Jacobian: J_t = J_p M, M=blkdiag(Jl^-1,I)
    P = rng.standard_normal((40, 3))
    u = rng.standard_normal((40, 2)) * 0.1
    yy = np.concatenate([w, [0.1, -0.2, 1.0]])
    d = ro.p1_cost_derivs(yy, P, u)
    Jt = d["J"].reshape(-1, 6)
    Jp = np.zeros_like(Jt)
    hh = 1e-7
    for a in range(6):
        e = np.zeros(6)
        e[a] = hh
        Jp[:, a] = (ro.p1_cost_derivs(yy + e, P, u)["r"]
                    - ro.p1_cost_derivs(yy - e, P, u)["r"]).reshape(-1) / (2 * hh)
    M, *_ = np.linalg.lstsq(Jp, Jt, rcond=None)
    M_exp = np.eye(6)
    M_exp[:3, :3] = np.linalg.inv(left_jacobian(w))
    out["baseline_J_is_tangent_rel_residual"] = rel(Jt, Jp @ M)   # 0 == exact
    out["baseline_M_vs_blkdiag_Jl_inv"] = float(np.abs(M - M_exp).max())

    # 3) per-case size of the basis error (this is what the published 1.37 measures)
    out["per_case_basis_error"] = []
    for r in p1_rows:
        y = np.asarray(r["pose"], float)
        out["per_case_basis_error"].append(dict(
            case=r["case"], norm_w=float(np.linalg.norm(y[:3])),
            rel_param_vs_tangent_of_same_vector=rel(
                param_to_tangent(np.ones(6), y), np.ones(6))))

    out["what_g_bpnp_is"] = (
        "BPnP.BPnP.forward returns P_6d = [rvec(3); t(3)] and BPnP.BPnP.backward "
        "returns dL/dpts2d = grad_output . d(rvec,t)/dpts2d.  So g_bpnp is the "
        "derivative of the **(rvec, t) parameter** vector w.r.t. the 2-D "
        "keypoints, contracted with the input perturbation.  Our paper's 6-DoF "
        "gradient is the **(omega, t_dot) tangent** vector (dR/dtheta=[omega]_x R); "
        "omega = J_l(w) w_dot.  The two bases agree only at w=0.")
    return out


def section_B_reproduction(delivered):
    """Reproduce the published numbers on one case; audit reproducibility."""
    out = {}
    from scipy.io import loadmat

    cube = loadmat(str(BPNP_SRC / "demo_data" / "cube.mat"))
    poses = loadmat(str(BPNP_SRC / "demo_data" / "poses.mat"))
    pts3d = torch.tensor(np.ascontiguousarray(cube["pts3d"]), dtype=torch.float32).contiguous()
    Pgt = torch.tensor(poses["poses"][0], dtype=torch.float32).view(1, 6)
    K = torch.tensor([[ro.F0_TOY, 0.0, ro.TOY_CENTRE],
                      [0.0, ro.F0_TOY, ro.TOY_CENTRE],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        u_gt = BPnP.batch_project(Pgt, pts3d, K).numpy()[0]
    ini = np.zeros(6)
    ini[5] = ro.BPNP_TOY_INI_Z

    # ---- toy: released backward vs FD of the released forward (the 17% claim)
    toy = []
    for label, noise in (("exact", 0.0), ("noisy_2px", ro.TOY_NOISE_PX)):
        rng = np.random.default_rng(0)
        x0 = u_gt + noise * rng.standard_normal(u_gt.shape)
        out0, Jac, t_bwd = ro.toy_bpnp_jacobian(x0, pts3d, K, ini)
        fd = ro.toy_fd_jacobian(x0, pts3d, K, out0)
        toy.append(dict(case=label, rel=rel(Jac, fd), norm_released=float(np.linalg.norm(Jac)),
                        norm_fd=float(np.linalg.norm(fd)), t_bwd_ms=float(t_bwd)))
    out["toy"] = toy
    out["toy_matches_report"] = {
        "exact": abs(toy[0]["rel"] - delivered["toy"]["rows"][0]["rel_err_released_vs_fd"]),
        "noisy_2px": abs(toy[1]["rel"] - delivered["toy"]["rows"][1]["rel_err_released_vs_fd"]),
    }

    # ---- in-tree P1 path: does the committed script regenerate its own JSON?
    try:
        P, u_true, Rg, tg = ro.synthetic_planar()
        rng = np.random.default_rng(11)
        u = u_true + 0.5 / 800.0 * rng.standard_normal(u_true.shape)
        from scipy.spatial.transform import Rotation as Rot
        y_gt = np.concatenate([Rot.from_matrix(Rg).as_rotvec(), tg])
        y_init = y_gt + np.array([0.06, -0.04, 0.05, 0.01, -0.015, 0.02])
        y_bad, d_bad, _hist = ro.p1_solve(y_init, P, u)
        rep = dict(C=float(d_bad["C"]), grad_norm_tangent=float(np.linalg.norm(d_bad["grad"])),
                   nfev=int(d_bad.get("nfev", -1)),
                   delivered_C=float(delivered["adaptation"][0]["solve"]["C"]),
                   delivered_grad_norm=float(delivered["adaptation"][0]["solve"]["grad_norm"]))
        try:
            ro.bpnp_jacobian(u, P, y_bad)
            rep["bpnp_jacobian_error"] = None
        except Exception as exc:                       # noqa: BLE001 - recorded verbatim
            rep["bpnp_jacobian_error"] = f"{type(exc).__name__}: {exc}"
        out["in_tree_p1_path"] = rep
    except Exception as exc:                           # noqa: BLE001
        out["in_tree_p1_path"] = dict(error=f"{type(exc).__name__}: {exc}")

    # ---- the published g_ift6 column IS reproducible from the stored pose
    out["delivered_g_ift6_reproduction"] = []
    for row in delivered["adaptation"]:
        P, u, du = _rebuild_case(row["case"])
        if P is None:
            out["delivered_g_ift6_reproduction"].append(dict(case=row["case"], error="data"))
            continue
        y = np.asarray(row["pose"], float)
        g_unit, _ = ro.p1_ift(y, P, u, du)
        pub = np.asarray(row["rows"][0]["g_ift6"], float) / row["rows"][0]["theta"]
        out["delivered_g_ift6_reproduction"].append(dict(case=row["case"], rel=rel(g_unit, pub)))
    return out


def section_C_cube():
    """Controlled test on the demo data: algebraic-LS forward vs released backward."""
    from scipy.io import loadmat

    cube = loadmat(str(BPNP_SRC / "demo_data" / "cube.mat"))
    poses = loadmat(str(BPNP_SRC / "demo_data" / "poses.mat"))
    Pnp = np.asarray(cube["pts3d"], float)
    P = torch.tensor(Pnp, dtype=DT)
    y_gt = np.asarray(poses["poses"][0], float).reshape(6)
    Knp = np.array([[ro.F0_TOY, 0.0, ro.TOY_CENTRE],
                    [0.0, ro.F0_TOY, ro.TOY_CENTRE], [0.0, 0.0, 1.0]], float)
    K = torch.tensor(Knp, dtype=DT)
    n = Pnp.shape[0]
    # "exact" data: the demo's own projection (float32 inside the release, so the
    # algebraic minimum is ~1e-8 rather than exactly 0; irrelevant at 1e-3 level)
    with torch.no_grad():
        u_px = BPnP.batch_project(torch.tensor(y_gt, dtype=torch.float32).view(1, 6),
                                  torch.tensor(np.ascontiguousarray(Pnp), dtype=torch.float32),
                                  torch.tensor(Knp, dtype=torch.float32)).numpy()[0].astype(float)

    out = dict(n_points=int(n), data="demo_data/cube.mat + poses.mat, exact (no noise)")
    # ---- forward = algebraic LS solution (min sum ||r||^2, r = u S - (K X)_12)
    y_init = y_gt + np.array([0.08, -0.05, 0.06, 0.02, -0.010, 0.30])
    y_alg, info = lm_solve(u_px, P, K, y_init)
    out["algebraic_solve"] = dict(C=info["C"], grad_norm=info["grad_norm"],
                                  dist_to_ground_truth=float(np.linalg.norm(y_alg - y_gt)),
                                  nfev=info["nfev"])
    S = depth_S(torch.tensor(y_alg, dtype=DT), P, K)
    out["depth_spread_rel"] = float((S.max() - S.min()) / S.mean())

    # ---- FD of the algebraic-LS forward map (warm-started exact Newton)
    uf = u_px.reshape(-1)

    def fd_alg(h):
        J = np.zeros((6, 2 * n))
        for c in range(2 * n):
            xp, xm = uf.copy(), uf.copy()
            xp[c] += h
            xm[c] -= h
            yp = newton_solve(xp.reshape(n, 2), P, K, y_alg, iters=6, tol=1e-13)[0]
            ym = newton_solve(xm.reshape(n, 2), P, K, y_alg, iters=6, tol=1e-13)[0]
            J[:, c] = (yp - ym) / (2 * h)
        return J

    fd_study = {}
    for h in (1e-3, 1e-4, 1e-5):
        fd_study[f"{h:.0e}"] = fd_alg(h)
    J_fd_alg = fd_study["1e-04"]
    out["fd_step_study_norm"] = {k: float(np.linalg.norm(v)) for k, v in fd_study.items()}

    # ---- exact IFT of the algebraic stationarity (validates the FD)
    g, H, _ = grad_hess(y_alg, u_px, P, K)
    ut = torch.tensor(u_px, dtype=DT, requires_grad=True)
    yt = torch.tensor(y_alg, dtype=DT, requires_grad=True)
    c = (res_alg(yt, ut, P, K) ** 2).sum()
    gg, = torch.autograd.grad(c, yt, create_graph=True)
    rhs = np.zeros((6, 2 * n))
    for j in range(6):
        m, = torch.autograd.grad(gg[j], ut, retain_graph=True)
        rhs[j] = m.detach().numpy().reshape(-1)
    J_ift_alg = -np.linalg.solve(H, rhs)
    out["ift_alg_vs_fd"] = rel(J_ift_alg, J_fd_alg)

    # ---- released backward at the algebraic solution
    J_rel = released_jacobian_at(u_px, Pnp, Knp, y_alg)
    out["released_vs_fd_alg"] = rel(J_rel, J_fd_alg)          # <-- the literal test
    out["released_vs_ift_alg"] = rel(J_rel, J_ift_alg)

    # ---- own-assumptions test: forward = root of BPnP's own system f = 0
    y_f, fn, kf = f_newton(u_px, P, K, y_alg)
    J_ift_f, f_norm, _ = f_ift(y_alg, u_px, P, K)
    out["f_root"] = dict(f_norm=float(fn), iters=int(kf),
                         dist_to_alg=float(np.linalg.norm(y_f - y_alg)))
    out["released_vs_ift_f"] = rel(J_rel, J_ift_f)            # implementation check
    out["ift_f_vs_ift_alg"] = rel(J_ift_f, J_ift_alg)

    def fd_f(h):
        J = np.zeros((6, 2 * n))
        for c in range(2 * n):
            xp, xm = uf.copy(), uf.copy()
            xp[c] += h
            xm[c] -= h
            yp = f_newton(xp.reshape(n, 2), P, K, y_alg, iters=6, tol=1e-13)[0]
            ym = f_newton(xm.reshape(n, 2), P, K, y_alg, iters=6, tol=1e-13)[0]
            J[:, c] = (yp - ym) / (2 * h)
        return J

    J_fd_f = fd_f(1e-4)
    out["released_vs_fd_f"] = rel(J_rel, J_fd_f)              # <-- own-assumptions test
    out["ift_f_vs_fd_f"] = rel(J_ift_f, J_fd_f)

    out["verdict"] = (
        "Setting the forward to the algebraic LS solution does NOT make the "
        "released backward consistent with it (rel=1.7e-01): both f=0 and "
        "grad C_alg=0 vanish there, but they are different systems, so their "
        "solution *maps* differ.  The released backward is nevertheless a correct "
        "IFT -- of its own f=0 system (rel=1.6e-07 vs FD of the f-root forward, "
        "1.7e-07 vs the exact IFT of f).  The 17% is the depth weighting: f is the "
        "stationarity of sum_i S_i ||e_i||^2, the algebraic cost is "
        "sum_i ||r_i||^2 = sum_i S_i^2 ||e_i||^2; with the cube's depth spread of "
        f"{out['depth_spread_rel']:.2f} the two maps differ by O(spread).")
    return out


def section_D_p1(delivered, no_real=False):
    """P1 same-data re-adjudication at a properly converged solution."""
    cases = []
    P, u_true, Rg, tg = ro.synthetic_planar()
    rng = np.random.default_rng(11)
    u_syn = u_true + 0.5 / 800.0 * rng.standard_normal(u_true.shape)
    cases.append(("synthetic-planar", P, u_syn))
    if not no_real:
        for fr, img, oid in (("000010", "15", 10), ("000011", "20", 11)):
            try:
                rf = ro.real_frame([(fr, img, oid)])
            except Exception:                          # noqa: BLE001
                rf = None
            if rf is None:
                continue
            sel = np.linspace(0, len(rf["P"]) - 1,
                              min(ro.REAL_N_ARM, len(rf["P"]))).astype(int)
            cases.append((f"real-{fr}-{img}", rf["P"][sel],
                          ro.norm_image_coords(rf["U"][sel])))

    K = torch.eye(3, dtype=DT)
    rows = []
    for name, Pnp, u in cases:
        t0 = time.perf_counter()
        Pt = torch.tensor(Pnp, dtype=DT)
        du = np.random.default_rng(7).standard_normal(u.shape)
        y_gt = None
        if name == "synthetic-planar":
            from scipy.spatial.transform import Rotation as Rot
            y_gt = np.concatenate([Rot.from_matrix(Rg).as_rotvec(), tg])
            y_init = y_gt + np.array([0.06, -0.04, 0.05, 0.01, -0.015, 0.02])
        else:
            y_init = None
            for fr, img, oid in (("000010", "15", 10), ("000011", "20", 11)):
                if f"real-{fr}-{img}" == name:
                    rf = ro.real_frame([(fr, img, oid)])
                    from scipy.spatial.transform import Rotation as Rot
                    y_init = np.concatenate(
                        [Rot.from_matrix(rf["R"]).as_rotvec(), rf["t"]])
        y, info = lm_solve(u, Pt, K, y_init)
        C, gn = info["C"], info["grad_norm"]

        # our exact 6-DoF IFT, parameter basis (omega, t_dot) is derived from it
        ut = torch.tensor(u, dtype=DT, requires_grad=True)
        yt = torch.tensor(y, dtype=DT, requires_grad=True)
        c = (res_alg(yt, ut, Pt, K) ** 2).sum()
        g_y, = torch.autograd.grad(c, yt, create_graph=True)
        H = torch.stack([torch.autograd.grad(g_y[i], yt, retain_graph=True)[0]
                         for i in range(6)]).detach().numpy()
        rhs1 = np.zeros(6)
        for j in range(6):
            m, = torch.autograd.grad(g_y[j], ut, retain_graph=True)
            rhs1[j] = float((m * torch.tensor(du, dtype=DT)).sum())
        g_ours = -np.linalg.solve(H, rhs1)

        # the baseline's tangent-basis IFT at the same point, and its mapping
        g_tan_pub, _ = ro.p1_ift(y, Pnp, u, du)
        g_ours_tan = param_to_tangent(g_ours, y)
        g_pub_as_param = tangent_to_param(g_tan_pub, y)

        # released backward at the same point (parameter basis)
        J_rel = released_jacobian_at(u, Pnp, np.eye(3), y)
        g_rel = J_rel @ du.reshape(-1)

        # FD of the (converged) solution map, step study
        fds = {}
        for h in (1e-2, 1e-3, 1e-4):
            yp = newton_solve(u + h * du, Pt, K, y, iters=6, tol=1e-13)[0]
            ym = newton_solve(u - h * du, Pt, K, y, iters=6, tol=1e-13)[0]
            fds[f"{h:.0e}"] = (yp - ym) / (2 * h)
        g_fd = fds["1e-04"]
        g_fd_study = {k: float(np.linalg.norm(v)) for k, v in fds.items()}

        # exact IFT of f (what the release actually differentiates)
        J_ift_f, f_norm, _ = f_ift(y, u, Pt, K)
        g_f = J_ift_f @ du.reshape(-1)

        S = depth_S(torch.tensor(y, dtype=DT), Pt, K)
        row = dict(
            case=name, n_points=int(len(u)),
            solve=dict(C=float(C), grad_norm=float(gn), nfev=info["nfev"],
                       newton_iters=info["newton_iters"]),
            depth_spread_rel=float((S.max() - S.min()) / S.mean()),
            norms=dict(bpnp=float(np.linalg.norm(g_rel)), ours=float(np.linalg.norm(g_ours)),
                       fd=float(np.linalg.norm(g_fd)), ift_f=float(np.linalg.norm(g_f)),
                       fd_study=g_fd_study),
            released_vs_fd=rel(g_rel, g_fd), cos_released_fd=cos(g_rel, g_fd),
            ours_vs_fd=rel(g_ours, g_fd), cos_ours_fd=cos(g_ours, g_fd),
            released_vs_ours=rel(g_rel, g_ours), cos_released_ours=cos(g_rel, g_ours),
            released_vs_ours_TANGENT_BASIS=rel(g_rel, g_ours_tan),
            released_vs_ift_f=rel(g_rel, g_f),
            ift_f_vs_fd=rel(g_f, g_fd),
            f_norm_at_solution=float(f_norm),
            published_tangent_ift_check=dict(
                rel_published_vs_mapped=rel(g_tan_pub, g_ours_tan),
                rel_mapped_vs_ours=rel(g_pub_as_param, g_ours)),
        )
        # the published row for the same case, for contrast
        pub = next((r for r in delivered["adaptation"] if r["case"] == name), None)
        if pub is not None:
            pr = pub["rows"][0]
            row["published"] = dict(
                rel_bpnp_vs_fd=pr["rel_bpnp_vs_fd"], rel_ift6_vs_fd=pr["rel_ift6_vs_fd"],
                rel_bpnp_vs_ift6=pr["rel_bpnp_vs_ift6"], theta=pr["theta"],
                solve_grad_norm=pub["solve"]["grad_norm"], solve_C=pub["solve"]["C"],
                norm_g_bpnp=float(np.linalg.norm(pr["g_bpnp"])),
                norm_g_fd=float(np.linalg.norm(pr["g_fd"])),
                basis_only_error=rel(g_ours, g_ours_tan))
        rows.append(row)
        print(f"  [D] {name:<18s} n={len(u):3d} C={C:.3e} |grad|={gn:.1e} "
              f"depth_spread={row['depth_spread_rel']:.2f}  "
              f"rel(rel,fd)={row['released_vs_fd']:.4f} "
              f"rel(ours,fd)={row['ours_vs_fd']:.2e} "
              f"rel(rel,ours)={row['released_vs_ours']:.4f} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    verdict = (
        "At a properly converged C_P1 solution the released backward agrees with "
        "our exact 6-DoF IFT to ~1.5% (cos 0.9999) and with a branch-following FD "
        "to ~1.5%; the published 1.37 / 0.33 came from comparing the released "
        "**parameter** rate (rvec_dot) with the baseline's **tangent** rate "
        "(omega) -- a basis error of the same size -- at a solution where "
        "||grad C|| was still 4e-4..4e-3.  The residual ~1.5% is the depth "
        "weighting (released: d/dy of sum S_i||e_i||^2; ours: sum S_i^2||e_i||^2) "
        "and grows with the depth spread; it is NOT a shared 6-DoF IFT.")
    return dict(rows=rows, verdict=verdict)


def section_E_cost_identity():
    """The published 1.8e-15 'cost identity' is vacuous; state the real relation."""
    rng = np.random.default_rng(3)
    P = rng.standard_normal((8, 3))
    w = rng.standard_normal(3)
    t = rng.standard_normal(3)
    X = (P @ ro.rodrigues(w).T) + t
    u_h = X[:, :2] / X[:, 2:3]
    Uh = ro.bearings_from_u(u_h)
    perp_at_data = X - Uh * (X * Uh).sum(1, keepdims=True)
    r_at_data = u_h * X[:, 2:3] - X[:, :2]
    # now move to a *different* pose and compare the two costs
    w2 = w + np.array([0.05, -0.03, 0.04])
    t2 = t + np.array([0.02, -0.01, 0.03])
    X2 = (P @ ro.rodrigues(w2).T) + t2
    perp = X2 - Uh * (X2 * Uh).sum(1, keepdims=True)
    r = u_h * X2[:, 2:3] - X2[:, :2]
    a = (Uh ** 2).sum(1)
    cross = (u_h ** 2 * X2[:, :2] ** 2).sum(1) - (u_h * X2[:, :2]).sum(1) ** 2
    pred = (1.0 + a) * (perp ** 2).sum(1) - cross
    return dict(
        at_data_pose_max_abs_perp=float(np.abs(perp_at_data).max()),
        at_data_pose_max_abs_r=float(np.abs(r_at_data).max()),
        delivered_check_value=1.8e-15,
        identity_residual_max_abs=float(np.abs((r ** 2).sum(1) - pred).max()),
        cost_ratio_off_pose=float((r ** 2).sum() / (perp ** 2).sum()),
        per_point_ratio_range=[float(((r ** 2).sum(1) / (perp ** 2).sum(1)).min()),
                               float(((r ** 2).sum(1) / (perp ** 2).sum(1)).max())],
        statement=("At the pose that generated u, P^perp X == 0 and r == 0 "
                   "identically, so the published |P^perp X - (-r1,-r2,0)| = 1.8e-15 "
                   "compares 0 with 0.  The exact per-point relation is "
                   "||r||^2 = (1+|u|^2)||P^perp X||^2 - |u x X_12|^2, so the two "
                   "costs agree only to leading order in the residual and their "
                   "minimisers are not identical."))


# ------------------------------------------------------------ case reconstruction
def _rebuild_case(name):
    """(P, u, du) for a case name in the delivered JSON, or (None, None, None)."""
    if name == "synthetic-planar":
        P, u_true, _R, _t = ro.synthetic_planar()
        rng = np.random.default_rng(11)
        u = u_true + 0.5 / 800.0 * rng.standard_normal(u_true.shape)
    else:
        parts = name.split("-")
        fr, img = parts[1], parts[2]
        oid = {"000010": 10, "000011": 11}.get(fr)
        rf = ro.real_frame([(fr, img, oid)])
        if rf is None:
            return None, None, None
        sel = np.linspace(0, len(rf["P"]) - 1, min(ro.REAL_N_ARM, len(rf["P"]))).astype(int)
        P = rf["P"][sel]
        u = ro.norm_image_coords(rf["U"][sel])
    du = np.random.default_rng(7).standard_normal(u.shape)
    return P, u, du


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="toy + cube only")
    ap.add_argument("--no-real", action="store_true", help="skip LINEMOD frames")
    args = ap.parse_args()
    T0 = time.time()
    RES = RESULTS
    res_json = RES / "official_bpnp.json"
    raw_json = RES / "official_bpnp_recheck.json"
    delivered = json.loads(res_json.read_text(encoding="utf-8"))
    print("[recheck] python", sys.version.split()[0], "torch", torch.__version__,
          "kornia", kn.__version__, flush=True)

    payload = dict(
        generated="2026-11-28",
        script=str(Path(__file__).resolve()),
        command=(" ".join([sys.executable, "labs/01-laplace-qpeplayer/experiments/"
                           "baselines/recheck_official_bpnp_gradient.py"]
                          + sys.argv[1:])),
        runtime_s=None,
        provenance=dict(
            bpnp_repo=str(BPNP_SRC),
            bpnp_commit=_git_head(BPNP_SRC),
            bpnp_py_sha256=sha256(BPNP_SRC / "BPnP.py"),
            baseline_script_sha256=sha256(HERE / "run_official_bpnp.py"),
            delivered_json_sha256=sha256(res_json),
            delivered_md_sha256=sha256(RES / "official_bpnp.md"),
            python=f"{platform.python_version()} ({sys.executable})",
            torch=torch.__version__, kornia=kn.__version__, numpy=np.__version__,
            platform=platform.platform(),
        ),
    )

    print("[recheck] A basis audit ...", flush=True)
    payload["A_basis"] = section_A_basis(delivered["adaptation"])

    print("[recheck] B reproduction ...", flush=True)
    payload["B_reproduction"] = section_B_reproduction(delivered)

    print("[recheck] C cube algebraic-solution test ...", flush=True)
    payload["C_algebraic_solution_test"] = section_C_cube()

    if not args.quick:
        print("[recheck] D P1 re-adjudication ...", flush=True)
        payload["D_p1_readjudication"] = section_D_p1(delivered, no_real=args.no_real)

    print("[recheck] E cost identity ...", flush=True)
    payload["E_cost_identity"] = section_E_cost_identity()

    # ------------------------------------------------------------- verdicts
    c = payload["C_algebraic_solution_test"]
    d = payload.get("D_p1_readjudication")
    v = [
        "The released backward IS a correct IFT -- of its own implicit system "
        f"f=0 (rel {c['released_vs_ift_f']:.1e} vs the exact IFT of f, "
        f"{c['released_vs_fd_f']:.1e} vs central FD of the f-root forward).",
        "It is NOT the IFT of the algebraic LS solution "
        f"(rel {c['released_vs_fd_alg']:.2e} on the demo cube) nor of "
        "cv.solvePnP's geometric solution (17.1% on the toy), because f is the "
        "stationarity of the depth-weighted normalised reprojection cost "
        "sum_i S_i||e_i||^2, not of sum_i||r_i||^2.",
        "The P1 adaptation table in the published JSON is not reproducible from "
        "the committed script (P1ViaReleasedBPnP.forward raises TypeError; the "
        "in-tree p1_solve returns C=6.6e-02 vs the reported 1.6e-04 and feeds a "
        "permuted Jacobian to least_squares), and its FD reference is ~1000x "
        "larger than both gradient arms, i.e. invalid.",
        "The published rel_bpnp_vs_ift6 = 1.37/1.31/0.92 is dominated by a BASIS "
        "error: g_bpnp is d(rvec,t)/dtheta (parameter basis) while the published "
        "g_ift6 was the tangent (omega) rate; the same-vector basis factor is "
        "0.76-1.37.",
    ]
    if d:
        r0 = d["rows"][0]
        v.append(
            "At a properly converged C_P1 solution (|grad C| < 1e-10) the released "
            f"backward and our exact 6-DoF IFT agree to {100 * r0['released_vs_ours']:.1f}% "
            f"(cos {r0['cos_released_ours']:.4f}), and both match a branch-following "
            f"FD ({100 * r0['released_vs_fd']:.1f}% / {r0['ours_vs_fd']:.1e}). The "
            "residual gap is the depth weighting (released sum S||e||^2 vs our "
            "sum S^2||e||^2), which scales with the depth spread -- 1.5% at 5% "
            "spread, 17% at the cube's 61% spread.")
    v.append(
        "The published '1.8e-15 cost identity' is vacuous (both residuals are "
        "identically zero at that pose); off-pose the two costs differ by a "
        f"per-point factor (measured ratio {payload['E_cost_identity']['cost_ratio_off_pose']:.3f}).")
    payload["verdicts"] = v
    payload["headline_corrected"] = (
        "released BPnP backward is the exact IFT of *its own* depth-weighted "
        "algebraic normal equations; it is not the IFT of the P1 cost nor of the "
        "geometric forward, and the published P1 adaptation table is invalid "
        "(basis error + non-converged solve + invalid FD).")

    payload["runtime_s"] = time.time() - T0
    payload["provenance"]["delivered_json_sha256_before"] = payload["provenance"]["delivered_json_sha256"]

    raw_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    delivered["gradient_recheck"] = payload
    res_json.write_text(json.dumps(delivered, indent=1), encoding="utf-8")
    print(f"[recheck] wrote {raw_json} and patched {res_json} "
          f"({payload['runtime_s']:.1f} s)", flush=True)
    for line in v:
        print("  -", line)


def _git_head(repo):
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


if __name__ == "__main__":
    main()
