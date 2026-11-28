#!/usr/bin/env python3
"""Anisotropic-weight (W_i) ablation for 3D-3D robust registration.

What this script measures
-------------------------
For per-observation SPD weights ``W_i`` the correct (GM-robust) objective is

    Phi(R,t) = sum_i rho( e_i ),      e_i = r_i^T W_i r_i,
    r_i      = R x_i + t - y_i,      rho(e) = c^2 e / (c^2 + e),

with the *same* kernel and GNC schedule as ``synthetic/common.py``
(c = 0.05, gamma = 1.4, mu0 = max(3 max_i e_i / c^2, 1), 5 mu=1 polish
steps).  At fixed scalar robust weights ``u_i = rho'(e_i)`` the inner
problem is

    min_{R,t} sum_i u_i r_i^T W_i r_i,

which is a *nonlinear* weighted least-squares problem in (R,t) because
``W_i`` is inside the residual.  It cannot be reduced to scalar weighted
Kabsch unless every ``W_i`` is a scalar multiple of the identity, so
``common.kabsch`` is *not* an exact inner solver for anisotropic W_i.
We therefore solve the inner problem with Gauss-Newton on the 6-DoF pose
using the Cholesky factor ``L_i`` of ``W_i`` and residuals

    z_i = sqrt(u_i) L_i^T (R x_i + t - y_i),   ||z_i||^2 = u_i e_i,

and cross-check the solver against ``scipy.optimize.least_squares``.

A scalar-Kabsch inner that merely reuses the (whitened) GM weights is
included ONLY as a clearly labelled approximation ``APPROX-sK`` to
quantify how far the scalar-Kabsch surrogate is from the correct
weighted objective.  It is not presented as a solution of min sum r_i^T W_i r_i.

Weight families (fixed per instance, independent of the data RNG)
----------------------------------------------------------------
* ``identity`` : W_i = I_3
* ``diag``     : W_i = diag(d1,d2,d3), d_k ~ U(0.5, 2)   (SPD)
* ``full``     : W_i = (A_i A_i^T + 0.05 I) / det(...)^(1/3), A_i ~ N(0,I)
                 (random Wishart + ridge, normalised to det = 1)

Outputs (this directory)
------------------------
results/anisotropic.json         aggregated table + correctness + paired stats
results/anisotropic_raw.jsonl    one line per (kind, p, trial, solver)
results/anisotropic.md           human-readable report (numbers from this run)
results/anisotropic.png          success / rotation error / runtime figure

Run (Slurm, from this directory):
  srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_anisotropic.py 2>&1 | \
    tee results/logs/anisotropic_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import os

# The linear algebra here is tiny (3x3, 300x6); multi-threaded BLAS only adds
# scheduling noise on the shared node.  Pin to one thread before numpy loads so
# the reported p50/p95 are reproducible (accuracy is unaffected).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent          # .../experiments
sys.path.insert(0, str(EXPERIMENTS / "synthetic"))
import common as cm  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)

BASE_SEED = 20261127
TRIALS = 25
P_LIST = [0.0, 0.3, 0.5, 0.7]
WEIGHT_KINDS = ["identity", "diag", "full"]
KIND_ID = {"identity": 0, "diag": 1, "full": 2}
SOLVERS = ["GNC-W", "IRLS-W", "APPROX-GNC-sK", "APPROX-IRLS-sK"]
APPROX = {"APPROX-GNC-sK", "APPROX-IRLS-sK"}
GN_MAX_ITER = 60
GN_TOL = 1e-10


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------
def skew(v):
    v = np.asarray(v, float)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def skew_batch(x):
    """(n,3) -> (n,3,3) cross-product matrices [x_i]_x."""
    n = len(x)
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -x[:, 2]
    S[:, 0, 2] = x[:, 1]
    S[:, 1, 0] = x[:, 2]
    S[:, 1, 2] = -x[:, 0]
    S[:, 2, 0] = -x[:, 1]
    S[:, 2, 1] = x[:, 0]
    return S


def so3_exp(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


# ---------------------------------------------------------------------------
# Anisotropic weights and whitening
# ---------------------------------------------------------------------------
def make_weights(rng, kind, n):
    """Return (n,3,3) SPD weights and their whitening factors S = L^T.

    ``S`` satisfies ||S_i r||^2 = r^T W_i r (L_i = cholesky(W_i), W = L L^T).
    """
    if kind == "identity":
        W = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    elif kind == "diag":
        d = rng.uniform(0.5, 2.0, size=(n, 3))       # eigenvalues in [0.5, 2]
        W = np.zeros((n, 3, 3))
        idx = np.arange(3)
        W[:, idx, idx] = d
    elif kind == "full":
        A = rng.normal(size=(n, 3, 3))
        M = A @ A.transpose(0, 2, 1) + 0.05 * np.eye(3)
        det = np.linalg.det(M)
        W = M / det[:, None, None] ** (1.0 / 3.0)    # det(W_i) = 1
    else:
        raise ValueError(kind)
    L = np.linalg.cholesky(W)
    S = L.transpose(0, 2, 1)                          # L^T
    return W, S


def whitened_sq(x, y, R, t, S):
    """e_i = ||S_i (R x_i + t - y_i)||^2 = r_i^T W_i r_i."""
    r = (R @ x.T).T + t - y
    z = np.einsum("nij,nj->ni", S, r)
    return np.einsum("ni,ni->n", z, z)


# ---------------------------------------------------------------------------
# Exact nonlinear inner solve: 6-DoF Gauss-Newton on the weighted problem
# ---------------------------------------------------------------------------
def gn_wls(x, y, S, u, R, t, XSKEW=None, max_iter=GN_MAX_ITER, tol=GN_TOL,
           return_info=False):
    """min_{R,t} sum_i u_i ||S_i(R x_i + t - y_i)||^2 by 6-DoF Gauss-Newton.

    Rotation is updated as R <- R exp([delta]) with an analytic Jacobian
    d(R exp([delta]) x_i)/d delta = -R [x_i]_x, and the linear system is
    re-anchored after every step (so this is full Gauss-Newton on the pose,
    not a single-chart linearisation).
    """
    n = len(x)
    if XSKEW is None:
        XSKEW = skew_batch(x)
    su = np.sqrt(np.asarray(u, float))
    R = np.array(R, float)
    t = np.array(t, float)
    SR = np.einsum("nij,jk->nik", S, R)              # S_i R
    gnorm = np.inf
    cost = np.inf
    it = 0
    for it in range(max_iter):
        r = (R @ x.T).T + t - y
        z = su[:, None] * np.einsum("nij,nj->ni", S, r)
        cost = float(np.einsum("ni,ni->", z, z))
        Jw = -su[:, None, None] * np.einsum("nij,njk->nik", SR, XSKEW)
        Jt = su[:, None, None] * S
        J = np.concatenate([Jw, Jt], axis=2).reshape(3 * n, 6)
        zf = z.reshape(3 * n)
        H = J.T @ J
        g = J.T @ zf
        gnorm = float(np.linalg.norm(g))
        if gnorm <= tol * max(1.0, cost):
            break
        try:
            delta = -np.linalg.solve(H + 1e-12 * np.eye(6), g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(H, g, rcond=None)[0]
        R = R @ so3_exp(delta[:3])
        t = t + delta[3:]
        # update S_i R for the new rotation
        SR = np.einsum("nij,jk->nik", S, R)
        if np.linalg.norm(delta) < tol:
            it += 1
            break
    # final cost / gradient at the returned pose
    r = (R @ x.T).T + t - y
    z = su[:, None] * np.einsum("nij,nj->ni", S, r)
    cost = float(np.einsum("ni,ni->", z, z))
    info = dict(gn_iters=it + 1, gn_cost=cost, gn_grad=gnorm)
    return (R, t, info) if return_info else (R, t)


# ---------------------------------------------------------------------------
# GNC-GM / IRLS with the exact anisotropic inner solve
# ---------------------------------------------------------------------------
def _gm_weights(e, mu, c):
    mc2 = mu * c * c
    return (mc2 / (mc2 + e)) ** 2


def solve_gnc_w(x, y, S, c=cm.C, gamma=cm.GNC_GAMMA,
                max_mu_iters=cm.GNC_MAX_MU_ITERS, polish=5, XSKEW=None):
    """GNC-GM with the exact nonlinear anisotropic inner solve.

    Identical schedule to common.solve_gnc, but e_i is the whitened
    squared residual and the inner problem is solved by gn_wls.
    """
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    e = whitened_sq(x, y, R, t, S)
    mu = max(3.0 * float(e.max()) / (c * c), 1.0)
    mu0 = mu
    n_solves = 0
    n_gn = 0
    n_mu = 0
    for _ in range(max_mu_iters):
        u = _gm_weights(e, mu, c)
        R, t, info = gn_wls(x, y, S, u, R, t, XSKEW=XSKEW, return_info=True)
        n_solves += 1
        n_gn += info["gn_iters"]
        n_mu += 1
        e = whitened_sq(x, y, R, t, S)
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
    for _ in range(polish):
        u = _gm_weights(e, 1.0, c)
        R, t, info = gn_wls(x, y, S, u, R, t, XSKEW=XSKEW, return_info=True)
        n_solves += 1
        n_gn += info["gn_iters"]
        e = whitened_sq(x, y, R, t, S)
    phi = float(np.sum(cm.C2 * e / (cm.C2 + e)))
    return R, t, dict(n_outer=n_mu, n_solves=n_solves, gn_iters_total=n_gn,
                      mu0=float(mu0), phi=phi, e=e)


def solve_irls_w(x, y, S, c=cm.C, iters=40, XSKEW=None):
    """IRLS-GM (no annealing) with the exact nonlinear anisotropic inner solve."""
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    n_gn = 0
    for _ in range(iters):
        e = whitened_sq(x, y, R, t, S)
        u = _gm_weights(e, 1.0, c)
        R, t, info = gn_wls(x, y, S, u, R, t, XSKEW=XSKEW, return_info=True)
        n_gn += info["gn_iters"]
    e = whitened_sq(x, y, R, t, S)
    phi = float(np.sum(cm.C2 * e / (cm.C2 + e)))
    return R, t, dict(n_outer=iters, n_solves=iters, gn_iters_total=n_gn,
                      mu0=1.0, phi=phi, e=e)


# ---------------------------------------------------------------------------
# Scalar-Kabsch surrogate (explicitly approximate)
# ---------------------------------------------------------------------------
def solve_gnc_scalar(x, y, S, c=cm.C, gamma=cm.GNC_GAMMA,
                     max_mu_iters=cm.GNC_MAX_MU_ITERS, polish=5):
    """GNC-GM whose inner solve is scalar weighted Kabsch.

    APPROXIMATION: this minimises sum_i u_i ||r_i||^2, NOT sum_i u_i r_i^T W_i r_i.
    It is exact only when W_i = w_i I.  Kept to quantify the gap.
    """
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    e = whitened_sq(x, y, R, t, S)
    mu = max(3.0 * float(e.max()) / (c * c), 1.0)
    mu0 = mu
    n_mu = 0
    for _ in range(max_mu_iters):
        u = _gm_weights(e, mu, c)
        R, t = cm.kabsch(x, y, u)
        e = whitened_sq(x, y, R, t, S)
        n_mu += 1
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
    for _ in range(polish):
        R, t = cm.kabsch(x, y, _gm_weights(e, 1.0, c))
        e = whitened_sq(x, y, R, t, S)
    phi = float(np.sum(cm.C2 * e / (cm.C2 + e)))
    return R, t, dict(n_outer=n_mu, n_solves=n_mu + polish, gn_iters_total=0,
                      mu0=float(mu0), phi=phi, e=e)


def solve_irls_scalar(x, y, S, c=cm.C, iters=40):
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    for _ in range(iters):
        e = whitened_sq(x, y, R, t, S)
        R, t = cm.kabsch(x, y, _gm_weights(e, 1.0, c))
    e = whitened_sq(x, y, R, t, S)
    phi = float(np.sum(cm.C2 * e / (cm.C2 + e)))
    return R, t, dict(n_outer=iters, n_solves=iters, gn_iters_total=0,
                      mu0=1.0, phi=phi, e=e)


# ---------------------------------------------------------------------------
# scipy cross-check
# ---------------------------------------------------------------------------
def scipy_wls(x, y, S, u, R0, t0, theta0=None, XSKEW=None):
    """scipy.optimize.least_squares on the same 6-param chart (independent check)."""
    from scipy.optimize import least_squares

    n = len(x)
    if XSKEW is None:
        XSKEW = skew_batch(x)
    su = np.sqrt(np.asarray(u, float))

    def res(theta):
        R = R0 @ so3_exp(theta[:3])
        t = t0 + theta[3:]
        r = (R @ x.T).T + t - y
        z = su[:, None] * np.einsum("nij,nj->ni", S, r)
        return z.ravel()

    def jac(theta):
        R = R0 @ so3_exp(theta[:3])
        SR = np.einsum("nij,jk->nik", S, R)
        Jw = -su[:, None, None] * np.einsum("nij,njk->nik", SR, XSKEW)
        Jt = su[:, None, None] * S
        return np.concatenate([Jw, Jt], axis=2).reshape(3 * n, 6)

    x0 = np.zeros(6) if theta0 is None else np.asarray(theta0, float)
    sol = least_squares(res, x0, jac=jac, method="trf",
                        ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=500)
    R = R0 @ so3_exp(sol.x[:3])
    t = t0 + sol.x[3:]
    return R, t, dict(cost2=2.0 * float(sol.cost), nfev=int(sol.nfev),
                      optimality=float(sol.optimality), status=int(sol.status))


def scipy_quat_wls(x, y, S, u, R0, t0):
    """Independent check in a *different* parameterisation: unit quaternion + t.

    Residual uses common.Rmat(q/norm(q)); the quaternion scale is a gauge
    direction, which scipy handles.  Returns the weighted least-squares cost.
    """
    from scipy.optimize import least_squares

    su = np.sqrt(np.asarray(u, float))

    def res(theta):
        q = theta[:4]
        q = q / np.linalg.norm(q)
        R = cm.Rmat(q)
        t = theta[4:]
        r = (R @ x.T).T + t - y
        z = su[:, None] * np.einsum("nij,nj->ni", S, r)
        return z.ravel()

    theta0 = np.concatenate([cm.quat_from_R(R0), np.asarray(t0, float)])
    sol = least_squares(res, theta0, method="trf", ftol=1e-15, xtol=1e-15,
                        gtol=1e-15, max_nfev=2000)
    q = sol.x[:4] / np.linalg.norm(sol.x[:4])
    R = cm.Rmat(q)
    t = sol.x[4:]
    z = res(sol.x)
    return R, t, dict(cost=float(np.einsum("i,i->", z, z)),
                      nfev=int(sol.nfev), optimality=float(sol.optimality))


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(p) for p in parts]])


def instance_seed(p, trial):
    """Data stream: same instances for all weight kinds at a given (p, trial)."""
    return np.random.default_rng(_seed(2, int(round(100 * p)), trial))


def weight_seed(kind, p, trial):
    return np.random.default_rng(_seed(3, KIND_ID[kind], int(round(100 * p)), trial))


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------
def run_trial(kind, p, trial):
    data_rng = instance_seed(p, trial)
    x, y, Rgt, tgt = cm.make_instance(data_rng, p, n=cm.N_DEFAULT,
                                      outlier_type="random")
    w_rng = weight_seed(kind, p, trial)
    W, S = make_weights(w_rng, kind, len(x))
    ev = np.linalg.eigvalsh(W)
    cond_med = float(np.median(ev[:, -1] / ev[:, 0]))
    XSKEW = skew_batch(x)

    recs = {}
    for solver in SOLVERS:
        t0 = time.perf_counter()
        if solver == "GNC-W":
            R, t, info = solve_gnc_w(x, y, S, XSKEW=XSKEW)
        elif solver == "IRLS-W":
            R, t, info = solve_irls_w(x, y, S, XSKEW=XSKEW)
        elif solver == "APPROX-GNC-sK":
            R, t, info = solve_gnc_scalar(x, y, S)
        else:
            R, t, info = solve_irls_scalar(x, y, S)
        dt = time.perf_counter() - t0
        re, te = cm.pose_errors(R, t, Rgt, tgt)
        recs[solver] = dict(
            weight=kind, p=p, trial=trial, solver=solver,
            success=cm.success(re, te), rot=re, trans=te,
            time_ms=1000.0 * dt, phi=info["phi"],
            n_outer=info["n_outer"], n_solves=info["n_solves"],
            gn_iters_total=info["gn_iters_total"], mu0=info["mu0"],
            cond_median=cond_med,
        )
        recs[solver]["_R"] = R
        recs[solver]["_t"] = t
        recs[solver]["_e"] = info["e"]
    return x, y, Rgt, tgt, W, S, recs


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def correctness_checks():
    """Validate the anisotropic solver against closed-form Kabsch / scipy.

    All checks use fresh fixed seeds and are independent of the main table.
    """
    out = {}
    n = 100
    rng = np.random.default_rng(_seed(9, 1))
    max_rot_kabsch = 0.0
    max_trans_kabsch = 0.0
    max_rot_wkabsch = 0.0
    max_trans_wkabsch = 0.0
    for k in range(20):
        x, y, Rgt, tgt = cm.make_instance(rng, 0.0, n=n)
        XSKEW = skew_batch(x)
        Rk, tk = cm.kabsch(x, y, np.ones(n))
        # start Gauss-Newton away from the closed-form optimum
        axis = rng.normal(size=3)
        R0 = Rk @ cm.rotation_about_axis(axis, np.deg2rad(20.0))
        t0 = tk + 0.05 * rng.normal(size=3)
        R, t = gn_wls(x, y, np.broadcast_to(np.eye(3), (n, 3, 3)), np.ones(n),
                      R0, t0, XSKEW=XSKEW)
        max_rot_kabsch = max(max_rot_kabsch, float(np.linalg.norm(R - Rk)))
        max_trans_kabsch = max(max_trans_kabsch, float(np.linalg.norm(t - tk)))
        # weighted closed form: scalar weights only (W = I)
        w = rng.uniform(0.05, 1.0, size=n)
        Rk, tk = cm.kabsch(x, y, w)
        R0 = Rk @ cm.rotation_about_axis(rng.normal(size=3), np.deg2rad(10.0))
        t0 = tk + 0.02 * rng.normal(size=3)
        R, t = gn_wls(x, y, np.broadcast_to(np.eye(3), (n, 3, 3)), w,
                      R0, t0, XSKEW=XSKEW)
        max_rot_wkabsch = max(max_rot_wkabsch, float(np.linalg.norm(R - Rk)))
        max_trans_wkabsch = max(max_trans_wkabsch, float(np.linalg.norm(t - tk)))

    out["gn_vs_kabsch_WI"] = dict(
        n_cases=20, max_rot_fro=max_rot_kabsch, max_trans=max_trans_kabsch,
        passed=bool(max_rot_kabsch < 1e-6 and max_trans_kabsch < 1e-6))
    out["gn_vs_weighted_kabsch_WI"] = dict(
        n_cases=20, max_rot_fro=max_rot_wkabsch, max_trans=max_trans_wkabsch,
        passed=bool(max_rot_wkabsch < 1e-6 and max_trans_wkabsch < 1e-6))

    # scipy cross-check on the anisotropic weighted problems
    scipy_rows = []
    rng = np.random.default_rng(_seed(9, 2))
    for kind in WEIGHT_KINDS:
        for k in range(3):
            p = 0.3
            x, y, Rgt, tgt = cm.make_instance(rng, p, n=n)
            _, S = make_weights(np.random.default_rng(_seed(9, 20 + KIND_ID[kind] * 10 + k)),
                                kind, n)
            XSKEW = skew_batch(x)
            Rk, tk = cm.kabsch(x, y, np.ones(n))
            # use the final GM weights from the Kabsch-init pose at mu=1
            e = whitened_sq(x, y, Rk, tk, S)
            u = _gm_weights(e, 1.0, cm.C)
            Rg, tg, gi = gn_wls(x, y, S, u, Rk, tk, XSKEW=XSKEW,
                                return_info=True)
            cost_gn = gi["gn_cost"]
            # scipy from the same init
            Rs, ts, si = scipy_wls(x, y, S, u, Rk, tk, XSKEW=XSKEW)
            cost_sp = si["cost2"]
            # scipy anchored at the GN solution (local optimality check)
            Rs2, ts2, si2 = scipy_wls(x, y, S, u, Rg, tg, XSKEW=XSKEW)
            cost_sp2 = si2["cost2"]
            scipy_rows.append(dict(
                weight=kind, case=k,
                cost_gn=cost_gn, cost_scipy_init=cost_sp,
                cost_scipy_at_gn=cost_sp2,
                rel_cost_diff_init=abs(cost_sp - cost_gn) / max(cost_gn, 1e-30),
                rel_cost_diff_at_gn=abs(cost_sp2 - cost_gn) / max(cost_gn, 1e-30),
                rot_diff_fro=float(np.linalg.norm(Rs - Rg)),
                trans_diff=float(np.linalg.norm(ts - tg)),
                rot_diff_at_gn_fro=float(np.linalg.norm(Rs2 - Rg)),
                trans_diff_at_gn=float(np.linalg.norm(ts2 - tg)),
                status=si["status"], status_at_gn=si2["status"],
            ))
    out["scipy_least_squares_crosscheck"] = scipy_rows

    # independent 7-parameter (unit quaternion + t) check on the same problems
    quat_rows = []
    rng = np.random.default_rng(_seed(9, 2))
    for kind in WEIGHT_KINDS:
        for k in range(3):
            x, y, Rgt, tgt = cm.make_instance(rng, 0.3, n=n)
            _, S = make_weights(np.random.default_rng(_seed(9, 20 + KIND_ID[kind] * 10 + k)),
                                kind, n)
            XSKEW = skew_batch(x)
            Rk, tk = cm.kabsch(x, y, np.ones(n))
            e = whitened_sq(x, y, Rk, tk, S)
            u = _gm_weights(e, 1.0, cm.C)
            Rg, tg, gi = gn_wls(x, y, S, u, Rk, tk, XSKEW=XSKEW,
                                return_info=True)
            Rq, tq, qi = scipy_quat_wls(x, y, S, u, Rk, tk)
            quat_rows.append(dict(
                weight=kind, case=k,
                cost_gn=gi["gn_cost"], cost_scipy=qi["cost"],
                rel_cost_diff=abs(qi["cost"] - gi["gn_cost"]) / max(gi["gn_cost"], 1e-30),
                rot_diff_fro=float(np.linalg.norm(Rq - Rg)),
                trans_diff=float(np.linalg.norm(tq - tg)),
                nfev=qi["nfev"],
            ))
    out["scipy_quaternion_crosscheck"] = quat_rows

    # full GNC-W vs common.solve_gnc on W = I (should agree)
    max_rot, max_trans = 0.0, 0.0
    rng = np.random.default_rng(_seed(9, 3))
    for trial in range(10):
        x, y, Rgt, tgt = cm.make_instance(rng, 0.5, n=n)
        S = np.broadcast_to(np.eye(3), (n, 3, 3))
        R1, t1, _ = solve_gnc_w(x, y, S, XSKEW=skew_batch(x))
        R2, t2 = cm.solve_gnc(x, y)
        max_rot = max(max_rot, float(np.linalg.norm(R1 - R2)))
        max_trans = max(max_trans, float(np.linalg.norm(t1 - t2)))
    out["gnc_w_vs_common_gnc_WI"] = dict(
        n_cases=10, max_rot_fro=max_rot, max_trans=max_trans,
        passed=bool(max_rot < 1e-6 and max_trans < 1e-6))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize(records, solver):
    rr = [r for r in records if r["solver"] == solver]
    re = np.array([r["rot"] for r in rr])
    te = np.array([r["trans"] for r in rr])
    tt = np.array([r["time_ms"] for r in rr])
    ph = np.array([r["phi"] for r in rr])
    return dict(
        n=len(rr),
        success=float(np.mean((re < cm.SUCCESS_ROT_DEG) & (te < cm.SUCCESS_TRANS))),
        rot_median=float(np.median(re)),
        rot_iqr=float(np.subtract(*np.percentile(re, [75, 25]))),
        rot_p90=float(np.percentile(re, 90)),
        trans_median=float(np.median(te)),
        trans_p90=float(np.percentile(te, 90)),
        time_ms_p50=float(np.median(tt)),
        time_ms_p95=float(np.percentile(tt, 95)),
        phi_median=float(np.median(ph)),
        cond_median=float(np.median([r["cond_median"] for r in rr])),
        gn_iters_mean=float(np.mean([r["gn_iters_total"] for r in rr])),
        n_outer_mean=float(np.mean([r["n_outer"] for r in rr])),
    )


def build_rows(per_cell):
    rows = []
    for kind in WEIGHT_KINDS:
        for p in P_LIST:
            recs = per_cell[(kind, p)]
            for solver in SOLVERS:
                row = dict(weight=kind, p=p, solver=solver,
                           approx=solver in APPROX)
                row.update(summarize(recs, solver))
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_pct(v):
    return f"{100.0 * v:6.1f}"


def build_markdown(rows, paired, inexact, correct, meta):
    def cell(weight, p, solver, key):
        for r in rows:
            if r["weight"] == weight and r["p"] == p and r["solver"] == solver:
                return r[key]
        raise KeyError((weight, p, solver, key))

    L = []
    L.append("# Anisotropic-weight $W_i$ ablation (3D-3D robust registration, GM/GNC)")
    L.append("")
    L.append("Numbers are generated by `run_anisotropic.py` (fixed seeds, "
             f"{meta['trials']} trials/cell, N={meta['n']}, p in {meta['p_list']}). "
             "Raw per-trial records: `anisotropic_raw.jsonl`.")
    L.append("")
    L.append("## 1. What is being solved (and what is not)")
    L.append("")
    L.append("The correct robust objective with per-observation SPD weights $W_i$ is")
    L.append("")
    L.append("$$\\Phi(R,t)=\\sum_i \\rho(e_i),\\qquad "
             "e_i=r_i^\\top W_i r_i,\\qquad r_i=R x_i+t-y_i,\\qquad "
             "\\rho(e)=\\frac{c^2 e}{c^2+e},\\ c=0.05.$$")
    L.append("")
    L.append("At fixed GM weights $u_i=\\rho'(e_i)$ the inner problem is "
             "$\\min_{R,t}\\sum_i u_i\\,r_i^\\top W_i r_i$. This is a **nonlinear** "
             "least-squares problem in the 6-DoF pose: the weights sit inside the "
             "residual, so it does **not** reduce to a scalar weighted Kabsch. "
             "`common.kabsch` solves $\\sum_i u_i\\|r_i\\|^2$, which coincides with "
             "the correct objective only when all $W_i$ are scalar multiples of $I$.")
    L.append("")
    L.append("This script solves the inner problem by 6-DoF Gauss-Newton on the "
             "Cholesky-whitened residuals $\\sqrt{u_i}\\,L_i^\\top(R x_i+t-y_i)$ "
             "($W_i=L_iL_i^\\top$), re-anchoring the rotation chart every iteration. "
             "`scipy.optimize.least_squares` is run on the same problem as an "
             "independent check. Rows labelled **APPROX-sK** replace the inner solve "
             "by scalar weighted Kabsch; they are an explicitly labelled "
             "approximation, not a solution of the anisotropic objective.")
    L.append("")
    L.append("Weight families: `identity` $W_i=I$; `diag` $W_i=\\mathrm{diag}(d_1,d_2,d_3)$, "
             "$d_k\\sim U(0.5,2)$; `full` $W_i=(A_iA_i^\\top+0.05I)/\\det(\\cdot)^{1/3}$, "
             "$A_i\\sim\\mathcal N(0,I)$ (random Wishart, $\\det W_i=1$). "
             "Weights are drawn from a seed stream independent of the data; the same "
             "instances are reused across the three weight families (paired comparison).")
    L.append("")
    L.append("## 2. Correctness checks")
    L.append("")
    L.append("| check | result | passed |")
    L.append("|---|---|---|")
    k = correct["gn_vs_kabsch_WI"]
    L.append(f"| GN-WLS with $W_i=I$, unit weights vs `common.kabsch` (20 cases, "
             f"init perturbed by 20 deg) | max $\\|R-R_{{kabsch}}\\|_F$ = "
             f"{k['max_rot_fro']:.2e}, max $\\|t-t_{{kabsch}}\\|$ = {k['max_trans']:.2e} | "
             f"{'yes' if k['passed'] else '**NO**'} |")
    k = correct["gn_vs_weighted_kabsch_WI"]
    L.append(f"| GN-WLS with $W_i=I$, random scalar weights vs weighted Kabsch "
             f"(20 cases) | max $\\|R-R_{{kabsch}}\\|_F$ = {k['max_rot_fro']:.2e}, "
             f"max $\\|t-t_{{kabsch}}\\|$ = {k['max_trans']:.2e} | "
             f"{'yes' if k['passed'] else '**NO**'} |")
    k = correct["gnc_w_vs_common_gnc_WI"]
    L.append(f"| full GNC-W ($W_i=I$) vs `common.solve_gnc` (10 cases, p=0.5) | "
             f"max $\|R-R_{{\mathrm{{common}}}}\|_F$ = {k['max_rot_fro']:.2e}, "
             f"max trans diff = {k['max_trans']:.2e} | "
             f"{'yes' if k['passed'] else '**NO**'} |")
    sp = correct["scipy_least_squares_crosscheck"]
    L.append(f"| GN vs `scipy.optimize.least_squares` on the anisotropic weighted "
             f"problem ({len(sp)} cases) | max rel. cost diff from same init = "
             f"{max(r['rel_cost_diff_init'] for r in sp):.2e}; at GN solution = "
             f"{max(r['rel_cost_diff_at_gn'] for r in sp):.2e}; max pose diff = "
             f"{max(max(r['rot_diff_fro'], r['trans_diff']) for r in sp):.2e} | "
             f"{'yes' if max(r['rel_cost_diff_init'] for r in sp) < 1e-9 else '**NO**'} |")
    sq = correct["scipy_quaternion_crosscheck"]
    L.append(f"| GN vs `scipy.optimize.least_squares` in a different parameterisation "
             f"(unit quaternion + t, {len(sq)} cases) | max rel. cost diff = "
             f"{max(r['rel_cost_diff'] for r in sq):.2e}; max pose diff = "
             f"{max(max(r['rot_diff_fro'], r['trans_diff']) for r in sq):.2e} | "
             f"{'yes' if max(r['rel_cost_diff'] for r in sq) < 1e-7 else '**NO**'} |")
    L.append("")
    L.append(f"The $W_i=I$ check is the required one: the nonlinear solver reproduces "
             f"closed-form Kabsch to well under $10^{{-6}}$ (max Frobenius error "
             f"{correct['gn_vs_kabsch_WI']['max_rot_fro']:.1e}).")
    L.append("")
    L.append("## 3. Main table")
    L.append("")
    L.append("Success = rotation error < 5 deg and translation error < 0.1. "
             "Errors are median (p90 in parentheses). $\\Phi$ is the robust objective "
             "$\\sum_i\\rho(e_i)$ at the returned pose (lower is better). $\\Phi$ values are "
             "comparable within one weight family; they are **not** comparable across "
             "families because $e_i$ itself scales with $W_i$. "
             "Time is wall-clock per solver call on a shared Slurm CPU node with "
             "BLAS/OpenMP pinned to one thread; absolute times can vary by tens of "
             "percent between runs (node contention), so compare within-run ratios.")
    L.append("")
    L.append("| $W_i$ | p | solver | success % | rot med (p90) deg | trans med (p90) | "
             "$\\Phi$ med | time p50 / p95 ms |")
    L.append("|---|---|---|---|---|---|---|---|")
    for w in WEIGHT_KINDS:
        for p in P_LIST:
            for s in SOLVERS:
                L.append(
                    f"| {w} | {p:.1f} | {s} | "
                    f"{fmt_pct(cell(w, p, s, 'success'))} | "
                    f"{cell(w, p, s, 'rot_median'):.4f} "
                    f"({cell(w, p, s, 'rot_p90'):.3f}) | "
                    f"{cell(w, p, s, 'trans_median'):.4f} "
                    f"({cell(w, p, s, 'trans_p90'):.4f}) | "
                    f"{cell(w, p, s, 'phi_median'):.3e} | "
                    f"{cell(w, p, s, 'time_ms_p50'):.3f} / "
                    f"{cell(w, p, s, 'time_ms_p95'):.3f} |")
    L.append("")
    L.append("### 3.0 Weight condition numbers")
    L.append("")
    L.append("Median condition number $\\lambda_{\\max}/\\lambda_{\\min}$ of the drawn "
             "$W_i$ per cell ($W_i=I$ has condition 1 by construction):")
    L.append("")
    L.append("| $W_i$ | " + " | ".join(f"p={p:.1f}" for p in P_LIST) + " |")
    L.append("|---|" + "---|" * len(P_LIST))
    for w in WEIGHT_KINDS:
        L.append(f"| {w} | " + " | ".join(
            f"{cell(w, p, 'GNC-W', 'cond_median'):.1f}" for p in P_LIST) + " |")
    L.append("")
    L.append("### 3.1 Runtime of the exact inner solve")
    L.append("")
    for w in WEIGHT_KINDS:
        for p in P_LIST:
            a = cell(w, p, "GNC-W", "time_ms_p50")
            b = cell(w, p, "APPROX-GNC-sK", "time_ms_p50")
            c = cell(w, p, "IRLS-W", "time_ms_p50")
            d = cell(w, p, "APPROX-IRLS-sK", "time_ms_p50")
            L.append(f"* `{w}` p={p:.1f}: GNC-W {a:.2f} ms vs APPROX-GNC-sK {b:.2f} ms "
                     f"({a / max(b, 1e-9):.1f}x); IRLS-W {c:.2f} ms vs "
                     f"APPROX-IRLS-sK {d:.2f} ms ({c / max(d, 1e-9):.1f}x)")
    L.append("")
    L.append("## 4. Does anisotropy help or hurt accuracy? (paired, same instances)")
    L.append("")
    L.append("For GNC-W, each anisotropic trial is paired with the identity trial on "
             "the same instance (same data seed, independent weight draw). "
             "$\\Delta$rot = rot(full/diag) - rot(identity), so negative means the "
             "anisotropic weighting improved accuracy. Fraction better = share of "
             "paired trials with strictly smaller rotation error.")
    L.append("")
    L.append("| comparison | p | median $\\Delta$rot (deg) | median $\\Delta$trans | "
             "rot better % | success (identity / anisotropic) |")
    L.append("|---|---|---|---|---|---|")
    for comp in ("diag", "full"):
        for p in P_LIST:
            r = paired[comp][p]
            L.append(f"| {comp} vs identity | {p:.1f} | {r['rot_delta_median']:+.4f} | "
                     f"{r['trans_delta_median']:+.4f} | {100 * r['rot_better_frac']:.1f} | "
                     f"{100 * r['success_identity']:.1f}% / {100 * r['success_other']:.1f}% |")
    L.append("")
    L.append("## 5. How inexact is the scalar-Kabsch inner?")
    L.append("")
    L.append("For each GNC-W trial we take the converged GM weights $u_i^*$, form the "
             "scalar weights $u_i^*$ and solve scalar weighted Kabsch. "
             "$J_{\\mathrm{sK}}=\\sum_i u_i^* e_i(R_{\\mathrm{sK}})$ is compared with the "
             "GN optimum $J_{\\mathrm{GN}}=\\sum_i u_i^* e_i(R_{\\mathrm{GN}})$. "
             "Ratio $J_{\\mathrm{sK}}/J_{\\mathrm{GN}}=1$ would mean the scalar inner is "
             "exact; $>1$ quantifies its error on the correct weighted objective.")
    L.append("")
    L.append("| $W_i$ | p | median ratio | p90 ratio | max ratio |")
    L.append("|---|---|---|---|---|")
    for w in WEIGHT_KINDS:
        for p in P_LIST:
            r = inexact[w][p]
            L.append(f"| {w} | {p:.1f} | {r['median']:.6f} | {r['p90']:.6f} | "
                     f"{r['max']:.6f} |")
    L.append("")
    L.append("## 6. Takeaways")
    L.append("")
    L += meta["takeaways"]
    L.append("")
    L.append("## 7. Reading and limitations")
    L.append("")
    L.append("* The nonlinear inner solve is only needed when some $W_i$ is genuinely "
             "anisotropic. When $W_i=w_i I$, scalar weighted Kabsch is exact and roughly "
             "an order of magnitude cheaper; production code should dispatch on that "
             "case. The runtime above is the straightforward GN implementation (no "
             "relative-cost stopping or LM damping), so treat the overhead as the "
             "unoptimised end of the range.")
    L.append("* The synthetic noise in `common.make_instance` is isotropic, so $W_i=I$ "
             "is the statistically correct weighting and any anisotropic $W_i$ is "
             "misspecified. The accuracy loss for diag/full in Section 4 is therefore "
             "expected and is not a solver defect: the solver is verified by the "
             "$W_i=I$ agreement with Kabsch, the scipy agreement, and the lower $\\Phi$ "
             "versus the scalar surrogate. Whether anisotropic weights *help* when they "
             "match the true noise covariance is a separate matched-covariance study, "
             "not run here.")
    L.append("")
    return "\n".join(L) + "\n"


def compute_takeaways(rows, paired, inexact):
    """Data-driven bullets (no hand-written numbers)."""
    def val(w, p, s, k):
        for r in rows:
            if r["weight"] == w and r["p"] == p and r["solver"] == s:
                return r[k]
        raise KeyError((w, p, s, k))

    out = []
    ratios_id = [inexact["identity"][p]["median"] for p in P_LIST]
    ratios_diag = [inexact["diag"][p]["median"] for p in P_LIST]
    ratios_full = [inexact["full"][p]["median"] for p in P_LIST]
    out.append(
        "* The scalar-Kabsch inner is exact for $W_i=I$ (median objective ratio "
        f"{max(ratios_id):.6f}) but not for anisotropic weights: median ratio "
        f"{min(ratios_diag):.4f}-{max(ratios_diag):.4f} (diag) and "
        f"{min(ratios_full):.4f}-{max(ratios_full):.4f} (full). Scalar Kabsch "
        "therefore does not solve $\\sum_i r_i^\\top W_i r_i$ for anisotropic $W_i$.")
    # correctness of the exact objective
    phi_better = 0
    for w in WEIGHT_KINDS:
        for p in P_LIST:
            a = val(w, p, "GNC-W", "phi_median")
            b = val(w, p, "APPROX-GNC-sK", "phi_median")
            if a <= b * (1.0 + 1e-9):
                phi_better += 1
    out.append(
        f"* GNC-W reaches the lower (or equal) robust objective $\\Phi$ than the "
        f"scalar-Kabsch surrogate in {phi_better}/{len(WEIGHT_KINDS) * len(P_LIST)} "
        "weight/p cells, confirming that the nonlinear inner solve minimises the "
        "weighted objective. A lower $\\Phi$ need not mean a lower pose error when "
        "$W_i$ is misspecified relative to the actual noise.")
    # accuracy effect
    for comp in ("diag", "full"):
        deltas = [paired[comp][p]["rot_delta_median"] for p in P_LIST]
        better = [paired[comp][p]["rot_better_frac"] for p in P_LIST]
        improves = (max(deltas) < -0.005) and (min(better) > 0.5)
        hurts = (min(deltas) > 0.005) and (max(better) < 0.5)
        if improves:
            verdict = "a small accuracy gain."
        elif hurts:
            verdict = ("a small accuracy **loss**; the anisotropic weights do not "
                       "improve accuracy here.")
        else:
            verdict = ("no consistent accuracy change (the sign varies with p).")
        out.append(
            f"* {comp.capitalize()} vs $W=I$ (GNC-W, paired instances): median rotation "
            f"change {min(deltas):+.3f} to {max(deltas):+.3f} deg across p, "
            f"anisotropic solution better on {100 * min(better):.0f}-"
            f"{100 * max(better):.0f}% of paired trials, i.e. {verdict} "
            "The synthetic noise is isotropic, so anisotropic $W_i$ is a misspecified "
            "model; the experiment tests solver correctness, not a claim that "
            "anisotropy helps this data.")
    succ = [val(w, p, "GNC-W", "success") for w in WEIGHT_KINDS for p in P_LIST]
    out.append(
        f"* GNC-W success across all (weight, p) cells is "
        f"{100 * min(succ):.0f}-{100 * max(succ):.0f}%; differences between weight "
        "families are in the sub-degree error regime, well inside the 5 deg / 0.1 "
        "success threshold.")
    rt = []
    for w in ("diag", "full"):
        for p in P_LIST:
            rt.append(val(w, p, "GNC-W", "time_ms_p50")
                      / max(val(w, p, "APPROX-GNC-sK", "time_ms_p50"), 1e-9))
    out.append(
        f"* The correct anisotropic inner solve costs {min(rt):.1f}-{max(rt):.1f}x the "
        "scalar-Kabsch approximation per GNC call (p50) and is still "
        f"{max(val(w, p, 'GNC-W', 'time_ms_p50') for w in ('diag', 'full') for p in P_LIST):.1f} ms "
        "at worst, which is the price of actually minimising the weighted objective.")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"identity": "#1f77b4", "diag": "#2ca02c", "full": "#d62728"}
    markers = {"GNC-W": "o", "IRLS-W": "s"}
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ps = [100 * p for p in P_LIST]
    for w in WEIGHT_KINDS:
        for s in ("GNC-W", "IRLS-W"):
            ys = [[r for r in rows if r["weight"] == w and r["p"] == p
                   and r["solver"] == s][0] for p in P_LIST]
            ls = "-" if s == "GNC-W" else "--"
            ax[0].plot(ps, [100 * y["success"] for y in ys], ls,
                       marker=markers[s], color=colors[w], label=f"{w} {s}", lw=1.4)
            ax[1].semilogy(ps, [max(y["rot_median"], 1e-3) for y in ys], ls,
                           marker=markers[s], color=colors[w], label=f"{w} {s}", lw=1.4)
            ax[2].semilogy(ps, [y["time_ms_p50"] for y in ys], ls,
                           marker=markers[s], color=colors[w], label=f"{w} {s}", lw=1.4)
    ax[0].set_ylabel("success rate (%)")
    ax[0].set_title("success (rot<5 deg, trans<0.1)")
    ax[0].set_ylim(-3, 103)
    ax[1].set_ylabel("median rotation error (deg, log)")
    ax[1].set_title("accuracy")
    ax[2].set_ylabel("time p50 (ms, log)")
    ax[2].set_title("runtime (exact nonlinear inner)")
    for a in ax:
        a.set_xlabel("outlier fraction (%)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "anisotropic.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("correctness checks ...", flush=True)
    correct = correctness_checks()
    for key in ("gn_vs_kabsch_WI", "gn_vs_weighted_kabsch_WI",
                "gnc_w_vs_common_gnc_WI"):
        if not correct[key]["passed"]:
            raise SystemExit(f"CORRECTNESS CHECK FAILED: {key} -> {correct[key]}")
    sp = correct["scipy_least_squares_crosscheck"]
    if max(r["rel_cost_diff_init"] for r in sp) > 1e-9:
        raise SystemExit("CORRECTNESS CHECK FAILED: scipy cross-check")
    sq = correct["scipy_quaternion_crosscheck"]
    if max(r["rel_cost_diff"] for r in sq) > 1e-7:
        raise SystemExit("CORRECTNESS CHECK FAILED: scipy quaternion cross-check")
    print("correctness ok", flush=True)

    # warm-up
    xw, yw, _, _ = cm.make_instance(np.random.default_rng(0), 0.3, n=cm.N_DEFAULT)
    for kind in WEIGHT_KINDS:
        _, Sw = make_weights(np.random.default_rng(0), kind, len(xw))
        solve_gnc_w(xw, yw, Sw, XSKEW=skew_batch(xw))
        solve_irls_w(xw, yw, Sw, XSKEW=skew_batch(xw))

    per_cell = {}
    all_recs = []
    inexact = {w: {} for w in WEIGHT_KINDS}
    paired = {w: {} for w in ("diag", "full")}
    for kind in WEIGHT_KINDS:
        for p in P_LIST:
            t0 = time.time()
            recs = []
            for trial in range(TRIALS):
                x, y, Rgt, tgt, W, S, r = run_trial(kind, p, trial)
                XSKEW = skew_batch(x)
                # scalar-inner inexactness measured on the GNC-W solution
                eg = r["GNC-W"]["_e"]
                ug = _gm_weights(eg, 1.0, cm.C)
                # exact minimiser of the fixed-weight anisotropic problem
                Rg2, tg2 = gn_wls(x, y, S, ug, r["GNC-W"]["_R"],
                                  r["GNC-W"]["_t"], XSKEW=XSKEW)
                J_gn = float(np.sum(ug * whitened_sq(x, y, Rg2, tg2, S)))
                Rk, tk = cm.kabsch(x, y, ug)
                J_sk = float(np.sum(ug * whitened_sq(x, y, Rk, tk, S)))
                r["GNC-W"]["scalar_ratio"] = J_sk / max(J_gn, 1e-300)
                for s in SOLVERS:
                    rr = {k2: v2 for k2, v2 in r[s].items() if not k2.startswith("_")}
                    rr["scalar_ratio"] = r[s].get("scalar_ratio")
                    recs.append(rr)
                all_recs.append(r["GNC-W"])
            per_cell[(kind, p)] = [
                {k2: v2 for k2, v2 in rr.items() if not k2.startswith("_")}
                for rr in recs]
            ratios = np.array([rr["scalar_ratio"] for rr in recs
                               if rr["solver"] == "GNC-W"])
            inexact[kind][p] = dict(median=float(np.median(ratios)),
                                    p90=float(np.percentile(ratios, 90)),
                                    max=float(np.max(ratios)))
            print(f"[{kind}] p={p:.1f} done in {time.time() - t0:.1f}s", flush=True)

    # paired comparison against identity on the same instances
    for comp in ("diag", "full"):
        for p in P_LIST:
            d_rot, d_tr, better, s_id, s_ot = [], [], [], [], []
            for trial in range(TRIALS):
                a = next(x for x in per_cell[("identity", p)]
                         if x["trial"] == trial and x["solver"] == "GNC-W")
                b = next(x for x in per_cell[(comp, p)]
                         if x["trial"] == trial and x["solver"] == "GNC-W")
                d_rot.append(b["rot"] - a["rot"])
                d_tr.append(b["trans"] - a["trans"])
                better.append(float(b["rot"] < a["rot"]))
                s_id.append(float(a["success"]))
                s_ot.append(float(b["success"]))
            paired[comp][p] = dict(
                rot_delta_median=float(np.median(d_rot)),
                trans_delta_median=float(np.median(d_tr)),
                rot_better_frac=float(np.mean(better)),
                success_identity=float(np.mean(s_id)),
                success_other=float(np.mean(s_ot)),
            )

    rows = build_rows(per_cell)
    meta = dict(trials=TRIALS, n=cm.N_DEFAULT, p_list=P_LIST,
                base_seed=BASE_SEED, c=cm.C, gamma=cm.GNC_GAMMA,
                gn_max_iter=GN_MAX_ITER)
    meta["takeaways"] = compute_takeaways(rows, paired, inexact)

    payload = dict(
        meta=meta,
        rows=rows,
        correctness=correct,
        inexact_scalar_inner=inexact,
        paired_vs_identity=paired,
        solver_notes=dict(
            correct="GNC-W / IRLS-W solve min sum_i u_i r_i^T W_i r_i by 6-DoF "
                    "Gauss-Newton on Cholesky-whitened residuals (same GNC schedule "
                    "as common.solve_gnc).",
            approx="APPROX-* replace the inner solve with scalar weighted Kabsch "
                   "(common.kabsch); exact only for W_i = w_i I, kept as a labelled "
                   "baseline to quantify the gap.",
        ),
    )
    cm.json_dump(OUT / "anisotropic.json", payload)
    with open(OUT / "anisotropic_raw.jsonl", "w") as f:
        for rr in all_recs:
            rec = {k2: v2 for k2, v2 in rr.items() if not k2.startswith("_")}
            f.write(json.dumps(rec) + "\n")
    cm.write_text(OUT / "anisotropic.md",
                  build_markdown(rows, paired, inexact, correct, meta))
    make_figure(rows)
    print(f"wrote {OUT}/anisotropic.{{json,md,png,jsonl}} in "
          f"{time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
