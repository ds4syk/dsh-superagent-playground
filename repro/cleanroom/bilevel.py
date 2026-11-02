"""Bilevel baseline calibration: Problem (P7) with Problem (P6) as inner layer.

Outer problem (Eq. P7):

    min_theta  || t*(theta) - t_gt ||_2^2 + || C*(theta)^T C_gt - I ||_F^2
    s.t.       {C*, t*} = argmin Problem (P6)

with ``theta`` the stereo baseline ``b``, solved by stochastic gradient descent
(learning rate ``1e-4``), initialised ``3 mm`` away from the true value, and
terminated when ``|grad|^2 < 1e-5`` (the upstream script's ``tol_grad_sq``;
the paper quotes ``|grad| < 1e-3``) or after 150 iterations.

Three inner solvers are compared, exactly as in the paper's Fig. 4 / Table V:

``sdpr``       SDPR-IS: globally optimal SDP relaxation + Theorem-2 implicit
               differentiation (no initialisation, no warm start needed).
``local_gt``   Gauss-Newton initialised at the ground-truth pose and
               warm-started with the previous solution.
``local_rnd``  Gauss-Newton initialised at the paper's random initialisation
               (Fig. 3b) and warm-started with the previous solution.

Per-iteration diagnostics of the SDP solve (tightness ratio, corank of the
certificate, KKT residual, solve/backward time, inner cost) are accumulated
for the statistics reported in ``REPORT.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np

import data_gen
import implicit_diff as idiff
import local_solver as ls
import sdp_pose
from camera_model import StereoCamera

__all__ = ["ExpConfig", "make_dataset", "run_trial", "make_camera"]


@dataclass
class ExpConfig:
    # -- problem size / data ------------------------------------------
    n_batch: int = 20
    n_runs: int = 50
    radius: float = 3.0
    board_dims: tuple = (1.0, 1.0)
    n_squares: tuple = (8, 8)
    alpha: float = np.pi / 4.0
    beta: float = np.pi / 4.0
    sigma_u: float = 0.5
    sigma_v: float = 0.5
    noise_mode: str = "independent"
    # -- camera / baseline -------------------------------------------
    fu: float = 484.5
    fv: float = 484.5
    b_true: float = 0.24
    b_offset: float = 0.003
    # -- outer optimisation (as in _scripts/stereo_cal.py) ------------
    lr: float = 1e-4
    max_iter: int = 150
    tol_grad_sq: float = 1e-5
    tol_loss: float = 1e-10
    diverge_tol: float = 1.0
    # -- inner SDP solver (paper: SCS, 1e-9) --------------------------
    sdp_solver: str = "SCS"
    sdp_eps: float = 1e-9
    sdp_max_iters: int = 100_000
    # -- inner local solver (Theseus GN configuration) ----------------
    gn_max_iter: int = 500
    gn_step_size: float = 0.2
    gn_abs_tol: float = 1e-10
    gn_rel_tol: float = 1e-6
    gn_polish: bool = True
    gn_polish_iter: int = 100
    # -- misc ---------------------------------------------------------
    seed0: int = 0
    record_diagnostics: bool = True


def make_camera(cfg: ExpConfig) -> StereoCamera:
    return StereoCamera(
        fu=cfg.fu,
        fv=cfg.fv,
        cu=0.0,
        cv=0.0,
        sigma_u=cfg.sigma_u,
        sigma_v=cfg.sigma_v,
    )


def make_dataset(cfg: ExpConfig, run_idx: int):
    """One problem instance: 20 poses, 8x8 board, noisy pixel measurements."""
    rng = np.random.default_rng(cfg.seed0 * 100_000 + run_idx)
    cam = make_camera(cfg)
    data = data_gen.get_calibration_data(
        n_batch=cfg.n_batch,
        board_dims=cfg.board_dims,
        n_squares=cfg.n_squares,
        radius=cfg.radius,
        beta=cfg.beta,
        alpha=cfg.alpha,
        cam=cam,
        b_true=cfg.b_true,
        rng=rng,
        noise_mode=cfg.noise_mode,
    )
    # separate RNG stream for the local-solver initialisation
    init_rng = np.random.default_rng(cfg.seed0 * 100_000 + 50_000 + run_idx)
    r_init, C_init = data_gen.get_random_poses(
        radius=cfg.radius, n_batch=cfg.n_batch, rng=init_rng
    )
    data["C_init_random"] = C_init
    data["r_init_random"] = r_init
    return data


# ----------------------------------------------------------------------
# inner solvers
# ----------------------------------------------------------------------
def _recompute_multipliers(solver: sdp_pose.SDPPoseSolver, x, Q_scaled):
    """Least-squares multipliers for the reduced constraint set (SDPR-CIFT).

    Used only as a safety net if the SDP dual certificate is not accurate
    enough (``||H x||`` or ``H >= 0`` fail); the default SDPR-IS path reuses
    the multipliers returned by the conic solver.
    """
    G_r, keep = solver.reduced_gradient_rows(x)
    lam, *_ = np.linalg.lstsq(G_r.T, -(Q_scaled @ x), rcond=None)
    H = np.array(Q_scaled, dtype=float)
    for li, i in enumerate(keep):
        H = H + lam[li] * solver.A_list[i]
    H = H + lam[len(keep)] * solver.A0
    return 0.5 * (H + H.T), lam


def _dQ_db_complex_step(cam, pixels, m, b, hc=1e-20):
    """``dQ/db`` for the (unscaled) objective matrix by the complex-step method."""
    meas_c, W_c = cam.inverse(pixels, b + 1j * hc)
    Qc = sdp_pose.build_objective_matrix(m, meas_c, W_c)
    dQ = np.imag(Qc) / hc
    return 0.5 * (dQ + dQ.T)


def _solve_inner_sdpr(data, cam, b, cfg, solvers, scales, prev_x=None):
    """Solve Problem (P6) for all poses with the SDPR layer at baseline ``b``."""
    m = data["r_l"]
    pixels = data["pixels"]
    C_gt = data["C_p0s"]
    r_gt = data["r_p0s"]
    losses = np.zeros(cfg.n_batch)
    grads = np.zeros(cfg.n_batch)
    inner_costs = np.zeros(cfg.n_batch)
    diags = dict(
        r_tight=np.zeros(cfg.n_batch),
        corank_H=np.zeros(cfg.n_batch, dtype=int),
        corank_H_strict=np.zeros(cfg.n_batch, dtype=int),
        eig2_eig1=np.zeros(cfg.n_batch),
        H_min_eig=np.zeros(cfg.n_batch),
        Hx_res=np.zeros(cfg.n_batch),
        orth_err=np.zeros(cfg.n_batch),
        primal=np.zeros(cfg.n_batch),
        dual=np.zeros(cfg.n_batch),
        t_solve=np.zeros(cfg.n_batch),
        t_backward=np.zeros(cfg.n_batch),
        n_retry=np.zeros(cfg.n_batch, dtype=int),
        fallback_H=np.zeros(cfg.n_batch, dtype=int),
    )
    for k in range(cfg.n_batch):
        pix = pixels[k]
        meas, W = cam.inverse(pix, b)
        Q = sdp_pose.build_objective_matrix(m, meas, W)
        Qs = sdp_pose.scale_objective(Q, scales[k])
        rec = solvers[k].solve(Qs)
        n_retry = 0
        if rec["status"] != "optimal":
            n_retry += 1
            solvers[k].eps = 1e-10
            rec = solvers[k].solve(Qs)
        x = rec["x"]
        H = rec["H"]
        fallback = 0
        if (not np.all(np.isfinite(H))) or rec["Hx_res"] > 1e-5 or rec["H_min_eig"] < -1e-7:
            H, _ = _recompute_multipliers(solvers[k], x, Qs)
            fallback = 1
        # derivative of the *scaled* objective (offset removed, frozen scale)
        dQ = scales[k] * _dQ_db_complex_step(cam, pix, m, b)
        dQ[0, 0] = 0.0
        loss_k, grad_x = idiff.loss_from_x_and_grad(x, C_gt[k], r_gt[k])
        G_r, _ = solvers[k].reduced_gradient_rows(x)
        t0 = time.perf_counter()
        bwd = idiff.qcqp_backward(H, G_r, x, dQ, grad_x)
        t_bwd = time.perf_counter() - t0
        losses[k] = loss_k
        grads[k] = bwd["grad_theta"]
        # inner (Problem P6) cost at the returned solution
        C_k, r_k = idiff.pose_from_x(x)
        e = (meas - C_k @ m + (C_k @ r_k)[:, None]).T
        inner_costs[k] = float(np.einsum("nij,ni,nj->", W, e, e))
        diags["r_tight"][k] = rec["r_tight"]
        diags["corank_H"][k] = rec["corank_H"]
        diags["corank_H_strict"][k] = rec["corank_H_strict"]
        diags["eig2_eig1"][k] = rec["eig2_eig1"]
        diags["H_min_eig"][k] = rec["H_min_eig"]
        diags["Hx_res"][k] = rec["Hx_res"]
        diags["orth_err"][k] = float(np.linalg.norm(C_k.T @ C_k - np.eye(3)))
        diags["primal"][k] = rec["primal_obj"]
        diags["dual"][k] = rec["dual_obj"]
        diags["t_solve"][k] = rec["t_solve"]
        diags["t_backward"][k] = t_bwd
        diags["n_retry"][k] = n_retry
        diags["fallback_H"][k] = fallback
    return losses, grads, inner_costs, diags


def _solve_inner_local(data, cam, b, cfg, prev, init, first):
    """Solve Problem (P6) for all poses with the Gauss-Newton local solver."""
    m = data["r_l"]
    pixels = data["pixels"]
    C_gt = data["C_p0s"]
    r_gt = data["r_p0s"]
    losses = np.zeros(cfg.n_batch)
    grads = np.zeros(cfg.n_batch)
    inner_costs = np.zeros(cfg.n_batch)
    poses = []
    diags = dict(
        cond=np.zeros(cfg.n_batch),
        g_norm=np.zeros(cfg.n_batch),
        n_gn_iter=np.zeros(cfg.n_batch, dtype=int),
        t_solve=np.zeros(cfg.n_batch),
        t_backward=np.zeros(cfg.n_batch),
        converged=np.zeros(cfg.n_batch, dtype=int),
        cost=np.zeros(cfg.n_batch),
        natural_step=np.zeros(cfg.n_batch),
        stationary=np.zeros(cfg.n_batch, dtype=int),
    )
    for k in range(cfg.n_batch):
        pix = pixels[k]
        t0 = time.perf_counter()
        meas, W = cam.inverse(pix, b)
        if first:
            C0 = init[0][k]
            r0 = init[1][k]
        else:
            C0, r0 = prev[k]
        res = ls.solve_pose_local(
            m, meas, W, C0, r0,
            max_iter=cfg.gn_max_iter,
            step_size=cfg.gn_step_size,
            abs_tol=cfg.gn_abs_tol,
            rel_tol=cfg.gn_rel_tol,
        )
        n_iter_forward = res["n_iter"]
        if cfg.gn_polish:
            # Continue the descent from the paper-configuration iterate to a
            # true stationary point (the paper's 1e-6 relative-error stopping
            # rule can leave the local methods ~1e-2 off the stationary point
            # for far-away local minima).  This does not change the basin and
            # makes the implicit derivative well defined.
            res = ls.solve_pose_local(
                m, meas, W, res["C"], res["r_p"],
                max_iter=cfg.gn_polish_iter,
                step_size=1.0,
                abs_tol=1e-14,
                rel_tol=1e-14,
            )
        t_solve = time.perf_counter() - t0
        C_k, r_k = res["C"], res["r_p"]
        poses.append((C_k.copy(), r_k.copy()))
        inner_costs[k] = res["cost"]
        loss_k, grad_C, grad_r = idiff.loss_pose_and_grad(C_k, r_k, C_gt[k], r_gt[k])
        losses[k] = loss_k
        t0 = time.perf_counter()
        der = ls.local_implicit_derivative(
            m, C_k, r_k,
            meas_fn=lambda bb, pix=pix: cam.inverse(pix, bb)[0],
            W_fn=lambda bb, pix=pix: cam.inverse(pix, bb)[1],
            b=b,
        )
        t_bwd = time.perf_counter() - t0
        grads[k] = float(np.sum(grad_C * der["dC"]) + grad_r @ der["dr_p"])
        diags["cond"][k] = der["cond"]
        diags["g_norm"][k] = der["g_norm"]
        diags["n_gn_iter"][k] = n_iter_forward
        diags["natural_step"][k] = res["natural_step"]
        diags["stationary"][k] = int(res["stationary"])
        diags["t_solve"][k] = t_solve
        diags["t_backward"][k] = t_bwd
        diags["converged"][k] = int(res["converged"])
        diags["cost"][k] = res["cost"]
    return losses, grads, inner_costs, diags, poses


# ----------------------------------------------------------------------
# one outer trial
# ----------------------------------------------------------------------
def run_trial(data, method: str, cfg: ExpConfig):
    """Run one bilevel trial and return the optimisation history."""
    cam = make_camera(cfg)
    m = data["r_l"]
    b_true = cfg.b_true
    b = b_true + cfg.b_offset

    solvers = None
    scales = None
    if method == "sdpr":
        solvers = [
            sdp_pose.SDPPoseSolver(
                solver=cfg.sdp_solver,
                eps=cfg.sdp_eps,
                max_iters=cfg.sdp_max_iters,
            )
            for _ in range(cfg.n_batch)
        ]
        # freeze the objective scaling per pose at the initial baseline
        scales = []
        for k in range(cfg.n_batch):
            meas, W = cam.inverse(data["pixels"][k], b)
            Q = sdp_pose.build_objective_matrix(m, meas, W)
            scales.append(sdp_pose.objective_scale(Q))
        scales = np.array(scales)

    prev = None
    if method == "local_gt":
        init = (data["C_p0s"], data["r_p0s"])
    elif method == "local_rnd":
        init = (data["C_init_random"], data["r_init_random"])
    else:
        init = None
    if method in ("local_gt", "local_rnd"):
        prev = [(init[0][k], init[1][k].reshape(3)) for k in range(cfg.n_batch)]

    hist = dict(
        status="max_iter",
        b=[],
        err=[],
        loss=[],
        grad=[],
        grad_sq=[],
        inner_cost=[],
        diag=[],
        time_iter=[],
        n_iter=0,
    )
    grad_sq = np.inf
    loss = np.inf
    n_iter = 0
    while grad_sq > cfg.tol_grad_sq and n_iter < cfg.max_iter and loss > cfg.tol_loss:
        t_start = time.perf_counter()
        if method == "sdpr":
            losses, grads, costs, diags = _solve_inner_sdpr(
                data, cam, b, cfg, solvers, scales)
        else:
            losses, grads, costs, diags, poses = _solve_inner_local(
                data, cam, b, cfg, prev, init, first=(n_iter == 0))
            prev = poses
        loss = float(np.sum(losses))
        grad = float(np.sum(grads))
        grad_sq = grad**2
        t_iter = time.perf_counter() - t_start
        b = b - cfg.lr * grad
        if not np.isfinite(b) or abs(b - b_true) > cfg.diverge_tol:
            # the outer parameter has escaped: count as a diverged run
            hist["status"] = "diverged"
            hist["b"].append(b)
            hist["err"].append(b - b_true)
            hist["loss"].append(loss)
            hist["grad"].append(grad)
            hist["grad_sq"].append(grad_sq)
            hist["inner_cost"].append(float(np.mean(costs)))
            hist["time_iter"].append(t_iter)
            if cfg.record_diagnostics:
                hist["diag"].append({k: np.array(vv) for k, vv in diags.items()})
            n_iter += 1
            hist["n_iter"] = n_iter
            break
        hist["b"].append(b)
        hist["err"].append(b - b_true)
        hist["loss"].append(loss)
        hist["grad"].append(grad)
        hist["grad_sq"].append(grad_sq)
        hist["inner_cost"].append(float(np.mean(costs)))
        hist["time_iter"].append(t_iter)
        if cfg.record_diagnostics:
            hist["diag"].append({k: np.array(vv) for k, vv in diags.items()})
        n_iter += 1
        hist["n_iter"] = n_iter
    else:
        hist["status"] = "max_iter"
    if grad_sq <= cfg.tol_grad_sq and hist["status"] == "max_iter":
        hist["status"] = "converged"
    return hist
