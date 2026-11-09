"""Jacobian validation and Table III/IV-style comparison.

Two sets of experiments:

**A. Jacobian of the solution w.r.t. the baseline** ``b``
    For 50 random problem instances (one ground-truth pose each, as in the
    paper's Fig. 3a) we compute

    * ``SDPR-IS``: Theorem-2 implicit differentiation of the QCQP, Eq. (16),
      reusing the SDP multipliers/certificate,
    * ``local-GT``: implicit differentiation of the local (Gauss-Newton)
      solution started at the ground truth,
    * a finite-difference reference of the SDP solution map, obtained with a
      much tighter SCS tolerance (the FD reference cannot be more accurate
      than the forward solver).

    The Eq. (29) relative Jacobian error ``||J_est - J_ref||_inf/||J_ref||_inf``
    is reported for SDPR-IS and local-GT.

**B. Jacobian of the solution w.r.t. the feature locations** ``m_k``
    This mirrors Tables III/IV of the paper (which compare Jacobians with
    respect to the feature locations).  Because the SDP and the local solver
    both find the global optimum when the local solver is initialised at the
    ground truth, their Jacobians should agree to high precision; we report
    the Eq. (29) difference of the full ``12 x 3N`` Jacobian, plus a
    finite-difference spot check of randomly chosen parameters.

Outputs: ``repro/results_clean/jacobians.json``, ``jacobians.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bilevel  # noqa: E402
import data_gen  # noqa: E402
import implicit_diff as idiff  # noqa: E402
import local_solver as ls  # noqa: E402
import sdp_pose  # noqa: E402
from camera_model import skew  # noqa: E402

C_IDX = sdp_pose.C_IDX
T_IDX = sdp_pose.T_IDX


# ----------------------------------------------------------------------
# helpers: derivatives of the QCQP objective matrix
# ----------------------------------------------------------------------
def Q_of_b(cam, pixels, m, b):
    meas, W = cam.inverse(pixels, b)
    return sdp_pose.build_objective_matrix(m, meas, W)


def dQ_db(cam, pixels, m, b, hc=1e-20):
    meas, W = cam.inverse(pixels, b + 1j * hc)
    Qc = sdp_pose.build_objective_matrix(m, meas, W)
    dQ = np.imag(Qc) / hc
    return 0.5 * (dQ + dQ.T)


def dQ_dm(m, y, W, k, j):
    """Derivative of the objective matrix w.r.t. ``m_k[j]`` (13x13)."""
    dQ = np.zeros((sdp_pose.N_VARS, sdp_pose.N_VARS))
    mk = m[:, k]
    Wk = W[k]
    ej = np.eye(3)[j]
    M = np.outer(ej, mk) + np.outer(mk, ej)          # d(m m^T)/dm_kj
    blk = np.einsum("il,ab->ialb", M, Wk).reshape(9, 9)
    dQ[1:10, 1:10] = blk
    Wy = Wk @ y[:, k]
    dQ[1 + 3 * j:4 + 3 * j, 0] = -Wy
    dQ[0, 1 + 3 * j:4 + 3 * j] = -Wy
    dQ[1 + 3 * j:4 + 3 * j, 10:13] = -Wk
    dQ[10:13, 1 + 3 * j:4 + 3 * j] = -Wk.T
    return dQ


# ----------------------------------------------------------------------
# SDPR Jacobians
# ----------------------------------------------------------------------
def sdp_pose_and_reference(cam, pixels, m, b, solver, tight_solver=None):
    Q = Q_of_b(cam, pixels, m, b)
    scale = sdp_pose.objective_scale(Q)
    qs = sdp_pose.scale_objective(Q, scale)
    rec = solver.solve(qs)
    if tight_solver is not None:
        rec_t = tight_solver.solve(qs)
    else:
        rec_t = rec
    return rec, rec_t, scale


def sdpr_jacobian_b(cam, pixels, m, b, rec, scale, solver):
    """J_b of the QCQP solution (Theorem 2)."""
    dQ = scale * dQ_db(cam, pixels, m, b)
    dQ[0, 0] = 0.0
    G_r, _ = solver.reduced_gradient_rows(rec["x"])
    jac = idiff.qcqp_jacobian(rec["H"], G_r, rec["x"], dQ)
    dC, dr = idiff.pose_jacobian_from_x(rec["x"], jac["dx"])
    return dC, dr


def sdpr_jacobian_m(cam, pixels, m, b, rec, scale, solver):
    """J_m (12 x 3N) of the QCQP solution, columns ordered (landmark, axis)."""
    meas, W = cam.inverse(pixels, b)
    y = meas
    N = m.shape[1]
    dQm = [dQ_dm(m, y, W, k, j) for k in range(N) for j in range(3)]
    G_r, _ = solver.reduced_gradient_rows(rec["x"])
    M = idiff.reduced_kkt_matrix(rec["H"], G_r)
    M_inv = np.linalg.inv(M)
    n = sdp_pose.N_VARS
    J = np.zeros((12, 3 * N))
    for idx, dQ in enumerate(dQm):
        dQ = scale * dQ
        dQ[0, 0] = 0.0
        rhs = np.zeros(n + G_r.shape[0])
        rhs[:n] = -(dQ @ rec["x"])
        dx = M_inv @ rhs
        dC, dr = idiff.pose_jacobian_from_x(rec["x"], dx[:n])
        J[:9, idx] = dC.reshape(-1)
        J[9:, idx] = dr.reshape(-1)
    return J


# ----------------------------------------------------------------------
# local Jacobians
# ----------------------------------------------------------------------
def local_solve(cam, pixels, m, b, cfg):
    meas, W = cam.inverse(pixels, b)
    res = ls.solve_pose_local(
        m, meas, W, cfg.C_gt, cfg.r_gt, max_iter=cfg.gn_max_iter,
        step_size=cfg.gn_step_size, abs_tol=cfg.gn_abs_tol,
        rel_tol=cfg.gn_rel_tol,
    )
    res = ls.solve_pose_local(
        m, meas, W, res["C"], res["r_p"], max_iter=cfg.gn_polish_iter,
        step_size=1.0, abs_tol=1e-15, rel_tol=1e-15,
    )
    return res


def _mat_inf_rel_diff(A, B):
    """Eq. (29) with the matrix infinity norm (max absolute row sum)."""
    num = np.max(np.sum(np.abs(A - B), axis=1))
    den = np.max(np.sum(np.abs(B), axis=1))
    return float(num / max(den, 1e-300))


def _local_J(prob, C, r_p, eps=1e-6):
    """Jacobian of the local stationarity conditions, 6x6."""
    J = np.zeros((6, 6))
    from camera_model import exp_so3

    for i in range(3):
        Cp = exp_so3(eps * np.eye(3)[i]) @ C
        Cm = exp_so3(-eps * np.eye(3)[i]) @ C
        J[:, i] = (prob.gradient(Cp, r_p) - prob.gradient(Cm, r_p)) / (2 * eps)
    for i in range(3):
        rp = r_p.copy()
        rm = r_p.copy()
        rp[i] += eps
        rm[i] -= eps
        J[:, 3 + i] = (prob.gradient(C, rp) - prob.gradient(C, rm)) / (2 * eps)
    return J


def local_jacobian_b(cam, pixels, m, b, res):
    der = ls.local_implicit_derivative(
        m, res["C"], res["r_p"],
        meas_fn=lambda bb: cam.inverse(pixels, bb)[0],
        W_fn=lambda bb: cam.inverse(pixels, bb)[1],
        b=b,
    )
    return der["dC"], der["dr_p"], der["cond"]


def local_jacobian_m(cam, pixels, m, b, res):
    meas, W = cam.inverse(pixels, b)
    prob = ls._LocalProblem(m, meas, W)
    C, r_p = res["C"], res["r_p"]
    J = _local_J(prob, C, r_p)
    J_inv = np.linalg.inv(J)
    N = m.shape[1]
    v = C @ (m - r_p.reshape(3)[:, None])
    e = meas - v
    We = np.einsum("nij,ni->nj", W, e.T)
    Jm = np.zeros((12, 3 * N))
    for k in range(N):
        for j in range(3):
            Cej = C[:, j]
            wc = W[k] @ Cej
            dg_d = 2.0 * (-np.cross(wc, v[:, k]) + np.cross(We[k], Cej))
            dg_r = -2.0 * (C.T @ wc)
            dg = np.concatenate([dg_d, dg_r])
            sol = -J_inv @ dg
            dC = skew(sol[:3]) @ C
            idx = 3 * k + j
            Jm[:9, idx] = dC.reshape(-1)
            Jm[9:, idx] = sol[3:]
    return Jm


# ----------------------------------------------------------------------
# main comparison loop
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--fd-trials", type=int, default=10)
    ap.add_argument("--fd-params", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir or os.path.join(here, "..", "results_clean"))
    os.makedirs(out_dir, exist_ok=True)

    cfg = bilevel.ExpConfig(n_batch=1)
    cam = bilevel.make_camera(cfg)
    b_true = cfg.b_true
    h_fd = 1e-3 * b_true

    solver = sdp_pose.SDPPoseSolver(solver="SCS", eps=1e-9)
    tight = sdp_pose.SDPPoseSolver(solver="SCS", eps=1e-12, max_iters=500000)

    rows = []
    print(f"Jacobian validation: {args.trials} trials")
    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed0 * 1000 + trial)
        data = data_gen.get_calibration_data(n_batch=1, rng=rng, cam=cam, b_true=b_true)
        m = data["r_l"]
        pixels = data["pixels"][0]
        cfg.C_gt = data["C_p0s"][0]
        cfg.r_gt = data["r_p0s"][0]

        # --- SDPR -----------------------------------------------------
        t0 = time.perf_counter()
        rec, rec_t, scale = sdp_pose_and_reference(
            cam, pixels, m, b_true, solver, tight_solver=None)
        t_solve = time.perf_counter() - t0
        t0 = time.perf_counter()
        dC_s, dr_s = sdpr_jacobian_b(cam, pixels, m, b_true, rec, scale, solver)
        t_bwd_sdpr = time.perf_counter() - t0
        Jm_sdpr = sdpr_jacobian_m(cam, pixels, m, b_true, rec, scale, solver)

        # --- FD reference of the SDP solution map --------------------
        h = h_fd
        rec_p = tight.solve(sdp_pose.scale_objective(Q_of_b(cam, pixels, m, b_true + h),
                                                    scale))
        rec_m = tight.solve(sdp_pose.scale_objective(Q_of_b(cam, pixels, m, b_true - h),
                                                    scale))
        Cp, rp = idiff.pose_from_x(rec_p["x"])
        Cm, rm = idiff.pose_from_x(rec_m["x"])
        dC_fd = (Cp - Cm) / (2 * h)
        dr_fd = (rp - rm) / (2 * h)

        # --- local ----------------------------------------------------
        t0 = time.perf_counter()
        res = local_solve(cam, pixels, m, b_true, cfg)
        t_local_solve = time.perf_counter() - t0
        t0 = time.perf_counter()
        dC_l, dr_l, cond_l = local_jacobian_b(cam, pixels, m, b_true, res)
        t_bwd_loc = time.perf_counter() - t0
        Jm_loc = local_jacobian_m(cam, pixels, m, b_true, res)

        # --- metrics --------------------------------------------------
        row = dict(
            trial=trial,
            sdpr_jac_err_vs_fd=idiff.relative_jacobian_difference(dC_s, dr_s, dC_fd, dr_fd),
            local_jac_err_vs_fd=idiff.relative_jacobian_difference(dC_l, dr_l, dC_fd, dr_fd),
            sdpr_vs_local_b=idiff.relative_jacobian_difference(dC_s, dr_s, dC_l, dr_l),
            jac_err_b_fd_norm=np.max(np.abs(np.concatenate(
                [(dC_s - dC_fd).reshape(-1), (dr_s - dr_fd).reshape(-1)]))),
            jac_err_b_scale=np.max(np.abs(np.concatenate(
                [dC_fd.reshape(-1), dr_fd.reshape(-1)]))),
            jac_diff_m_linf=_mat_inf_rel_diff(Jm_sdpr, Jm_loc),
            jac_diff_m_mean_abs=float(np.mean(np.abs(Jm_sdpr - Jm_loc))),
            t_sdp_solve=t_solve,
            t_backward_sdpr=t_bwd_sdpr,
            t_local_solve=t_local_solve,
            t_backward_local=t_bwd_loc,
            cond_local=cond_l,
            r_tight=rec["r_tight"],
            corank_H=rec["corank_H"],
            corank_H_strict=rec["corank_H_strict"],
            H_min_eig=rec["H_min_eig"],
            Hx_res=rec["Hx_res"],
            pose_diff_trans=float(np.linalg.norm(idiff.pose_from_x(rec["x"])[1]
                                                 - res["r_p"])),
            pose_diff_rot=float(np.linalg.norm(
                idiff.pose_from_x(rec["x"])[0].T @ res["C"] - np.eye(3))),
        )

        # --- FD spot check of the m-Jacobian -------------------------
        row["fd_check_m_sdpr"] = np.nan
        row["fd_check_m_local"] = np.nan
        if trial < args.fd_trials:
            N = m.shape[1]
            idxs = rng.choice(3 * N, size=min(args.fd_params, 3 * N), replace=False)
            errs_s, errs_l, scales = [], [], []
            for idx in idxs:
                k, j = divmod(int(idx), 3)
                hm = 1e-5
                mp = m.copy()
                mm = m.copy()
                mp[j, k] += hm
                mm[j, k] -= hm
                rpp = tight.solve(sdp_pose.scale_objective(Q_of_b(cam, pixels, mp, b_true), scale))
                rmm = tight.solve(sdp_pose.scale_objective(Q_of_b(cam, pixels, mm, b_true), scale))
                Cpp, rpp_ = idiff.pose_from_x(rpp["x"])
                Cmm, rmm_ = idiff.pose_from_x(rmm["x"])
                col_fd = np.concatenate([((Cpp - Cmm) / (2 * hm)).reshape(-1),
                                         (rpp_ - rmm_) / (2 * hm)])
                col_s = np.concatenate([Jm_sdpr[:9, idx], Jm_sdpr[9:, idx]])
                col_l = np.concatenate([Jm_loc[:9, idx], Jm_loc[9:, idx]])
                errs_s.append(np.max(np.abs(col_s - col_fd)))
                errs_l.append(np.max(np.abs(col_l - col_fd)))
                scales.append(np.max(np.abs(col_fd)))
            sc = np.mean(scales)
            row["fd_check_m_sdpr"] = float(np.mean(errs_s) / sc)
            row["fd_check_m_local"] = float(np.mean(errs_l) / sc)

        rows.append(row)
        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1}/{args.trials}: "
                  f"sdpr-vs-fd {row['sdpr_jac_err_vs_fd']:.2e}, "
                  f"local-vs-fd {row['local_jac_err_vs_fd']:.2e}, "
                  f"m-jac diff {row['jac_diff_m_linf']:.2e}")

    # ---- summary ------------------------------------------------------
    keys = ["sdpr_jac_err_vs_fd", "local_jac_err_vs_fd", "sdpr_vs_local_b",
            "jac_diff_m_linf", "jac_diff_m_mean_abs", "fd_check_m_sdpr",
            "fd_check_m_local", "t_sdp_solve", "t_backward_sdpr",
            "t_local_solve", "t_backward_local", "r_tight", "corank_H",
            "corank_H_strict", "H_min_eig", "Hx_res", "pose_diff_trans",
            "pose_diff_rot", "cond_local"]
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        summary[k] = dict(mean=float(np.mean(v)) if v.size else float("nan"),
                          std=float(np.std(v)) if v.size else float("nan"),
                          median=float(np.median(v)) if v.size else float("nan"),
                          max=float(np.max(v)) if v.size else float("nan"))
    summary["n_trials"] = len(rows)
    summary["settings"] = dict(
        b_true=b_true,
        fd_step_b=h_fd,
        sdp_solver="SCS eps=1e-9 (1e-12 for FD reference)",
        local_solver="LM, step 0.2, rel tol 1e-6 + tight polish",
    )
    with open(os.path.join(out_dir, "jacobians.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=2, default=float)
    import csv

    with open(os.path.join(out_dir, "jacobians.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSummary (mean +- std):")
    for k in keys:
        s = summary[k]
        print(f"  {k:26s} {s['mean']:.3e} +- {s['std']:.3e}  (median {s['median']:.3e})")
    print(f"\nwrote {out_dir}/jacobians.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
