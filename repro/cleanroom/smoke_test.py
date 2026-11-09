"""Self-checks for the clean-room implementation.

Run with

    /home/syk/miniconda3/envs/sdpr-clean/bin/python repro/cleanroom/smoke_test.py

Checks (each printed with PASS/FAIL):

1. the 24 homogeneous constraints vanish on ``{(1, C, t) : C in SO(3)}``,
2. ``x^T Q x`` equals the matrix-weighted cost of Problem (P6),
3. the inverse-model Jacobian ``G`` matches finite differences, and
4. the SDP relaxation is tight, its certificate ``H`` is PSD with
   ``H x = 0`` and primal/dual objectives agree,
5. analytic implicit Jacobian ``dx/db`` matches finite differences,
6. the analytic outer-loss gradient matches finite differences,
7. the loss gradient w.r.t. ``x`` matches finite differences,
8. single-solve and backward timings.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_gen  # noqa: E402
import implicit_diff as idiff  # noqa: E402
import sdp_pose  # noqa: E402
from camera_model import StereoCamera, exp_so3  # noqa: E402

TOL = 1e-8
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    rng = np.random.default_rng(7)
    cam = StereoCamera()
    b_true = 0.24

    # ---------------------------------------------------------------
    # 1. constraints vanish on the feasible set
    # ---------------------------------------------------------------
    A_list = sdp_pose.constraint_matrices()
    A0 = sdp_pose.homogenising_matrix()
    worst = 0.0
    for _ in range(20):
        C = exp_so3(rng.standard_normal(3))
        x = np.concatenate([[1.0], C.T.reshape(9), rng.standard_normal(3)])
        worst = max(worst, max(abs(x @ A @ x) for A in A_list))
        assert abs(x @ A0 @ x - 1.0) < 1e-14
    check("constraints vanish on SO(3)", worst < 1e-12, f"max|x^T A x|={worst:.2e}")

    # rank of the full constraint Jacobian (should be codim = 7)
    C = exp_so3(rng.standard_normal(3))
    x = np.concatenate([[1.0], C.T.reshape(9), rng.standard_normal(3)])
    G_all = np.vstack([A @ x for A in A_list] + [A0 @ x])
    sv = np.linalg.svd(G_all, compute_uv=False)
    r_full = int(np.sum(sv > 1e-8 * sv[0]))
    check("constraint Jacobian rank == codim(Omega) == 7", r_full == 7,
          f"rank={r_full} sv={np.round(sv[:9], 3)}")

    # ---------------------------------------------------------------
    # 2. objective matrix
    # ---------------------------------------------------------------
    data = data_gen.get_calibration_data(n_batch=1, rng=rng, cam=cam, b_true=b_true)
    pixels = data["pixels"][0]
    m = data["r_l"]
    C_gt = data["C_p0s"][0]
    r_gt = data["r_p0s"][0].reshape(3)
    meas, W = cam.inverse(pixels, b_true)
    Q = sdp_pose.build_objective_matrix(m, meas, W)
    C = exp_so3(rng.standard_normal(3))
    t = rng.standard_normal(3)
    x = np.concatenate([[1.0], C.T.reshape(9), t])
    e = (meas - C @ m + t[:, None]).T
    cost_manual = np.einsum("nij,ni,nj->", W, e, e)
    check("x^T Q x == matrix-weighted cost",
          abs(x @ Q @ x - cost_manual) < 1e-10 * max(1.0, abs(cost_manual)),
          f"{x @ Q @ x:.6e} vs {cost_manual:.6e}")

    # ---------------------------------------------------------------
    # 3. inverse-model Jacobian vs finite differences
    # ---------------------------------------------------------------
    eps = 1e-7
    G = cam.inverse_jacobian(pixels, b_true, meas=meas)
    # G is d m / d(u_l, v_l, d);  pixels are (u_l, v_l, u_r, v_r)
    err = 0.0
    for j in range(4):
        pp = np.array(pixels, dtype=float)
        pm = np.array(pixels, dtype=float)
        pp[j] += eps
        pm[j] -= eps
        mp, _ = cam.inverse_meas_only(pp, b_true)
        mm, _ = cam.inverse_meas_only(pm, b_true)
        dmd = (mp - mm) / (2 * eps)
        col = np.zeros((pixels.shape[1], 3))
        if j == 0:
            col = G[:, :, 0] + G[:, :, 2]
        elif j == 1:
            col = G[:, :, 1]
        elif j == 2:
            col = -G[:, :, 2]
        err = max(err, np.max(np.abs(dmd.T - col)))
    check("inverse-model Jacobian vs FD", err < 1e-6, f"max err={err:.2e}")

    Sig = cam.pixel_covariance()
    check("pixel covariance PSD", np.all(np.linalg.eigvalsh(Sig) > 0),
          f"eig={np.round(np.linalg.eigvalsh(Sig), 4)}")
    check("W symmetric positive definite",
          np.allclose(W, np.swapaxes(W, 1, 2)) and np.all(np.linalg.eigvalsh(W) > 0), "")

    # ---------------------------------------------------------------
    # 4. SDP solve (paper setting: SCS, 1e-9)
    # ---------------------------------------------------------------
    solver = sdp_pose.SDPPoseSolver(solver="SCS", eps=1e-9)
    scale = sdp_pose.objective_scale(Q)
    Qs0 = sdp_pose.scale_objective(Q, scale)
    rec = solver.solve(Qs0)
    print(f"    SDP: status={rec['status']} r_tight={rec['r_tight']:.3e} "
          f"corank(H)={rec['corank_H']} H_min_eig={rec['H_min_eig']:.2e} "
          f"||Hx||={rec['Hx_res']:.2e} primal={rec['primal_obj']:.8f} "
          f"dual={rec['dual_obj']:.8f} t_solve={1e3*rec['t_solve']:.2f}ms")
    check("SDP tight (eig ratio > 1e5)", rec["r_tight"] > 1e5, f"{rec['r_tight']:.2e}")
    check("certificate PSD", rec["H_min_eig"] > -1e-7, f"{rec['H_min_eig']:.2e}")
    check("certificate Hx == 0", rec["Hx_res"] < 1e-5, f"{rec['Hx_res']:.2e}")
    check("strong duality (primal == dual)",
          abs(rec["primal_obj"] - rec["dual_obj"]) < 1e-6 * max(1, abs(rec["primal_obj"])),
          f"{rec['primal_obj']:.8f} vs {rec['dual_obj']:.8f}")
    Cr, rp = idiff.pose_from_x(rec["x"])
    check("orthogonality of recovered C",
          np.linalg.norm(Cr.T @ Cr - np.eye(3)) < 1e-6, f"{np.linalg.norm(Cr.T @ Cr - np.eye(3)):.2e}")

    # ---------------------------------------------------------------
    # 5. implicit Jacobian vs finite differences
    #    (reference solves use a much tighter SCS tolerance, because the
    #     finite-difference reference cannot be more accurate than the
    #     forward solver)
    # ---------------------------------------------------------------
    b0 = b_true + 3e-3     # the experiment's initial baseline

    def Q_of_b(bb):
        meas_, W_ = cam.inverse(pixels, bb)
        return sdp_pose.build_objective_matrix(m, meas_, W_)

    Q0 = Q_of_b(b0)
    scale = sdp_pose.objective_scale(Q0)
    sdp_ref = sdp_pose.SDPPoseSolver(solver="SCS", eps=1e-12, max_iters=500000)
    sdp_ref_lo = sdp_pose.SDPPoseSolver(solver="SCS", eps=1e-9)
    rec = sdp_ref_lo.solve(sdp_pose.scale_objective(Q0, scale))
    rec_ref = sdp_ref.solve(sdp_pose.scale_objective(Q0, scale))

    # dQ/db by the complex-step method (the model is analytic in b)
    hc = 1e-20
    Qc = Q_of_b(b0 + 1j * hc)
    dQ_db = np.imag(Qc) / hc
    dQ_db = 0.5 * (dQ_db + dQ_db.T)
    dQ_scaled = scale * dQ_db
    dQ_scaled[0, 0] = 0.0

    G_r, _ = solver.reduced_gradient_rows(rec["x"])
    jac = idiff.qcqp_jacobian(rec["H"], G_r, rec["x"], dQ_scaled)
    dC_a, dr_a = idiff.pose_jacobian_from_x(rec["x"], jac["dx"])
    hfd = 1e-3 * b0

    def solve_pose(bb, use_ref=True):
        s = sdp_ref if use_ref else sdp_ref_lo
        rb = s.solve(sdp_pose.scale_objective(Q_of_b(bb), scale))
        return idiff.pose_from_x(rb["x"])

    Cpp, rpp = solve_pose(b0 + hfd)
    Cmm, rmm = solve_pose(b0 - hfd)
    rel = idiff.relative_jacobian_difference(
        dC_a, dr_a, (Cpp - Cmm) / (2 * hfd), (rpp.reshape(3) - rmm.reshape(3)) / (2 * hfd))
    # Jacobian built from the tighter solver's certificate
    G_r_ref, _ = sdp_ref.reduced_gradient_rows(rec_ref["x"])
    jac_ref = idiff.qcqp_jacobian(rec_ref["H"], G_r_ref, rec_ref["x"], dQ_scaled)
    dC_r, dr_r = idiff.pose_jacobian_from_x(rec_ref["x"], jac_ref["dx"])
    rel_ref = idiff.relative_jacobian_difference(
        dC_r, dr_r, (Cpp - Cmm) / (2 * hfd), (rpp.reshape(3) - rmm.reshape(3)) / (2 * hfd))
    print(f"    Jacobian vs FD: rel diff={rel:.2e} (SCS 1e-9) / {rel_ref:.2e} (SCS 1e-12);"
          f" cond(M)={jac['cond']:.1e}, rank(Gr)={np.linalg.matrix_rank(G_r)}")
    check("implicit Jacobian vs FD (tight reference)", rel_ref < 1e-5, f"rel={rel_ref:.2e}")
    check("implicit Jacobian vs FD (paper solver)", rel < 1e-3, f"rel={rel:.2e}")

    # ---------------------------------------------------------------
    # 6. backward rule vs FD of the outer loss
    # ---------------------------------------------------------------
    loss, grad_x = idiff.loss_from_x_and_grad(rec["x"], C_gt, r_gt)

    def loss_of_b(bb):
        C_, r_ = solve_pose(bb)
        return idiff.loss_pose_and_grad(C_, r_, C_gt, r_gt)[0]

    g_fd = (loss_of_b(b0 + hfd) - loss_of_b(b0 - hfd)) / (2 * hfd)
    G_r, _ = sdp_ref_lo.reduced_gradient_rows(rec["x"])
    bwd = idiff.qcqp_backward(rec["H"], G_r, rec["x"], dQ_scaled, grad_x)
    relg = abs(bwd["grad_theta"] - g_fd) / max(1e-12, abs(g_fd))
    check("backward rule == Jacobian contraction",
          abs(bwd["grad_theta"] - bwd["grad_theta_direct"]) < 1e-8 *
          max(1.0, abs(bwd["grad_theta"])), f"{bwd['grad_theta']:.10f}")
    print(f"    outer-loss gradient: analytic={bwd['grad_theta']:.6e} fd={g_fd:.6e} rel={relg:.2e}")
    check("outer-loss gradient vs FD", relg < 1e-3, f"rel={relg:.2e}")

    # ---------------------------------------------------------------
    # 7. grad_x vs FD
    # ---------------------------------------------------------------
    gx = np.zeros(13)
    for i in range(13):
        xp = np.array(rec["x"], dtype=float)
        xm = np.array(rec["x"], dtype=float)
        xp[i] += 1e-7
        xm[i] -= 1e-7
        lp = idiff.loss_from_x_and_grad(xp, C_gt, r_gt)[0]
        lm = idiff.loss_from_x_and_grad(xm, C_gt, r_gt)[0]
        gx[i] = (lp - lm) / 2e-7
    check("loss grad w.r.t. x vs FD",
          np.max(np.abs(gx - grad_x)) < 1e-4 * max(1.0, np.max(np.abs(gx))),
          f"max err={np.max(np.abs(gx-grad_x)):.2e}")

    # ---------------------------------------------------------------
    # 8. timing
    # ---------------------------------------------------------------
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        solver.solve(sdp_pose.scale_objective(Q0, scale))
    t_solve = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        idiff.qcqp_backward(rec["H"], G_r, rec["x"], dQ_scaled, grad_x)
    t_bwd = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        Q_of_b(b0)
    t_q = (time.perf_counter() - t0) / n
    print(f"    timing: Q build {1e3*t_q:.3f} ms, SDP solve {1e3*t_solve:.2f} ms, "
          f"backward {1e3*t_bwd:.3f} ms")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)-n_fail}/{len(results)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
