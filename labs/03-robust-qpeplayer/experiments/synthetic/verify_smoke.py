#!/usr/bin/env python3
"""Known-truth / closed-form / finite-difference checks before the main tables.

Checks
------
[1] Kabsch on noiseless data: rotation and translation errors ~ machine eps.
[2] GNC at 0% outliers vs Kabsch: both success; poses nearly identical;
    the GM cost of the GNC solution is not larger than that of Kabsch.
[3] RANSAC on all-inlier data vs Kabsch: poses nearly identical.
[4] GNC mu-schedule: mu0 formula, monotone mu, final Phi norm, weights in [0,1].
[5] Coherent majority (70%): RANSAC max-consensus picks the *outlier* rigid
    transform; this is the documented model-identifiability failure
    (Prop 3.7), not an implementation bug.
[6] Exact Riemannian Hessian H_rob vs geodesic finite differences of F_rob.
[7] lambda_i = q^T g_{i,q} = 4 u_i^T W_i r_i != 4 e_i.
[8] Exact IFT Jacobian dz*/dy_i vs central FD; naive (frozen weight) vs FD.
[9] RANSAC adaptive stopping: empirical success vs
    1-(1-(1-p)^3)^k theory (k = iterations actually spent).

Run (Slurm):
  srun --partition=cpu --time=00:10:00 --ntasks=1 \
    /home/syk/miniconda3/envs/WM/bin/python verify_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def fd_hessian(x, y, q, t, Bq, h=1e-4, c=cm.C):
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)

    def Fval(qq, tt):
        terms = cm.robust_terms(x, y, qq, tt)
        rho, _, _ = cm.gm_terms(terms.e, c)
        return float(rho.sum())

    f0 = Fval(q, t)
    Q = np.zeros((6, 6))

    def qform(xi):
        qp, tp = cm.exp_map(q, t, h * xi, Bq)
        qm, tm = cm.exp_map(q, t, -h * xi, Bq)
        return (Fval(qp, tp) - 2.0 * f0 + Fval(qm, tm)) / (h * h)

    for j in range(6):
        for k in range(j, 6):
            ej = np.zeros(6)
            ek = np.zeros(6)
            ej[j] = 1.0
            ek[k] = 1.0
            Q[j, k] = Q[k, j] = (qform(ej + ek) - qform(ej - ek)) / 4.0
    return Q


def main():
    out = []

    def p(s=""):
        out.append(s)
        print(s, flush=True)

    p("=== Robust-QPEPLayer synthetic smoke / known-truth checks ===")
    p(f"python={sys.version.split()[0]} numpy={np.__version__}")
    p("")

    # ---------------------------------------------------------------- [1]
    p("[1] Kabsch on noiseless data (closed-form check)")
    rng = np.random.default_rng(101)
    max_re = max_te = max_fro = 0.0
    for _ in range(20):
        x, _, Rgt, tgt = cm._clean_instance(rng, 100)
        y = (Rgt @ x.T).T + tgt
        R, t = cm.kabsch(x, y, np.ones(len(x)))
        max_re = max(max_re, cm.rot_err_deg(R, Rgt))
        max_fro = max(max_fro, float(np.linalg.norm(R - Rgt)))
        max_te = max(max_te, float(np.linalg.norm(t - tgt)))
    p(f"    max ||R-Rgt||_F = {max_fro:.3e}; max translation error = {max_te:.3e} m")
    p(f"    arccos-based rotation error = {max_re:.3e} deg "
      f"(floor ~1e-6 deg from arccos precision near 1)")
    p(f"    PASS: {max_fro < 1e-10 and max_te < 1e-10}")
    p("")

    # ---------------------------------------------------------------- [2]
    p("[2] GNC at 0% outliers vs Kabsch (should be ~identical)")
    re_g, re_k, te_g, te_k, dR_pose, dt_pose, dF = [], [], [], [], [], [], []
    for trial in range(20):
        rng = np.random.default_rng([2026, 0, trial])
        x, y, Rgt, tgt = cm.make_instance(rng, 0.0)
        Rk, tk = cm.kabsch(x, y, np.ones(len(x)))
        Rg, tg = cm.solve_gnc(x, y)
        re_k.append(cm.rot_err_deg(Rk, Rgt))
        re_g.append(cm.rot_err_deg(Rg, Rgt))
        te_k.append(float(np.linalg.norm(tk - tgt)))
        te_g.append(float(np.linalg.norm(tg - tgt)))
        dR_pose.append(cm.rot_err_deg(Rg, Rk))
        dt_pose.append(float(np.linalg.norm(tg - tk)))
        qg = cm.quat_from_R(Rg)
        qk = cm.quat_from_R(Rk)
        Fg = cm.gm_terms(cm.robust_terms(x, y, qg, tg).e)[0].sum()
        Fk = cm.gm_terms(cm.robust_terms(x, y, qk, tk).e)[0].sum()
        dF.append(float(Fg - Fk))
    p(f"    median rot err: Kabsch {np.median(re_k):.4f} deg, GNC {np.median(re_g):.4f} deg")
    p(f"    median trans err: Kabsch {np.median(te_k):.5f} m, GNC {np.median(te_g):.5f} m")
    p(f"    pose difference GNC-vs-Kabsch: rot p50 {np.median(dR_pose):.3e} deg, "
      f"max {max(dR_pose):.3e} deg; trans p50 {np.median(dt_pose):.3e} m, "
      f"max {max(dt_pose):.3e} m")
    p(f"    max (F_gnc - F_kabsch) over 20 trials = {max(dF):.3e} (should be <= ~0)")
    max_fro = 0.0
    for trial in range(20):
        rng = np.random.default_rng([2026, 0, trial])
        x, y, Rgt, tgt = cm.make_instance(rng, 0.0)
        Rk, tk = cm.kabsch(x, y, np.ones(len(x)))
        Rg, tg = cm.solve_gnc(x, y)
        max_fro = max(max_fro, float(np.linalg.norm(Rg - Rk)))
    p(f"    max ||R_gnc - R_kabsch||_F = {max_fro:.3e}")
    p(f"    PASS: {np.median(dR_pose) < 0.05 and np.median(dt_pose) < 1e-3 and max(dF) <= 1e-12}")
    p("")

    # ---------------------------------------------------------------- [3]
    p("[3] RANSAC on all-inlier data vs Kabsch")
    dR, dt = [], []
    for trial in range(20):
        rng = np.random.default_rng([2026, 1, trial])
        x, y, Rgt, tgt = cm.make_instance(rng, 0.0)
        Rk, tk = cm.kabsch(x, y, np.ones(len(x)))
        rng_r = np.random.default_rng([77, trial])
        Rr, tr = cm.solve_ransac(x, y, rng_r, adaptive=True)
        dR.append(float(np.linalg.norm(Rr - Rk)))
        dt.append(float(np.linalg.norm(tr - tk)))
    p(f"    pose difference RANSAC-vs-Kabsch: ||dR||_F p50 {np.median(dR):.3e}, "
      f"|dt| p50 {np.median(dt):.3e} m")
    p(f"    PASS: {np.median(dR) < 1e-8 and np.median(dt) < 1e-10}")
    p("")

    # ---------------------------------------------------------------- [4]
    p("[4] GNC mu-schedule and final first-order residual")
    for p_out in [0.0, 0.3, 0.7]:
        rows = []
        for trial in range(10):
            rng = np.random.default_rng([2026, 2, trial])
            x, y, Rgt, tgt = cm.make_instance(rng, p_out)
            R, t, info = cm.solve_gnc(x, y, return_info=True)
            q = cm.quat_from_R(R)
            Bq = cm.tangent_basis(q)
            terms = cm.robust_terms(x, y, q, t)
            H, lam, gtot, Bfull = cm.robust_hessian_exact(terms, q, Bq)
            phi = float(np.linalg.norm(Bfull.T @ gtot))
            w = cm.gm_terms(terms.e)[1]
            rows.append((info["mu0"], info["n_mu"], phi, w.min(), w.max(), lam))
        arr = np.array(rows)
        p(f"    p={p_out:.1f}: mu0 {arr[:,0].min():.3g}..{arr[:,0].max():.3g}, "
          f"n_mu {arr[:,1].min():.0f}..{arr[:,1].max():.0f}, "
          f"||Phi|| p50 {np.median(arr[:,2]):.2e}, "
          f"w in [{arr[:,3].min():.3g},{arr[:,4].max():.3g}], "
          f"lambda_rob p50 {np.median(arr[:,5]):+.3f}")
    p("    NOTE: solve_gnc uses 5 exact-kernel IRLS steps, so ||Phi|| ~1e-6;")
    p("          the gradient experiments polish to 1e-13 with Newton.")
    p(f"    PASS: weights in [0,1] and ||Phi|| < 1e-4")
    p("")

    # ---------------------------------------------------------------- [5]
    p("[5] Coherent majority (70% outliers): which rigid transform wins?")
    re_out, re_gt, picks, rec_out, seps = [], [], [], [], []
    for trial in range(20):
        rng = np.random.default_rng([2026, 3, trial])
        x, y, Rgt, tgt, meta = cm.make_instance(rng, 0.7, outlier_type="coherent",
                                                return_meta=True)
        Rout, tout = meta["Rout"], meta["tout"]
        seps.append(cm.rot_err_deg(Rout, Rgt))
        Rr, tr = cm.solve_ransac(x, y, np.random.default_rng([2026, 4, trial]), adaptive=True)
        e_gt = cm.rot_err_deg(Rr, Rgt)
        t_gt = float(np.linalg.norm(tr - tgt))
        e_out = cm.rot_err_deg(Rr, Rout)
        t_out = float(np.linalg.norm(tr - tout))
        re_gt.append(e_gt)
        re_out.append(e_out)
        rec_out.append(float(e_out < cm.SUCCESS_ROT_DEG and t_out < cm.SUCCESS_TRANS))
        picks.append("outlier" if (e_out < e_gt and t_out < t_gt) else "gt")
    p(f"    GT-vs-outlier transform separation: rot p50 {np.median(seps):.2f} deg")
    p(f"    RANSAC pose error: to GT p50 {np.median(re_gt):.2f} deg, "
      f"to outlier transform p50 {np.median(re_out):.2f} deg")
    p(f"    picked outlier side in {picks.count('outlier')}/20 trials; "
      f"recovered outlier pose (<5 deg,<0.1 m) in {int(sum(rec_out))}/20")
    p(f"    PASS: {picks.count('outlier') >= 18 and sum(rec_out) >= 18}")
    p("")

    # ---------------------------------------------------------------- [6]
    p("[6] Exact H_rob vs geodesic FD Hessian of F_rob (N=30, boundary mix)")
    rng = np.random.default_rng(606)
    x, y, Rgt, tgt = cm.make_instance(rng, 0.4, n=30, outlier_type="boundary")
    Rg, tg = cm.solve_gnc(x, y)
    q, t, H, pn, nit = cm.newton_polish(x, y, cm.quat_from_R(Rg), tg)
    Bq = cm.tangent_basis(q)
    H_fd = fd_hessian(x, y, q, t, Bq, h=1e-5)
    rel = np.linalg.norm(H - H_fd) / max(1.0, np.linalg.norm(H_fd))
    H_fd2 = fd_hessian(x, y, q, t, Bq, h=5e-6)
    H_rich = (4.0 * H_fd2 - H_fd) / 3.0
    rel_rich = np.linalg.norm(H - H_rich) / max(1.0, np.linalg.norm(H_rich))
    eig = np.linalg.eigvalsh(H)
    p(f"    ||Phi|| after Newton = {pn:.2e} ({nit} Newton iters)")
    p(f"    ||H_rob - H_fd||_F / ||H_fd||_F = {rel:.3e}  (central FD, h=1e-5)")
    p(f"    ||H_rob - H_fd||_F / ||H_fd||_F = {rel_rich:.3e}  (Richardson extrapolation)")
    p(f"    eig(H_rob) = {np.array2string(eig, precision=4)}  (lambda_min = {eig.min():+.4e})")
    p(f"    PASS: {rel < 1e-5 and rel_rich < 1e-7 and eig.min() > 0}")
    p("")

    # ---------------------------------------------------------------- [6b]
    p("[6b] Same N=6 instance as verify/riemann_hessian_check.py, corrected build_A")
    rng6 = np.random.default_rng(7)
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    t_true = np.array([0.10, -0.10, 0.20])
    N6 = 6
    p6 = rng6.normal(size=(N6, 3))
    y6 = np.array([cm.Rmat(q_true) @ p6[i] + t_true + 0.10 * rng6.normal(size=3)
                   for i in range(N6)])
    W6 = []
    for _ in range(N6):
        A6 = rng6.normal(size=(3, 3))
        W6.append(A6.T @ A6 + 0.15 * np.eye(3))
    q6 = rng6.normal(size=4)
    q6 /= np.linalg.norm(q6)
    t6 = np.array([0.15, -0.12, 0.22])
    A6_list = [cm.build_A(p6[i]) for i in range(N6)]
    terms6 = cm.robust_terms(p6, y6, q6, t6, W=W6, A_list=A6_list)
    Bq6 = cm.tangent_basis(q6)
    H6, lam6, gtot6, B6 = cm.robust_hessian_exact(terms6, q6, Bq6, c=1.0)
    H6_doc = H6 + lam6 * np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def F6(qq, tt):
        return float(cm.gm_terms(cm.robust_terms(p6, y6, qq, tt, W=W6,
                                                 A_list=A6_list).e, c=1.0)[0].sum())

    f60 = F6(q6, t6)
    Q6 = np.zeros((6, 6))
    h6 = 1e-5

    def qf6(xi):
        qp, tp = cm.exp_map(q6, t6, h6 * xi, Bq6)
        qm, tm = cm.exp_map(q6, t6, -h6 * xi, Bq6)
        return (F6(qp, tp) - 2.0 * f60 + F6(qm, tm)) / (h6 * h6)

    for j in range(6):
        for k in range(j, 6):
            ej = np.zeros(6)
            ek = np.zeros(6)
            ej[j] = 1.0
            ek[k] = 1.0
            Q6[j, k] = Q6[k, j] = (qf6(ej + ek) - qf6(ej - ek)) / 4.0
    rel6 = np.linalg.norm(H6 - Q6) / max(1.0, np.linalg.norm(Q6))
    rel6_doc = np.linalg.norm(H6_doc - Q6) / max(1.0, np.linalg.norm(Q6))
    lam_i6 = np.array([q6 @ terms6.g[i, :4] for i in range(N6)])
    lam_u6 = np.array([4.0 * terms6.u[i] @ W6[i] @ terms6.r[i] for i in range(N6)])
    p(f"    lambda_rob = {lam6:+.6f}")
    p(f"    ||H_rob - H_fd||/||H_fd|| = {rel6:.3e}")
    p(f"    ||H_doc - H_fd||/||H_fd|| = {rel6_doc:.3e}  (missing -lambda)")
    p(f"    max |lambda_i - 4u^T W r| = {np.max(np.abs(lam_i6 - lam_u6)):.3e}")
    p(f"    max |lambda_i - 4 e_i|    = {np.max(np.abs(lam_i6 - 4*terms6.e)):.3e}")
    p(f"    PASS: {rel6 < 1e-4 and rel6_doc > 0.1}")
    p("")

    # ---------------------------------------------------------------- [7]
    p("[7] lambda_i identity: q^T g_{i,q} = 4 u_i^T W_i r_i, not 4 e_i")
    terms = cm.robust_terms(x, y, q, t)
    lam_i = np.array([q @ terms.g[i, :4] for i in range(len(x))])
    lam_u = np.array([4.0 * terms.u[i] @ terms.r[i] for i in range(len(x))])
    p(f"    max |lambda_i - 4 u_i^T W_i r_i| = {np.max(np.abs(lam_i - lam_u)):.3e}")
    p(f"    max |lambda_i - 4 e_i|           = {np.max(np.abs(lam_i - 4 * terms.e)):.3e}")
    p(f"    PASS: {np.max(np.abs(lam_i - lam_u)) < 1e-12}")
    p("")

    # ---------------------------------------------------------------- [8]
    p("[8] Exact IFT vs central FD vs naive (per-observation Jacobian dz*/dy_i)")
    terms = cm.robust_terms(x, y, q, t)
    H, lam, gtot, Bfull = cm.robust_hessian_exact(terms, q, Bq)
    Hn, lamn = cm.robust_hessian_naive(terms, q, Bq)
    rows = []
    for i in range(len(x)):
        RHS = cm.dphi_dy_exact(terms, i, Bfull)
        Jex = -np.linalg.solve(H, RHS)
        RHSn = cm.dphi_dy_naive(terms, i, Bfull)
        Jna = -np.linalg.solve(Hn, RHSn)
        Jfd = cm.fd_jacobian_y(x, y, i, q, t, Bq, h=1e-5)
        rows.append((terms.e[i] / cm.C2, np.linalg.norm(Jex - Jfd) / max(1e-12, np.linalg.norm(Jfd)),
                     np.linalg.norm(Jna - Jfd) / max(1e-12, np.linalg.norm(Jfd)),
                     float(np.sum(Jex * Jfd) / max(1e-12, np.linalg.norm(Jex) * np.linalg.norm(Jfd))),
                     float(np.sum(Jna * Jfd) / max(1e-12, np.linalg.norm(Jna) * np.linalg.norm(Jfd)))))
    arr = np.array(rows)
    for lo, hi, name in [(0, 1 / 3, "inlier e<c^2/3"), (1 / 3, 3, "boundary c^2/3..3c^2"),
                         (3, 1e9, "gross e>3c^2")]:
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.any():
            p(f"    {name:24s} n={m.sum():2d}  exact relerr p50 {np.median(arr[m,1]):.2e}  "
              f"cos p50 {np.median(arr[m,3]):+.6f} | naive relerr p50 {np.median(arr[m,2]):.2e}  "
              f"cos p50 {np.median(arr[m,4]):+.6f}")
    ok8 = bool(np.median(arr[:, 1]) < 1e-6 and np.median(arr[:, 3]) > 0.9999)
    p(f"    PASS: {ok8}")
    p("")

    # ---------------------------------------------------------------- [9]
    p("[9] Adaptive RANSAC: empirical success vs 1-(1-(1-p)^3)^k theory")
    for p_out in [0.1, 0.5, 0.7, 0.8]:
        succ, ks = [], []
        for trial in range(50):
            rng = np.random.default_rng([2026, 5, trial])
            x, y, Rgt, tgt = cm.make_instance(rng, p_out)
            Rr, tr, info = cm.solve_ransac(x, y, np.random.default_rng([2026, 6, trial]),
                                           adaptive=True, return_info=True)
            succ.append(float(cm.success(*cm.pose_errors(Rr, tr, Rgt, tgt))))
            ks.append(info["n_iters"])
        theory = np.mean(1.0 - (1.0 - (1.0 - p_out) ** 3) ** np.asarray(ks))
        p(f"    p={p_out:.1f}: success {100*np.mean(succ):6.1f}%  "
          f"mean iters {np.mean(ks):6.1f}  theory {100*theory:6.1f}%  (n=50)")
    p("")

    text = "\n".join(out)
    cm.write_text(OUT / "verify_smoke.txt", text + "\n")
    p(f"wrote {OUT / 'verify_smoke.txt'}")


if __name__ == "__main__":
    main()
