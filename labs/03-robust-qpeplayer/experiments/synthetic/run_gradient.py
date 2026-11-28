#!/usr/bin/env python3
"""6D gradient table: exact IFT vs Naive IRLS-StopGrad vs finite differences.

Setting
-------
N=30, c=0.05, sigma=0.01; 10 instances with 40% boundary outliers
(e in [c^2/3, 3c^2]) and 10 instances with 20% gross random outliers, so
that all three bins (inlier / boundary / gross) are populated.  Each
instance is solved by GNC-GM and polished by Newton to ||Phi|| ~ 1e-13.

For every observation i we compare the 6x3 Jacobian dz*/dy_i:

* exact      : -H_rob^{-1} dPhi/dy_i            (Theorem 3.2/3.3)
* naive      : -H_naive^{-1} dPhi_naive/dy_i    (freeze w_i = rho'(e_i))
* no-lambda  : -H_doc^{-1} dPhi/dy_i            (drop the -lambda_rob term)
* FD         : central difference of the polished fixed-point branch
               (h=1e-5; the fixed point is re-solved to 1e-13 each side)

Metrics per bin: relative Frobenius error, cosine, and the sign of the
radial response (mapping-level and pose-level translation).

Outputs
-------
results/gradient_table.json
results/gradient_table.md
results/gradient_table.png
results/gradient_sign.png

Run (Slurm):
  srun --partition=cpu --time=00:30:00 --ntasks=1 \
    /home/syk/miniconda3/envs/WM/bin/python run_gradient.py 2>&1 | \
    tee results/logs/gradient_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

BINS = [(0.0, 1.0 / 3.0, "inlier (<c^2/3)"),
        (1.0 / 3.0, 3.0, "boundary (c^2/3..3c^2)"),
        (3.0, 1e9, "gross (>3c^2)")]
FD_H = 1e-5


def hessian_nolambda(terms, q, Bq):
    n = len(terms.e)
    _, drho, d2rho = cm.gm_terms(terms.e)
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)
    Htot = np.zeros((7, 7))
    for i in range(n):
        Htot += drho[i] * terms.H[i] + d2rho[i] * np.outer(terms.g[i], terms.g[i])
    return Bfull.T @ Htot @ Bfull, Bfull


def prepare(seed, p, otype, n=30):
    rng = np.random.default_rng(seed)
    x, y, Rgt, tgt = cm.make_instance(rng, p, n=n, outlier_type=otype)
    Rg, tg = cm.solve_gnc(x, y)
    q = cm.quat_from_R(Rg)
    A_list = [cm.build_A(x[i]) for i in range(n)]
    q, t, H, pn, nit = cm.newton_polish(x, y, q, tg, A_list=A_list)
    Bq = cm.tangent_basis(q)
    terms = cm.robust_terms(x, y, q, t, A_list=A_list)
    H, lam_rob, gtot, Bfull = cm.robust_hessian_exact(terms, q, Bq)
    Hn, lamn = cm.robust_hessian_naive(terms, q, Bq)
    Hnl, _ = hessian_nolambda(terms, q, Bq)
    return dict(x=x, y=y, Rgt=Rgt, tgt=tgt, q=q, t=t, A=A_list, Bq=Bq,
                terms=terms, H=H, Hn=Hn, Hnl=Hnl, Bfull=Bfull,
                pn=pn, nit=nit, lam_rob=lam_rob, lamn=lamn)


def analyse_instance(inst, fd_h=FD_H):
    terms = inst["terms"]
    H, Hn, Hnl, Bfull = inst["H"], inst["Hn"], inst["Hnl"], inst["Bfull"]
    rows = []
    for i in range(len(terms.e)):
        RHS = cm.dphi_dy_exact(terms, i, Bfull)
        RHSn = cm.dphi_dy_naive(terms, i, Bfull)
        Jex = -np.linalg.solve(H, RHS)
        Jna = -np.linalg.solve(Hn, RHSn)
        Jnl = -np.linalg.solve(Hnl, RHS)
        Jfd = cm.fd_jacobian_y(inst["x"], inst["y"], i, inst["q"], inst["t"],
                               inst["Bq"], h=fd_h, A_list=inst["A"])
        e = terms.e[i]
        rnorm = float(np.linalg.norm(terms.r[i]))
        if rnorm > 1e-12:
            rhat = terms.r[i] / rnorm
        else:
            rhat = np.zeros(3)
        a_ex = float(rhat @ RHS[3:, :] @ rhat)
        a_na = float(rhat @ RHSn[3:, :] @ rhat)
        b_ex = float(rhat @ Jex[3:, :] @ rhat)
        b_na = float(rhat @ Jna[3:, :] @ rhat)
        row = dict(i=i, e=e, e_over_c2=e / cm.C2, rnorm=rnorm,
                   lam_parallel=float(cm.lambda_parallel(e)),
                   radial_mapping_exact=a_ex, radial_mapping_naive=a_na,
                   radial_pose_exact=b_ex, radial_pose_naive=b_na)
        for name, J in [("exact", Jex), ("naive", Jna), ("nolambda", Jnl)]:
            d = J - Jfd
            nfd = float(np.linalg.norm(Jfd))
            nd = float(np.linalg.norm(d))
            row[f"relerr_{name}"] = nd / max(nfd, 1e-12)
            row[f"abs_{name}"] = nd
            denom = float(np.linalg.norm(J) * nfd)
            row[f"cos_{name}"] = float(np.sum(J * Jfd) / denom) if denom > 1e-18 else float("nan")
            row[f"norm_{name}"] = float(np.linalg.norm(J))
        row["norm_fd"] = float(np.linalg.norm(Jfd))
        rows.append(row)
    return rows


def summarize(rows):
    out = []
    for lo, hi, label in BINS:
        m = [r for r in rows if lo <= r["e_over_c2"] < hi]
        if not m:
            continue
        for method in ["exact", "naive", "nolambda"]:
            rel = np.array([r[f"relerr_{method}"] for r in m])
            cos = np.array([r[f"cos_{method}"] for r in m])
            cos = cos[np.isfinite(cos)]
            out.append(dict(bin=label, method=method, n=len(m),
                            relerr_p50=float(np.median(rel)),
                            relerr_p90=float(np.percentile(rel, 90)),
                            cos_p50=float(np.median(cos)) if len(cos) else float("nan"),
                            cos_min=float(np.min(cos)) if len(cos) else float("nan")))
    return out


def step_size_table(inst, obs_idx):
    x, y = inst["x"], inst["y"]
    terms = inst["terms"]
    RHS = cm.dphi_dy_exact(terms, obs_idx, inst["Bfull"])
    RHSn = cm.dphi_dy_naive(terms, obs_idx, inst["Bfull"])
    Jex = -np.linalg.solve(inst["H"], RHS)
    Jna = -np.linalg.solve(inst["Hn"], RHSn)
    rows = []
    for h in [1e-4, 1e-5, 1e-6, 1e-7]:
        Jfd = cm.fd_jacobian_y(x, y, obs_idx, inst["q"], inst["t"], inst["Bq"],
                               h=h, A_list=inst["A"])
        nfd = np.linalg.norm(Jfd)
        rows.append(dict(h=h,
                         relerr_exact=float(np.linalg.norm(Jex - Jfd) / max(nfd, 1e-12)),
                         relerr_naive=float(np.linalg.norm(Jna - Jfd) / max(nfd, 1e-12)),
                         cos_exact=float(np.sum(Jex * Jfd) / max(np.linalg.norm(Jex) * nfd, 1e-18)),
                         cos_naive=float(np.sum(Jna * Jfd) / max(np.linalg.norm(Jna) * nfd, 1e-18))))
    return rows


def fd_hessian_check(inst, h=1e-5):
    x, y, q, t, Bq = inst["x"], inst["y"], inst["q"], inst["t"], inst["Bq"]

    def Fval(qq, tt):
        return float(cm.gm_terms(cm.robust_terms(x, y, qq, tt, A_list=inst["A"]).e)[0].sum())

    f0 = Fval(q, t)
    Q = np.zeros((6, 6))

    def qf(xi):
        qp, tp = cm.exp_map(q, t, h * xi, Bq)
        qm, tm = cm.exp_map(q, t, -h * xi, Bq)
        return (Fval(qp, tp) - 2 * f0 + Fval(qm, tm)) / (h * h)

    for j in range(6):
        for k in range(j, 6):
            ej = np.zeros(6)
            ek = np.zeros(6)
            ej[j] = 1.0
            ek[k] = 1.0
            Q[j, k] = Q[k, j] = (qf(ej + ek) - qf(ej - ek)) / 4.0
    return Q


def main():
    t_start = time.time()
    instances = []
    seeds_boundary = [(9100 + k) for k in range(10)]
    seeds_random = [(9200 + k) for k in range(10)]
    for k, seed in enumerate(seeds_boundary):
        print(f"boundary instance {k} ...", flush=True)
        instances.append(("boundary", prepare(seed, 0.4, "boundary")))
    for k, seed in enumerate(seeds_random):
        print(f"random instance {k} ...", flush=True)
        instances.append(("random", prepare(seed, 0.2, "random")))

    all_rows = []
    inst_rows = []
    skipped = 0
    for tag, inst in instances:
        eig = np.linalg.eigvalsh(inst["H"])
        lam_min = float(eig.min())
        cond = float(eig.max() / max(lam_min, 1e-12))
        ok = lam_min > 1e-6
        if not ok:
            skipped += 1
            print(f"  skip {tag}: lambda_min={lam_min:.2e}", flush=True)
            continue
        rows = analyse_instance(inst)
        for r in rows:
            r["instance"] = tag
        all_rows += rows
        inst_rows.append(dict(tag=tag, n=len(inst["x"]), phi_norm=inst["pn"],
                              newton_iters=inst["nit"], lambda_min=lam_min,
                              cond=cond, lam_rob=inst["lam_rob"]))
        print(f"  {tag}: lam_min={lam_min:.3f} cond={cond:.2f} "
              f"e/c2 range {min(r['e_over_c2'] for r in rows):.3g}.."
              f"{max(r['e_over_c2'] for r in rows):.3g}", flush=True)

    bins = summarize(all_rows)
    sign_flips = []
    for lo, hi, label in BINS:
        m = [r for r in all_rows if lo <= r["e_over_c2"] < hi]
        if not m:
            continue
        fm = [r["radial_mapping_exact"] * r["radial_mapping_naive"] < 0 for r in m]
        fp = [r["radial_pose_exact"] * r["radial_pose_naive"] < 0 for r in m]
        sign_flips.append(dict(bin=label, n=len(m),
                               mapping=float(np.mean(fm)), pose_t=float(np.mean(fp))))
    # representative per-observation table: last boundary instance
    rep_inst = None
    for tag, inst in reversed(instances):
        if tag == "boundary":
            rep_inst = inst
            break
    rep_rows = analyse_instance(rep_inst)
    rep_rows.sort(key=lambda r: r["e_over_c2"])
    rep = [{k: v for k, v in r.items() if k in
            ("i", "e_over_c2", "lam_parallel", "radial_mapping_exact", "radial_mapping_naive",
             "radial_pose_exact", "radial_pose_naive", "relerr_exact", "relerr_naive",
             "cos_exact", "cos_naive")} for r in rep_rows]
    # step-size table on the boundary observation with e closest to c^2
    target = min(rep_rows, key=lambda r: abs(r["e_over_c2"] - 1.0))
    steps = step_size_table(rep_inst, target["i"])
    # Hessian FD check on a boundary instance
    Hfd = fd_hessian_check(rep_inst, h=1e-5)
    Hdoc = rep_inst["H"] + rep_inst["lam_rob"] * np.diag([1., 1., 1., 0., 0., 0.])
    rel_h = float(np.linalg.norm(rep_inst["H"] - Hfd) / np.linalg.norm(Hfd))
    rel_doc = float(np.linalg.norm(Hdoc - Hfd) / np.linalg.norm(Hfd))
    rel_naive = float(np.linalg.norm(rep_inst["Hn"] - Hfd) / np.linalg.norm(Hfd))
    hess = dict(lambda_min=float(np.linalg.eigvalsh(rep_inst["H"]).min()),
                relerr_vs_fd=rel_h, relerr_doc_vs_fd=rel_doc,
                relerr_naive_vs_fd=rel_naive, lam_rob=float(rep_inst["lam_rob"]),
                n_obs=len(rep_rows))
    # lambda identity
    terms = rep_inst["terms"]
    lam_i = np.array([rep_inst["q"] @ terms.g[i, :4] for i in range(len(terms.e))])
    lam_u = np.array([4.0 * terms.u[i] @ terms.r[i] for i in range(len(terms.e))])
    lam_id = dict(max_abs_lambda_vs_4uWr=float(np.max(np.abs(lam_i - lam_u))),
                  max_abs_lambda_vs_4e=float(np.max(np.abs(lam_i - 4 * terms.e))))
    # mapping identity a_exact = -lambda_parallel
    map_err = max(abs(r["radial_mapping_exact"] + r["lam_parallel"]) for r in all_rows)
    data = dict(
        settings=dict(N=30, c=cm.C, sigma=cm.SIGMA, fd_h=FD_H,
                      instances_boundary=len(seeds_boundary),
                      instances_random=len(seeds_random), skipped=skipped),
        instances=inst_rows, bins=bins, sign_flips=sign_flips, representative=rep,
        step_sizes=steps, hessian_check=hess, lambda_identity=lam_id,
        mapping_identity_max_err=float(map_err),
        total_obs=len(all_rows),
    )
    cm.json_dump(OUT / "gradient_table.json", data)

    write_markdown(data)
    make_figures(data, all_rows)
    print(f"wrote gradient_table.* in {time.time()-t_start:.1f}s")


def write_markdown(d):
    L = []
    L.append("# 6D gradient table: exact IFT vs Naive IRLS-StopGrad vs finite differences")
    L.append("")
    L.append(f"N={d['settings']['N']}, c={d['settings']['c']}, sigma={d['settings']['sigma']}, "
             f"FD step h={d['settings']['fd_h']}. "
             f"{d['settings']['instances_boundary']} boundary (40%) + "
             f"{d['settings']['instances_random']} random (20%) instances; "
             f"{d['total_obs']} observations; skipped {d['settings']['skipped']} "
             "near-singular instances.")
    L.append("")
    L.append("Definitions: exact = -H_rob^{-1} dPhi/dy_i; naive = -H_naive^{-1} "
             "dPhi_naive/dy_i with w_i frozen and H_naive the exact frozen-weight "
             "Hessian (including the sphere term); no-lambda = exact RHS but "
             "H_doc without -lambda_rob diag(I_3,0). FD re-solves the polished "
             "fixed point to 1e-13 on each side.")
    L.append("")
    L.append("## Metrics by residual bin")
    L.append("")
    L.append("| bin | method | n | relerr p50 | relerr p90 | cos p50 | cos min |")
    L.append("|---|---|---|---|---|---|---|")
    for r in d["bins"]:
        L.append(f"| {r['bin']} | {r['method']} | {r['n']} | {r['relerr_p50']:.2e} | "
                 f"{r['relerr_p90']:.2e} | {r['cos_p50']:+.6f} | {r['cos_min']:+.6f} |")
    L.append("")
    L.append("> The exact Jacobian matches FD to ~5e-8 in the inlier and boundary "
             "bins, and to ~1e-4 (relative) for gross outliers where ||dz*/dy_i|| "
             "is tiny and FD suffers cancellation. The naive Jacobian is already "
             "biased for inliers (~20% relative error), degrades to ~100% at the "
             "boundary, and is anti-correlated for gross outliers. `no-lambda` is a "
             "small but systematic bias here (~1e-3 relative); the constraint term is "
             "a much larger fraction of H itself when H is not data-dominated (see "
             "the Hessian lines below), so it cannot be dropped on principle.")
    L.append("")
    L.append("## Sign flips by residual bin (exact vs naive)")
    L.append("")
    L.append("| bin | n | mapping radial a | pose translation radial b |")
    L.append("|---|---|---|---|")
    for r in d["sign_flips"]:
        L.append(f"| {r['bin']} | {r['n']} | {100*r['mapping']:.0f}% | {100*r['pose_t']:.0f}% |")
    L.append("")
    L.append("> Mapping level: a_exact = -lambda_parallel(e) is provably opposite to "
             "a_naive = -2 rho'(e) for every e>c^2/3. Pose level: b_exact = "
             "rhat^T (dt*/dy_i) rhat is not forced to flip (H_rob^{-1} mixes "
             "directions, Theorem 3.3(d)), but in this well-conditioned 6D sample it "
             "flips for 100% of the boundary/gross observations.")
    L.append("")
    L.append("## Representative boundary instance (per observation)")
    L.append("")
    L.append("| i | e/c^2 | lambda_parallel | a_exact (map) | a_naive (map) | "
             "b_exact (pose t) | b_naive (pose t) | relerr exact | relerr naive | cos exact | cos naive |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["representative"]:
        L.append(f"| {r['i']} | {r['e_over_c2']:.3f} | {r['lam_parallel']:+.4f} | "
                 f"{r['radial_mapping_exact']:+.5f} | {r['radial_mapping_naive']:+.5f} | "
                 f"{r['radial_pose_exact']:+.5f} | {r['radial_pose_naive']:+.5f} | "
                 f"{r['relerr_exact']:.2e} | {r['relerr_naive']:.2e} | "
                 f"{r['cos_exact']:+.5f} | {r['cos_naive']:+.5f} |")
    L.append("")
    L.append("> `a_*` is the mapping-level radial response rhat^T (dPhi_t/dy_i) rhat; "
             "exactly -lambda_parallel(e_i). `b_*` is the pose-level translation "
             "response rhat^T (dt*/dy_i) rhat; H_rob^{-1} mixes directions, so b is "
             "not forced to flip sign by lambda_parallel alone (Theorem 3.3(d)). "
             "The sign-flip column in the summary table counts e>c^2/3 observations.")
    L.append("")
    L.append("## FD step-size check (one boundary observation)")
    L.append("")
    L.append("| h | relerr exact | relerr naive | cos exact | cos naive |")
    L.append("|---|---|---|---|---|")
    for r in d["step_sizes"]:
        L.append(f"| {r['h']:.0e} | {r['relerr_exact']:.2e} | {r['relerr_naive']:.2e} | "
                 f"{r['cos_exact']:+.8f} | {r['cos_naive']:+.8f} |")
    L.append("")
    L.append("## Hessian / lambda checks")
    L.append("")
    L.append(f"* lambda_min(H_rob) on representative instance: "
             f"{d['hessian_check']['lambda_min']:+.4f}")
    L.append(f"* ||H_rob - H_fd||_F / ||H_fd||_F: {d['hessian_check']['relerr_vs_fd']:.2e}")
    L.append(f"* ||H_doc - H_fd||_F / ||H_fd||_F (missing -lambda): "
             f"{d['hessian_check']['relerr_doc_vs_fd']:.2e}  "
             f"(lambda_rob={d['hessian_check']['lam_rob']:+.3f})")
    L.append(f"* ||H_naive - H_fd||_F / ||H_fd||_F (frozen weights): "
             f"{d['hessian_check']['relerr_naive_vs_fd']:.2e}")
    L.append(f"* max |lambda_i - 4 u_i^T W_i r_i|: "
             f"{d['lambda_identity']['max_abs_lambda_vs_4uWr']:.2e}")
    L.append(f"* max |lambda_i - 4 e_i|: {d['lambda_identity']['max_abs_lambda_vs_4e']:.2e}")
    L.append(f"* max |a_exact + lambda_parallel|: {d['mapping_identity_max_err']:.2e}")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("* FD differentiates the *polished fixed-point branch*, not a finite GNC "
             "unroll; this is the correct oracle for the IFT derivative. The GNC output "
             "itself has ||Phi||~1e-7 because it uses 5 final IRLS steps.")
    L.append("* The naive Hessian keeps the exact sphere term, so exact-vs-naive isolates "
             "the weight-sensitivity (rho'') error, not the constraint error. The "
             "`no-lambda` row separately isolates the constraint error.")
    L.append("* `relerr` can be large for gross outliers where ||dz*/dy_i|| is tiny; the "
             "cosine column and absolute errors are the meaningful diagnostics there.")
    cm.write_text(OUT / "gradient_table.md", "\n".join(L) + "\n")


def make_figures(d, all_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    bins = [b[2] for b in BINS]
    methods = ["exact", "naive", "nolambda"]
    colors = {"exact": "tab:red", "naive": "tab:blue", "nolambda": "tab:green"}
    width = 0.25
    for k, method in enumerate(methods):
        vals = []
        for label in bins:
            r = next((z for z in d["bins"] if z["bin"] == label and z["method"] == method), None)
            vals.append(r["relerr_p50"] if r else np.nan)
        ax[0].bar(np.arange(len(bins)) + (k - 1) * width, vals, width, label=method,
                  color=colors[method])
    ax[0].set_yscale("log")
    ax[0].set_xticks(np.arange(len(bins)))
    ax[0].set_xticklabels(["inlier", "boundary", "gross"])
    ax[0].set_ylabel("median relative error vs FD (log)")
    ax[0].set_title("Jacobian relative error")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    for k, method in enumerate(methods):
        vals = []
        for label in bins:
            r = next((z for z in d["bins"] if z["bin"] == label and z["method"] == method), None)
            vals.append(r["cos_p50"] if r else np.nan)
        ax[1].bar(np.arange(len(bins)) + (k - 1) * width, vals, width, label=method,
                  color=colors[method])
    ax[1].set_xticks(np.arange(len(bins)))
    ax[1].set_xticklabels(["inlier", "boundary", "gross"])
    ax[1].set_ylim(-0.4, 1.05)
    ax[1].set_ylabel("median cosine vs FD")
    ax[1].set_title("Jacobian cosine")
    ax[1].grid(alpha=0.3, axis="y")

    e = np.linspace(0, 4, 800)
    ax[2].plot(e, cm.lambda_parallel(e * cm.C2), lw=2, color="k")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].axvline(1 / 3, color="r", ls="--", lw=1)
    ax[2].axvline(1, color="g", ls=":", lw=1)
    ax[2].axvline(3, color="g", ls=":", lw=1)
    ax[2].set_xlabel("e / c^2")
    ax[2].set_ylabel("lambda_parallel(e)")
    ax[2].set_title("lambda_parallel: zero at e=c^2/3, min -1/2 at e=c^2")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_table.png", dpi=160)
    plt.close(fig)

    # sign scatter: mapping and pose radial responses for e>c^2/3
    sel = [r for r in all_rows if r["e_over_c2"] > 1 / 3]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter([r["radial_mapping_naive"] for r in sel],
                  [r["radial_mapping_exact"] for r in sel],
                  c=[r["e_over_c2"] for r in sel], cmap="viridis", s=18)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("naive mapping radial response")
    ax[0].set_ylabel("exact mapping radial response")
    ax[0].set_title("mapping level: opposite signs")
    ax[0].grid(alpha=0.3)
    ax[1].scatter([r["radial_pose_naive"] for r in sel],
                  [r["radial_pose_exact"] for r in sel],
                  c=[r["e_over_c2"] for r in sel], cmap="viridis", s=18)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("naive pose radial response")
    ax[1].set_ylabel("exact pose radial response")
    ax[1].set_title("pose level: H^{-1} mixes directions")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_sign.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
