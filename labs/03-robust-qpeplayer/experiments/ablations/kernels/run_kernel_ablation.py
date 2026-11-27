#!/usr/bin/env python3
"""Kernel ablation for the GNC robust outer loop (GM / Cauchy / Welsch).

All three kernels share the same weighted-Kabsch inner solver and the same
bandwidth-annealing schedule; only the weight map changes.  This isolates
the kernel choice from the inner solver and from the outlier protocol.

GM is the paper's default.  Cauchy and Welsch are standard redescending
alternatives; the experiment reports whether the default is actually a
good choice on random vs boundary outliers.

Run:
  srun --partition=cpu --time=00:25:00 --ntasks=1 --cpus-per-task=2 \
    /home/syk/miniconda3/envs/WM/bin/python run_kernel_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(SYN))
import common as cm  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
C = 0.05
N = 100
TRIALS = 50
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TYPES = ["random", "boundary"]
KERNELS = ["GM", "Cauchy", "Welsch"]
GAMMA = 1.4
MAX_MU_ITERS = 40


def weight(kernel, e, a):
    if kernel == "GM":
        return (a / (a + e)) ** 2
    if kernel == "Cauchy":
        return a / (a + e)
    if kernel == "Welsch":
        return np.exp(-e / a)
    raise ValueError(kernel)


def solve_gnc_kernel(x, y, kernel="GM", c=C, gamma=GAMMA, max_mu_iters=MAX_MU_ITERS):
    """GNC with a kernel-specific weight map and weighted-Kabsch inner."""
    R, t = cm.kabsch(x, y, np.ones(len(x)))
    e = cm.residuals(x, y, R, t) ** 2
    mu = max(3.0 * e.max() / (c * c), 1.0)
    for _ in range(max_mu_iters):
        a = mu * c * c
        w = weight(kernel, e, a)
        R, t = cm.kabsch(x, y, w)
        e = cm.residuals(x, y, R, t) ** 2
        if mu <= 1.0 + 1e-12:
            break
        mu = max(1.0, mu / gamma)
    # exact-kernel IRLS polish
    for _ in range(5):
        w = weight(kernel, e, c * c)
        R, t = cm.kabsch(x, y, w)
        e = cm.residuals(x, y, R, t) ** 2
    return R, t


def main():
    t_start = time.time()
    rows = []
    for otype in TYPES:
        for p in P_LIST:
            for kernel in KERNELS:
                rng = np.random.default_rng([20261127, 31, {"random": 1, "boundary": 2}[otype],
                                             int(round(100 * p))])
                res = dict(re=[], te=[], time=[])
                for trial in range(TRIALS):
                    x, y, Rgt, tgt = cm.make_instance(rng, p, n=N, outlier_type=otype)
                    t0 = time.perf_counter()
                    R, t = solve_gnc_kernel(x, y, kernel=kernel)
                    dt = time.perf_counter() - t0
                    re, te = cm.pose_errors(R, t, Rgt, tgt)
                    res["re"].append(re)
                    res["te"].append(te)
                    res["time"].append(dt)
                s = cm.summarize_errors(res["re"], res["te"], res["time"])
                rows.append(dict(outlier_type=otype, p=p, kernel=kernel,
                                 gamma=GAMMA, trials=TRIALS, **s))
                print(f"{otype:8s} p={p:.1f} {kernel:7s} "
                      f"succ={100*s['success']:.0f}% rot={s['rot_median']:.3f} "
                      f"t={s['time_ms_p50']:.2f}ms ({time.time()-t_start:.1f}s)", flush=True)

    cm.json_dump(OUT / "kernel_ablation.json", dict(
        config=dict(n=N, c=C, trials=TRIALS, p=P_LIST, types=TYPES,
                    kernels=KERNELS, gamma=GAMMA), rows=rows))

    L = ["# Kernel ablation: GM vs Cauchy vs Welsch", "",
         f"N={N}, c={C}, {TRIALS} trials/cell, gamma={GAMMA}, identical weighted-Kabsch inner "
         "and annealing schedule. Success = rot<5 deg and trans<0.1 m.", ""]
    for otype in TYPES:
        L.append(f"## {otype} outliers: success rate")
        L.append("")
        L.append("| kernel | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
        L.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
        for k in KERNELS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["kernel"] == k
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{100*r['success']:.0f}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
        L.append(f"## {otype} outliers: median rotation error (deg)")
        L.append("")
        L.append("| kernel | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
        L.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
        for k in KERNELS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["kernel"] == k
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{r['rot_median']:.2f}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
        L.append(f"## {otype} outliers: runtime p50 (ms)")
        L.append("")
        L.append("| kernel | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |")
        L.append("|---|" + "|".join(["---"] * len(P_LIST)) + "|")
        for k in KERNELS:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["outlier_type"] == otype and z["kernel"] == k
                         and abs(z["p"] - p) < 1e-9)
                cells.append(f"{r['time_ms_p50']:.2f}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Reading", "",
          "* All kernels are run with the same schedule, so differences are kernel shape, "
          "not tuning. GM is the default because it has a closed-form BR dual and is smooth; "
          "Cauchy is heavier-tailed near zero and Welsch is flat near zero.",
          "* A kernel that wins on random outliers but loses on boundary outliers would be a "
          "bad default for this paper's gradient story; report both columns."]
    (OUT / "kernel_ablation.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"GM": "tab:red", "Cauchy": "tab:blue", "Welsch": "tab:green"}
    for col, otype in enumerate(TYPES):
        for k in KERNELS:
            rs = [next(z for z in rows if z["outlier_type"] == otype and z["kernel"] == k
                       and abs(z["p"] - p) < 1e-9) for p in P_LIST]
            ax[0, col].plot([100 * p for p in P_LIST], [100 * r["success"] for r in rs],
                            "o-", color=colors[k], label=k, lw=1.5)
            ax[1, col].semilogy([100 * p for p in P_LIST],
                                np.maximum([r["rot_median"] for r in rs], 1e-3),
                                "o-", color=colors[k], label=k, lw=1.5)
        ax[0, col].set_title(f"{otype}: success")
        ax[1, col].set_title(f"{otype}: median rotation error")
        for row in ax[:, col]:
            row.set_xlabel("outlier fraction (%)")
            row.grid(alpha=0.3, which="both")
        ax[0, col].set_ylabel("success (%)")
        ax[1, col].set_ylabel("rot error (deg, log)")
        ax[0, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kernel_ablation.png", dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
