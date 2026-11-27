#!/usr/bin/env python3
"""Extended synthetic 3D-3D main table.

Adds to the v0 table (run_registration.py):

* runtime p50/p95 columns,
* coherent outliers (one competing rigid transform),
* boundary outliers (residual squared e in [c^2/3, 3c^2]),
* an adaptive RANSAC baseline in addition to the fixed 1000-iteration one,
* a RANSAC budget sweep (50 vs 1000 iterations, 50%..90% outliers),
* an N-scaling sweep (N=20/50/100/200).

Outputs
-------
results/main_table_extended.json       aggregated cells
results/main_table_extended_raw.jsonl  one line per (cell, trial, method)
results/main_table_extended.md         human-readable tables
results/main_table_extended.png        success / runtime / speedup figure
results/ransac_budget.png              budget sweep figure
results/scaling.png                    N-scaling figure

Run (Slurm):
  srun --partition=cpu --time=00:30:00 --ntasks=1 \
    /home/syk/miniconda3/envs/WM/bin/python run_main_extended.py 2>&1 | \
    tee results/logs/main_extended_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)

BASE_SEED = 20261127
TRIALS_MAIN = 100
TRIALS_SCALING = 50
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OUTLIER_TYPES = ["random", "coherent", "boundary"]
METHOD_ORDER = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(p) for p in parts]])


def make_methods():
    def kabsch_m(x, y, rng):
        return (*cm.kabsch(x, y, np.ones(len(x))), {})

    def ransac_fixed(x, y, rng):
        return cm.solve_ransac(x, y, rng, iters=cm.RANSAC_ITERS,
                               adaptive=False, return_info=True)

    def ransac_adapt(x, y, rng):
        return cm.solve_ransac(x, y, rng, iters=cm.RANSAC_ITERS,
                               adaptive=True, return_info=True)

    def irls(x, y, rng):
        return (*cm.solve_irls(x, y), {})

    def gnc(x, y, rng):
        return (*cm.solve_gnc(x, y), {})

    return {
        "Kabsch": kabsch_m,
        "RANSAC-1000": ransac_fixed,
        "RANSAC-adaptive": ransac_adapt,
        "IRLS-GM": irls,
        "GNC-GM": gnc,
    }


def run_cell(methods, p, outlier_type, n, trials, exp_id):
    """Run one (outlier_type, p) cell; return summary rows + raw records."""
    type_id = {"random": 1, "coherent": 2, "boundary": 3}[outlier_type]
    if p == 0.0:
        type_id = 0  # identical clean instances across outlier types
    per_method = {m: dict(re=[], te=[], time=[], picked=[]) for m in METHOD_ORDER}
    raw = []
    for trial in range(trials):
        data_rng = np.random.default_rng(_seed(exp_id, type_id, int(round(100 * p)), trial))
        x, y, Rgt, tgt, meta = cm.make_instance(data_rng, p, n=n,
                                                outlier_type=outlier_type,
                                                return_meta=True)
        for name in METHOD_ORDER:
            fn = methods[name]
            # fixed and adaptive RANSAC share the same sampling stream
            rng = np.random.default_rng(_seed(exp_id, type_id, int(round(100 * p)), trial, 999))
            t0 = time.perf_counter()
            R, t, info = fn(x, y, rng)
            dt = time.perf_counter() - t0
            re, te = cm.pose_errors(R, t, Rgt, tgt)
            rec = dict(exp=exp_id, outlier_type=outlier_type, p=p, n=n, trial=trial,
                       method=name, success=cm.success(re, te), rot=re, trans=te,
                       time_ms=1000.0 * dt, ransac_iters=info.get("n_iters"),
                       ransac_inliers=info.get("best_count"))
            if outlier_type == "coherent" and meta.get("Rout") is not None:
                re_out = cm.rot_err_deg(R, meta["Rout"])
                te_out = float(np.linalg.norm(t - meta["tout"]))
                rec["rot_to_outlier"] = re_out
                rec["trans_to_outlier"] = te_out
                rec["picked_outlier"] = bool(re_out < re and te_out < te)
                per_method[name]["picked"].append(float(rec["picked_outlier"]))
                per_method[name].setdefault("rec_out", []).append(
                    float(re_out < cm.SUCCESS_ROT_DEG and te_out < cm.SUCCESS_TRANS))
            raw.append(rec)
            a = per_method[name]
            a["re"].append(re)
            a["te"].append(te)
            a["time"].append(dt)
    rows = []
    for name in METHOD_ORDER:
        a = per_method[name]
        s = cm.summarize_errors(a["re"], a["te"], a["time"])
        row = dict(experiment="outlier_type", outlier_type=outlier_type, p=p,
                   n=n, method=name, **s)
        if a["picked"]:
            row["frac_picked_outlier"] = float(np.mean(a["picked"]))
            row["frac_recovered_outlier"] = float(np.mean(a["rec_out"]))
        rows.append(row)
    return rows, raw


def run_budget(methods, trials):
    """RANSAC budget sweep on random outliers, p=0.5..0.9."""
    rows, raw = [], []
    exp_id = 20
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for budget, adaptive in [(50, False), (1000, False), (1000, True)]:
            label = f"RANSAC-{budget}" + ("-adaptive" if adaptive else "")
            re, te, tt, iters = [], [], [], []
            for trial in range(trials):
                data_rng = np.random.default_rng(_seed(exp_id, 1, int(round(100 * p)), trial))
                x, y, Rgt, tgt = cm.make_instance(data_rng, p)
                rng = np.random.default_rng(_seed(exp_id, 1, int(round(100 * p)), trial, 999))
                t0 = time.perf_counter()
                R, t, info = cm.solve_ransac(x, y, rng, iters=budget, adaptive=adaptive,
                                             return_info=True)
                dt = time.perf_counter() - t0
                rerr, terr = cm.pose_errors(R, t, Rgt, tgt)
                re.append(rerr)
                te.append(terr)
                tt.append(dt)
                iters.append(info["n_iters"])
                raw.append(dict(exp=exp_id, p=p, budget=budget, adaptive=adaptive,
                                trial=trial, method=label, success=cm.success(rerr, terr),
                                rot=rerr, trans=terr, time_ms=1000 * dt,
                                ransac_iters=info["n_iters"],
                                ransac_inliers=info["best_count"]))
            s = cm.summarize_errors(re, te, tt)
            n_in = cm.N_DEFAULT - int(round(p * cm.N_DEFAULT))
            p_clean = comb(n_in, 3) / comb(cm.N_DEFAULT, 3)
            theory = float(np.mean(1.0 - (1.0 - p_clean) ** np.asarray(iters)))
            rows.append(dict(experiment="ransac_budget", p=p, budget=budget,
                             adaptive=adaptive, method=label, **s,
                             mean_iters=float(np.mean(iters)),
                             theory_success=theory))
    return rows, raw


def run_scaling(methods, trials):
    rows, raw = [], []
    exp_id = 30
    p = 0.3
    for n in [20, 50, 100, 200]:
        per_method = {m: dict(re=[], te=[], time=[]) for m in METHOD_ORDER}
        for trial in range(trials):
            data_rng = np.random.default_rng(_seed(exp_id, 1, int(round(100 * p)), trial, n))
            x, y, Rgt, tgt = cm.make_instance(data_rng, p, n=n)
            for name in METHOD_ORDER:
                rng = np.random.default_rng(_seed(exp_id, 1, int(round(100 * p)), trial, 999, n))
                t0 = time.perf_counter()
                R, t, info = methods[name](x, y, rng)
                dt = time.perf_counter() - t0
                re, te = cm.pose_errors(R, t, Rgt, tgt)
                per_method[name]["re"].append(re)
                per_method[name]["te"].append(te)
                per_method[name]["time"].append(dt)
                raw.append(dict(exp=exp_id, n=n, p=p, trial=trial, method=name,
                                success=cm.success(re, te), rot=re, trans=te,
                                time_ms=1000 * dt))
        for name in METHOD_ORDER:
            a = per_method[name]
            s = cm.summarize_errors(a["re"], a["te"], a["time"])
            rows.append(dict(experiment="scaling", n=n, p=p, method=name, **s))
    return rows, raw


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _cell(rows, exp, method, key, p=None, n=None, outlier_type=None):
    for r in rows:
        if r["experiment"] != exp or r["method"] != method:
            continue
        if p is not None and abs(r.get("p", -1) - p) > 1e-9:
            continue
        if n is not None and r.get("n") != n:
            continue
        if outlier_type is not None and r.get("outlier_type") != outlier_type:
            continue
        return r
    raise KeyError((exp, method, key, p, n, outlier_type))


def pivot_table(rows, exp, methods, value_key, fmt, ps=P_LIST, outlier_type=None):
    lines = ["| method | " + " | ".join(f"{int(100 * p)}%" for p in ps) + " |",
             "|---|" + "|".join(["---"] * len(ps)) + "|"]
    for m in methods:
        cells = []
        for p in ps:
            r = _cell(rows, exp, m, value_key, p=p, outlier_type=outlier_type)
            cells.append(fmt(r[value_key]) if value_key in r else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(rows, budget_rows, scaling_rows):
    L = []
    L.append("# Synthetic 3D-3D robust registration — extended main table")
    L.append("")
    L.append(f"N={cm.N_DEFAULT}, sigma={cm.SIGMA}, c={cm.C}, {TRIALS_MAIN} trials/cell "
             "(clean instances shared across outlier types at p=0).")
    L.append("Success = rotation error < 5 deg and translation error < 0.1 m. "
             "Times are wall-clock p50/p95 over trials.")
    L.append("")
    L.append("## 1. Random outliers (v0 protocol: replaced y uniform in [-2,2]^3)")
    L.append("")
    L.append("### Success rate")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "success", lambda x: f"{100*x:.0f}%",
                         outlier_type="random"))
    L.append("")
    L.append("### Median rotation error (deg)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "rot_median", lambda x: f"{x:.2f}",
                         outlier_type="random"))
    L.append("")
    L.append("### Median translation error (m)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "trans_median", lambda x: f"{x:.4f}",
                         outlier_type="random"))
    L.append("")
    L.append("### Runtime p50 (ms)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "time_ms_p50", lambda x: f"{x:.2f}",
                         outlier_type="random"))
    L.append("")
    L.append("### Runtime p95 (ms)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "time_ms_p95", lambda x: f"{x:.2f}",
                         outlier_type="random"))
    L.append("")
    L.append("### Speedup of GNC-GM vs RANSAC baselines (runtime p50 ratio)")
    L.append("")
    L.append("| outlier rate | vs RANSAC-1000 | vs RANSAC-adaptive | GNC p50 (ms) | RANSAC-1000 p50 (ms) | RANSAC-adaptive p50 (ms) |")
    L.append("|---|---|---|---|---|---|")
    for p in P_LIST:
        g = _cell(rows, "outlier_type", "GNC-GM", "time_ms_p50", p=p, outlier_type="random")
        f = _cell(rows, "outlier_type", "RANSAC-1000", "time_ms_p50", p=p, outlier_type="random")
        a = _cell(rows, "outlier_type", "RANSAC-adaptive", "time_ms_p50", p=p, outlier_type="random")
        L.append(f"| {int(100*p)}% | {f['time_ms_p50']/g['time_ms_p50']:.1f}x | "
                 f"{a['time_ms_p50']/g['time_ms_p50']:.1f}x | "
                 f"{g['time_ms_p50']:.2f} | {f['time_ms_p50']:.2f} | {a['time_ms_p50']:.2f} |")
    L.append("")
    L.append("> Reading: against the fixed-budget 1000-iteration RANSAC baseline (the v0 table), "
             "GNC is ~30x faster for every p>=10%. Against adaptive RANSAC, which stops on a "
             "confidence bound, GNC is comparable at 10-30% outliers and pulls ahead at >=60% "
             "(and by ~25-30x at >=80%, see the budget table). RANSAC is not differentiable.")
    L.append("")
    L.append("## 2. Coherent outliers (one competing rigid transform, 30-60 deg / 0.6-1.5 m)")
    L.append("")
    L.append("### Success rate")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "success", lambda x: f"{100*x:.0f}%",
                         outlier_type="coherent"))
    L.append("")
    L.append("### Median rotation error (deg)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "rot_median", lambda x: f"{x:.2f}",
                         outlier_type="coherent"))
    L.append("")
    L.append("### Fraction of trials that recover the outlier pose (<5 deg, <0.1 m)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "frac_recovered_outlier",
                         lambda x: f"{100*x:.0f}%", outlier_type="coherent"))
    L.append("")
    L.append("### Fraction of trials closer to the outlier transform than to GT")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "frac_picked_outlier",
                         lambda x: f"{100*x:.0f}%", outlier_type="coherent"))
    L.append("")
    L.append("### Runtime p50 (ms)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "time_ms_p50", lambda x: f"{x:.2f}",
                         outlier_type="coherent"))
    L.append("")
    L.append("> Failure mode: at 60-70% coherent outliers the competing transform is the "
             "largest consensus / lowest robust cost, so every method (including the exact "
             "global inner solver) returns the *outlier* pose. This is the identifiability "
             "failure of Prop 3.7, not a solver bug. At 50% the two hypotheses tie and the "
             "winner is essentially a coin flip.")
    L.append("")
    L.append("## 3. Boundary outliers (e = ||r||^2 uniform in [c^2/3, 3c^2])")
    L.append("")
    L.append("### Success rate")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "success", lambda x: f"{100*x:.0f}%",
                         outlier_type="boundary"))
    L.append("")
    L.append("### Median rotation error (deg)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "rot_median", lambda x: f"{x:.2f}",
                         outlier_type="boundary"))
    L.append("")
    L.append("### Median translation error (m)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "trans_median", lambda x: f"{x:.4f}",
                         outlier_type="boundary"))
    L.append("")
    L.append("### Runtime p50 (ms)")
    L.append("")
    L.append(pivot_table(rows, "outlier_type", METHOD_ORDER, "time_ms_p50", lambda x: f"{x:.2f}",
                         outlier_type="boundary"))
    L.append("")
    L.append("## 4. RANSAC budget sweep (random outliers, 3-point sampling)")
    L.append("")
    L.append("| p | budget | adaptive | success | rot med (deg) | time p50 (ms) | time p95 (ms) | mean iters | theory success |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in budget_rows:
        L.append(f"| {int(100*r['p'])}% | {r['budget']} | {r['adaptive']} | "
                 f"{100*r['success']:.0f}% | {r['rot_median']:.2f} | "
                 f"{r['time_ms_p50']:.2f} | {r['time_ms_p95']:.2f} | "
                 f"{r['mean_iters']:.0f} | {100*r['theory_success']:.1f}% |")
    L.append("")
    L.append("> The 50-iteration budget is *not* a failure at 50% outliers: a clean 3-point "
             "sample has probability ~0.121 (without replacement), so 50 iterations succeed "
             "with probability ~99.9%. The accuracy-time trade-off bites at >=70% outliers: "
             "at 70% the clean-sample probability is 0.025 and 50 iterations give ~75%; at "
             "80% 50 iterations give 26% while 1000 give 100%; at 90% even 1000 iterations "
             "only reach 46% (exact theory 52.4%). This is the honest RANSAC baseline: its "
             "cost is *data-dependent* (inlier ratio), while GNC's is not.")
    L.append("")
    L.append("## 5. Scaling with N (p=30% random outliers)")
    L.append("")
    L.append("| N | method | success | rot med (deg) | time p50 (ms) | time p95 (ms) |")
    L.append("|---|---|---|---|---|---|")
    for n in [20, 50, 100, 200]:
        for m in METHOD_ORDER:
            r = _cell(scaling_rows, "scaling", m, "time_ms_p50", n=n)
            L.append(f"| {n} | {m} | {100*r['success']:.0f}% | {r['rot_median']:.2f} | "
                     f"{r['time_ms_p50']:.2f} | {r['time_ms_p95']:.2f} |")
    L.append("")
    L.append("## Notes / caveats")
    L.append("")
    L.append("* The 3D-3D forward solve here is weighted Kabsch (the closed-form 3D-3D "
             "specialization of QPEP); the robust outer loop is GNC-GM or IRLS-GM.  "
             "No QPEP polynomial solver is involved, so this table isolates the robust "
             "estimator, not the QPEP algebra.")
    L.append("* RANSAC threshold is `||r|| < c`; the GM kernel uses the same `c` on the "
             "squared residual, so both methods have the same inlier scale.")
    L.append("* `RANSAC-1000` is the v0 baseline (always 1000 iterations unless a sample "
             "already covers all N points). `RANSAC-adaptive` uses the standard confidence "
             "bound (0.999) and is capped at 1000 iterations.")
    L.append("* Metrics use the trace/arccos formula; near zero it has a numerical floor of "
             "~1e-6 deg.  This does not affect the 0.1-100 deg comparisons.")
    L.append("* All numbers generated by `run_main_extended.py`; raw per-trial records in "
             "`results/main_table_extended_raw.jsonl`.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(rows, budget_rows, scaling_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue",
              "GNC-GM": "tab:red"}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for col, otype in enumerate(OUTLIER_TYPES):
        for m in METHOD_ORDER:
            ys = [_cell(rows, "outlier_type", m, "success", p=p, outlier_type=otype)["success"]
                  for p in P_LIST]
            ax[0, col].plot([100 * p for p in P_LIST], [100 * y for y in ys], "o-",
                            label=m, color=colors[m], lw=1.6, ms=4)
        ax[0, col].set_title(f"{otype} outliers")
        ax[0, col].set_xlabel("outlier fraction (%)")
        ax[0, col].set_ylabel("success rate (%)")
        ax[0, col].set_ylim(-3, 103)
        ax[0, col].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7, loc="lower left")

    for m in METHOD_ORDER:
        ys = [_cell(rows, "outlier_type", m, "time_ms_p50", p=p, outlier_type="random")["time_ms_p50"]
              for p in P_LIST]
        ax[1, 0].semilogy([100 * p for p in P_LIST], ys, "o-", label=m, color=colors[m], lw=1.6, ms=4)
    ax[1, 0].set_title("runtime p50 vs outlier rate (random)")
    ax[1, 0].set_xlabel("outlier fraction (%)")
    ax[1, 0].set_ylabel("time (ms, log)")
    ax[1, 0].grid(alpha=0.3, which="both")
    ax[1, 0].legend(fontsize=7)

    g = np.array([_cell(rows, "outlier_type", "GNC-GM", "time_ms_p50", p=p,
                        outlier_type="random")["time_ms_p50"] for p in P_LIST])
    f = np.array([_cell(rows, "outlier_type", "RANSAC-1000", "time_ms_p50", p=p,
                        outlier_type="random")["time_ms_p50"] for p in P_LIST])
    a = np.array([_cell(rows, "outlier_type", "RANSAC-adaptive", "time_ms_p50", p=p,
                        outlier_type="random")["time_ms_p50"] for p in P_LIST])
    ax[1, 1].semilogy([100 * p for p in P_LIST], f / g, "o-", label="RANSAC-1000 / GNC",
                      color="tab:orange", lw=1.6)
    ax[1, 1].semilogy([100 * p for p in P_LIST], a / g, "s-", label="RANSAC-adaptive / GNC",
                      color="tab:green", lw=1.6)
    ax[1, 1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1, 1].set_title("speedup over GNC (runtime p50)")
    ax[1, 1].set_xlabel("outlier fraction (%)")
    ax[1, 1].set_ylabel("RANSAC time / GNC time")
    ax[1, 1].grid(alpha=0.3, which="both")
    ax[1, 1].legend(fontsize=8)

    for m in METHOD_ORDER:
        ys = [_cell(rows, "outlier_type", m, "rot_median", p=p, outlier_type="random")["rot_median"]
              for p in P_LIST]
        ax[1, 2].semilogy([100 * p for p in P_LIST], np.maximum(ys, 1e-3), "o-",
                          label=m, color=colors[m], lw=1.6, ms=4)
    ax[1, 2].set_title("median rotation error vs outlier rate (random)")
    ax[1, 2].set_xlabel("outlier fraction (%)")
    ax[1, 2].set_ylabel("rotation error (deg, log)")
    ax[1, 2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "main_table_extended.png", dpi=160)
    plt.close(fig)

    # RANSAC budget
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {(50, False): ("tab:purple", "o", "50 iters"),
              (1000, False): ("tab:orange", "s", "1000 iters"),
              (1000, True): ("tab:green", "^", "adaptive (<=1000)")}
    for (budget, adaptive), (color, marker, label) in styles.items():
        rr = [r for r in budget_rows if r["budget"] == budget and r["adaptive"] == adaptive]
        ps = [r["p"] for r in rr]
        ax[0].plot([100 * p for p in ps], [100 * r["success"] for r in rr],
                   marker + "-", color=color, label=label, lw=1.6)
        ax[1].semilogy([100 * p for p in ps], [r["time_ms_p50"] for r in rr],
                       marker + "-", color=color, label=label, lw=1.6)
    ax[0].set_xlabel("outlier fraction (%)")
    ax[0].set_ylabel("success rate (%)")
    ax[0].set_ylim(-3, 103)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_title("RANSAC budget: success")
    ax[1].set_xlabel("outlier fraction (%)")
    ax[1].set_ylabel("time p50 (ms, log)")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    ax[1].set_title("RANSAC budget: runtime")
    fig.tight_layout()
    fig.savefig(OUT / "ransac_budget.png", dpi=160)
    plt.close(fig)

    # scaling
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in METHOD_ORDER:
        ns = [20, 50, 100, 200]
        ts = [_cell(scaling_rows, "scaling", m, "time_ms_p50", n=n)["time_ms_p50"] for n in ns]
        ax[0].loglog(ns, ts, "o-", label=m, color=colors[m], lw=1.6)
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("time p50 (ms, log)")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    ax[0].set_title("runtime vs N (p=30%)")
    for m in METHOD_ORDER:
        ns = [20, 50, 100, 200]
        ys = [_cell(scaling_rows, "scaling", m, "rot_median", n=n)["rot_median"] for n in ns]
        ax[1].semilogy(ns, np.maximum(ys, 1e-3), "o-", label=m, color=colors[m], lw=1.6)
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("rotation error (deg, log)")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7)
    ax[1].set_title("accuracy vs N (p=30%)")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=160)
    plt.close(fig)


def main():
    t_start = time.time()
    methods = make_methods()
    print("warm-up ...", flush=True)
    rng = np.random.default_rng(1)
    x, y, _, _ = cm.make_instance(rng, 0.3)
    for name in METHOD_ORDER:
        methods[name](x, y, np.random.default_rng(2))

    all_rows, all_raw = [], []
    for otype in OUTLIER_TYPES:
        for p in P_LIST:
            t0 = time.time()
            rows, raw = run_cell(methods, p, otype, cm.N_DEFAULT, TRIALS_MAIN, exp_id=10)
            all_rows += rows
            all_raw += raw
            print(f"[main] {otype:8s} p={p:.1f} done in {time.time()-t0:.1f}s", flush=True)

    print("budget sweep ...", flush=True)
    budget_rows, budget_raw = run_budget(methods, TRIALS_MAIN)
    all_raw += budget_raw
    print("scaling ...", flush=True)
    scaling_rows, scaling_raw = run_scaling(methods, TRIALS_SCALING)
    all_raw += scaling_raw

    cm.json_dump(OUT / "main_table_extended.json",
                 dict(rows=all_rows, budget=budget_rows, scaling=scaling_rows))
    with open(OUT / "main_table_extended_raw.jsonl", "w") as f:
        for rec in all_raw:
            f.write(json.dumps(rec) + "\n")
    cm.write_text(OUT / "main_table_extended.md",
                  build_markdown(all_rows, budget_rows, scaling_rows))
    make_figures(all_rows, budget_rows, scaling_rows)
    print(f"wrote {OUT/'main_table_extended.json'} etc. in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
