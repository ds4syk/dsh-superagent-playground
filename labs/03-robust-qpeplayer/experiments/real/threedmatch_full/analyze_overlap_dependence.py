#!/usr/bin/env python3
"""Paired test: is the GNC-GM vs RANSAC advantage overlap-dependent?

Reads ``results/overlap_bins_raw.jsonl`` (written by ``run_overlap_bins.py``)
and tests, separately per overlap bin:

* success delta (GNC - RANSAC) with an exact McNemar test on the paired
  trials, at p in {0, 0.3, 0.5, 0.7};
* paired rotation-error difference (GNC - RANSAC) and the per-pair ratio,
  Wilcoxon signed-rank on per-pair medians plus a bootstrap CI over the
  independent fragment pairs;
* runtime speedup (RANSAC p50 / GNC p50) with a bootstrap CI over pairs;
* the interaction ``low - high`` of the per-bin rotation deltas, i.e. a direct
  test of "is the advantage overlap-dependent?";
* the clean-geometry trend (base inlier fraction, p=0 rotation error) that
  explains whatever dependence is observed.

The bootstrap resamples *fragment pairs* (clusters), not individual trials,
because the 8 trials of one pair share the same geometry.  All numbers in the
output JSON are computed here; nothing is hand-entered.

Output: ``results/overlap_dependence.json``
Run:
  /home/syk/miniconda3/envs/WM/bin/python analyze_overlap_dependence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RAW = RES / "overlap_bins_raw.jsonl"
OUT = RES / "overlap_dependence.json"

BINS = ["low", "mid", "high"]
P_LIST = [0.0, 0.3, 0.5, 0.7]
N_BOOT = 4000
SEED = 20261128


def load_raw():
    with open(RAW) as f:
        return [json.loads(line) for line in f]


def paired(rows, bin_, p, otype="random"):
    """Per-pair paired arrays for GNC-GM and RANSAC-1000 at one (bin, p)."""
    sel = [r for r in rows if r["bin"] == bin_ and r["outlier_type"] == otype
           and abs(r["p"] - p) < 1e-9
           and r["method"] in ("GNC-GM", "RANSAC-1000")]
    by = {}
    for r in sel:
        by.setdefault((r["scene"], r["pair"], r["trial"]), {})[r["method"]] = r

    out = dict(diff_rot={}, ratio_rot={}, time_g={}, time_r={}, succ={})
    for (scene, pair, trial), d in by.items():
        if "GNC-GM" not in d or "RANSAC-1000" not in d:
            continue
        g, k = d["GNC-GM"], d["RANSAC-1000"]
        key = (scene, pair)
        out["diff_rot"].setdefault(key, []).append(g["rot"] - k["rot"])
        out["ratio_rot"].setdefault(key, []).append(
            g["rot"] / k["rot"] if k["rot"] > 1e-12 else 1.0)
        out["time_g"].setdefault(key, []).append(g["time_ms"])
        out["time_r"].setdefault(key, []).append(k["time_ms"])
        out["succ"].setdefault(key, []).append((bool(g["success"]),
                                                bool(k["success"])))
    return out


def boot_ci(per_pair, stat=np.median, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI of ``stat`` over independent fragment-pair clusters.

    ``per_pair`` maps pair -> list of per-trial values; each resample draws
    pairs with replacement and pools all trials of the drawn pairs.
    """
    keys = list(per_pair)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = rng.integers(0, len(keys), len(keys))
        pooled = np.concatenate([np.asarray(per_pair[keys[i]], float) for i in take])
        vals.append(float(stat(pooled)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(rows, bin_, p, otype="random"):
    pr = paired(rows, bin_, p, otype=otype)
    if not pr["diff_rot"]:
        return None
    base = [r for r in rows if r["bin"] == bin_ and r["outlier_type"] == otype
            and abs(r["p"] - p) < 1e-9]
    rot_g = float(np.median([r["rot"] for r in base if r["method"] == "GNC-GM"]))
    rot_r = float(np.median([r["rot"] for r in base
                             if r["method"] == "RANSAC-1000"]))
    succ = [x for v in pr["succ"].values() for x in v]
    g = np.array([x[0] for x in succ], float)
    r = np.array([x[1] for x in succ], float)
    diffs = np.array([np.median(v) for v in pr["diff_rot"].values()], float)
    ratios = np.array([np.median(v) for v in pr["ratio_rot"].values()], float)
    try:
        w_p = float(wilcoxon(diffs).pvalue) if np.any(np.abs(diffs) > 0) else 1.0
    except ValueError:
        w_p = 1.0
    speedups = {k: [np.median(pr["time_r"][k]) / np.median(pr["time_g"][k])]
                for k in pr["time_r"]}
    disc_g = int(np.sum(g * (1 - r)))
    disc_r = int(np.sum((1 - g) * r))
    n_disc = disc_g + disc_r
    return dict(
        n_pairs=int(len(diffs)),
        n_trials=int(len(g)),
        success_gnc=float(g.mean()),
        success_ransac=float(r.mean()),
        delta_success=float(g.mean() - r.mean()),
        discordant_gnc_only=disc_g,
        discordant_ransac_only=disc_r,
        mcnemar_p=float(binomtest(disc_g, n_disc, 0.5).pvalue) if n_disc else 1.0,
        rot_median_gnc=rot_g,
        rot_median_ransac=rot_r,
        rot_ratio_of_medians=rot_g / rot_r if rot_r > 0 else None,
        rot_delta_median=float(np.median(diffs)),
        rot_delta_ci=boot_ci(pr["diff_rot"]),
        rot_ratio_median=float(np.median(ratios)),
        rot_ratio_ci=boot_ci(pr["ratio_rot"]),
        wilcoxon_p=w_p,
        time_gnc_ms_p50=float(np.median([np.median(v) for v in pr["time_g"].values()])),
        time_ransac_ms_p50=float(np.median(
            [np.median(v) for v in pr["time_r"].values()])),
        speedup_median=float(np.median([v[0] for v in speedups.values()])),
        speedup_ci=boot_ci(speedups),
    )


def interaction(rows, p_lo="low", p_hi="high", p=None, otype="random"):
    a = paired(rows, p_lo, p, otype=otype)
    b = paired(rows, p_hi, p, otype=otype)
    if not (a["diff_rot"] and b["diff_rot"]):
        return None
    ka, kb = list(a["diff_rot"]), list(b["diff_rot"])
    delta_lo = float(np.median([np.median(a["diff_rot"][k]) for k in ka]))
    delta_hi = float(np.median([np.median(b["diff_rot"][k]) for k in kb]))
    rng = np.random.default_rng(SEED + int(round(1000 * p)))
    vals = []
    for _ in range(N_BOOT):
        ia = rng.integers(0, len(ka), len(ka))
        ib = rng.integers(0, len(kb), len(kb))
        va = np.median([np.median(a["diff_rot"][ka[i]]) for i in ia])
        vb = np.median([np.median(b["diff_rot"][kb[i]]) for i in ib])
        vals.append(float(va - vb))
    return dict(
        p=p,
        rot_delta_low=delta_lo,
        rot_delta_high=delta_hi,
        interaction_low_minus_high=delta_lo - delta_hi,
        interaction_ci=[float(np.percentile(vals, 2.5)),
                        float(np.percentile(vals, 97.5))],
    )


def main():
    rows = load_raw()
    out = dict(config=dict(
        n_raw=int(len(rows)), bins=BINS, p=P_LIST, n_boot=N_BOOT,
        bootstrap_unit="fragment pair (all 8 trials of a pair pooled)",
        tests=("exact McNemar on paired success; Wilcoxon signed-rank on "
               "per-pair median rotation difference; bootstrap CI over pairs"),
    ))
    cells = {}
    for p in P_LIST:
        for b in BINS:
            s = summarize(rows, b, p)
            if s:
                cells[f"{b}|{p:.1f}"] = s
    out["cells"] = cells
    out["interaction"] = {f"{p:.1f}": interaction(rows, p=p)
                          for p in P_LIST if interaction(rows, p=p)}
    # secondary stress cell: coherent outliers at 50%, where the full-set table
    # leaves the success ceiling (GNC 46% vs RANSAC 3% pooled)
    out["coherent_05"] = {b: summarize(rows, b, 0.5, otype="coherent")
                          for b in BINS}
    out["interaction_coherent_05"] = interaction(rows, p=0.5, otype="coherent")

    # clean geometry trend (p=0)
    tabs = json.loads((RES / "overlap_bins.json").read_text())["tables"]
    clean = {}
    for b in BINS:
        t = next((r for r in tabs if r["bin"] == b
                  and r["outlier_type"] == "random" and r["p"] == 0.0
                  and r["method"] == "GNC-GM"), None)
        s = cells.get(f"{b}|0.0")
        if t and s:
            clean[b] = dict(n_pairs=s["n_pairs"], n_trials=s["n_trials"],
                            base_inlier=t["base_inlier"],
                            overlap_mean=t["overlap_mean"],
                            clean_rot_median=s["rot_median_gnc"],
                            clean_rot_ransac=s["rot_median_ransac"])
    out["clean_geometry"] = clean

    OUT.write_text(json.dumps(out, indent=2))

    for p in P_LIST:
        print(f"== random p={p}")
        for b in BINS:
            s = cells.get(f"{b}|{p:.1f}")
            if not s:
                print(f"  {b:4s}: no pairs")
                continue
            print(f"  {b:4s}: pairs={s['n_pairs']:2d} trials={s['n_trials']:3d} "
                  f"succ={s['success_gnc']:.2f}/{s['success_ransac']:.2f} "
                  f"d={s['delta_success']:+.2f} (disc {s['discordant_gnc_only']}/"
                  f"{s['discordant_ransac_only']}, p={s['mcnemar_p']:.3g}) "
                  f"rot={s['rot_median_gnc']:.2f}/{s['rot_median_ransac']:.2f} "
                  f"d_rot={s['rot_delta_median']:+.3f} "
                  f"CI=[{s['rot_delta_ci'][0]:+.3f},{s['rot_delta_ci'][1]:+.3f}] "
                  f"wilcoxon_p={s['wilcoxon_p']:.3g} "
                  f"speedup={s['speedup_median']:.1f}x "
                  f"CI=[{s['speedup_ci'][0]:.1f},{s['speedup_ci'][1]:.1f}]")
    print("== coherent p=0.5")
    for b in BINS:
        s = out["coherent_05"].get(b)
        if not s:
            continue
        print(f"  {b:4s}: pairs={s['n_pairs']:2d} succ G/R="
              f"{s['success_gnc']:.2f}/{s['success_ransac']:.2f} "
              f"d={s['delta_success']:+.2f} disc={s['discordant_gnc_only']}/"
              f"{s['discordant_ransac_only']} p={s['mcnemar_p']:.3g} "
              f"rot={s['rot_median_gnc']:.2f}/{s['rot_median_ransac']:.2f}")
    ic = out["interaction_coherent_05"]
    if ic:
        print(f"  interaction low-high: {ic['interaction_low_minus_high']:+.3f} "
              f"CI=[{ic['interaction_ci'][0]:+.3f},{ic['interaction_ci'][1]:+.3f}]")
    print("== interaction (low - high, per-pair median rot delta)")
    for p, v in out["interaction"].items():
        if v:
            print(f"  p={p}: {v['interaction_low_minus_high']:+.3f} "
                  f"CI=[{v['interaction_ci'][0]:+.3f},{v['interaction_ci'][1]:+.3f}]")
    print("== clean geometry")
    for b, v in clean.items():
        print(f"  {b:4s}: pairs={v['n_pairs']} base_inlier={v['base_inlier']:.3f} "
              f"clean_rot={v['clean_rot_median']:.3f} "
              f"overlap={v['overlap_mean']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
