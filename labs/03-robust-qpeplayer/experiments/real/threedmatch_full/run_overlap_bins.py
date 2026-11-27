#!/usr/bin/env python3
"""Low / mid / high overlap study on the full 3DMatch set.

Question
--------
The full-set table (``run_full_scenes.py``) only samples the 8 highest-overlap
fragment pairs of each scene, and the 3DMatch benchmark itself only lists
pairs above roughly 30% overlap.  Does GNC-GM's advantage over RANSAC hold
when the base geometry is a genuinely low-overlap fragment pair, and is the
advantage overlap-dependent?

Design
------
1. For **every** pair in each scene's ``gt.log`` we compute the *uncapped*
   GT-NN overlap fraction (``full_common.overlap_fraction``): sample 2000
   source points in fragment id1, map them into id2 with inv(gt.log), and
   count the fraction whose nearest neighbour in id2 is within tau=0.06 m.
   The parent loader's helper caps the returned count at N and therefore
   saturates at 4x overlap; this estimator does not.
2. Bin the pairs: low < 0.5, mid 0.5-0.8, high > 0.8.
3. Pick up to 3 pairs per bin per scene, spread over the bin's overlap range.
4. Run the identical robust protocol on those pairs: N=300 (padded by
   resampling for the few pairs that yield fewer valid GT-NN pairs), c=0.05 m,
   random outliers at p in {0, 0.3, 0.5, 0.7}, plus coherent-outlier cells
   at p in {0.3, 0.5} as a secondary stress check, 8 trials/cell, 5 methods.
5. Report success / rotation error / runtime by overlap bin, the GNC-vs-RANSAC
   success delta and speedup per bin, and per-pair success vs overlap with a
   Spearman correlation.  The clean (p=0) rows measure how the real GT-NN
   residual and inlier fraction depend on overlap.

Outputs (``results/``): ``overlap_bins.json``, ``overlap_bins.md``,
``overlap_bins.png``, ``overlap_bins_raw.jsonl``, ``overlap_scan.json`` (all
pairs of all scenes), ``logs/``.

Run (Slurm):
  srun --partition=cpu --time=00:50:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python run_overlap_bins.py 2>&1 | \
    tee results/logs/overlap_bins_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_common as F  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

N = F.N
C = F.C
TAU = F.TAU
VOXEL = F.VOXEL
P_LIST = [0.0, 0.3, 0.5, 0.7]
COHERENT_P = [0.3, 0.5]
TRIALS = 8
PER_BIN = 3
N_PROBE = 2000
BINS = [("low", None, 0.5), ("mid", 0.5, 0.8), ("high", 0.8, None)]
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261127


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(x) for x in parts]])


def bin_label(v):
    return F.overlap_bin(v, low=0.5, high=0.8)


def phase_scan():
    scan, scenes = {}, {}
    t0 = time.time()
    for si, scene in enumerate(F.SCENES):
        clouds = F.L.load_scene(F.ROOT / scene, voxel=VOXEL)
        pairs = F.L.parse_gt_log(F.ROOT / f"{scene}-evaluation" / "gt.log")
        rows = F.scan_scene_overlaps(clouds, pairs, n_probe=N_PROBE, tau=TAU, seed=si)
        for r in rows:
            r["bin"] = bin_label(r["overlap"])
        vals = np.array([r["overlap"] for r in rows])
        counts = {b: int(sum(1 for r in rows if r["bin"] == b)) for b in ["low", "mid", "high"]}
        scan[scene] = rows
        scenes[scene] = dict(
            short=F.SCENE_SHORT[scene], n_fragments=len(clouds), n_pairs=len(rows),
            overlap_quantiles={str(q): float(np.percentile(vals, q))
                               for q in [0, 10, 25, 50, 75, 90, 100]},
            bin_counts=counts,
            med_res_median=float(np.median([r["med_res"] for r in rows
                                            if r["med_res"] is not None])),
        )
        print(f"[scan] {scene}: {len(rows)} pairs, bins {counts}, "
              f"overlap median {np.median(vals):.3f} ({time.time()-t0:.1f}s)", flush=True)
    return scan, scenes


def select_pairs(scan):
    """Pick bin pairs as *copies* of the scan rows.

    ``main`` attaches a numpy ``Tinv`` to every selected pair; if the selected
    dicts were the same objects as ``scan`` rows, writing ``overlap_scan.json``
    would fail on the ndarray.  Copying keeps the scan artifact pure JSON.
    """
    by_scene = {}
    for scene, rows in scan.items():
        for r in rows:
            r["bin"] = bin_label(r["overlap"])
        picked = F.pick_bin_pairs(rows, per_bin=PER_BIN)
        by_scene[scene] = {b: [dict(r) for r in picked[b]] for b in picked}
    return by_scene


def run_cell(clouds, pair, p, otype, trial, si, pair_id, bin_idx, methods):
    key = {"random": 1, "coherent": 2}[otype]
    rng = np.random.default_rng(_seed(si * 1000 + pair_id, int(round(100 * p)), key, trial))
    s = F.clean_pair_sample(clouds, pair, rng)
    if s is None:
        return []
    x, y0, d0 = s["x"], s["y"], s["d"]
    Rgt, tgt = s["Rgt"], s["tgt"]
    base_in = float(np.mean(d0 < C))
    if otype == "random":
        y = F.inject_random(x, y0, rng, p, clouds[pair["id2"]])
    else:
        y = y0.copy()
        n_out = int(round(p * len(x)))
        if n_out > 0:
            from common import rotation_about_axis
            idx = rng.choice(len(x), n_out, replace=False)
            axis = rng.normal(size=3)
            ang = np.deg2rad(rng.uniform(30.0, 60.0))
            R_delta = rotation_about_axis(axis, ang)
            Rout = R_delta @ Rgt
            t_shift = rng.normal(size=3)
            t_shift = t_shift / np.linalg.norm(t_shift) * rng.uniform(0.5, 1.5)
            tout = R_delta @ tgt + t_shift
            y[idx] = (Rout @ x[idx].T).T + tout
    recs = []
    for name in METHODS:
        rrng = np.random.default_rng(_seed(7, si * 1000 + pair_id,
                                           int(round(100 * p)), trial))
        t1 = time.perf_counter()
        R, t, info = methods[name](x, y, rrng)
        dt = time.perf_counter() - t1
        re, te = F.cm.pose_errors(R, t, Rgt, tgt)
        recs.append(dict(scene=pair["scene"], scene_short=F.SCENE_SHORT[pair["scene"]],
                         pair=f"{pair['id1']}-{pair['id2']}", bin=bin_idx,
                         overlap=pair["overlap"], n_valid=s["n_valid"],
                         p=p, outlier_type=otype, trial=trial, method=name, n=len(x),
                         base_inlier=base_in, success=F.cm.success(re, te),
                         rot=re, trans=te, time_ms=1000.0 * dt))
    return recs


def phase_run(scan, selected):
    methods = F.make_methods(c=C)
    raw = []
    t0 = time.time()
    for si, scene in enumerate(F.SCENES):
        clouds = F.L.load_scene(F.ROOT / scene, voxel=VOXEL)
        for b in ["low", "mid", "high"]:
            counter = 0
            for pi, pair in enumerate(selected[scene][b]):
                pair = dict(pair)
                if pair.get("Tinv") is None:
                    raise RuntimeError("missing Tinv")
                pair["scene"] = scene
                pair_id = {"low": 0, "mid": 1, "high": 2}[b] * 10 + pi
                counter += 1
                for p in P_LIST:
                    for trial in range(TRIALS):
                        raw += run_cell(clouds, pair, p, "random", trial, si,
                                        pair_id, b, methods)
                for p in COHERENT_P:
                    for trial in range(TRIALS):
                        raw += run_cell(clouds, pair, p, "coherent", trial, si,
                                        pair_id, b, methods)
            print(f"[run] {scene} bin={b}: {time.time()-t0:.1f}s, rows={len(raw)}",
                  flush=True)
    return raw


def aggregate(raw, selected):
    tables = {}
    for b in ["low", "mid", "high"]:
        for otype, ps in [("random", P_LIST), ("coherent", COHERENT_P)]:
            for p in ps:
                for m in METHODS:
                    sel = [r for r in raw if r["bin"] == b and r["outlier_type"] == otype
                           and r["method"] == m and abs(r["p"] - p) < 1e-9]
                    if not sel:
                        continue
                    s = F.cm.summarize_errors([r["rot"] for r in sel],
                                              [r["trans"] for r in sel],
                                              [r["time_ms"] / 1000.0 for r in sel])
                    tables[f"{b}|{otype}|{p:.1f}|{m}"] = dict(
                        bin=b, outlier_type=otype, p=p, method=m,
                        n_trials=len(sel),
                        overlap_mean=float(np.mean([r["overlap"] for r in sel])),
                        base_inlier=float(np.mean([r["base_inlier"] for r in sel])),
                        n_pairs=len({(r["scene"], r["pair"]) for r in sel}), **s)
    return tables


def advantage(tables):
    """GNC vs RANSAC-1000 per bin at each p, plus runtime ratio."""
    out = {}
    for b in ["low", "mid", "high"]:
        for p in P_LIST:
            g = tables.get(f"{b}|random|{p:.1f}|GNC-GM")
            r = tables.get(f"{b}|random|{p:.1f}|RANSAC-1000")
            i = tables.get(f"{b}|random|{p:.1f}|IRLS-GM")
            if not (g and r):
                continue
            out[f"{b}|{p:.1f}"] = dict(
                bin=b, p=p,
                success_gnc=g["success"], success_ransac=r["success"],
                success_irls=i["success"] if i else None,
                delta_gnc_minus_ransac=g["success"] - r["success"],
                rot_gnc=g["rot_median"], rot_ransac=r["rot_median"],
                time_gnc=g["time_ms_p50"], time_ransac=r["time_ms_p50"],
                speedup_ransac_over_gnc=(r["time_ms_p50"] / g["time_ms_p50"]
                                          if g["time_ms_p50"] > 0 else None),
            )
    return out


def pair_level(raw):
    """Per selected pair: success of each method at p=0.5 random, plus clean stats."""
    out = []
    for b in ["low", "mid", "high"]:
        pairs = sorted({(r["scene"], r["pair"]) for r in raw
                        if r["bin"] == b and r["outlier_type"] == "random"})
        for scene, pair in pairs:
            row = dict(scene=scene, pair=pair, bin=b)
            base = [r for r in raw if r["scene"] == scene and r["pair"] == pair
                    and r["outlier_type"] == "random" and abs(r["p"] - 0.0) < 1e-9]
            if base:
                row["overlap"] = float(np.mean([r["overlap"] for r in base]))
                row["n_valid"] = int(np.median([r["n_valid"] for r in base]))
                row["base_inlier"] = float(np.mean([r["base_inlier"] for r in base]))
                row["clean_rot"] = float(np.median([r["rot"] for r in base
                                                    if r["method"] == "GNC-GM"]))
            for m in METHODS:
                cell = [r for r in raw if r["scene"] == scene and r["pair"] == pair
                        and r["outlier_type"] == "random" and r["method"] == m
                        and abs(r["p"] - 0.5) < 1e-9]
                if cell:
                    row[f"succ_{m}"] = float(np.mean([r["success"] for r in cell]))
            out.append(row)
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None, None
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def fmt_rho(rho, p, what="success"):
    if rho is None:
        return f"undefined ({what} is constant across the selected pairs)"
    return f"{rho:+.2f} (p={p:.3f})"


def write_markdown(scan_summary, tables, adv, pl, config):
    lines = ["# 3DMatch low / mid / high overlap study (all 8 scenes)", ""]
    lines.append(f"Overlap = uncapped GT-NN fraction (fraction of 2000 sampled source points "
                 f"mapped by inv(gt.log) with a target NN within tau={TAU} m). Bins: "
                 f"low < 0.5, mid 0.5-0.8, high > 0.8. N={N}, c={C} m, random outliers at "
                 f"p in {P_LIST}, plus coherent outliers at p in {COHERENT_P}; "
                 f"{TRIALS} trials/cell; up to {PER_BIN} pairs per bin per scene.")
    lines.append("")
    lines.append("## Overlap distribution of all gt.log pairs")
    lines.append("")
    lines.append("| scene | fragments | pairs | low | mid | high | overlap p0 / p50 / p100 |")
    lines.append("|---|---|---|---|---|---|---|")
    for scene, s in scan_summary.items():
        q = s["overlap_quantiles"]
        lines.append(f"| {s['short']} | {s['n_fragments']} | {s['n_pairs']} | "
                     f"{s['bin_counts']['low']} | {s['bin_counts']['mid']} | "
                     f"{s['bin_counts']['high']} | {q['0']:.2f} / {q['50']:.2f} / {q['100']:.2f} |")
    lines.append("")
    lines.append("## Success rate by overlap bin (random outliers)")
    lines.append("")
    hdr = "| bin | p | " + " | ".join(METHODS) + " |"
    lines.append(hdr)
    lines.append("|---|---|" + "|".join(["---"] * len(METHODS)) + "|")
    for b in ["low", "mid", "high"]:
        for p in P_LIST:
            cells = []
            for m in METHODS:
                t = tables.get(f"{b}|random|{p:.1f}|{m}")
                cells.append(f"{100 * t['success']:.0f}" if t else "—")
            lines.append(f"| {b} | {int(100 * p)}% | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Median rotation error by overlap bin (random outliers, deg)")
    lines.append("")
    lines.append(hdr)
    lines.append("|---|---|" + "|".join(["---"] * len(METHODS)) + "|")
    for b in ["low", "mid", "high"]:
        for p in P_LIST:
            cells = []
            for m in METHODS:
                t = tables.get(f"{b}|random|{p:.1f}|{m}")
                cells.append(f"{t['rot_median']:.2f}" if t else "—")
            lines.append(f"| {b} | {int(100 * p)}% | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## GNC advantage vs overlap (random outliers)")
    lines.append("")
    lines.append("| bin | p | GNC success | RANSAC-1000 success | Δ success | GNC ms p50 | RANSAC ms p50 | speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, a in adv.items():
        lines.append(f"| {a['bin']} | {int(100 * a['p'])}% | {100 * a['success_gnc']:.0f} | "
                     f"{100 * a['success_ransac']:.0f} | {100 * a['delta_gnc_minus_ransac']:+.0f} | "
                     f"{a['time_gnc']:.2f} | {a['time_ransac']:.1f} | "
                     f"{a['speedup_ransac_over_gnc']:.1f}x |")
    lines.append("")
    lines.append("## Coherent outliers by overlap bin (success %)")
    lines.append("")
    lines.append(hdr)
    lines.append("|---|---|" + "|".join(["---"] * len(METHODS)) + "|")
    for b in ["low", "mid", "high"]:
        for p in COHERENT_P:
            cells = []
            for m in METHODS:
                t = tables.get(f"{b}|coherent|{p:.1f}|{m}")
                cells.append(f"{100 * t['success']:.0f}" if t else "—")
            lines.append(f"| {b} | {int(100 * p)}% | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Coherent outliers by overlap bin (median rotation error, deg)")
    lines.append("")
    lines.append(hdr)
    lines.append("|---|---|" + "|".join(["---"] * len(METHODS)) + "|")
    for b in ["low", "mid", "high"]:
        for p in COHERENT_P:
            cells = []
            for m in METHODS:
                t = tables.get(f"{b}|coherent|{p:.1f}|{m}")
                cells.append(f"{t['rot_median']:.2f}" if t else "—")
            lines.append(f"| {b} | {int(100 * p)}% | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-pair overlap dependence")
    lines.append("")
    pairs_at = [r for r in pl if "succ_GNC-GM" in r and "succ_RANSAC-1000" in r]
    rho_g, pg = spearman([r["overlap"] for r in pairs_at],
                         [r["succ_GNC-GM"] for r in pairs_at])
    rho_r, pr = spearman([r["overlap"] for r in pairs_at],
                         [r["succ_RANSAC-1000"] for r in pairs_at])
    rho_c, pc = spearman([r["overlap"] for r in pairs_at], [r["clean_rot"] for r in pairs_at])
    lines.append(f"* GNC success vs pair overlap at p=0.5: Spearman rho="
                 f"{fmt_rho(rho_g, pg, 'GNC success')}")
    lines.append(f"* RANSAC-1000 success vs pair overlap at p=0.5: rho="
                 f"{fmt_rho(rho_r, pr, 'RANSAC success')}")
    lines.append(f"* clean median rotation error vs pair overlap: rho={rho_c:+.2f} (p={pc:.3f})")
    lines.append("")
    lines.append("| scene | pair | bin | overlap | n_valid | base inlier | clean rot | GNC succ p=0.5 | RANSAC succ p=0.5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(pl, key=lambda r: r["overlap"]):
        lines.append(f"| {F.SCENE_SHORT[r['scene']]} | {r['pair']} | {r['bin']} | "
                     f"{r['overlap']:.2f} | {r.get('n_valid', '—')} | "
                     f"{r.get('base_inlier', float('nan')):.2f} | {r.get('clean_rot', float('nan')):.3f} | "
                     f"{r.get('succ_GNC-GM', float('nan')):.2f} | "
                     f"{r.get('succ_RANSAC-1000', float('nan')):.2f} |")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("* Base correspondences are GT-NN oracle matches on real fragments, so low overlap "
                 "does not degrade *correctness* of base matches; it changes the residual "
                 "distribution and the share of boundary correspondences near c. The clean p=0 "
                 "rows and `base_inlier` quantify that real-geometry effect.")
    lines.append("* Pairs whose valid GT-NN count is below N=300 are padded by resampling with "
                 "replacement; `n_valid` is reported per pair.")
    lines.append("* The overlap estimator is a surface-proximity proxy with tau=0.06 m; it can "
                 "overestimate overlap when large planar structures (floors/ceilings) are shared "
                 "by fragments that see different parts of the room.")
    return "\n".join(lines) + "\n"


def make_figure(tables, adv, pl, scan, scan_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue",
              "GNC-GM": "tab:red"}
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    width = 0.16
    xpos = np.arange(3)
    blabels = ["low\n(<0.5)", "mid\n(0.5-0.8)", "high\n(>0.8)"]

    def bin_bars(ax, key, col, title, ylabel, scale=1.0, log=False, legend=False):
        for k, m in enumerate(METHODS):
            vals = []
            for b in ["low", "mid", "high"]:
                t = tables.get(f"{b}|{key}|{m}")
                vals.append(scale * t[col] if t else np.nan)
            ax.bar(xpos + (k - 2) * width, vals, width, color=colors[m], label=m)
        ax.set_xticks(xpos)
        ax.set_xticklabels(blabels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if log:
            ax.set_yscale("log")
        if legend:
            ax.legend(fontsize=6, ncol=2)

    # 1: per-pair success vs overlap at p=0.5
    ax = axes[0, 0]
    pts = [r for r in pl if "succ_GNC-GM" in r]
    for m in ["RANSAC-1000", "GNC-GM", "IRLS-GM"]:
        ax.scatter([r["overlap"] for r in pts], [r[f"succ_{m}"] for r in pts],
                   s=22, color=colors[m], label=m, alpha=0.75)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.axvline(0.8, color="k", ls="--", lw=0.8)
    ax.set_xlabel("GT-NN overlap")
    ax.set_ylabel("success at p=0.5 random")
    ax.set_title("Per-pair success vs overlap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # 2: success by bin at p=0.5
    bin_bars(axes[0, 1], "random|0.5", "success",
             "Random outliers p=0.5: success by overlap bin",
             "success (%)", scale=100.0, legend=True)

    # 3: clean rotation by bin (p=0) -- the real overlap dependence
    bin_bars(axes[0, 2], "random|0.0", "rot_median",
             "Clean (p=0): median rotation error by overlap bin",
             "median rotation error (deg)")

    # 4: coherent p=0.5 success by bin (stress cell where the ceiling breaks)
    bin_bars(axes[0, 3], "coherent|0.5", "success",
             "Coherent outliers p=0.5: success by overlap bin",
             "success (%)", scale=100.0, legend=True)

    # 5: rotation error by bin at p=0.5
    bin_bars(axes[1, 0], "random|0.5", "rot_median",
             "Random outliers p=0.5: rotation error",
             "median rotation error (deg)")

    # 6: runtime by bin at p=0.5
    bin_bars(axes[1, 1], "random|0.5", "time_ms_p50",
             "Random outliers p=0.5: runtime by overlap bin",
             "runtime p50 (ms, log)", log=True)

    # 7: per-scene overlap distributions of all gt.log pairs
    ax = axes[1, 2]
    scenes = list(scan)
    data = [[r["overlap"] for r in scan[s]] for s in scenes]
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axhline(0.8, color="k", ls="--", lw=0.8)
    ax.set_xticks(np.arange(1, len(scenes) + 1))
    ax.set_xticklabels([F.SCENE_SHORT[s] for s in scenes], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("GT-NN overlap (all pairs)")
    ax.set_title("Overlap distribution per scene"); ax.grid(alpha=0.3, axis="y")

    # 8: GNC minus RANSAC median rotation error at p=0.5, by bin
    ax = axes[1, 3]
    deltas = []
    for b in ["low", "mid", "high"]:
        g = tables.get(f"{b}|random|0.5|GNC-GM")
        r = tables.get(f"{b}|random|0.5|RANSAC-1000")
        deltas.append(1000 * (g["rot_median"] - r["rot_median"]) if g and r else np.nan)
    ax.bar(xpos, deltas, 0.5, color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(blabels)
    ax.set_ylabel("GNC - RANSAC median rot error (mdeg)")
    ax.set_title("Rotation advantage at p=0.5 (negative = GNC better)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("3DMatch full set: low / mid / high overlap bins")
    fig.tight_layout()
    return fig


def main():
    t0 = time.time()
    scan, scan_summary = phase_scan()
    selected = select_pairs(scan)
    # attach Tinv to selected pairs (recover inv(gt) per pair)
    for scene, bins in selected.items():
        gt = F.L.parse_gt_log(F.ROOT / f"{scene}-evaluation" / "gt.log")
        lut = {(id1, id2): T for id1, id2, n, T in gt}
        for b in ["low", "mid", "high"]:
            for r in bins[b]:
                r["Tinv"] = np.linalg.inv(lut[(r["id1"], r["id2"])])
    n_sel = sum(len(v[b]) for v in selected.values() for b in ["low", "mid", "high"])
    print(f"selected {n_sel} pairs across {len(selected)} scenes "
          f"({(time.time()-t0):.1f}s)", flush=True)
    raw = phase_run(scan, selected)
    tables = aggregate(raw, selected)
    adv = advantage(tables)
    pl = pair_level(raw)
    config = dict(scenes=F.SCENES, n=N, c=C, tau=TAU, voxel=VOXEL, p=P_LIST,
                  coherent_p=COHERENT_P, trials=TRIALS, per_bin=PER_BIN,
                  n_probe=N_PROBE, bin_edges=[0.5, 0.8], methods=METHODS,
                  data_root=str(F.ROOT))
    F.cm.json_dump(OUT / "overlap_bins.json", dict(
        config=config, scene_summary=scan_summary,
        selected_pairs={F.SCENE_SHORT[s]: {b: [{k: v for k, v in r.items() if k != "Tinv"}
                                              for r in selected[s][b]]
                                           for b in ["low", "mid", "high"]}
                        for s in selected},
        tables=list(tables.values()), advantage=adv, pair_level=pl))
    with open(OUT / "overlap_scan.json", "w") as f:
        json.dump({F.SCENE_SHORT[s]: rows for s, rows in scan.items()}, f, indent=2)
    with open(OUT / "overlap_bins_raw.jsonl", "w") as f:
        for rec in raw:
            f.write(json.dumps(rec) + "\n")
    (OUT / "overlap_bins.md").write_text(write_markdown(scan_summary, tables, adv, pl, config))
    fig = make_figure(tables, adv, pl, scan, scan_summary)
    fig.savefig(OUT / "overlap_bins.png", dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", OUT / "overlap_bins.{json,md,png}", f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
