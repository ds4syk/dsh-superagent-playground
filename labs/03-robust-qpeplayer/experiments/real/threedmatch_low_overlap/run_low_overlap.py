#!/usr/bin/env python3
"""3DMatch *true low overlap* (uncapped overlap 0.20-0.50) robust-protocol study.

Motivation
----------
The existing full-set study (``../threedmatch_full/``) bins pairs as
low < 0.5 / mid 0.5-0.8 / high > 0.8.  Its "low" bin (mean overlap ~0.39)
still mixes pairs down to ~0.03, and the official 3DMatch release ships no
per-pair overlap, so the label "low overlap" needs an explicit, uncapped
number attached to it.  This study:

1. rescan **every** gt.log pair of all 8 official scene sets with the uncapped
   GT-NN overlap estimator (2000 probes, tau = 0.06 m) and write
   ``results/overlap_scan.json`` (same estimator/seed as the parent script, so
   the values are directly comparable);
2. select pairs with uncapped overlap in **[0.20, 0.50]** - genuinely low
   geometric overlap, not the 0.95 "high-overlap" base-inlier bin - up to 4 per
   scene spread over the band;
3. run the identical robust protocol (N=300, c=0.05 m, random outliers at
   p in {0, 0.3, 0.5, 0.7}, **10 trials/cell**, 5 methods) on those pairs;
4. run a matched high-overlap control (overlap >= 0.80, up to 2 pairs/scene,
   same 10 trials/cell) so the comparison against the existing high-overlap
   table is apples-to-apples;
5. record per-pair clean GT-NN inlier fraction, uncapped kept-correspondence
   count, and whether the 300-point budget had to be padded.

The key question is not whether a robust solver recovers the pose (it does),
but whether *geometric* overlap degrades the *correspondence quality* that the
oracle GT-NN base provides.  If the clean base inlier fraction stays > 0.9,
the "low overlap" label describes the fraction of the scene surface shared by
the two fragments, not the fraction of outlier correspondences - a distinction
the paper must state explicitly.

Read-only imports: ``../threedmatch_full/full_common.py``, the parent
``../threedmatch/load_3dmatch.py`` and shared ``../../synthetic/common.py``.

Run (bounded local process; the Slurm cpu node was occupied):
  taskset -c 0-3 /home/syk/miniconda3/envs/WM/bin/python run_low_overlap.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "threedmatch_full"
sys.path.insert(0, str(FULL))
import full_common as F  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

N = F.N
C = F.C
TAU = F.TAU
VOXEL = F.VOXEL
P_LIST = [0.0, 0.3, 0.5, 0.7]
TRIALS = 10
N_PROBE = 2000
LOW_BAND = (0.20, 0.50)
HIGH_MIN = 0.80
PER_SCENE_LOW = 4
PER_SCENE_HIGH = 2
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
BASE_SEED = 20261128


def _seed(*parts):
    return np.random.SeedSequence([BASE_SEED, *[int(x) for x in parts]])


def band_label(v):
    if v < LOW_BAND[0]:
        return "very_low"
    if v <= LOW_BAND[1]:
        return "low_20_50"
    if v > HIGH_MIN:
        return "high"
    return "mid"


def phase_scan():
    scan, scenes = {}, {}
    t0 = time.time()
    for si, scene in enumerate(F.SCENES):
        clouds = F.L.load_scene(F.ROOT / scene, voxel=VOXEL)
        pairs = F.L.parse_gt_log(F.ROOT / f"{scene}-evaluation" / "gt.log")
        rows = F.scan_scene_overlaps(clouds, pairs, n_probe=N_PROBE, tau=TAU, seed=si)
        for r in rows:
            r["band"] = band_label(r["overlap"])
        vals = np.array([r["overlap"] for r in rows])
        in_low = [r for r in rows if LOW_BAND[0] <= r["overlap"] <= LOW_BAND[1]]
        in_high = [r for r in rows if r["overlap"] > HIGH_MIN]
        scenes[scene] = dict(
            short=F.SCENE_SHORT[scene], n_fragments=len(clouds), n_pairs=len(rows),
            overlap_quantiles={str(q): float(np.percentile(vals, q))
                               for q in [0, 10, 25, 50, 75, 90, 100]},
            band_counts={b: int(sum(1 for r in rows if r["band"] == b))
                         for b in ["very_low", "low_20_50", "mid", "high"]},
            n_pairs_low_band=len(in_low), n_pairs_high_band=len(in_high))
        scan[scene] = rows
        print(f"[scan] {scene}: {len(rows)} pairs, band "
              f"{scenes[scene]['band_counts']}, "
              f"low[0.2,0.5]={len(in_low)}, high>0.8={len(in_high)} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return scan, scenes


def pick(rows, lo, hi, k, inclusive_hi=True):
    """Up to k rows with lo <= overlap <= hi, spread over the band's range."""
    sel = [r for r in rows
           if r["overlap"] >= lo and (r["overlap"] <= hi if inclusive_hi else r["overlap"] < hi)]
    sel = sorted(sel, key=lambda r: r["overlap"])
    if len(sel) <= k:
        return sel
    pos = np.linspace(0, len(sel) - 1, k).round().astype(int)
    return [sel[i] for i in sorted(set(pos.tolist()))]


def select(scan):
    gt = {}
    for scene in F.SCENES:
        pairs = F.L.parse_gt_log(F.ROOT / f"{scene}-evaluation" / "gt.log")
        gt[scene] = {(int(a), int(b)): T for a, b, _n, T in pairs}
    out = {}
    for scene in F.SCENES:
        rows = scan[scene]
        low = pick(rows, LOW_BAND[0], LOW_BAND[1], PER_SCENE_LOW)
        high = pick(rows, HIGH_MIN, 1.0001, PER_SCENE_HIGH)
        sel = []
        for band, items in [("low_20_50", low), ("high", high)]:
            for r in items:
                T = gt[scene][(int(r["id1"]), int(r["id2"]))]
                sel.append(dict(r, band=band, Tinv=np.linalg.inv(T)))
        out[scene] = sel
    return out


def run_cell(clouds, pair, p, trial, si, pair_id, methods):
    rng = np.random.default_rng(
        _seed(si * 1000 + pair_id, int(round(100 * p)), 1, trial))
    s = F.clean_pair_sample(clouds, pair, rng)
    if s is None:
        return []
    x, y0, d0 = s["x"], s["y"], s["d"]
    Rgt, tgt = s["Rgt"], s["tgt"]
    base_in = float(np.mean(d0 < C))
    y = F.inject_random(x, y0, rng, p, clouds[pair["id2"]])
    recs = []
    for name in METHODS:
        rrng = np.random.default_rng(
            _seed(7, si * 1000 + pair_id, int(round(100 * p)), trial))
        t1 = time.perf_counter()
        R, t, _info = methods[name](x, y, rrng)
        dt = time.perf_counter() - t1
        re, te = F.cm.pose_errors(R, t, Rgt, tgt)
        recs.append(dict(scene=pair["scene"], scene_short=F.SCENE_SHORT[pair["scene"]],
                         pair=f"{pair['id1']}-{pair['id2']}", band=pair["band"],
                         overlap=pair["overlap"], n_kept_2000=pair["n_kept"],
                         med_res=pair["med_res"], n_valid=s["n_valid"],
                         p=p, outlier_type="random", trial=trial, method=name,
                         n=len(x), base_inlier=base_in,
                         success=F.cm.success(re, te), rot=re, trans=te,
                         time_ms=1000.0 * dt))
    return recs


def phase_run(scan, selected):
    methods = F.make_methods(c=C)
    raw = []
    t0 = time.time()
    for si, scene in enumerate(F.SCENES):
        clouds = F.L.load_scene(F.ROOT / scene, voxel=VOXEL)
        for pi, pair in enumerate(selected[scene]):
            pair = dict(pair)
            pair["scene"] = scene
            bid = {"low_20_50": 0, "high": 1}[pair["band"]]
            pair_id = bid * 10 + pi
            for p in P_LIST:
                for trial in range(TRIALS):
                    raw += run_cell(clouds, pair, p, trial, si, pair_id, methods)
            print(f"[run] {scene} {pair['band']} pair={pair['id1']}-{pair['id2']} "
                  f"ov={pair['overlap']:.3f} n_kept={pair['n_kept']} "
                  f"({time.time()-t0:.1f}s, rows={len(raw)})", flush=True)
    return raw


def aggregate(raw):
    tables = []
    for band in ["low_20_50", "high"]:
        for p in P_LIST:
            for m in METHODS:
                sel = [r for r in raw if r["band"] == band and r["method"] == m
                       and abs(r["p"] - p) < 1e-9]
                if not sel:
                    continue
                s = F.cm.summarize_errors([r["rot"] for r in sel],
                                          [r["trans"] for r in sel],
                                          [r["time_ms"] / 1000.0 for r in sel])
                tables.append(dict(
                    band=band, p=p, method=m, n_trials=len(sel),
                    overlap_mean=float(np.mean([r["overlap"] for r in sel])),
                    overlap_min=float(np.min([r["overlap"] for r in sel])),
                    overlap_max=float(np.max([r["overlap"] for r in sel])),
                    base_inlier=float(np.mean([r["base_inlier"] for r in sel])),
                    base_inlier_min=float(np.min([r["base_inlier"] for r in sel])),
                    n_valid_median=float(np.median([r["n_valid"] for r in sel])),
                    n_kept_median=float(np.median([r["n_kept_2000"] for r in sel])),
                    padded_frac=float(np.mean([r["n_valid"] < N for r in sel])),
                    n_pairs=len({(r["scene"], r["pair"]) for r in sel}), **s))
    return tables


def pair_level(raw):
    out = []
    for band in ["low_20_50", "high"]:
        keys = sorted({(r["scene"], r["pair"]) for r in raw if r["band"] == band})
        for scene, pair in keys:
            base = [r for r in raw if r["scene"] == scene and r["pair"] == pair
                    and r["band"] == band and abs(r["p"]) < 1e-9]
            row = dict(scene=scene, scene_short=F.SCENE_SHORT[scene], pair=pair,
                       band=band)
            if base:
                row.update(overlap=float(np.mean([r["overlap"] for r in base])),
                           n_kept_2000=int(np.median([r["n_kept_2000"] for r in base])),
                           n_valid=int(np.median([r["n_valid"] for r in base])),
                           base_inlier=float(np.mean([r["base_inlier"] for r in base])),
                           clean_rot_gnc=float(np.median(
                               [r["rot"] for r in base if r["method"] == "GNC-GM"])))
            for m in METHODS:
                for p in P_LIST:
                    cell = [r for r in raw if r["scene"] == scene and r["pair"] == pair
                            and r["band"] == band and r["method"] == m
                            and abs(r["p"] - p) < 1e-9]
                    if cell:
                        row[f"succ_{m}_p{p:.1f}"] = float(np.mean(
                            [r["success"] for r in cell]))
            out.append(row)
    return out


def advantage(tables):
    idx = {(t["band"], t["p"], t["method"]): t for t in tables}
    out = []
    for band in ["low_20_50", "high"]:
        for p in P_LIST:
            g, r_, i = (idx.get((band, p, m)) for m in
                        ["GNC-GM", "RANSAC-1000", "IRLS-GM"])
            if not (g and r_):
                continue
            out.append(dict(
                band=band, p=p,
                success_gnc=g["success"], success_ransac=r_["success"],
                success_irls=i["success"] if i else None,
                delta_gnc_minus_ransac=g["success"] - r_["success"],
                rot_gnc=g["rot_median"], rot_ransac=r_["rot_median"],
                time_gnc=g["time_ms_p50"], time_ransac=r_["time_ms_p50"],
                speedup_ransac_over_gnc=(r_["time_ms_p50"] / g["time_ms_p50"]
                                         if g["time_ms_p50"] > 0 else None)))
    return out


def high_reference():
    """Existing high-overlap bin (>0.8) from the parent 8-trial table, for context."""
    path = FULL / "results" / "overlap_bins.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    want = {}
    for t in d.get("tables", []):
        if t.get("bin") == "high" and t.get("outlier_type") == "random":
            want[f"{t['p']:.1f}|{t['method']}"] = dict(
                p=t["p"], method=t["method"], n_trials=t["n_trials"],
                overlap_mean=t["overlap_mean"], base_inlier=t["base_inlier"],
                success=t["success"], rot_median=t["rot_median"],
                trans_median=t["trans_median"], time_ms_p50=t["time_ms_p50"])
    return dict(source="../threedmatch_full/results/overlap_bins.json",
                trials=d.get("config", {}).get("trials"), tables=want)


def main():
    t0 = time.time()
    scan, scan_summary = phase_scan()
    # same keying/format as the parent's documented overlap_scan.json
    with open(OUT / "overlap_scan.json", "w") as f:
        json.dump({F.SCENE_SHORT[s]: rows for s, rows in scan.items()}, f, indent=2)
    print(f"[scan] wrote overlap_scan.json ({len(scan)} scenes, "
          f"{sum(len(v) for v in scan.values())} pairs) in {time.time()-t0:.1f}s",
          flush=True)

    selected = select(scan)
    n_sel = {F.SCENE_SHORT[s]: len(v) for s, v in selected.items()}
    print(f"[select] {sum(n_sel.values())} pairs: {n_sel}", flush=True)

    raw = phase_run(scan, selected)
    print(f"[run] {len(raw)} per-trial rows in {time.time()-t0:.1f}s", flush=True)

    tables = aggregate(raw)
    adv = advantage(tables)
    pl = pair_level(raw)

    with open(OUT / "low_overlap_raw.jsonl", "w") as f:
        for rec in raw:
            f.write(json.dumps(rec) + "\n")

    out = dict(
        config=dict(scenes=F.SCENES, n=N, c=C, tau=TAU, voxel=VOXEL,
                    p=P_LIST, trials=TRIALS, n_probe=N_PROBE,
                    low_band=list(LOW_BAND), high_min=HIGH_MIN,
                    per_scene_low=PER_SCENE_LOW, per_scene_high=PER_SCENE_HIGH,
                    methods=METHODS, base_seed=BASE_SEED),
        scan_summary=scan_summary,
        selected_pairs={F.SCENE_SHORT[s]: [
            {k: v for k, v in r.items() if k != "Tinv"} for r in rows]
            for s, rows in selected.items()},
        tables=tables, advantage=adv, pair_level=pl,
        high_reference=high_reference(),
    )
    with open(OUT / "low_overlap.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # restore the parent's documented overlap_scan.json (it was gitignored and
    # absent from this worktree); same estimator/seed -> same numbers.
    parent_scan = FULL / "results" / "overlap_scan.json"
    shutil.copyfile(OUT / "overlap_scan.json", parent_scan)

    write_markdown(out)
    print(f"[done] wrote low_overlap.{{json,md}}, raw jsonl, overlap_scan.json "
          f"in {time.time()-t0:.1f}s", flush=True)


def write_markdown(o):
    cfg = o["config"]
    L = ["# 3DMatch true low overlap (uncapped 0.20-0.50) robust protocol", ""]
    L.append(f"Overlap = uncapped GT-NN fraction (2000 uniform source points, mapped by "
             f"inv(gt.log), nearest target NN < tau={cfg['tau']} m). "
             f"Band low_20_50 = [{cfg['low_band'][0]:.2f}, {cfg['low_band'][1]:.2f}]; "
             f"control high > {cfg['high_min']:.2f}. N={cfg['n']}, c={cfg['c']} m, "
             f"random outliers at p in {cfg['p']}, {cfg['trials']} trials/cell, "
             f"up to {cfg['per_scene_low']} low + {cfg['per_scene_high']} high pairs per "
             f"scene. Success = rot < 5 deg and |t| < 0.1 m.")
    L.append("")
    L.append("## Overlap scan of all 8 official scene sets")
    L.append("")
    L.append("| scene | fragments | gt pairs | <0.20 | **0.20-0.50** | 0.50-0.80 | >0.80 | p0 / p50 / p100 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sc in F.SCENES:
        s = o["scan_summary"][sc]
        q = s["overlap_quantiles"]
        L.append(f"| {s['short']} | {s['n_fragments']} | {s['n_pairs']} | "
                 f"{s['band_counts']['very_low']} | {s['n_pairs_low_band']} | "
                 f"{s['band_counts']['mid']} | {s['n_pairs_high_band']} | "
                 f"{q['0']:.2f} / {q['50']:.2f} / {q['100']:.2f} |")
    L.append("")
    L.append("## Selected low-overlap pairs: GT-NN base quality")
    L.append("")
    L.append("| scene | pair | overlap | n_kept/2000 | n_valid (of 300) | clean base inlier | padded |")
    L.append("|---|---|---|---|---|---|---|")
    for r in o["pair_level"]:
        if r["band"] != "low_20_50":
            continue
        L.append(f"| {r['scene_short']} | {r['pair']} | {r['overlap']:.3f} | "
                 f"{r['n_kept_2000']} | {r['n_valid']} | {r['base_inlier']:.3f} | "
                 f"{'yes' if r['n_valid'] < cfg['n'] else 'no'} |")
    L.append("")
    L.append("## Robust success: true low overlap vs matched high-overlap control")
    L.append("")
    L.append("| p | method | low success | low rot med | low t med | high success | high rot med | high t med |")
    L.append("|---|---|---|---|---|---|---|---|")
    idx = {(t["band"], t["p"], t["method"]): t for t in o["tables"]}
    for p in cfg["p"]:
        for m in cfg["methods"]:
            lo, hi = idx.get(("low_20_50", p, m)), idx.get(("high", p, m))
            if not lo:
                continue
            hsucc = f"{hi['success']:.2f}" if hi else "-"
            hrot = f"{hi['rot_median']:.2f}" if hi else "-"
            htr = f"{hi['trans_median']:.3f}" if hi else "-"
            L.append(f"| {p:.1f} | {m} | {lo['success']:.2f} | {lo['rot_median']:.2f} | "
                     f"{lo['trans_median']:.3f} | {hsucc} | {hrot} | {htr} |")
    L.append("")
    L.append("## GNC vs RANSAC (true low overlap)")
    L.append("")
    L.append("| p | succ GNC | succ RANSAC-1000 | succ IRLS | Δ(GNC-RANSAC) | RANSAC/GNC time |")
    L.append("|---|---|---|---|---|---|")
    for a in o["advantage"]:
        if a["band"] != "low_20_50":
            continue
        L.append(f"| {a['p']:.1f} | {a['success_gnc']:.2f} | {a['success_ransac']:.2f} | "
                 f"{a['success_irls']:.2f} | {a['delta_gnc_minus_ransac']:+.2f} | "
                 f"{a['speedup_ransac_over_gnc']:.1f}x |")
    L.append("")
    if o["high_reference"]:
        hr = o["high_reference"]
        L.append(f"## Existing high-overlap bin reference ({hr['trials']} trials/cell)")
        L.append("")
        L.append("| p | method | overlap mean | base inlier | success | rot med |")
        L.append("|---|---|---|---|---|---|")
        for key in sorted(hr["tables"], key=lambda k: (float(k.split('|')[0]),
                                                       k.split('|')[1])):
            t = hr["tables"][key]
            L.append(f"| {t['p']:.1f} | {t['method']} | {t['overlap_mean']:.3f} | "
                     f"{t['base_inlier']:.3f} | {t['success']:.2f} | "
                     f"{t['rot_median']:.2f} |")
        L.append("")
    L.append("## Reading of the result")
    L.append("")
    lo0 = idx.get(("low_20_50", 0.0, "GNC-GM"))
    L.append(
        "The clean GT-NN base inlier fraction is measured on **oracle** matches: every "
        "correspondence already passes the tau gate by construction, so geometric "
        "overlap cannot lower it below ~0.9 unless the probe budget is too small. "
        "What low overlap *does* reduce is the number of usable correspondences "
        f"(n_kept/2000 and n_valid); below overlap ~0.25 the 300-point budget is padded "
        "by resampling, which is recorded in every row. Therefore in this protocol "
        "'low overlap' is a **geometric** label (shared surface fraction), not a "
        "**correspondence-quality** label (inlier ratio). A learned front-end with "
        "real descriptor noise is out of scope, and the paper should not present these "
        "cells as evidence that low-overlap registration is easy for a real matcher."
        if lo0 else "No low-band cells were produced.")
    L.append("")
    (OUT / "low_overlap.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
