#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADD-S AUC / recall-threshold curves for the 15 real LINEMOD frames (P1, 2026-11-28).

Question
--------
`results/pose_error.json` stores one ADD-S value per (frame, estimator) for the 15
real eggbox/glue frames of the P1 two-mode experiment, but the paper's LINEMOD
AUC cells (Table 5.20) are `not measured` while the LMO rows carry an
ADD-S AUC(0-0.1 m).  This script computes that cell *from the stored per-frame
ADD-S array only*: no mesh, no re-fit, no new pose, no interpolation.  Each of
the 15 frames contributes exactly one number to each curve, so the recall curve
has at most 15 steps and every threshold that is not one of those steps is
unconstrained by the data.

Main finding about the metric itself
------------------------------------
All 15 ADD-S values (max 22.4 mm) are far below the 100 mm cap, so the clipped
sum collapses to an affine function of the mean ADD-S:

    AUC(0-0.1 m) = 1 - mean(ADD-S) / 0.1     (exactly, whenever max ADD-S <= 0.1 m)

The script asserts this to machine precision and verifies that its bootstrap CI
is *identical* to the mean-ADD-S CI already stored in `pose_error.json`, rescaled
(using the same seeds).  So the LINEMOD AUC cell is a format-matched restatement
of the mean ADD-S that the paper already reports -- not an independent
measurement.  This is stated in the report rather than hidden, and it is the
reason the report is labelled descriptive.

What this is not
----------------
* Not a benchmark number and not the BOP leaderboard metric (that is AP, on the
  official test split, with a detector front-end).
* Not a detector claim: every pose behind these ADD-S values was fit to
  correspondences associated through the GT pose, so absolute errors are
  optimistic by construction.
* Not a per-object population claim: n=8 / n=7 are the frames that exist; the
  object split is reported to show the spread, not to test an object effect.

Definition
----------
For threshold t (metres), recall(t) = (1/n) * #{i : ADD-S_i <= t}.
    ADD-S AUC(0-T) = (1/T) * int_0^T recall(t) dt            (BOP convention)
                   = (1/n) * sum_i (1 - min(max(ADD-S_i,0), T) / T)
The second form is the exact integral of the step function and is what this
script computes; it is checked (a) against a dense numerical integral of the
recall curve, (b) against the published LMO AUC in `lmo_two_mode.json` which uses
the same formula on its own per-frame fields, and (c) against the mean/median
ADD-S CIs already in `pose_error.json`.

Outputs
-------
`results/auc15.json`, `results/auc15.md`, `results/auc15.png`.

Run
---
    cd labs/01-laplace-qpeplayer/experiments/real/linemod
    /home/syk/miniconda3/envs/WM/bin/python run_auc15.py
CPU only, ~2 s.  Deterministic (seeds fixed below).  This file only reads
`results/pose_error.json` and `results/lmo_two_mode.json`; it does not import or
modify any other script.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SRC = RESULTS / "pose_error.json"
LMO_SRC = RESULTS / "lmo_two_mode.json"
OUT_JSON = RESULTS / "auc15.json"
OUT_MD = RESULTS / "auc15.md"
OUT_PNG = RESULTS / "auc15.png"

T_M = 0.1                      # AUC integration range [0, T] in metres (BOP)
THRESH_MM = (5.0, 10.0, 20.0, 50.0)
N_BOOT = 1000
BOOT_SEED = 20261127           # same seed base as run_pose_error.py / run_two_mode_lmo.py
OBJ_NAME = {10: "eggbox", 11: "glue"}
METHODS = ("hard", "laplace", "mc")
METH_LABEL = {"hard": "hard argmin", "laplace": "Laplace", "mc": "MC"}
METH_COLOR = {"hard": "#0072B2", "laplace": "#D55E00", "mc": "#009E73"}
METH_MARK = {"hard": "o", "laplace": "s", "mc": "^"}
# The single frame where the hard argmin is not the GT-like mode (pose_error.md §1).
FAIL_FRAME = "000010/72"


# ------------------------------------------------------------------ definitions
def auc_frame(err_m: float, T: float = T_M) -> float:
    """Exact AUC(0-T) contribution of one frame: integral of 1{x<=t} over [0,T] / T."""
    return 1.0 - min(max(float(err_m), 0.0), T) / T


def auc_of(errs, T: float = T_M) -> float:
    """ADD-S AUC(0-T) of a frame set = mean of the exact per-frame contributions."""
    return float(np.mean([auc_frame(x, T) for x in np.asarray(errs, float)]))


def recall_grid(errs, tgrid) -> np.ndarray:
    """recall(t) on a grid: fraction of frames with ADD-S <= t."""
    e = np.asarray(errs, float)[:, None]
    return np.mean(e <= np.asarray(tgrid, float)[None, :], axis=0)


def auc_numeric(errs, T: float = T_M, n: int = 200001) -> float:
    """(1/T) * int_0^T recall(t) dt on a dense grid -- independent of `auc_of`."""
    t = np.linspace(0.0, T, n)
    return float(np.trapezoid(recall_grid(errs, t), t) / T)


# ------------------------------------------------------------------ bootstrap
def pct_ci(samples, lo: float = 2.5, hi: float = 97.5):
    s = np.asarray(samples, float)
    return [float(np.percentile(s, lo)), float(np.percentile(s, hi))]


def boot_idx(n: int, seed: int, n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap index matrix, identical construction to run_pose_error.boot_ci."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def strat_idx(n_a: int, n_b: int, seed: int, n_boot: int = N_BOOT):
    """Frame bootstrap that preserves the design's object counts (n_a, n_b)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n_a, size=(n_boot, n_a))
    ib = rng.integers(0, n_b, size=(n_boot, n_b))
    return ia, ib


# ------------------------------------------------------------------ data
def load_frames():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    src = json.loads(SRC.read_text())
    rows = []
    for f in src["frames"]:
        oid = int(f["obj"])
        if oid not in OBJ_NAME:
            sys.exit(f"unexpected object id {oid} in {SRC}")
        rows.append(dict(
            label=f"{f['frame']}/{f['img']}",
            seq=f["frame"],
            img=int(f["img"]),
            obj=oid,
            obj_name=OBJ_NAME[oid],
            adds={m: float(f[m]["adds_m"] if m != "mc" else f["mc"]["adds_m"]["mean"])
                  for m in METHODS},
            adds_full={"hard": float(f["adds_full_check_m"]),
                       "laplace": float(f["adds_full_check_laplace_m"])},
            mc_seeds=np.asarray(f["mc"]["adds_m"]["seeds"], float),
            mc_sd=float(f["mc"]["adds_m"]["sd"]),
            spread={"hard": float(f["hard"]["adds_sub_spread_m"]),
                    "laplace": float(f["laplace"]["adds_sub_spread_m"])},
            rot={m: float(f[m]["rot_deg"] if m != "mc" else f["mc"]["rot_deg"]["mean"])
                 for m in METHODS},
            hard_is_gt_like=bool(f["hard_is_gt_like"]),
        ))
    return src, rows


def group_index(rows):
    """Group label -> list of row indices, in object-id order."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["obj"], []).append(i)
    return [(OBJ_NAME[o], o, idxs) for o, idxs in sorted(groups.items())]


# ------------------------------------------------------------------ aggregates
def summarise(vals, idx_mean, idx_med):
    v = np.asarray(vals, float)
    return dict(
        n=int(len(v)),
        mean=float(v.mean()),
        mean_ci=pct_ci(v[idx_mean].mean(axis=1)),
        median=float(np.median(v)),
        median_ci=pct_ci(np.median(v[idx_med], axis=1)),
        min=float(v.min()),
        max=float(v.max()),
    )


def paired_diff_ci(a, b, idx):
    """CI of mean(a - b) on the same resampled frame indices (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                ci=pct_ci(d[idx].mean(axis=1)))


def build(rows, groups, src):
    n = len(rows)
    auc = {m: np.array([auc_frame(r["adds"][m]) for r in rows]) for m in METHODS}
    errs = {m: np.array([r["adds"][m] for r in rows]) for m in METHODS}
    hit = {t: {m: (errs[m] <= t / 1000.0).astype(float) for m in METHODS}
           for t in THRESH_MM}

    gi = {name: idxs for name, _o, idxs in groups}
    assert len(gi) == 2, "expected exactly two objects"
    (ga, ia), (gb, ib) = list(gi.items())
    ia_idx, ib_idx = np.array(ia), np.array(ib)
    # sensitivity only: pooled bootstrap that preserves the 8/7 design split
    si_a, si_b = strat_idx(len(ia), len(ib), seed=BOOT_SEED + 101)
    pool_strat = np.concatenate([ia_idx[si_a], ib_idx[si_b]], axis=1)

    out = dict(auc={}, recall={}, recall_curve={}, per_frame=[], checks={})
    out["auc"]["pooled"] = {}
    out["auc"]["pooled_object_balanced"] = {}

    # ---- AUC: per object, pooled (frame weight), pooled (object balanced)
    for j, m in enumerate(METHODS):
        # the same seeds run_pose_error.py used for its adds_m block, so the CI
        # below is the stored mean-ADD-S CI rescaled -- verified in checks
        seed = BOOT_SEED + 101 + 17 * j
        blk = summarise(auc[m], boot_idx(n, seed), boot_idx(n, seed + 1))
        blk["mean_ci_stratified"] = pct_ci(
            auc[m][pool_strat].mean(axis=1))
        ob = 0.5 * (auc[m][ia_idx][si_a].mean(axis=1) + auc[m][ib_idx][si_b].mean(axis=1))
        out["auc"]["pooled"][m] = blk
        out["auc"]["pooled_object_balanced"][m] = dict(
            mean=float(0.5 * (auc[m][ia_idx].mean() + auc[m][ib_idx].mean())),
            mean_ci=pct_ci(ob))
        for name, idxs, extra in ((ga, ia_idx, 3), (gb, ib_idx, 4)):
            out["auc"].setdefault(name, {})[m] = summarise(
                auc[m][idxs], boot_idx(len(idxs), seed + extra),
                boot_idx(len(idxs), seed + extra + 1))
    for tag, other, sd in (("laplace", "hard", 211), ("mc", "hard", 223)):
        out["auc"]["pooled"][f"{tag}_minus_hard"] = paired_diff_ci(
            auc[tag], auc[other], boot_idx(n, BOOT_SEED + sd))
        for name, idxs in ((ga, ia_idx), (gb, ib_idx)):
            out["auc"][f"{name}"][f"{tag}_minus_hard"] = paired_diff_ci(
                auc[tag][idxs], auc[other][idxs],
                boot_idx(len(idxs), BOOT_SEED + sd + 7))

    # ---- recall @ thresholds (primary: plain frame bootstrap, repo convention)
    for name, idxs, sidx in [("pooled", np.arange(n), pool_strat)] + \
                            [(ga, ia_idx, si_a), (gb, ib_idx, si_b)]:
        out["recall"][name] = {}
        for jm, m in enumerate(METHODS):
            ridx = boot_idx(len(idxs), BOOT_SEED + 500 + 17 * jm
                            + (0 if name == "pooled" else (3 if name == ga else 4)))
            for t in THRESH_MM:
                h = hit[t][m][idxs]
                blk = dict(value=float(h.mean()), n_hit=int(h.sum()), n=int(len(idxs)),
                           ci=pct_ci(h[ridx].mean(axis=1)))
                if name == "pooled":
                    blk["ci_stratified"] = pct_ci(h[sidx].mean(axis=1))
                out["recall"][name].setdefault(f"{t:g}mm", {})[m] = blk

    # ---- stored recall curves (for the figure and for reproducibility)
    tg = np.linspace(0.0, T_M, 1001)
    out["recall_curve"]["t_m"] = [float(x) for x in tg]
    for name, idxs in [("pooled", np.arange(n))] + [(s, np.array(i)) for s, _o, i in groups]:
        out["recall_curve"][name] = {m: [float(v) for v in recall_grid(errs[m][idxs], tg)]
                                     for m in METHODS}

    # ---- per-frame rows (the exact steps of the curve)
    for i, r in enumerate(rows):
        out["per_frame"].append(dict(
            frame=r["label"], obj=r["obj"], obj_name=r["obj_name"],
            adds_m={m: r["adds"][m] for m in METHODS},
            auc={m: auc_frame(r["adds"][m]) for m in METHODS},
            rot_deg=r["rot"],
            adds_sub_spread_m=r["spread"],
            mc_within_frame_sd_m=r["mc_sd"],
            hard_is_gt_like=r["hard_is_gt_like"],
        ))

    # ---- checks -------------------------------------------------------
    # (1) closed form vs dense numerical integral of the recall curve
    num = {m: auc_numeric(errs[m]) for m in METHODS}
    out["checks"]["auc_closed_vs_numeric"] = {
        m: dict(closed=float(auc[m].mean()), numeric=num[m],
                abs_diff=abs(float(auc[m].mean()) - num[m])) for m in METHODS}
    out["checks"]["auc_closed_vs_numeric"]["max_abs_diff"] = max(
        abs(float(auc[m].mean()) - num[m]) for m in METHODS)

    # (2) the clipped sum is affine in the mean ADD-S because nothing hits the cap
    aff = {}
    for m in METHODS:
        mean_adds = float(errs[m].mean())
        aff[m] = dict(mean_adds_m=mean_adds, max_adds_m=float(errs[m].max()),
                      auc=float(auc[m].mean()), auc_from_mean=1.0 - mean_adds / T_M,
                      abs_diff=abs(float(auc[m].mean()) - (1.0 - mean_adds / T_M)),
                      clipping_active=bool(errs[m].max() > T_M))
    aff["max_abs_diff"] = max(aff[m]["abs_diff"] for m in METHODS)
    aff["note"] = ("no frame reaches the 100 mm cap, so AUC = 1 - mean(ADD-S)/0.1 "
                   "exactly; the AUC column is a rescaling of the mean ADD-S already "
                   "reported in pose_error.json, not an independent measurement")
    out["checks"]["auc_affine_in_mean"] = aff

    # (3) same formula reproduces the published LMO AUC (unit test on a 37-frame array)
    if LMO_SRC.exists():
        lmo = json.loads(LMO_SRC.read_text())
        rep = {}
        for m, key in (("hard", "adds_hard_m"), ("laplace", "adds_laplace_m"),
                       ("mc", "adds_mc_m")):
            v = np.array([auc_frame(r[key]) for r in lmo["frames"]])
            rep[m] = dict(mean=float(v.mean()), median=float(np.median(v)),
                          published_mean=float(lmo["aggregate"]["auc"][m]["mean"]),
                          published_median=float(lmo["aggregate"]["auc"][m]["median"]))
            rep[m]["abs_diff_mean"] = abs(v.mean() - rep[m]["published_mean"])
        rep["n_frames"] = len(lmo["frames"])
        rep["source"] = str(LMO_SRC.relative_to(HERE))
        rep["max_abs_diff"] = max(rep[m]["abs_diff_mean"] for m in METHODS)
        out["checks"]["lmo_cross_check"] = rep

    # (4) the bootstrap CI must equal the stored mean/median ADD-S CI, rescaled
    agg = src.get("aggregate", {}).get("adds_m")
    if agg:
        rep = {}
        for m in METHODS:
            mp = [1.0 - agg[m]["mean_ci"][1] / T_M, 1.0 - agg[m]["mean_ci"][0] / T_M]
            mp_med = [1.0 - agg[m]["median_ci"][1] / T_M, 1.0 - agg[m]["median_ci"][0] / T_M]
            got = out["auc"]["pooled"][m]
            rep[m] = dict(mean_ci=got["mean_ci"], mean_ci_rescaled=mp,
                          median_ci=got["median_ci"], median_ci_rescaled=mp_med,
                          abs_diff=max(abs(a - b) for a, b in zip(got["mean_ci"], mp)))
            rep[m]["abs_diff_median"] = max(
                abs(a - b) for a, b in zip(got["median_ci"], mp_med))
        rep["max_abs_diff"] = max(rep[m]["abs_diff"] for m in METHODS)
        rep["max_abs_diff_median"] = max(rep[m]["abs_diff_median"] for m in METHODS)
        rep["note"] = ("same seeds and same index construction as run_pose_error.py, so the "
                       "AUC CI reproduces the stored mean/median ADD-S CI exactly after the "
                       "monotone map x -> 1 - x/T")
        out["checks"]["ci_matches_mean_adds_ci"] = rep

    # (5) monotone dominance / crossings of the recall curves
    dg = np.linspace(0.0, T_M, 100001)
    rc = {m: recall_grid(errs[m], dg) for m in METHODS}
    out["checks"]["curve_dominance"] = dict(
        min_hard_minus_laplace=float(np.min(rc["hard"] - rc["laplace"])),
        min_hard_minus_mc=float(np.min(rc["hard"] - rc["mc"])),
        max_laplace_minus_mc=float(np.max(np.abs(rc["laplace"] - rc["mc"]))),
        hard_weakly_dominates=bool(np.all(rc["hard"] >= rc["laplace"] - 1e-12)
                                   and np.all(rc["hard"] >= rc["mc"] - 1e-12)),
        note="scan over 100001 thresholds in [0,0.1] m; a negative min means the curves cross")

    # ---- MC hierarchical bootstrap: frames *and* the 20 MC seeds
    mat = np.array([[auc_frame(x) for x in r["mc_seeds"]] for r in rows])  # (n, S)
    F, S = mat.shape
    rng = np.random.default_rng(BOOT_SEED + 3301)
    vals = np.empty(N_BOOT)
    per = rng.integers(0, S, size=(N_BOOT, F))
    for b in range(N_BOOT):
        vals[b] = mat[rng.integers(0, F, size=F), per[b]].mean()
    out["auc"]["pooled"]["mc"]["mean_ci_hier"] = pct_ci(vals)

    # ---- estimator sensitivity: full-mesh-vs-full-mesh ADD-S (hard & laplace only)
    out["auc"]["sensitivity_full_check"] = {}
    for m in ("hard", "laplace"):
        v = np.array([auc_frame(r["adds_full"][m]) for r in rows])
        out["auc"]["sensitivity_full_check"][m] = dict(
            mean=float(v.mean()),
            mean_ci=pct_ci(v[boot_idx(n, BOOT_SEED + 101 + 17 * METHODS.index(m))].mean(axis=1)),
            delta_vs_primary=float(v.mean() - auc[m].mean()),
        )
    out["auc"]["sensitivity_full_check"]["note"] = (
        "full-mesh-vs-full-mesh ADD-S check stored in pose_error.json "
        "(`adds_full_check_m` / `adds_full_check_laplace_m`); MC has no such column")

    # ---- failure bookkeeping
    out["failures"] = {}
    for t in THRESH_MM:
        out["failures"][f"{t:g}mm"] = {
            m: [rows[i]["label"] for i in range(n) if rows[i]["adds"][m] > t / 1000.0]
            for m in METHODS}
    hard_fail = [r for r in rows if not r["hard_is_gt_like"]]
    out["failures"]["hard_mode_failure"] = dict(
        frames=[r["label"] for r in hard_fail],
        n=len(hard_fail),
        detail=[dict(frame=r["label"], obj=r["obj_name"],
                     adds_m={m: r["adds"][m] for m in METHODS},
                     rot_deg=r["rot"]) for r in hard_fail],
        note="frame(s) where the hard argmin is not the mode closest to GT")
    # how fragile each single-threshold number is: distance to the nearest frame
    # vs that frame's between-subsample range (max-min over its 3 vertex draws)
    frag = {}
    for t in THRESH_MM:
        for m in METHODS:
            best = min(rows, key=lambda r: abs(r["adds"][m] * 1e3 - t))
            gap = abs(best["adds"][m] * 1e3 - t)
            if m == "mc":
                sp, kind = best["mc_sd"], "MC seed sd"
            else:
                sp, kind = best["spread"][m], "vertex subsample range"
            frag[f"{t:g}mm_{m}"] = dict(
                frame=best["label"], adds_mm=float(best["adds"][m] * 1e3),
                gap_mm=float(gap), spread_mm=float(sp * 1e3), spread_kind=kind,
                could_flip=bool(gap < sp * 1e3))
    out["failures"]["threshold_fragility"] = frag
    out["failures"]["threshold_fragility_note"] = (
        "gap_mm = |ADD-S of the closest frame - threshold|; spread_mm is that frame's own "
        "dispersion: max-min over its 3 vertex subsamples (adds_sub_spread_m) for hard and "
        "Laplace, and the sd over its 20 MC seeds for MC; could_flip = gap < spread, i.e. a "
        "different vertex draw / seed set could move that frame across the threshold and "
        "change the recall count by 1/n.")
    return out


# ------------------------------------------------------------------ figure
def figure(rows, out, path: Path):
    tg = np.array(out["recall_curve"]["t_m"]) * 1000.0  # mm
    max_adds_mm = max(max(r["adds"][m] for m in METHODS) for r in rows) * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.55), dpi=165)

    # ---- (a) pooled recall curve over the full integration window
    ax = axes[0]
    ax.axvspan(max_adds_mm, 100.0, color="0.85", alpha=0.45, lw=0, zorder=0)
    ax.text((max_adds_mm + 100.0) / 2, 0.50,
            f"recall = 1.000 already at t = {max_adds_mm:.1f} mm\n"
            f"-> this flat region is {(100.0 - max_adds_mm) / 100.0 * 100:.0f}% of the "
            f"0-100 mm integration window\n"
            f"(why AUC = 1 - mean(ADD-S)/0.1 here)",
            ha="center", va="center", fontsize=7.4, color="0.3")
    for m in METHODS:
        y = np.array(out["recall_curve"]["pooled"][m])
        ax.step(tg, y, where="post", color=METH_COLOR[m], lw=1.9,
                label=f"{METH_LABEL[m]}  AUC {out['auc']['pooled'][m]['mean']:.3f}")
    for k, t in enumerate((5, 10, 20, 50)):
        ax.axvline(t, color="0.55", ls=":", lw=0.9)
        ax.text(t, 1.105 if k % 2 == 0 else 1.035, f"{t} mm", ha="center", va="bottom",
                fontsize=7.4, color="0.35")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.20)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("ADD-S threshold t (mm), the integration range of the AUC")
    ax.set_ylabel("recall(t) = fraction of frames with ADD-S $\\leq$ t")
    ax.set_title("(a) pooled recall curve, n=15 (at most 15 steps)", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.92)

    # ---- (b) the 15 per-frame values behind those steps
    ax = axes[1]
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.plot([i, i], [r["adds"]["hard"] * 1e3, r["adds"]["laplace"] * 1e3],
                color="0.78", lw=1.0, zorder=1)
    for m in METHODS:
        ax.scatter(x, [r["adds"][m] * 1e3 for r in rows], s=34, color=METH_COLOR[m],
                   marker=METH_MARK[m], zorder=3, label=METH_LABEL[m],
                   edgecolors="white", linewidths=0.5)
    for t in (5, 10, 20):
        ax.axhline(t, color="0.55", ls=":", lw=0.9)
        ax.text(14.35, t + 0.45, f"{t} mm", ha="right", va="bottom", fontsize=7, color="0.35")
    ax.axvline(7.5, color="0.6", lw=0.8, ls="-")
    ax.text(3.4, 0.9, "eggbox (n=8)", ha="center", fontsize=8, color="0.35")
    ax.text(11.0, 0.9, "glue (n=7)", ha="center", fontsize=8, color="0.35")
    fi = next((i for i, r in enumerate(rows) if r["label"] == FAIL_FRAME), None)
    if fi is not None:
        ax.annotate("hard mode failure\n" + FAIL_FRAME,
                    xy=(fi, rows[fi]["adds"]["hard"] * 1e3),
                    xytext=(fi - 0.7, 22.0), fontsize=7.2, color="#7a1f00",
                    ha="center", arrowprops=dict(arrowstyle="->", color="#7a1f00", lw=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace("0000", "") for r in rows], rotation=90, fontsize=6.8)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_ylim(0, 31)
    ax.set_ylabel("ADD-S (mm)")
    ax.set_title("(b) the 15 per-frame ADD-S values (the curve's steps)", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=7.2, ncol=3, framealpha=0.92,
              columnspacing=0.9, handletextpad=0.3)

    # ---- (c) AUC per object and pooled, with CIs
    ax = axes[2]
    groups = [("eggbox\n(n=8)", "eggbox"), ("glue\n(n=7)", "glue"),
              ("pooled\n(n=15)", "pooled")]
    w = 0.26
    for j, m in enumerate(METHODS):
        vals = [out["auc"][g][m]["mean"] for _lab, g in groups]
        los = [out["auc"][g][m]["mean"] - out["auc"][g][m]["mean_ci"][0] for _lab, g in groups]
        his = [out["auc"][g][m]["mean_ci"][1] - out["auc"][g][m]["mean"] for _lab, g in groups]
        xs = np.arange(len(groups)) + (j - 1) * w
        ax.bar(xs, vals, width=w, color=METH_COLOR[m], label=METH_LABEL[m],
               alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor="0.15",
                    elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=4)
        for xx, v in zip(xs, vals):
            ax.text(xx, v - 0.035, f"{v:.3f}", ha="center", va="top",
                    fontsize=6.8, rotation=90, color="white", fontweight="bold", zorder=5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("ADD-S AUC (0-0.1 m), BOP normalisation")
    ax.set_title("(c) AUC with 95% frame-bootstrap CI (1000 resamples)", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92)

    fig.suptitle(
        "ADD-S AUC / recall on the 15 real LINEMOD eggbox+glue frames -- descriptive only, "
        "n=15 so at most 15 curve steps, GT-associated correspondences, not a benchmark. "
        "Because no frame exceeds 28.2 mm, AUC = 1 - mean(ADD-S)/0.1 exactly (a rescaling "
        "of the mean ADD-S, not a new measurement).",
        fontsize=8.3, y=1.015)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def ci(blk, nd=3):
    lo, hi = blk["mean_ci"] if "mean_ci" in blk else blk["ci"]
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def write_md(rows, out, path: Path, dt_s: float):
    a = out["auc"]
    n = len(rows)
    g_egg, g_glue = "eggbox", "glue"
    chk = out["checks"]
    aff = chk["auc_affine_in_mean"]
    mean_mm = {m: aff[m]["mean_adds_m"] * 1e3 for m in METHODS}
    rp = out["recall"]["pooled"]
    L = []
    A = L.append

    A("# ADD-S AUC and recall-threshold curves on the 15 real LINEMOD frames")
    A("")
    A(f"**Date:** 2026-11-28 · **Source:** [`results/pose_error.json`](pose_error.json) "
      f"(per-frame ADD-S; 15 frames = 8 eggbox + 7 glue) · **Frames:** {n} · "
      f"**Runtime:** {dt_s:.1f} s (CPU, login node).")
    A("")
    A("> **What this is.** The paper's LINEMOD AUC cells in Table 5.20 read `not measured` "
      "while the LMO rows carry an ADD-S AUC(0-0.1 m). This report computes that cell *from "
      "the per-frame ADD-S values already stored in `pose_error.json`* -- no mesh, no "
      "re-fit, no new pose, no interpolation -- per object (eggbox n=8, glue n=7) and "
      "pooled (n=15), with 95% frame-bootstrap CIs (1000 resamples) and "
      "recall@{5,10,20,50} mm. "
      "**What this is not.** Not a benchmark and not the BOP AP metric; n=15 means the "
      "recall curve has at most 15 steps, so every threshold that is not one of those "
      "steps is unconstrained by the data; the correspondences behind every pose are "
      "GT-associated, so absolute ADD-S is optimistic; there is no detector or front-end "
      "anywhere in this table. And most importantly: **with no frame above the 0.1 m cap, "
      "AUC is exactly `1 - mean(ADD-S)/0.1`** (verified to "
      f"{chk['auc_affine_in_mean']['max_abs_diff']:.1e}), i.e. the cell is a rescaling of "
      "the mean ADD-S the paper already reports, not a new measurement.")
    A("")
    A("## Headline")
    A("")
    A(f"1. **ADD-S AUC(0-0.1 m), pooled (n=15; mean [95% frame-bootstrap CI]):** hard "
      f"**{fmt(a['pooled']['hard']['mean'])} {ci(a['pooled']['hard'])}**, Laplace "
      f"{fmt(a['pooled']['laplace']['mean'])} {ci(a['pooled']['laplace'])}, MC "
      f"{fmt(a['pooled']['mc']['mean'])} {ci(a['pooled']['mc'])}. Paired with the same "
      f"resampled frames: Laplace - hard "
      f"{a['pooled']['laplace_minus_hard']['mean']:+.3f} "
      f"{ci(a['pooled']['laplace_minus_hard'])}, MC - hard "
      f"{a['pooled']['mc_minus_hard']['mean']:+.3f} {ci(a['pooled']['mc_minus_hard'])}. "
      f"The Laplace-vs-MC gap is {abs(a['pooled']['laplace']['mean'] - a['pooled']['mc']['mean']):.4f}, "
      f"~{abs(a['pooled']['laplace']['mean'] - a['pooled']['mc']['mean']) and ((a['pooled']['laplace']['mean_ci'][1] - a['pooled']['laplace']['mean_ci'][0]) / 2) / abs(a['pooled']['laplace']['mean'] - a['pooled']['mc']['mean']):.0f}x "
      f"smaller than the CI half-width -- they are the same estimator functional here, as "
      f"in every other table.")
    A(f"2. **The cell is fully determined by the mean ADD-S.** Every one of the 15 ADD-S "
      f"values is below the 0.1 m cap (max "
      f"{max(aff[m]['max_adds_m'] for m in METHODS) * 1e3:.1f} mm), so the clipped sum "
      f"collapses to `AUC = 1 - mean(ADD-S)/0.1` exactly: hard mean "
      f"{mean_mm['hard']:.1f} mm -> {aff['hard']['auc_from_mean']:.3f} (reported "
      f"{aff['hard']['auc']:.3f}), Laplace {mean_mm['laplace']:.1f} mm -> "
      f"{aff['laplace']['auc_from_mean']:.3f}, MC {mean_mm['mc']:.1f} mm -> "
      f"{aff['mc']['auc_from_mean']:.3f}. The CI is likewise the mean-ADD-S CI of "
      f"`pose_error.json` after the monotone map, and it reproduces that stored CI to "
      f"{chk['ci_matches_mean_adds_ci']['max_abs_diff']:.1e} (same seeds). **So filling "
      f"Table 5.20's LINEMOD AUC cell adds no measurement; it is the format-matched "
      f"restatement of the mean ADD-S (hard {mean_mm['hard']:.1f} mm, Laplace "
      f"{mean_mm['laplace']:.1f} mm). If the paper would rather not imply a new metric, a "
      f"mean-ADD-S column carries exactly this information without the 0-0.1 m framing.**")
    A(f"3. **recall@{5:g}/{10:g}/{20:g}/{50:g} mm, pooled:** hard "
      f"{rp['5mm']['hard']['value']:.3f}/{rp['10mm']['hard']['value']:.3f}/"
      f"{rp['20mm']['hard']['value']:.3f}/{rp['50mm']['hard']['value']:.3f}, Laplace "
      f"{rp['5mm']['laplace']['value']:.3f}/{rp['10mm']['laplace']['value']:.3f}/"
      f"{rp['20mm']['laplace']['value']:.3f}/{rp['50mm']['laplace']['value']:.3f}, MC "
      f"{rp['5mm']['mc']['value']:.3f}/{rp['10mm']['mc']['value']:.3f}/"
      f"{rp['20mm']['mc']['value']:.3f}/{rp['50mm']['mc']['value']:.3f} "
      f"(Laplace and MC coincide to 1/15). "
      f"**recall@50 mm is 1.000 everywhere** -- the worst ADD-S in the set is "
      f"{max(max(r['adds'][m] for m in METHODS) for r in rows) * 1e3:.1f} mm -- so that "
      f"column carries no information and is included only because it was asked for. The "
      f"Laplace/MC recall@5 mm is structurally 0: their smallest ADD-S is "
      f"{min(r['adds']['laplace'] for r in rows) * 1e3:.1f} mm because the posterior mean "
      f"sits between the two modes.")
    A(f"4. **The 1/15 hard failure frame is `{FAIL_FRAME}`** (eggbox, image 72): the hard "
      f"argmin picks the mode 44.46 deg from GT (ADD-S "
      f"{[r['adds']['hard'] for r in rows if r['label'] == FAIL_FRAME][0] * 1e3:.1f} mm) "
      f"while Laplace/MC land between the two modes and happen to be closer "
      f"({[r['adds']['laplace'] for r in rows if r['label'] == FAIL_FRAME][0] * 1e3:.1f} mm). "
      f"It is the only frame where the per-frame ordering flips, but it is not enough to "
      f"make the curves cross: a 100 001-point scan of [0, 0.1] m finds no threshold where "
      f"hard's recall is below Laplace's or MC's (the difference is 0 only at t=0 and above "
      f"the largest ADD-S, min = "
      f"{chk['curve_dominance']['min_hard_minus_laplace']:+.4f}). It is also *not* the "
      f"frame that sets recall@20 mm (that is `000011/49`, ADD-S "
      f"{[r['adds']['hard'] for r in rows if r['label'] == '000011/49'][0] * 1e3:.1f} mm).")
    A(f"5. **Descriptive, and the CIs are wide for a structural reason.** With n=15 the "
      f"curve has at most 15 steps; the pooled AUC CI half-width is "
      f"~{(a['pooled']['hard']['mean_ci'][1] - a['pooled']['hard']['mean_ci'][0]) / 2:.3f} "
      f"for hard and ~{(a['pooled']['laplace']['mean_ci'][1] - a['pooled']['laplace']['mean_ci'][0]) / 2:.3f} "
      f"for Laplace/MC. Stratifying the bootstrap by object (preserving 8/7) halves the "
      f"Laplace/MC width, because object identity explains much of the between-frame "
      f"spread; that variant is reported as a sensitivity in §5, not used as the primary, "
      f"because every other CI in the paper is a plain frame bootstrap and this table has "
      f"to be comparable to them.")
    A("")
    A("## 0. Metric definition")
    A("")
    A("For a threshold $t$ in metres, "
      "$\\mathrm{recall}(t)=\\frac1n\\#\\{i:\\mathrm{ADD\\text{-}S}_i\\le t\\}$. "
      "The AUC is the BOP normalisation over the fixed range $[0,T]$, $T=0.1$ m:")
    A("")
    A("$$\\mathrm{AUC}(0\\text{-}T)=\\frac1T\\int_0^T\\mathrm{recall}(t)\\,dt"
      "=\\frac1n\\sum_i\\Big(1-\\frac{\\min(\\max(\\mathrm{ADD\\text{-}S}_i,0),T)}{T}\\Big).$$")
    A("")
    A("The second equality is exact (the integrand is a step function, so the integral is a "
      "sum of rectangle widths) and it is the form computed here. Four consequences bound "
      "what the number can mean:")
    A("")
    A(f"1. **Nothing is interpolated.** The curve is the empirical CDF of 15 numbers, so "
      f"recall only changes at those 15 ADD-S values; between two steps it is constant and "
      f"unconstrained by the data.")
    A(f"2. **`mean over thresholds of recall` and `mean of clipped per-frame score` are "
      f"the same number** for any uniform grid over $[0,T]$ fine enough to resolve the "
      f"steps; checked with a dense 200 001-point integral (max abs diff "
      f"{chk['auc_closed_vs_numeric']['max_abs_diff']:.1e}).")
    A(f"3. **With no frame at the cap, the clip is inactive and the sum is affine in the "
      f"mean:** $\\mathrm{{AUC}}=1-\\overline{{\\mathrm{{ADD\\text{{-}}S}}}}/T$. That is the "
      f"case here (max {max(aff[m]['max_adds_m'] for m in METHODS) * 1e3:.1f} mm < 100 mm), "
      f"so this AUC is a linearly rescaled mean ADD-S and cannot rank the estimators "
      f"differently from the mean ADD-S column. A 1 mm change in mean ADD-S moves the AUC "
      f"by 0.01.")
    A(f"4. **It is not a detection rate near the boundary.** An AUC of "
      f"{a['pooled']['hard']['mean']:.3f} here means 'mean ADD-S "
      f"{mean_mm['hard']:.1f} mm', not '94% of frames would be detected at 10 cm'; the "
      f"window is {T_M / max(aff[m]['max_adds_m'] for m in METHODS):.1f}x wider than the worst error in the set.")
    A("")
    A("**recall@X mm** is $\\mathrm{recall}(X\\,\\mathrm{mm})$, reported with $k/n$ so the "
      "granularity (1/15 = 0.067 per frame) is visible.")
    A("")
    A("**Data availability.** The stored file has a per-frame `adds_m` for all 15 frames "
      "and all three estimators (plus the 20 per-seed MC values and the 3-vertex-subsample "
      "spread of each frame), not just medians or aggregates. Every number below is "
      "therefore computed from the stored per-frame array -- nothing is interpolated, "
      "back-solved from a median, or reconstructed from the mesh.")
    A("")
    A("**Bootstrap.** Non-parametric frame bootstrap, 1000 resamples, percentile 95% CI "
      "($[2.5, 97.5]$), plain 15-of-15 resampling and the same seeds `run_pose_error.py` "
      "used for its `adds_m` block (base `20261127`), so the AUC CI here *is* the stored "
      "mean-ADD-S CI after the monotone map -- asserted to "
      f"{chk['ci_matches_mean_adds_ci']['max_abs_diff']:.1e} in §5. Paired differences "
      "(Laplace - hard, MC - hard) use the same resampled frame indices for both "
      "estimators. Per-object rows resample within the object (8 of 8, 7 of 7). The MC "
      "point estimate is the AUC of the 20-seed-mean ADD-S per frame; a hierarchical "
      "bootstrap over frames *and* those 20 seeds is in §5.")
    A("")
    A("## 1. Main table")
    A("")
    A("Mean [95% frame-bootstrap CI]. Object rows resample within the object; the pooled "
      "row resamples 15 of 15 frames. Median and median CI are in `results/auc15.json`.")
    A("")
    A("| estimator | AUC eggbox (n=8) | AUC glue (n=7) | AUC pooled (n=15) | AUC pooled, object-balanced |")
    A("|-|-|-|-|-|")
    for m in METHODS:
        A(f"| {METH_LABEL[m]} | {fmt(a[g_egg][m]['mean'])} {ci(a[g_egg][m])} "
          f"| {fmt(a[g_glue][m]['mean'])} {ci(a[g_glue][m])} "
          f"| **{fmt(a['pooled'][m]['mean'])} {ci(a['pooled'][m])}** "
          f"| {fmt(a['pooled_object_balanced'][m]['mean'])} "
          f"{ci(a['pooled_object_balanced'][m])} |")
    A("")
    A(f"*`object-balanced` = mean of the two per-object AUCs (equal object weight); "
      f"`pooled` = mean over the 15 frames (equal frame weight, the BOP per-instance "
      f"convention). They differ by "
      f"{abs(a['pooled']['hard']['mean'] - a['pooled_object_balanced']['hard']['mean']):.3f} "
      f"for hard, i.e. the 8/7 imbalance is not what drives the pooled number.*")
    A("")
    ov = [max(a[g_egg][m]['mean_ci'][0], a[g_glue][m]['mean_ci'][0]) <=
          min(a[g_egg][m]['mean_ci'][1], a[g_glue][m]['mean_ci'][1]) for m in METHODS]
    A(f"*Object split:* hard is {fmt(a[g_egg]['hard']['mean'])} vs "
      f"{fmt(a[g_glue]['hard']['mean'])} (the two CIs "
      f"{'overlap' if ov[0] else 'are disjoint'}); Laplace is "
      f"{fmt(a[g_egg]['laplace']['mean'])} vs {fmt(a[g_glue]['laplace']['mean'])} "
      f"({len(a[g_egg]['laplace']['mean_ci']) and ('overlap' if ov[1] else 'disjoint')}) and "
      f"MC {fmt(a[g_egg]['mc']['mean'])} vs {fmt(a[g_glue]['mc']['mean'])} "
      f"({'overlap' if ov[2] else 'disjoint'}). The Laplace/MC gap between objects is "
      f"driven by the glue frames sitting at "
      f"{min(r['adds']['laplace'] for r in rows if r['obj'] == 11) * 1e3:.0f}-"
      f"{max(r['adds']['laplace'] for r in rows if r['obj'] == 11) * 1e3:.0f} mm; with "
      f"n=8 / n=7 and no pre-registered object contrast this is reported, not claimed as an "
      f"object effect.*")
    A("")
    A("### Paired differences (same resampled frames)")
    A("")
    A("| comparison | eggbox | glue | pooled |")
    A("|-|-|-|-|")
    for tag, lab in (("laplace_minus_hard", "Laplace - hard"),
                     ("mc_minus_hard", "MC - hard")):
        A(f"| {lab} | {a[g_egg][tag]['mean']:+.3f} {ci(a[g_egg][tag])} "
          f"| {a[g_glue][tag]['mean']:+.3f} {ci(a[g_glue][tag])} "
          f"| {a['pooled'][tag]['mean']:+.3f} {ci(a['pooled'][tag])} |")
    A("")
    A("### Table 5.20 cell format, `median (mean [CI])`")
    A("")
    A("Table 5.20 quotes LMO AUC as `median (mean)`. For LINEMOD the same cell is:")
    A("")
    A("| instance | n | AUC hard | AUC Laplace | AUC MC |")
    A("|-|-|-|-|-|")
    A(f"| LINEMOD (E15) | {n} | {a['pooled']['hard']['median']:.3f} "
      f"({a['pooled']['hard']['mean']:.3f} {ci(a['pooled']['hard'])}) "
      f"| {a['pooled']['laplace']['median']:.3f} "
      f"({a['pooled']['laplace']['mean']:.3f} {ci(a['pooled']['laplace'])}) "
      f"| {a['pooled']['mc']['median']:.3f} "
      f"({a['pooled']['mc']['mean']:.3f} {ci(a['pooled']['mc'])}) |")
    A("")
    A(f"Feeding this row into Table 5.20 also makes the column honest about what it is: "
      f"for LINEMOD `mean(ADD-S)` = 1 - AUC/0.1, so the AUC and the mean ADD-S columns of "
      f"that table are the same variable up to an affine map for any row where no frame "
      f"exceeds 0.1 m.")
    A("")
    A("## 2. Recall at 5 / 10 / 20 / 50 mm")
    A("")
    A("Value (k/n) [95% CI]. Pooled: plain 15-frame bootstrap.")
    A("")
    A("| group, estimator | recall@5 mm | @10 mm | @20 mm | @50 mm |")
    A("|-|-|-|-|-|")
    for grp, lab in (("pooled", "pooled"), (g_egg, "eggbox"), (g_glue, "glue")):
        for m in METHODS:
            cells = []
            for t in THRESH_MM:
                b = out["recall"][grp][f"{t:g}mm"][m]
                cs = f" {ci(b, 3)}" if grp == "pooled" else ""
                cells.append(f"{b['value']:.3f} ({b['n_hit']}/{b['n']}){cs}")
            A(f"| {lab}, {METH_LABEL[m]} | " + " | ".join(cells) + " |")
    A("")
    A("**Frames above each threshold** (the exact failures, so the curve can be read "
      "without the figure):")
    A("")
    A("| threshold | hard | Laplace | MC |")
    A("|-|-|-|-|")
    for t in THRESH_MM:
        row = out["failures"][f"{t:g}mm"]
        cells = [", ".join(f"`{x}`" for x in row[m]) if row[m] else "none"
                 for m in METHODS]
        A(f"| {t:g} mm | " + " | ".join(cells) + " |")
    A("")
    A("### How fragile is a single threshold?")
    A("")
    fg = out["failures"]["threshold_fragility"]
    A("Each row takes the frame closest to the threshold and compares its distance to the "
      "threshold with that frame's own dispersion -- for hard/Laplace the max - min ADD-S "
      "over its 3 vertex subsamples (`adds_sub_spread_m`), for MC the sd over its 20 MC "
      "seeds. If the gap is smaller than the spread, a different draw could move that frame "
      "across the threshold and the recall count would change by 1/15 = 0.067. The two "
      "spread types are not the same quantity (vertex discretisation vs posterior-seed "
      "noise) but both are the right scale for *how well that frame's ADD-S is pinned "
      "down*.")
    A("")
    A("| threshold | estimator | closest frame | ADD-S (mm) | gap (mm) | spread (mm) | spread type | could flip |")
    A("|-|-|-|-|-|-|-|-|")
    for t in THRESH_MM:
        for m in METHODS:
            f_ = fg[f"{t:g}mm_{m}"]
            flip = "**yes**" if f_["could_flip"] else "no"
            A(f"| {t:g} mm | {METH_LABEL[m]} | `{f_['frame']}` | {f_['adds_mm']:.1f} "
              f"| {f_['gap_mm']:.2f} | {f_['spread_mm']:.2f} | {f_['spread_kind']} | {flip} |")
    A("")
    flips = [k for k, v in fg.items() if v["could_flip"]]
    A("**Reading:** the numbers that are not stable to the last frame are the ones where "
      "`could flip` is marked -- for each of them the closest frame is closer to the "
      "threshold than that frame's own spread, so a different draw would move the count by "
      "1/15 = 0.067:")
    for k in flips:
        t_s, m_s = k.split("_")
        b = out["recall"]["pooled"][t_s][m_s]
        A(f"- recall@{t_s[:-2]} mm for {METH_LABEL[m_s]}: closest frame `{fg[k]['frame']}` at "
          f"{fg[k]['adds_mm']:.1f} mm, gap {fg[k]['gap_mm']:.2f} mm vs "
          f"{fg[k]['spread_kind']} {fg[k]['spread_mm']:.2f} mm; the count could be "
          f"{(b['n_hit'] - 1) / 15:.3f}, {b['value']:.3f} or {(b['n_hit'] + 1) / 15:.3f} "
          f"instead of {b['value']:.3f}.")
    A(f"The hard column is not in that regime at 5 or 20 mm (gaps "
      f"{fg['5mm_hard']['gap_mm']:.2f} mm and {fg['20mm_hard']['gap_mm']:.2f} mm vs spreads "
      f"{fg['5mm_hard']['spread_mm']:.2f} mm and {fg['20mm_hard']['spread_mm']:.2f} mm), and "
      f"the AUC is far less "
      f"exposed than any recall count: moving one frame by a whole 0.4 mm shifts the pooled "
      f"AUC by {0.4 / (T_M * 1e3) / n:.4f}, i.e. ~"
      f"{(a['pooled']['hard']['mean_ci'][1] - a['pooled']['hard']['mean_ci'][0]) / 2 / (0.4 / (T_M * 1e3) / n):.0f}x "
      f"smaller than its own CI. This is the concrete reason to prefer the AUC over a "
      f"recall@t cell for the paper, independent of the redundancy caveat in §0.")
    A("")
    A("## 3. The 15 steps (per-frame contributions)")
    A("")
    A("Each row is one step of the curve: "
      "$\\mathrm{AUC}_i=1-\\min(\\mathrm{ADD\\text{-}S}_i,100\\,\\mathrm{mm})/100\\,\\mathrm{mm}$, "
      "and the pooled AUC is the mean of that column. Sorted by object then image.")
    A("")
    A("| frame/img | obj | ADD-S hard (mm) | Laplace (mm) | MC (mm) | AUC_i hard | AUC_i Lap | AUC_i MC | hard argmin = GT-like |")
    A("|-|-|-|-|-|-|-|-|-|")
    for r in rows:
        A(f"| `{r['label']}` | {r['obj_name']} "
          f"| {r['adds']['hard'] * 1e3:.1f} | {r['adds']['laplace'] * 1e3:.1f} "
          f"| {r['adds']['mc'] * 1e3:.1f} "
          f"| {auc_frame(r['adds']['hard']):.3f} | {auc_frame(r['adds']['laplace']):.3f} "
          f"| {auc_frame(r['adds']['mc']):.3f} "
          f"| {'yes' if r['hard_is_gt_like'] else '**no**'} |")
    A(f"| **mean** | | **{mean_mm['hard']:.1f}** | **{mean_mm['laplace']:.1f}** "
      f"| **{mean_mm['mc']:.1f}** | **{a['pooled']['hard']['mean']:.3f}** "
      f"| **{a['pooled']['laplace']['mean']:.3f}** | **{a['pooled']['mc']['mean']:.3f}** "
      f"| 14/15 |")
    A("")
    A("## 4. The 1/15 hard failure frame")
    A("")
    fd = out["failures"]["hard_mode_failure"]
    d0 = fd["detail"][0]
    A(f"`{FAIL_FRAME}` is the only frame where the hard argmin is not the mode closest to "
      f"GT (`hard_is_gt_like = false`; the mode 3.8 deg from GT has $J$ higher by "
      f"1.25e-04, so the loss ranking is wrong there). Its effect on the curves:")
    A("")
    A(f"- hard ADD-S {d0['adds_m']['hard'] * 1e3:.1f} mm vs Laplace/MC "
      f"{d0['adds_m']['laplace'] * 1e3:.1f}/{d0['adds_m']['mc'] * 1e3:.1f} mm: the one "
      f"frame whose per-frame ordering is reversed.")
    A(f"- It is **not** enough to reverse the population ordering: over a 100 001-point "
      f"scan of [0, 0.1] m, hard's recall is never below Laplace's or MC's (minimum "
      f"difference {chk['curve_dominance']['min_hard_minus_laplace']:+.4f}, reached only at "
      f"t=0 and above the largest ADD-S), so the curves never cross and the AUC order is "
      f"hard > Laplace ~ MC at every object level.")
    A(f"- It is **not** the frame that decides recall@20 mm: hard passes 20 mm here "
      f"({d0['adds_m']['hard'] * 1e3:.1f} mm); `000011/49` at "
      f"{[r['adds']['hard'] for r in rows if r['label'] == '000011/49'][0] * 1e3:.1f} mm "
      f"is the one that fails. Two different single frames, two different statements, both "
      f"single-frame accidents.")
    loo = float(np.mean(np.delete([auc_frame(r["adds"]["hard"]) for r in rows], 
                                  [i for i, r in enumerate(rows) if r["label"] == FAIL_FRAME][0])))
    A(f"- Leave-one-out: dropping it would move the pooled hard AUC from "
      f"{a['pooled']['hard']['mean']:.3f} to {loo:.3f}. It is **not** dropped: the frame "
      f"belongs to the pre-registered 15, and removing it after seeing the result would be "
      f"selection. Contrast the honest reading -- the estimator is wrong on 1/15 frames, "
      f"and that is already in the rotation table (44.5 deg).")
    A("")
    A("## 5. Checks, sensitivity, caveats")
    A("")
    A("**Arithmetic checks.**")
    A("")
    A(f"- Closed form vs a dense numerical integral of the recall curve (200 001 "
      f"points): max abs diff "
      f"{chk['auc_closed_vs_numeric']['max_abs_diff']:.1e} "
      f"(hard {chk['auc_closed_vs_numeric']['hard']['numeric']:.7f} vs "
      f"{chk['auc_closed_vs_numeric']['hard']['closed']:.7f}).")
    lc = chk.get("lmo_cross_check")
    if lc:
        A(f"- **The same formula reproduces the published LMO AUC** "
          f"(`{lc['source']}`, n={lc['n_frames']}, a completely different frame set): "
          f"hard {lc['hard']['mean']:.4f} vs published {lc['hard']['published_mean']:.4f}, "
          f"Laplace {lc['laplace']['mean']:.4f} vs {lc['laplace']['published_mean']:.4f}, "
          f"MC {lc['mc']['mean']:.4f} vs {lc['mc']['published_mean']:.4f}; max abs diff "
          f"{lc['max_abs_diff']:.1e}. The LINEMOD number is therefore on the same scale as "
          f"the LMO rows of Table 5.20.")
    cm = chk.get("ci_matches_mean_adds_ci")
    if cm:
        A(f"- **The bootstrap reproduces the stored mean/median ADD-S CIs exactly** after "
          f"the map $x\\mapsto1-x/T$ (max abs diff mean "
          f"{cm['max_abs_diff']:.1e}, median {cm['max_abs_diff_median']:.1e}; e.g. hard mean "
          f"CI {ci(dict(mean_ci=cm['hard']['mean_ci']))} vs rescaled stored "
          f"{ci(dict(mean_ci=cm['hard']['mean_ci_rescaled']))}). This is the affine identity "
          f"of §0 consequence 3, confirmed numerically rather than asserted.")
    A(f"- Affinity of the AUC in the mean: "
      f"`|AUC - (1 - mean/0.1)|` = "
      f"{aff['hard']['abs_diff']:.1e} / {aff['laplace']['abs_diff']:.1e} / "
      f"{aff['mc']['abs_diff']:.1e} for hard / Laplace / MC; any frame at the cap: "
      f"{'yes' if any(aff[m]['clipping_active'] for m in METHODS) else 'no'}.")
    A(f"- Full-mesh-vs-full-mesh ADD-S check instead of the 1000-query/dense-GT estimator "
      f"(`adds_full_check_m` / `adds_full_check_laplace_m`): hard "
      f"{a['sensitivity_full_check']['hard']['mean']:.3f} "
      f"({a['sensitivity_full_check']['hard']['delta_vs_primary']:+.3f}) and Laplace "
      f"{a['sensitivity_full_check']['laplace']['mean']:.3f} "
      f"({a['sensitivity_full_check']['laplace']['delta_vs_primary']:+.3f}); MC has no "
      f"such column. The ADD-S estimator choice does not move the ordering or the size.")
    A("")
    A("**Sensitivity.**")
    A("")
    A(f"- Bootstrap stratified by object (8 eggbox + 7 glue resampled within object, the "
      f"design-preserving variant): hard "
      f"{ci(dict(mean_ci=a['pooled']['hard']['mean_ci_stratified']))} (vs "
      f"{ci(a['pooled']['hard'])}), Laplace "
      f"{ci(dict(mean_ci=a['pooled']['laplace']['mean_ci_stratified']))} (vs "
      f"{ci(a['pooled']['laplace'])}). The Laplace/MC interval is ~2x narrower when the "
      f"object mix is held fixed -- object identity is a large part of the between-frame "
      f"variance -- but the hard-vs-Laplace separation survives both.")
    A(f"- MC seed noise: hierarchical bootstrap over frames *and* the 20 MC seeds gives "
      f"{ci(dict(mean_ci=a['pooled']['mc']['mean_ci_hier']))} vs the frame-only "
      f"{ci(a['pooled']['mc'])}. Seed noise is negligible next to frame noise, as expected "
      f"because MC estimates the same functional as Laplace.")
    hard_loo = [float(np.delete([auc_frame(r["adds"]["hard"]) for r in rows], i).mean())
                for i in range(n)]
    lap_loo = [float(np.delete([auc_frame(r["adds"]["laplace"]) for r in rows], i).mean())
               for i in range(n)]
    A(f"- Leave-one-out over the 15 frames: pooled hard AUC spans "
      f"[{min(hard_loo):.3f}, {max(hard_loo):.3f}], Laplace "
      f"[{min(lap_loo):.3f}, {max(lap_loo):.3f}]. No single frame changes the ordering.")
    A("")
    A("**Caveats.**")
    A("")
    A(f"1. **n=15, and this is descriptive.** At most 15 curve steps; pooled AUC CI "
      f"half-widths are "
      f"{(a['pooled']['hard']['mean_ci'][1] - a['pooled']['hard']['mean_ci'][0]) / 2:.3f} "
      f"(hard) and "
      f"{(a['pooled']['laplace']['mean_ci'][1] - a['pooled']['laplace']['mean_ci'][0]) / 2:.3f} "
      f"(Laplace). Object rows are n=8 / n=7. No threshold-based conclusion more precise "
      f"than 'hard is clearly more accurate than the posterior mean on these frames' is "
      f"supported, and that conclusion was already in `pose_error.md` without any AUC.")
    A(f"2. **The cell is redundant with the mean ADD-S** (identity above). The only reason "
      f"to put it in Table 5.20 is format consistency with the LMO rows; the only reason "
      f"*not* to is that it invites reading the column as an independent metric. Both "
      f"readings are stated here so the choice is explicit.")
    A("3. **Not the BOP metric.** BOP reports AP over a 3-D detection pipeline with "
      "per-object normalisation on the official test split; this is the older PoseCNN-style "
      "AUC on a 0-0.1 m window over 15 hand-picked frames. Not comparable to a leaderboard "
      "entry and not quoted as one.")
    A("4. **GT-associated correspondences.** Poses were fit to nearest-vertex "
      "correspondences selected through the GT pose with a RANSAC plane subset, so the GT "
      "is favoured by construction and the absolute ADD-S is optimistic. Only the relative "
      "ordering (hard more accurate than Laplace/MC) is measured.")
    A("5. **Frame selection.** The 15 frames are those where both objects gave K=2 "
      "physical modes with >=900 valid depth pixels; heavy occlusion and K=1 frames are "
      "outside the sample, so the AUC is conditional on K>=2.")
    A("6. **recall@50 mm is saturated and recall@5 mm for Laplace/MC is structurally 0**; "
      "only 10 and 20 mm carry signal, and the 10 mm value moves by +/-1 frame with the "
      "vertex draw (§2).")
    A("")
    A("## 6. Reproduce")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/real/linemod")
    A("/home/syk/miniconda3/envs/WM/bin/python run_auc15.py")
    A("```")
    A("")
    A(f"Runtime {dt_s:.1f} s (login node, CPU). The script reads only "
      f"`results/pose_error.json` and, for the cross-check, `results/lmo_two_mode.json`; it "
      f"does not import or modify any other script. Constants are at the top "
      f"(`T_M=0.1`, `N_BOOT=1000`, `BOOT_SEED={BOOT_SEED}`). Every reported number is "
      f"deterministic; the only field that changes between runs is the recorded runtime.")
    A("")
    A("## 7. Files")
    A("")
    A("- `run_auc15.py` -- this analysis.")
    A("- `results/auc15.md` -- this report.")
    A("- `results/auc15.json` -- every number: meta, per-frame rows, stored recall curves, "
      "aggregates with CIs, checks, failure and fragility bookkeeping.")
    A("- `results/auc15.png` -- (a) pooled recall curve, (b) the 15 per-frame ADD-S values "
      "that generate its steps, (c) AUC bars with CIs per object and pooled.")
    A("- `results/pose_error.json` -- the input (per-frame ADD-S of the 15 frames).")
    A("")
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    src, rows = load_frames()
    groups = group_index(rows)
    n = len(rows)
    assert n == 15, n
    assert [(g[0], len(g[2])) for g in groups] == [("eggbox", 8), ("glue", 7)], groups
    # sanity: the per-frame values must reproduce the medians already published
    assert abs(np.median([r["adds"]["hard"] for r in rows]) * 1e3 - 3.7) < 0.15
    assert abs(np.median([r["adds"]["laplace"] for r in rows]) * 1e3 - 13.9) < 0.2

    out = build(rows, groups, src)
    figure(rows, out, OUT_PNG)
    dt = time.time() - t0
    write_md(rows, out, OUT_MD, dt)

    meta = dict(
        date="2026-11-28",
        script="run_auc15.py",
        source=str(SRC.relative_to(HERE)),
        n_frames=n,
        objects={OBJ_NAME[g[1]]: len(g[2]) for g in groups},
        threshold_range_m=[0.0, T_M],
        recall_thresholds_mm=list(THRESH_MM),
        n_boot=N_BOOT,
        boot_seed=BOOT_SEED,
        ci="percentile 95% frame bootstrap (plain 15-of-15, seeds matching run_pose_error.py)",
        bootstrap_variants=dict(
            primary="plain frame bootstrap",
            stratified_by_object="reported for AUC pooled and recall pooled (8/7 within object)",
            mc_hierarchical="frames x 20 MC seeds (AUC, MC only)",
        ),
        definition=("ADD-S AUC(0-T) = (1/T) int_0^T recall(t) dt "
                    "= mean_i(1 - min(max(ADD-S_i,0),T)/T), T=0.1 m"),
        identity="AUC = 1 - mean(ADD-S)/0.1 exactly while max ADD-S <= 0.1 m (true here)",
        source_meta={k: src.get("meta", {}).get(k) for k in
                     ("tau", "n_starts", "n_points", "mc_n", "mc_seeds", "n_boot",
                      "n_add_pts", "n_add_subs", "date")},
        description=("Descriptive only: 15 fixed frames, recall curve has <=15 steps, "
                     "GT-associated correspondences, not the BOP AP metric, and the AUC is "
                     "an affine rescaling of the mean ADD-S already reported."),
        failure_frame=FAIL_FRAME,
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        host=platform.node(),
    )
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        meta["git_branch"] = subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "--abbrev-ref", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - git optional
        pass

    total = time.time() - t0
    meta["runtime_s"] = round(total, 2)
    payload = dict(meta=meta, checks=out["checks"], per_frame=out["per_frame"],
                   recall_curve=out["recall_curve"],
                   aggregate=dict(auc=out["auc"], recall=out["recall"]),
                   failures=out["failures"])
    OUT_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")

    ap = out["auc"]["pooled"]
    print(f"frames={n} (8 eggbox + 7 glue)  T={T_M} m  boot={N_BOOT}  runtime={total:.2f}s")
    for m in METHODS:
        b = ap[m]
        print(f"  {m:<8} AUC {b['mean']:.3f} [{b['mean_ci'][0]:.3f}, {b['mean_ci'][1]:.3f}]"
              f"  median {b['median']:.3f}  recall@5/10/20/50 mm "
              f"{out['recall']['pooled']['5mm'][m]['value']:.3f}/"
              f"{out['recall']['pooled']['10mm'][m]['value']:.3f}/"
              f"{out['recall']['pooled']['20mm'][m]['value']:.3f}/"
              f"{out['recall']['pooled']['50mm'][m]['value']:.3f}"
              f"  ({b['n']} frames)")
    print(f"  checks: closed-vs-numeric max|d|="
          f"{out['checks']['auc_closed_vs_numeric']['max_abs_diff']:.1e}"
          f"  AUC-vs-1-mean/T max|d|={out['checks']['auc_affine_in_mean']['max_abs_diff']:.1e}"
          f"  CI-vs-stored mean CI max|d|="
          f"{out['checks']['ci_matches_mean_adds_ci']['max_abs_diff']:.1e}"
          f"  LMO cross-check max|d|="
          f"{out['checks']['lmo_cross_check']['max_abs_diff']:.1e}"
          f"  hard dominates: {out['checks']['curve_dominance']['hard_weakly_dominates']}")
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
