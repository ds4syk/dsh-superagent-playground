#!/usr/bin/env python3
"""P1 end-to-end toy training through the Laplace-QPEP layer (CPU, synthetic instance).

Why this file exists
--------------------
``run_main.py`` measures per-theta gradient accuracy on the two-mode crossing
instance, but its gradient-descent fit branch is disabled (``FIT_SEEDS = 0``)
and was never run.  This script is the actual end-to-end test:

1. **Fit through the layer.**  Learn the scalar input ``theta`` (the
   branch-crossing parameter) so that the layer output pose matches a target
   pose.  Two targets:
     * ``mix``  -- the average of the two *global-branch* poses at theta = 0
       (the target named in README section "optimisation fit is a diagnostic").
       At finite tau this is *not* exactly representable: the Laplace layer
       puts ~8% mass on the two excited minima, so the loss floor is ~1.1e-2,
       far above the 1e-4 success threshold.
     * ``star`` -- the pose of the global branch at theta* = 0.10.  This one is
       representable (loss = 0 at theta*), so it is the positive control.
   Methods: hard argmin, multi-modal Laplace (analytic dpose through the
   fixed closed-form ``core.dlogdetH_dtheta`` path) and mode-level Monte Carlo
   (score function, n = 200).  20 random initial theta in [-0.2, 0.2],
   150 steps, lr = 0.01, global-norm gradient clipping at 1.0.

2. **Gradient variance.**  dpose/dtheta at theta = 0 and 0.01 across 20 seeds
   for hard / Laplace / MC.  At theta = 0 the hard argmin sits on an exact tie:
   the derivative does not exist, and the two one-sided branch values are
   reported explicitly.

3. **Throughput.**  Per-sample forward+backward wall time for the three
   methods (median over 20 repetitions), decomposition into mode enumeration /
   forward / backward, and a loop batch test at B = 1/8/64 (this implementation
   enumerates modes per sample, it is *not* vectorised across a batch).

4. One 6-panel figure (``results/e2e_fit.png``), plus ``e2e_fit.json`` and
   ``e2e_fit.md``.

Everything reuses the committed layer code (``instance.py``, ``qpep_core.py``,
``run_main.py``); this file only orchestrates, records and plots.  CPU only,
single-threaded BLAS for stable timings.
"""
from __future__ import annotations

import os

# Stable CPU timings: pin BLAS/OpenMP to one thread before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import instance as inst_mod          # noqa: E402
import qpep_core as core             # noqa: E402
import run_main as R                 # noqa: E402

OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- configuration
INSTANCE_SEED = 20261127
N_STARTS = R.N_STARTS  # 128: same cold enumeration as main_table.md
TAU = R.TAU            # 0.01
N_MC = R.N_MC          # 200 mode samples
THETA_STAR = 0.10      # target theta for the "star" pose target
LR = 0.01
CLIP = 1.0
STEPS = 120
N_FIT_SEEDS = 20
N_VAR_SEEDS = 20
VAR_THETAS = (0.0, 0.01)
N_THROUGHPUT_REPS = 20
BATCH_SIZES = (1, 8, 64)
CONTROL_CLIPS = (1.0, 0.1, 0.02)
CONTROL_SEEDS = 8
THETA_BOUND = 0.5      # hard clip on theta inside the optimiser (same as run_main)


# ---------------------------------------------------------------- json helpers

def json_safe(x):
    """Recursively replace non-finite floats with None (strict JSON)."""
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    return x


# ---------------------------------------------------------------- layer wrappers

def layer_eval(I, method, theta, rng):
    """One forward+backward call of the layer.  Returns pose, dpose, K."""
    if method == "hard":
        d = R.pose_and_derivs(I, theta)
        k = int(np.argmin(d["J"]))
        return dict(pose=d["poses"][k], dpose=d["dposes"][k], K=len(d["J"]), k=k)
    if method == "laplace":
        out = R.laplace(I, theta)
        return dict(pose=out["pose"], dpose=out["dpose"], K=len(out["w"]),
                    w=out["w"])
    if method == "mc":
        out = R.mc(I, theta, rng=rng)
        return dict(pose=out["pose"], dpose=out["dpose"], K=len(out["p"]),
                    p=out["p"])
    raise ValueError(method)


def enum_modes(I, theta, seed, ns=N_STARTS):
    """Cold mode enumeration with an *explicit* random-start seed.

    ``Instance.find_modes`` uses a fixed internal rng (seed + 7); passing
    ``starts`` lets us vary the enumeration seed, which is the only stochastic
    ingredient for the hard / Laplace gradients.
    """
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(ns, 4))
    return I.find_modes(I.M(theta), starts=starts)


# ---------------------------------------------------------------- targets

def build_targets(I):
    """Return the two target poses and their (method-specific) loss floors."""
    m0 = I.modes0                       # 4 minima, sorted by J
    glob = np.argsort(m0["J"])[:2]      # the two *global* branches at theta=0
    pA, pB = I.pose(m0["q"][glob[0]]), I.pose(m0["q"][glob[1]])
    p_mix = 0.5 * (pA + pB)

    m_star = I.find_modes(I.M(THETA_STAR), n_starts=N_STARTS)
    k_star = int(np.argmin(m_star["J"]))
    p_star = I.pose(m_star["q"][k_star])

    # hard floor for mix: it can only ever output one branch pose
    floor_hard_mix = float(np.sum((pA - p_mix) ** 2))

    # Laplace floor on a dense grid around the crossing (the fold makes the
    # minimum sharp; scan it rather than assuming theta=0 is optimal).
    grid = np.unique(np.concatenate([np.linspace(-0.05, 0.05, 101),
                                     np.array([0.0, -1.3e-3, 1.3e-3])]))
    lap_loss = []
    lap_dp = []
    for th in grid:
        L = R.laplace(I, th)
        lap_loss.append(float(np.sum((L["pose"] - p_mix) ** 2)))
        lap_dp.append(float(np.linalg.norm(L["dpose"])))
    lap_loss = np.array(lap_loss)
    lap_dp = np.array(lap_dp)
    i_best = int(np.argmin(lap_loss))
    floor_laplace_mix = float(lap_loss[i_best])
    theta_best_mix = float(grid[i_best])

    # MC floor at the Laplace optimum: <loss> = bias^2 + (1/n) Tr Var_p(pose).
    L0 = R.laplace(I, theta_best_mix)
    p_at = L0["modes"]["poses"]
    w_at = L0["w"]
    mean_pose = w_at @ p_at
    var = float(np.sum(w_at * np.sum((p_at - mean_pose) ** 2, axis=1)))
    mc_noise = var / N_MC

    # star floor on a fine grid around theta* (should be ~0 at theta*)
    grid2 = np.linspace(THETA_STAR - 0.01, THETA_STAR + 0.01, 41)
    star_loss = []
    for th in grid2:
        L = R.laplace(I, th)
        star_loss.append(float(np.sum((L["pose"] - p_star) ** 2)))
    star_loss = np.array(star_loss)
    floor_laplace_star = float(star_loss.min())

    return dict(
        mix=dict(
            kind="mix", name="average of the two global-branch poses at theta=0",
            pose=p_mix,
            floor_hard=floor_hard_mix,
            floor_laplace=floor_laplace_mix,
            floor_mc=float(floor_laplace_mix + mc_noise),
            theta_best=theta_best_mix,
            mc_noise_floor=float(mc_noise),
            w_at_best=L0["w"].tolist(),
            var_pose=var,
            note=("finite-tau Laplace mixes ~8% weight onto the two excited "
                  "minima (J gap 0.034 = 3.4 tau), so the branch average has a "
                  "representability floor instead of loss 0"),
        ),
        star=dict(
            kind="star", name=f"pose of the global branch at theta*={THETA_STAR:.2f}",
            pose=p_star,
            theta_target=THETA_STAR,
            floor_hard=0.0,
            floor_laplace=floor_laplace_star,
            floor_mc=None,   # measured from the fit itself (no cheap exact form)
            note=("representable target: the global branch at theta* is unique "
                  "(gap 0.72), so loss(theta*) = 0"),
        ),
    )


# ---------------------------------------------------------------- fit

def fit_run(I, target_pose, method, theta0, seed, steps, lr, clip):
    """Plain gradient descent on the scalar theta; returns the visited curve."""
    rng = np.random.default_rng(seed)
    th = float(theta0)
    losses, thetas = [], []
    for _ in range(steps):
        out = layer_eval(I, method, th, rng)
        diff = out["pose"] - target_pose
        loss = float(diff @ diff)
        g = 2.0 * (diff @ out["dpose"])
        gn = float(np.linalg.norm(g))
        if gn > clip:
            g = g * (clip / gn)
        losses.append(loss)
        thetas.append(th)
        th = float(np.clip(th - lr * g, -THETA_BOUND, THETA_BOUND))
    # one extra evaluated point at the final theta, so "final loss" is measured
    out = layer_eval(I, method, th, rng)
    diff = out["pose"] - target_pose
    losses.append(float(diff @ diff))
    thetas.append(th)
    return np.array(losses), np.array(thetas)


def fit_records(I, targets, steps, n_seeds, lr, clip):
    rows = []
    for tname in ("mix", "star"):
        target = targets[tname]
        for method in ("hard", "laplace", "mc"):
            seeds = []
            for s in range(n_seeds):
                rng = np.random.default_rng(100 + s)
                theta0 = float(rng.uniform(-0.2, 0.2))
                t0 = time.perf_counter()
                losses, thetas = fit_run(I, target["pose"], method, theta0, s,
                                          steps, lr, clip)
                dt = time.perf_counter() - t0
                step_cross = thetas[:-1] * thetas[1:] < 0
                dloss = np.abs(np.diff(losses))
                hit = np.flatnonzero(losses < 1e-4)
                seeds.append(dict(
                    seed=s, theta0=theta0, theta_final=float(thetas[-1]),
                    loss_init=float(losses[0]), loss_final=float(losses[-1]),
                    loss_min=float(losses.min()),
                    steps_to_1e4=(int(hit[0]) if len(hit) else None),
                    n_crossings=int(step_cross.sum()),
                    max_abs_dloss=float(dloss.max()),
                    max_abs_dloss_at_cross=(float(dloss[step_cross].max())
                                            if step_cross.any() else None),
                    n_spikes=int(np.sum(dloss > 1.0)),
                    wall_s=float(dt),
                    losses=[round(float(x), 12) for x in losses],
                    thetas=[round(float(x), 12) for x in thetas],
                ))
            rows.append(dict(target=tname, method=method, seeds=seeds))
            print(f"  fit {tname:4s} {method:7s} "
                  f"median final {np.median([r['loss_final'] for r in seeds]):.3e} "
                  f"success {np.mean([r['loss_final'] < 1e-4 for r in seeds]):.0%}",
                  flush=True)
    return rows


def summarize_fit(rows, targets):
    summary = []
    for rec in rows:
        seeds = rec["seeds"]
        tname, method = rec["target"], rec["method"]
        target = targets[tname]
        fin = np.array([r["loss_final"] for r in seeds])
        init = np.array([r["loss_init"] for r in seeds])
        steps_ok = [r["steps_to_1e4"] for r in seeds if r["steps_to_1e4"] is not None]
        n_cross = np.array([r["n_crossings"] for r in seeds])
        max_dl = np.array([r["max_abs_dloss"] for r in seeds])
        max_dlc = [r["max_abs_dloss_at_cross"] for r in seeds
                   if r["max_abs_dloss_at_cross"] is not None]
        side = np.array([r["theta0"] > 0 for r in seeds])
        floor = float({"hard": target["floor_hard"],
                       "laplace": target["floor_laplace"],
                       "mc": (target["floor_mc"] if target.get("floor_mc") is not None
                              else target["floor_laplace"])}[method])
        summary.append(dict(
            target=tname, method=method, n=len(seeds),
            loss_init_median=float(np.median(init)),
            loss_final_median=float(np.median(fin)),
            loss_final_mean=float(np.mean(fin)),
            loss_final_min=float(fin.min()),
            loss_final_max=float(fin.max()),
            success_rate=float(np.mean(fin < 1e-4)),
            success_rate_pos_init=float(np.mean(fin[side] < 1e-4)) if side.any() else None,
            success_rate_neg_init=float(np.mean(fin[~side] < 1e-4)) if (~side).any() else None,
            reached_any_rate=float(np.mean([r["loss_min"] < 1e-4 for r in seeds])),
            steps_to_1e4_median=(float(np.median(steps_ok)) if steps_ok else None),
            steps_to_1e4_iqr=(list(np.percentile(steps_ok, [25, 75])) if steps_ok else None),
            n_seeds_reached=len(steps_ok),
            cross_rate=float(np.mean(n_cross > 0)),
            cross_count_median=float(np.median(n_cross)),
            max_abs_dloss_median=float(np.median(max_dl)),
            max_abs_dloss_max=float(max_dl.max()),
            max_abs_dloss_at_cross_median=(float(np.median(max_dlc)) if max_dlc else None),
            floor_laplace=floor,
            gap_to_floor_median=(float(np.median(fin) / floor)
                                 if floor > 1e-12 else None),
            theta_final_median=float(np.median([r["theta_final"] for r in seeds])),
            success_by_side=dict(
                pos=[int(s) for s in side],
                rate_pos=float(np.mean(fin[side] < 1e-4)) if side.any() else None,
                rate_neg=float(np.mean(fin[~side] < 1e-4)) if (~side).any() else None,
            ),
        ))
    return summary


# ---------------------------------------------------------------- gradient variance

def _vec_stats(vals):
    vals = np.asarray(vals, float)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0, ddof=1)
    return dict(
        mean_norm=float(np.linalg.norm(mean)),
        std_frob=float(np.linalg.norm(std)),
        rel_std=float(np.linalg.norm(std) / max(np.linalg.norm(mean), 1e-300)),
        std_max_coord=float(std.max()),
        n_distinct=int(len(np.unique(np.round(vals, 8), axis=0))),
        mean=mean.tolist(),
        std=std.tolist(),
    )


def gradient_variance(I, theta, n_seeds, dpose_ref=None):
    """dpose/dtheta across 20 seeds for the three methods.

    * hard / Laplace: vary the *mode-enumeration* seed (their only stochastic
      ingredient).  Laplace is analytic, so this measures numerical jitter.
    * MC: vary the sampling seed at a fixed cold mode set.  Additionally run
      MC with enumeration variation to show the combined noise.
    """
    ms = enum_modes(I, theta, 9000)          # fixed cold set for MC
    d_fixed = R.pose_and_derivs(I, theta, m=ms)

    hard, lap, mc_samp = [], [], []
    for s in range(n_seeds):
        m = enum_modes(I, theta, 5000 + s)
        d = R.pose_and_derivs(I, theta, m=m)
        k = int(np.argmin(d["J"]))
        hard.append(d["dposes"][k])
        lap.append(R.laplace(I, theta, m=m)["dpose"])
        mc_samp.append(R.mc(I, theta, rng=np.random.default_rng(7000 + s),
                            d=d_fixed)["dpose"])
    mc_comb = [R.mc(I, theta, rng=np.random.default_rng(8000 + s),
                    d=R.pose_and_derivs(I, theta,
                                        m=enum_modes(I, theta, 6000 + s)))["dpose"]
               for s in range(n_seeds)]

    out = dict(
        theta=float(theta),
        hard=dict(_vec_stats(hard), note="tie-breaking dependent"),
        laplace=dict(_vec_stats(lap), note="deterministic analytic gradient"),
        mc=dict(_vec_stats(mc_samp), note="20 sampling seeds, fixed mode set"),
        mc_combined=dict(_vec_stats(mc_comb), note="sampling + enumeration seeds"),
    )

    # hard at the exact tie: both one-sided branch derivatives
    d0 = d_fixed
    order = np.argsort(d0["J"])[:2]
    dA, dB = d0["dposes"][order[0]], d0["dposes"][order[1]]
    delta = dA - dB
    out["hard_branch"] = dict(
        dpose_branch_a=dA.tolist(), dpose_branch_b=dB.tolist(),
        delta_norm=float(np.linalg.norm(delta)),
        rel_ambiguity=float(np.linalg.norm(delta)
                            / max(np.linalg.norm(0.5 * (dA + dB)), 1e-300)),
        J_gap=float(abs(d0["J"][order[0]] - d0["J"][order[1]])),
    )
    if dpose_ref is not None:
        out["mc_bias_rel_vs_laplace"] = float(
            np.linalg.norm(np.mean(mc_samp, axis=0) - dpose_ref)
            / max(np.linalg.norm(dpose_ref), 1e-300))
    return out


# ---------------------------------------------------------------- throughput

def _time_call(fn, reps):
    fn()                                     # warm-up (page cache, numba-free)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.array(ts)


def forward_only(I, theta):
    """Mode enumeration + per-mode poses (no derivatives)."""
    m = I.find_modes(I.M(theta), n_starts=N_STARTS)
    return np.array([I.pose(q) for q in m["q"]])


def throughput(I, reps):
    out = dict(reps=reps)
    for tag, theta in (("theta0", 0.0), ("theta01", THETA_STAR)):
        t_enum = _time_call(lambda th=theta: I.find_modes(I.M(th), n_starts=N_STARTS),
                            reps)
        t_fwd = _time_call(lambda th=theta: forward_only(I, th), reps)
        methods = {}
        for method in ("hard", "laplace", "mc"):
            fn = {
                "hard": lambda th=theta: R.hard(I, th),
                "laplace": lambda th=theta: R.laplace(I, th),
                "mc": lambda th=theta: R.mc(I, th, rng=np.random.default_rng(0)),
            }[method]
            ts = _time_call(fn, reps)
            methods[method] = dict(
                median_ms=float(1e3 * np.median(ts)),
                iqr_ms=[float(1e3 * np.percentile(ts, 25)),
                        float(1e3 * np.percentile(ts, 75))],
                min_ms=float(1e3 * ts.min()),
                enum_share=float(min(1.0, np.median(t_enum)
                                     / max(np.median(ts), 1e-300))),
            )
        out[tag] = dict(
            enum_median_ms=float(1e3 * np.median(t_enum)),
            forward_median_ms=float(1e3 * np.median(t_fwd)),
            methods=methods,
        )

    # loop batch test (Laplace; different theta per sample)
    batch = []
    for B in BATCH_SIZES:
        ths = np.linspace(THETA_STAR - 0.02, THETA_STAR + 0.02, B)
        nr = 10 if B <= 8 else 5
        ts = _time_call(lambda ths=ths: [R.laplace(I, th) for th in ths], nr)
        batch.append(dict(
            B=B, reps=nr,
            total_ms_median=float(1e3 * np.median(ts)),
            per_sample_ms=float(1e3 * np.median(ts) / B),
        ))
    out["batch"] = batch
    out["note"] = ("mode enumeration is per-sample in this implementation; a "
                   "vectorised multi-instance Newton (batched over B different M) "
                   "is not implemented, so the loop batch gives no speed-up")
    return out


# ---------------------------------------------------------------- landscape

def landscape(I, targets):
    ths = np.unique(np.concatenate([
        np.linspace(-0.2, 0.2, 41),
        np.linspace(-0.02, 0.02, 81),
        np.array([-1.3e-3, 1.3e-3, 0.0]),
    ]))
    loss, dp, K, hard_loss = [], [], [], []
    for th in ths:
        L = R.laplace(I, th)
        loss.append(float(np.sum((L["pose"] - targets["mix"]["pose"]) ** 2)))
        dp.append(float(np.linalg.norm(L["dpose"])))
        K.append(len(L["w"]))
        H = R.hard(I, th)
        hard_loss.append(float(np.sum((H["pose"] - targets["mix"]["pose"]) ** 2)))
    return dict(theta=ths.tolist(), loss_mix=loss, dpose_norm=dp,
                K=K, loss_hard_mix=hard_loss,
                fold_theta=float(ths[int(np.argmax(dp))]),
                dpose_max=float(np.max(dp)), dpose_min=float(np.min(dp)))


# ---------------------------------------------------------------- clip control

def clip_control(I, targets, steps, n_seeds):
    rows = []
    for clip in CONTROL_CLIPS:
        finals = []
        for s in range(n_seeds):
            rng = np.random.default_rng(100 + s)   # paired inits across clips
            theta0 = float(rng.uniform(-0.2, 0.2))
            losses, thetas = fit_run(I, targets["mix"]["pose"], "laplace",
                                     theta0, s, steps, LR, clip)
            finals.append(dict(seed=s, theta0=theta0, loss_final=float(losses[-1]),
                               theta_final=float(thetas[-1]),
                               loss_min=float(losses.min())))
        arr = np.array([r["loss_final"] for r in finals])
        rows.append(dict(clip=clip, seeds=finals,
                         loss_median=float(np.median(arr)),
                         loss_min=float(arr.min()),
                         n_below_floor=float(np.mean(arr < 1e-4)),
                         note=("step = lr*clip = %.1e; crossing width "
                               "tau/slope ~ 1.4e-3" % (LR * clip))))
        print(f"  clip {clip:4.2f}: median final {np.median(arr):.3e} "
              f"min {arr.min():.3e}", flush=True)
    return rows


# ---------------------------------------------------------------- plotting

def make_plot(path, fit_rows, summary, targets, var, thr, land, ctrl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6))
    ax = axes.ravel()
    colors = {"hard": "C3", "laplace": "C0", "mc": "C2"}
    labels = {"hard": "hard argmin", "laplace": "Laplace (analytic)",
              "mc": "mode-level MC (n=200)"}

    def curve(ax_, tname, title, ylim=None):
        recs = [r for r in fit_rows if r["target"] == tname]
        style = {"hard": dict(lw=1.0, ls="-"), "mc": dict(lw=1.2, ls="--"),
                 "laplace": dict(lw=2.0, ls="-")}
        # draw hard/mc first so the (usually overlapping) Laplace median stays visible
        for method in ("hard", "mc", "laplace"):
            rec = next((r for r in recs if r["method"] == method), None)
            if rec is None:
                continue
            L = np.array([r["losses"] for r in rec["seeds"]])   # (S, T+1)
            med = np.median(L, axis=0)
            q1, q3 = np.percentile(L, [25, 75], axis=0)
            x = np.arange(len(med))
            ax_.semilogy(x, med, color=colors[method], label=labels[method],
                         **style[method])
            ax_.fill_between(x, np.maximum(q1, 1e-16), np.maximum(q3, 1e-16),
                             color=colors[method], alpha=0.10, lw=0)
        ax_.axhline(1e-4, color="k", ls=":", lw=0.9)
        ax_.text(2, 1.4e-4, "success threshold 1e-4", fontsize=7)
        if tname == "mix":
            for m, val in (("hard", "floor_hard"), ("laplace", "floor_laplace"),
                           ("mc", "floor_mc")):
                if targets["mix"].get(val) is not None:
                    ax_.axhline(targets["mix"][val], color=colors[m], ls="--",
                                lw=0.8, alpha=0.6)
        ax_.set_xlabel("step"); ax_.set_ylabel("pose loss (log)")
        ax_.set_title(title, fontsize=9)
        ax_.grid(alpha=0.25)
        ax_.legend(fontsize=7)
        if ylim:
            ax_.set_ylim(*ylim)

    curve(ax[0], "mix", "A. fit target = average of global branches (theta=0)")

    # B. star: final loss vs initial theta (the methods coincide away from the
    # crossing, so the three scatters overlap; an inset shows the loss curves
    # restricted to the initialisations on the correct side).
    S = {(r["target"], r["method"]): r for r in summary}
    pct = lambda v: "—" if v is None else f"{100*v:.0f}%"
    star_recs = {r["method"]: r for r in fit_rows if r["target"] == "star"}
    for method, marker in (("hard", "o"), ("laplace", "x"), ("mc", "+")):
        rec = star_recs.get(method)
        if rec is None:
            continue
        jit = {"hard": -0.003, "laplace": 0.0, "mc": 0.003}[method]
        x = np.array([s["theta0"] for s in rec["seeds"]]) + jit
        y = np.maximum([s["loss_final"] for s in rec["seeds"]], 1e-16)
        ax[1].scatter(x, y, s=26, marker=marker, color=colors[method],
                      label=labels[method], alpha=0.85, lw=1.2)
    ax[1].axhline(1e-4, color="k", ls=":", lw=0.9)
    ax[1].axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-5, 40.0)
    ax[1].set_xlabel("initial theta0"); ax[1].set_ylabel("final loss")
    ax[1].set_title(f"B. star target: final loss vs init "
                    f"(theta*={THETA_STAR:.2f}, methods coincide)", fontsize=9)
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=6.5, loc="center left")
    txt = "\n".join(f"{m}: +side {pct(S[('star', m)]['success_by_side']['rate_pos'])} / "
                    f"-side {pct(S[('star', m)]['success_by_side']['rate_neg'])}"
                    for m in ("hard", "laplace", "mc") if ("star", m) in S)
    ax[1].text(0.02, 0.05, txt, transform=ax[1].transAxes, fontsize=6.5,
               va="bottom", bbox=dict(fc="white", ec="0.7", alpha=0.8))
    ax[1].text(0.02, 0.80, "theta0<0: stuck at the branch wall (loss 8.2)",
               transform=ax[1].transAxes, fontsize=6.5)
    # inset: median (+IQR) loss curve for correct-side initialisations only
    axin = ax[1].inset_axes([0.40, 0.10, 0.57, 0.33])
    for method in ("hard", "laplace", "mc"):
        rec = star_recs.get(method)
        if rec is None:
            continue
        pos = [s for s in rec["seeds"] if s["theta0"] > 0]
        if not pos:
            continue
        L = np.array([s["losses"] for s in pos])
        med = np.median(L, axis=0)
        q1, q3 = np.percentile(L, [25, 75], axis=0)
        xx = np.arange(len(med))
        axin.semilogy(xx, med, color=colors[method], lw=1.0)
        axin.fill_between(xx, np.maximum(q1, 1e-16), np.maximum(q3, 1e-16),
                          color=colors[method], alpha=0.12, lw=0)
    axin.axhline(1e-4, color="k", ls=":", lw=0.7)
    axin.set_title("positive-inits only", fontsize=6)
    axin.tick_params(labelsize=6)
    axin.grid(alpha=0.25)

    # C. gradient variance bars
    ths = [v["theta"] for v in var]
    width = 0.26
    for i, method in enumerate(("hard", "laplace", "mc")):
        vals = [v[method]["std_frob"] for v in var]
        xpos = np.arange(len(ths)) + (i - 1) * width
        ax[2].bar(xpos, np.maximum(vals, 1e-16), width, color=colors[method],
                  label=labels[method],
                  hatch=("//" if method == "hard" else None))
    ax[2].set_yscale("log")
    ax[2].set_xticks(np.arange(len(ths)))
    ax[2].set_xticklabels([f"theta={t:g}" for t in ths])
    ax[2].set_ylabel("std of dpose/dtheta (log)")
    ax[2].set_title("C. std of dpose/dtheta over 20 seeds\n"
                    "hatched: hard at theta=0 is a 2-value tie (undefined)",
                    fontsize=9)
    ax[2].grid(alpha=0.25, axis="y")
    ax[2].legend(fontsize=7)

    # D. landscape
    th = np.array(land["theta"])
    l = np.array(land["loss_mix"])
    dp = np.array(land["dpose_norm"])
    ax[3].semilogy(th, np.maximum(l, 1e-16), "-", color="C0", lw=1.2,
                   label="Laplace loss (mix)")
    ax[3].set_xscale("symlog", linthresh=1e-3)
    ax[3].axhline(targets["mix"]["floor_laplace"], color="C0", ls="--", lw=0.7)
    ax[3].axhline(1e-4, color="k", ls=":", lw=0.9)
    ax[3].set_xlabel("theta (symlog, linthresh=1e-3)")
    ax[3].set_ylabel("loss", color="C0")
    ax[3].set_title("D. mix-target landscape: flat far away, steep walls + fold\n"
                    "at the crossing (blue = loss, red = |dpose|)", fontsize=9)
    axd = ax[3].twinx()
    mask = np.abs(th) > 0
    axd.set_yscale("log")
    axd.plot(th[mask], np.maximum(dp[mask], 1e-6), ".", ms=2, color="C3", alpha=0.6)
    axd.set_ylabel("|dpose|", color="C3")
    axd.tick_params(axis="y", colors="C3")
    for sgn in (-1.0, 1.0):
        ax[3].axvline(sgn * abs(land["fold_theta"]), color="C3", ls=":", lw=0.8,
                      alpha=0.7)
    ax[3].text(1.6e-3, 2e-2, "fold", color="C3", fontsize=7)
    ax[3].grid(alpha=0.25)

    # E. clip control
    xpos = np.arange(len(ctrl))
    for i, row in enumerate(ctrl):
        vals = np.array([r["loss_final"] for r in row["seeds"]])
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax[4].scatter(np.full(len(vals), i) + jitter, np.maximum(vals, 1e-16),
                      s=14, color="C0", alpha=0.7)
        ax[4].plot([i - 0.15, i + 0.15], [row["loss_median"]] * 2, "k-", lw=1.5)
    ax[4].set_yscale("log")
    ax[4].set_xticks(xpos)
    ax[4].set_xticklabels([f"clip={row['clip']:g}" for row in ctrl])
    ax[4].axhline(targets["mix"]["floor_laplace"], color="C0", ls="--", lw=0.8)
    ax[4].axhline(1e-4, color="k", ls=":", lw=0.9)
    ax[4].set_ylabel("final loss (Laplace, mix)")
    ax[4].set_title("E. step-size dilemma: no single clip does both\n"
                    "travel (step too small) and resolve the crossing (step too big)",
                    fontsize=9)
    ax[4].grid(alpha=0.25, axis="y")

    # F. throughput
    methods = ("hard", "laplace", "mc")
    for tag, hatch in (("theta0", None), ("theta01", "//")):
        vals = [thr[tag]["methods"][m]["median_ms"] for m in methods]
        enum = thr[tag]["enum_median_ms"]
        xpos = np.arange(len(methods)) + (-0.19 if tag == "theta0" else 0.19)
        ax[5].bar(xpos, vals, 0.34, color=[colors[m] for m in methods],
                  hatch=hatch, alpha=0.9 if tag == "theta0" else 0.6)
        ax[5].bar(xpos, [enum] * len(methods), 0.34, color="0.35", alpha=0.5)
    ax[5].set_xticks(np.arange(len(methods)))
    ax[5].set_xticklabels(methods)
    ax[5].set_ylabel("forward+backward (ms)")
    ax[5].set_title("F. throughput: dark = mode enumeration share\n"
                    "solid = theta=0, hatched = theta=0.1", fontsize=9)
    ax[5].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- markdown

def fmt(x, spec=".3e"):
    if x is None:
        return "—"
    if isinstance(x, float) and not np.isfinite(x):
        return "N/A"
    return format(x, spec)


def write_md(path, meta, targets, summary, var, thr, land, ctrl, wall):
    L = []
    A = L.append
    S = {(r["target"], r["method"]): r for r in summary}
    mix_lap, mix_hard = S[("mix", "laplace")], S[("mix", "hard")]
    star_lap, star_hard = S[("star", "laplace")], S[("star", "hard")]
    fmt_pct = lambda x: "—" if x is None else f"{100*x:.0f}%"
    A("# P1 end-to-end toy training through the Laplace-QPEP layer")
    A("")
    A("> **一句话结论**：在合成双峰实例、$\\tau=0.01$、150 步、"
      "$\\mathrm{lr}=0.01$、clip=1.0 下，"
      "**两分支平均位姿目标在有限 $\\tau$ 下不可表示**"
      f"（Laplace 下限 {fmt(mix_lap['floor_laplace'])}，hard 结构性下限 "
      f"{fmt(targets['mix']['floor_hard'])}；"
      f"`mix` 成功率 Laplace {fmt_pct(mix_lap['success_rate'])} / hard "
      f"{fmt_pct(mix_hard['success_rate'])}）。"
      f"$\\theta^\\ast={THETA_STAR:.2f}$ 的位姿目标从正确一侧可达"
      f"（Laplace 正侧 {fmt_pct(star_lap['success_by_side']['rate_pos'])}、"
      f"负侧 {fmt_pct(star_lap['success_by_side']['rate_neg'])}；hard 正侧 "
      f"{fmt_pct(star_hard['success_by_side']['rate_pos'])}、负侧 "
      f"{fmt_pct(star_hard['success_by_side']['rate_neg'])}）——"
      "错误一侧被分支交叉挡住：位姿对 $\\theta$ 近乎不敏感，梯度 $\\sim10^{-2}$，"
      "150 步内无法穿越。梯度方差：Laplace 解析梯度跨种子 std $\\approx0$，"
      f"MC 相对 std {fmt(np.mean([v['mc']['rel_std'] for v in var]))}，"
      "hard 在精确 tie 处导数不存在、两支相对相差 "
      f"{fmt(var[0]['hard_branch']['rel_ambiguity'], '.2f')}。")
    A("")
    A(f"Generated by `code/run_e2e_fit.py` on {meta['host']} "
      f"(git `{meta['git_head']}`, numpy {meta['numpy']}), CPU only, "
      f"single-thread BLAS.  Total wall {wall:.1f} s.")
    A("")
    A("Evidence level: `独立复现` (self-built synthetic instance, no paper claim).  "
      "Maps to `protocol.md` E3 (multi-modal Laplace vs hard argmin: convergence, "
      "gradient variance, failure) and §4.3/§4.4 (gradient variance across "
      "root-enumeration perturbations, loss spikes at the branch crossing, "
      "root-enumeration share of runtime).")
    A("")
    A("## 0. Setup")
    A("")
    A(f"- Instance: `instance.py` seed {meta['instance_seed']}, cold mode "
      f"enumeration with `N_STARTS={meta['n_starts']}` (same as `main_table.md`).")
    A(f"- Layer: $\\tau={TAU}$, mode-level MC with n={N_MC}; Laplace uses the "
      "closed-form `core.dlogdetH_dtheta` path (fixed 2026-11-27).")
    A(f"- Fit: {meta['steps']} gradient steps, lr={meta['lr']}, global-norm "
      f"clip={meta['clip']}, {meta['n_fit_seeds']} random initial "
      "$\\theta\\in[-0.2,0.2]$; $\\theta$ hard-clipped to $[-0.5,0.5]$.  "
      "One extra forward at the final $\\theta$ defines `loss_final`.")
    A("- CPU only; no GPU.  Throughput pins OpenMP/BLAS to one thread.")
    A("")
    A("Targets:")
    A("")
    A("| target | definition | hard floor | Laplace floor | MC floor |")
    A("|-|-|-|-|-|")
    A(f"| `mix` | average of the two global-branch poses at theta=0 | "
      f"{fmt(targets['mix']['floor_hard'])} | {fmt(targets['mix']['floor_laplace'])} "
      f"| {fmt(targets['mix']['floor_mc'])} |")
    A(f"| `star` | pose of the global branch at theta*={THETA_STAR:.2f} | 0 | "
      f"{fmt(targets['star']['floor_laplace'])} | 0 (representable) |")
    A("")
    A("The `mix` floor is not an optimisation artefact: at $\\theta=0$ the "
      f"Laplace weights are {np.round(targets['mix']['w_at_best'], 3).tolist()} "
      "(two global minima + the two excited minima, $J$ gap $0.034=3.4\\tau$), "
      f"so the layer output differs from the pure branch average by "
      f"{fmt(np.sqrt(targets['mix']['floor_laplace']))} in pose norm.  "
      f"MC adds a $\\mathrm{{Var}}/n$ floor of {fmt(targets['mix']['mc_noise_floor'])}.")
    A("")
    A("## 1. Fit through the layer")
    A("")
    A("| target | method | final loss median | mean | min | success (<1e-4) | "
      "steps-to-1e-4 (median [IQR]) | reached ever | crossing seeds | max |Δloss| | "
      "max |Δloss| at crossing |")
    A("|-|-|-|-|-|-|-|-|-|-|-|")
    for r in summary:
        steps = ("—" if r["steps_to_1e4_median"] is None
                 else f"{r['steps_to_1e4_median']:.0f} "
                      f"[{r['steps_to_1e4_iqr'][0]:.0f},{r['steps_to_1e4_iqr'][1]:.0f}]")
        A(f"| `{r['target']}` | {r['method']} | {fmt(r['loss_final_median'])} | "
          f"{fmt(r['loss_final_mean'])} | {fmt(r['loss_final_min'])} | "
          f"{r['success_rate']*100:.0f}% | {steps} | "
          f"{r['reached_any_rate']*100:.0f}% | {r['cross_rate']*100:.0f}% | "
          f"{fmt(r['max_abs_dloss_median'], '.2f')} | "
          f"{fmt(r['max_abs_dloss_at_cross_median'], '.2f')} |")
    A("")
    A("Success by initial side (positive vs negative theta):")
    A("")
    A("| target | method | success, theta0>0 | success, theta0<0 |")
    A("|-|-|-|-|")
    for r in summary:
        A(f"| `{r['target']}` | {r['method']} | "
          f"{fmt(r['success_by_side']['rate_pos'], '.0%')} | "
          f"{fmt(r['success_by_side']['rate_neg'], '.0%')} |")
    A("")
    A("Reading:")
    A("")
    A("- `mix`: hard cannot represent a two-branch average at all (floor "
      f"{fmt(targets['mix']['floor_hard'])}, exactly $\\tfrac14\\|p_A-p_B\\|^2$); "
      f"its median final loss is {fmt(mix_hard['loss_final_median'])}.  "
      "Laplace is the only method whose floor is near the target, but the floor "
      f"itself is {fmt(targets['mix']['floor_laplace'])}, so `loss<1e-4` is "
      "**unreachable by construction**; the relevant metric is the gap to the "
      f"floor (Laplace median final {fmt(mix_lap['loss_final_median'])}).  "
      "With clip=1.0 the Laplace step near the crossing is "
      "$\\mathrm{lr}\\cdot\\mathrm{clip}=10^{-2}$, ~7x the crossing width "
      "$\\tau/\\mathrm{slope}\\approx1.4\\times10^{-3}$, so trajectories "
      "oscillate across the tie (large $|\\Delta\\mathrm{loss}|$ at crossing) "
      "instead of settling at the floor.")
    A("- `star`: from $\\theta_0>0$ the target branch is the global one; from "
      "$\\theta_0<0$ the *other* global branch is deeper, the layer pose is "
      "almost insensitive to $\\theta$ ($|\\mathrm{dpose}|\\approx0.2$ far from "
      "the tie), and the gradient (~1e-2) never crosses the tie inside "
      f"{meta['steps']} steps.  This is the branch-crossing failure in "
      "optimisation form: the landscape is flat on the wrong side and "
      "discontinuous at the tie.  On the correct side success is limited by "
      "the same flatness: the geometric rate is "
      "$1-2\\,\\mathrm{lr}\\,\\|\\mathrm{dpose}\\|^2\\approx0.999$ at "
      "$|\\theta|\\approx0.2$, so far initialisations need far more than "
      f"{meta['steps']} steps (median steps-to-1e-4 among successes: "
      f"{fmt(star_lap['steps_to_1e4_median'], '.0f')}).")
    A("- **The three methods coincide away from the crossing.**  For "
      "$|\\theta|\\gtrsim10^{-2}$ the Laplace weights collapse to one mode and "
      "MC samples only that mode, so hard / Laplace / MC produce identical "
      "poses *and* gradients; the `star` target trajectories are bit-identical "
      "across methods.  Only the `mix` target (at $\\theta=0$) separates them.")
    A("")
    A("## 2. Gradient variance at fixed theta")
    A("")
    A("dpose/dtheta in R^12; sample std over 20 seeds (ddof=1), Frobenius norm.  "
      "hard/Laplace vary the mode-enumeration seed (their only stochastic "
      "ingredient); MC varies the sampling seed at a fixed cold mode set.")
    A("")
    A("| theta | method | ||mean dpose|| | std (frob) | relative std | note |")
    A("|-|-|-|-|-|-|")
    for v in var:
        A(f"| {v['theta']:g} | hard | {fmt(v['hard']['mean_norm'])} | "
          f"{fmt(v['hard']['std_frob'])} | {fmt(v['hard']['rel_std'])} | "
          + ("**N/A — exact tie, derivative undefined**; the two one-sided "
             f"branch values differ by {fmt(v['hard_branch']['delta_norm'])} "
             f"(rel {fmt(v['hard_branch']['rel_ambiguity'], '.2f')}); across 20 "
             f"enumeration seeds the argmin flips between them "
             f"(n_distinct={v['hard']['n_distinct']}, "
             f"std={fmt(v['hard']['std_frob'])})"
             if v['theta'] == 0.0 else "unique argmin, ordinary derivative") + " |")
        A(f"| {v['theta']:g} | laplace | {fmt(v['laplace']['mean_norm'])} | "
          f"{fmt(v['laplace']['std_frob'])} | {fmt(v['laplace']['rel_std'])} | "
          "deterministic analytic gradient; std is numerical jitter |")
        A(f"| {v['theta']:g} | mc | {fmt(v['mc']['mean_norm'])} | "
          f"{fmt(v['mc']['std_frob'])} | {fmt(v['mc']['rel_std'])} | "
          "score-function sampling noise; "
          f"bias vs Laplace {fmt(v.get('mc_bias_rel_vs_laplace'), '.2e')} |")
    A("")
    A("At $\\theta=0$ the hard argmin sits on an exact two-fold tie "
      f"($|J_A-J_B|={fmt(var[0]['hard_branch']['J_gap'], '.1e')}$).  The two "
      "one-sided derivatives differ by "
      f"{fmt(var[0]['hard_branch']['rel_ambiguity'], '.2f')} relative — the "
      "derivative **does not exist**.  A straight-through implementation reports "
      "one of the two values, and *which* one flips with the mode-enumeration "
      "jitter (the argmin is decided by a ~1e-16 numerical $J$ difference), so "
      "the 20-seed std is not zero even though nothing physical is random.  "
      "Laplace's analytic gradient is well defined and its 20-seed std is "
      "machine-level.  MC is the only method with genuine stochastic variance, "
      "and it is large in absolute terms because the score carries "
      "$\\dot J/\\tau\\sim10^{2}$.")
    A("")
    A("## 3. Throughput")
    A("")
    A("| method | fwd+bwd @ theta=0 (ms) | fwd+bwd @ theta=0.1 (ms) | "
      "mode-enum share @ theta=0 |")
    A("|-|-|-|-|")
    for m in ("hard", "laplace", "mc"):
        a = thr["theta0"]["methods"][m]
        b = thr["theta01"]["methods"][m]
        A(f"| {m} | {a['median_ms']:.2f} [{a['iqr_ms'][0]:.2f},{a['iqr_ms'][1]:.2f}] | "
          f"{b['median_ms']:.2f} [{b['iqr_ms'][0]:.2f},{b['iqr_ms'][1]:.2f}] | "
          f"{100*a['enum_share']:.0f}% |")
    A("")
    A(f"Cold `find_modes({N_STARTS})` alone: "
      f"{thr['theta0']['enum_median_ms']:.2f} ms @ theta=0, "
      f"{thr['theta01']['enum_median_ms']:.2f} ms @ theta=0.1.  "
      "**Mode enumeration dominates** (it is the Newton + classify batch); the "
      "per-mode IFT/logdet/pose work adds only a few ms.  At the crossing the "
      "enumeration is slower because the Newton needs more iterations near the "
      "fold.")
    A("")
    A("Loop batch test (Laplace, a different theta per sample):")
    A("")
    A("| B | total (ms) | per-sample (ms) |")
    A("|-|-|-|")
    for b in thr["batch"]:
        A(f"| {b['B']} | {b['total_ms_median']:.2f} | {b['per_sample_ms']:.2f} |")
    A("")
    A("No batching speed-up: `find_modes` is per-sample (batched over *starts of "
      "one instance*, not over instances).  A vectorised multi-instance Newton "
      "is the obvious next step but is not implemented here.")
    A("")
    A("## 4. Step-size / conditioning control (Laplace on `mix`)")
    A("")
    A("| clip | step = lr*clip | final loss median | min | seeds <1e-4 |")
    A("|-|-|-|-|-|")
    for row in ctrl:
        A(f"| {row['clip']:g} | {LR*row['clip']:.1e} | {fmt(row['loss_median'])} | "
          f"{fmt(row['loss_min'])} | {row['n_below_floor']*100:.0f}% |")
    A("")
    A("Landscape: $|\\mathrm{dpose}|$ spans "
      f"{land['dpose_min']:.2f}..{land['dpose_max']:.1f} "
      f"(max at $|\\theta|={abs(land['fold_theta']):.2e}$, the fold reported in "
      "`README.md`), i.e. a curvature range of ~"
      f"{(land['dpose_max']/land['dpose_min'])**2:.1e} between the flat region and "
      "the crossing.  A single fixed step cannot both traverse the flat region "
      "and resolve the fold; the clip sweep shows exactly this trade-off "
      "(clip=1 oscillates at the crossing; clip=0.02 gets stuck far away unless "
      "the init is already near 0).")
    A("")
    A("## 5. Caveats")
    A("")
    A("- The `mix` target is **not representable** at $\\tau=0.01$; its 1e-4 "
      "success criterion is unattainable for every method.  Do not read the "
      "0% success as an optimisation bug alone — the Laplace floor is 1.1e-2.")
    A("- The `star` target is self-consistency (fit back to a pose the layer "
      "itself produces at $\\theta^\\ast$), not a real-image training signal.")
    A("- Scalar $\\theta$ only: the synthetic instance has a one-parameter "
      "$M(\\theta)$ family, so a 2D input is not available without changing the "
      "instance.")
    A("- Hard's `dpose` is the selected branch's derivative (straight-through); "
      "it is not the derivative of the argmin map at the tie.")
    A("- MC is mode-level sampling, not a continuous SE(3) MCMC; it measures "
      "discrete mode-weight sampling noise only.")
    A("- Timings are single-thread CPU on a shared login node; absolute values "
      "depend on load, use the ratios and the enum share.")
    A("- `N_STARTS=128` cold enumeration may still miss a fragile mode exactly at "
      "the fold; the fold bin is a known limitation of the real-root proxy "
      "(see `results/fold_6d.md`).")
    A("")
    A("## 6. Reproduction")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/code")
    A("/home/syk/miniconda3/envs/WM/bin/python run_e2e_fit.py")
    A("```")
    A("")
    A("Outputs: `results/e2e_fit.json`, `results/e2e_fit.md`, "
      "`results/e2e_fit.png`.")
    Path(path).write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=N_FIT_SEEDS)
    ap.add_argument("--control-seeds", type=int, default=CONTROL_SEEDS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run, writes results/_smoke_e2e.*")
    args = ap.parse_args()
    steps = args.steps
    n_seeds = args.seeds
    n_ctrl = args.control_seeds
    prefix = "_smoke_e2e" if args.smoke else "e2e_fit"

    t_start = time.perf_counter()
    print("building instance ...", flush=True)
    t0 = time.perf_counter()
    I = inst_mod.Instance(seed=INSTANCE_SEED)
    targets = build_targets(I)
    print(f"  [phase {time.perf_counter()-t0:.1f}s] "
          f"mix floor: hard {targets['mix']['floor_hard']:.4f} "
          f"laplace {targets['mix']['floor_laplace']:.4e} "
          f"mc {targets['mix']['floor_mc']:.4e}", flush=True)
    print(f"  star floor: laplace {targets['star']['floor_laplace']:.4e}", flush=True)

    print("fit runs ...", flush=True)
    t0 = time.perf_counter()
    fit_rows = fit_records(I, targets, steps, n_seeds, LR, CLIP)
    summary = summarize_fit(fit_rows, targets)
    print(f"  [phase {time.perf_counter()-t0:.1f}s]", flush=True)

    print("gradient variance ...", flush=True)
    t0 = time.perf_counter()
    var = []
    for th in VAR_THETAS:
        v = gradient_variance(I, th, N_VAR_SEEDS,
                              dpose_ref=R.laplace(I, th)["dpose"])
        var.append(v)
        print(f"  theta {th:g}: hard std {v['hard']['std_frob']:.2e} "
              f"lap std {v['laplace']['std_frob']:.2e} "
              f"mc std {v['mc']['std_frob']:.2e} "
              f"(rel {v['mc']['rel_std']:.2e})", flush=True)
    print(f"  [phase {time.perf_counter()-t0:.1f}s]", flush=True)

    print("throughput ...", flush=True)
    t0 = time.perf_counter()
    thr = throughput(I, N_THROUGHPUT_REPS if not args.smoke else 3)
    print(f"  [phase {time.perf_counter()-t0:.1f}s]", flush=True)

    print("landscape ...", flush=True)
    t0 = time.perf_counter()
    land = landscape(I, targets)
    print(f"  [phase {time.perf_counter()-t0:.1f}s]", flush=True)

    print("clip control ...", flush=True)
    t0 = time.perf_counter()
    ctrl = clip_control(I, targets, steps, n_ctrl)
    print(f"  [phase {time.perf_counter()-t0:.1f}s]", flush=True)

    meta = dict(
        instance_seed=INSTANCE_SEED, n_starts=N_STARTS, tau=TAU, n_mc=N_MC,
        theta_star=THETA_STAR, lr=LR, clip=CLIP, steps=steps,
        n_fit_seeds=n_seeds, n_var_seeds=N_VAR_SEEDS,
        control_clips=list(CONTROL_CLIPS), control_seeds=n_ctrl,
        python=sys.version.split()[0], numpy=np.__version__,
        host=platform.node(),
        git_head=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=str(HERE)).stdout.strip(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        blas_threads=os.environ.get("OMP_NUM_THREADS"),
    )

    wall = time.perf_counter() - t_start
    payload = dict(
        meta=meta, wall_seconds=float(wall),
        targets={k: {kk: vv for kk, vv in v.items() if kk != "pose"}
                 for k, v in targets.items()},
        fit=fit_rows, fit_summary=summary, variance=var,
        throughput=thr, landscape=land, clip_control=ctrl,
    )
    with open(OUT / f"{prefix}.json", "w") as f:
        json.dump(json_safe(payload), f, indent=1)

    make_plot(OUT / f"{prefix}.png", fit_rows, summary, targets, var, thr,
              land, ctrl)
    write_md(OUT / f"{prefix}.md", meta, targets, summary, var, thr, land, ctrl,
             wall)
    print(f"wrote {OUT / (prefix + '.json')}, {OUT / (prefix + '.md')}, "
          f"{OUT / (prefix + '.png')}  (wall {wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
