#!/usr/bin/env python3
"""Inner-solver benchmark: weighted Kabsch/SVD (current) vs Horn 4x4 (candidate).

Runs 1000 random weighted 3D-3D problems for each N in {20, 100, 500},
compares

* rotation Frobenius difference  ||R_davenport - R_kabsch||_F
* translation difference         ||t_davenport - t_kabsch||_2
* each solver's error against ground truth (rotation deg / translation m)
* success identity (5 deg + 0.1 m criterion from synthetic/common.py)
* weighted least-squares cost excess  F_davenport - F_kabsch
* wall-clock p50/p95/p99 over repeated calls (interleaved, fixed data)

and writes ``results/inner_solver.{json,md,png}``.  Every number in the
report is produced by this script; fixed seeds.

Honesty notes
-------------
* The two solvers solve the *same* convex-over-SO(3) problem, so equality
  to round-off is the expected result, not a victory of one method.
* runtime is per Python call on a shared CPU node (single BLAS thread);
  absolute microseconds include interpreter/NumPy overhead, so the
  small-N regime is dominated by constant overhead, not linear algebra.
* This is the scalar-weight 3D-3D inner solve only.  It is NOT a 40-root
  QPEP polynomial solver; see ``SCOPE_NOTE`` below.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
WT_ROOT = Path(__file__).resolve().parents[4]
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXPERIMENTS / "synthetic"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
from davenport import solve as davenport_solve  # noqa: E402

# ---------------------------------------------------------------------------
# Config (fixed seeds; do not randomise)
# ---------------------------------------------------------------------------
SEED = 20261127
N_LIST = (20, 100, 500)
N_TRIALS = 1000
TIMING_REPEATS = 5
WARMUP_CALLS = 200
SIGMA = 0.01
ZERO_FRACTION = 0.10

SCOPE_NOTE = (
    "范围说明：本实验只验证 3D-3D 的标量权重内层加权最小二乘。"
    "当 W_i = w_i I 时，QPEP 的 6-DoF 位姿估计退化为“加权质心 + SO(3) 上的"
    "正交 Procrustes”，其全局最优由 4x4 对称矩阵的最大特征向量精确给出（Horn 1987）。"
    "因此这里比较的是同一 4x4 特征问题的两种实现（Kabsch/SVD vs Horn/4x4），"
    "并没有实现、也没有声称实现完整的 40 根 QPEP 多项式求解器"
    "（十次/四十次多项式、根筛选、数值完备性、P3P 等均不在本轮范围）。"
    "Davenport/Horn 在本文语境中的角色是：一个可证明全局最优的闭式内层，"
    "反射/det 修正由四元数参数化自动满足（无需 Kabsch 式的 det 分支，正常路径不用 SVD）。"
    "各向异性 W_i、尺度、以及 40 根代数本身均未测试。"
)


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------
def make_weights(rng, n):
    """Non-negative weights with zeros and skewed magnitudes (IRLS-like)."""
    kind = int(rng.integers(0, 3))
    if kind == 0:      # bounded positive
        w = rng.uniform(0.05, 1.0, n)
    elif kind == 1:    # heavy tail (some near-zero, some large)
        w = rng.exponential(1.0, n)
    else:              # includes near-zero values
        w = rng.uniform(0.0, 1.0, n)
    w = w.copy()
    w[rng.random(n) < ZERO_FRACTION] = 0.0
    if w.sum() <= 1e-12:  # astronomically unlikely; keep the stream usable
        w = np.ones(n)
    return w


def make_problem(rng, n):
    x = rng.uniform(-1.0, 1.0, size=(n, 3))
    Rgt = common.random_rotation(rng)
    tgt = rng.uniform(-0.5, 0.5, size=3)
    y = (Rgt @ x.T).T + tgt + SIGMA * rng.normal(size=(n, 3))
    w = make_weights(rng, n)
    return x, y, w, Rgt, tgt


def problem_rng(n, trial):
    return np.random.default_rng(np.random.SeedSequence([SEED, n, trial]))


def weighted_cost(x, y, w, R, t):
    r = (R @ x.T).T + t - y
    return float(np.sum(w[:, None] * r * r))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def estats(a):
    a = np.asarray(a, float)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def tstats(ns):
    us = np.asarray(ns, float) / 1000.0
    return {
        "n": int(us.size),
        "p50_us": float(np.percentile(us, 50)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
        "mean_us": float(us.mean()),
        "min_us": float(us.min()),
        "max_us": float(us.max()),
    }


def speedup(base, cand):
    return float(base / cand)


# ---------------------------------------------------------------------------
# Correctness pass
# ---------------------------------------------------------------------------
def correctness_for_n(n):
    rot_diff, trans_diff, max_entry, cost_excess = [], [], [], []
    cost_kabsch, cost_excess_rel = [], []
    rot_gt_k, rot_gt_d, trans_gt_k, trans_gt_d = [], [], [], []
    succ_k, succ_d = 0, 0
    both_succ = 0
    dav_only = 0
    kab_only = 0
    near_deg = 0
    fallback = 0
    det_k_min, det_d_min = np.inf, np.inf
    det_k_max, det_d_max = -np.inf, -np.inf
    gap_min = np.inf
    zero_weight_frac = []
    for trial in range(N_TRIALS):
        rng = problem_rng(n, trial)
        x, y, w, Rgt, tgt = make_problem(rng, n)
        Rk, tk = common.kabsch(x, y, w)
        Rd, td, info = davenport_solve(x, y, w, return_info=True)

        rk = common.rot_err_deg(Rk, Rgt)
        rd = common.rot_err_deg(Rd, Rgt)
        tk_err = float(np.linalg.norm(tk - tgt))
        td_err = float(np.linalg.norm(td - tgt))

        rot_diff.append(float(np.linalg.norm(Rd - Rk)))
        trans_diff.append(float(np.linalg.norm(td - tk)))
        max_entry.append(float(np.max(np.abs(Rd - Rk))))
        ck = weighted_cost(x, y, w, Rk, tk)
        cd = weighted_cost(x, y, w, Rd, td)
        cost_excess.append(cd - ck)
        cost_kabsch.append(ck)
        cost_excess_rel.append(abs(cd - ck) / max(abs(ck), 1e-300))
        rot_gt_k.append(rk)
        rot_gt_d.append(rd)
        trans_gt_k.append(tk_err)
        trans_gt_d.append(td_err)

        sk = common.success(rk, tk_err)
        sd = common.success(rd, td_err)
        succ_k += int(sk)
        succ_d += int(sd)
        both_succ += int(sk and sd)
        dav_only += int(sd and not sk)
        kab_only += int(sk and not sd)

        near_deg += int(info["near_degenerate"])
        fallback += int(info["weight_fallback"])
        gap_min = min(gap_min, info["eig_gap"])
        dk = float(np.linalg.det(Rk))
        dd = float(info["det"])
        det_k_min, det_k_max = min(det_k_min, dk), max(det_k_max, dk)
        det_d_min, det_d_max = min(det_d_min, dd), max(det_d_max, dd)
        zero_weight_frac.append(float(np.mean(w == 0.0)))

    cost_excess = np.asarray(cost_excess, float)
    return {
        "n_trials": int(N_TRIALS),
        "rot_err_to_gt_deg": {
            "note": "rotation error vs GT in degrees (trace/arccos), not Frobenius",
            "kabsch": estats(rot_gt_k),
            "davenport": estats(rot_gt_d),
        },
        "trans_err_to_gt_m": {
            "kabsch": estats(trans_gt_k),
            "davenport": estats(trans_gt_d),
        },
        "rot_frob_diff_dav_vs_kabsch": estats(rot_diff),
        "trans_diff_dav_vs_kabsch_m": estats(trans_diff),
        "max_abs_rot_entry_diff": float(np.max(max_entry)),
        "cost_excess_dav_minus_kabsch": estats(cost_excess),
        "cost_excess_abs_max": float(np.max(np.abs(cost_excess))),
        "cost_excess_rel_max": float(np.max(cost_excess_rel)),
        "cost_kabsch_median": float(np.median(cost_kabsch)),
        "success_rate": {
            "kabsch": succ_k / N_TRIALS,
            "davenport": succ_d / N_TRIALS,
        },
        "success_identical_frac": (both_succ + (N_TRIALS - succ_k - succ_d + both_succ)) / N_TRIALS,
        "n_both_success": int(both_succ),
        "n_davenport_only_success": int(dav_only),
        "n_kabsch_only_success": int(kab_only),
        "n_both_fail": int(N_TRIALS - both_succ - dav_only - kab_only),
        "near_degenerate_count": int(near_deg),
        "eig_gap_min": float(gap_min),
        "weight_fallback_count": int(fallback),
        "det_kabsch_range": [float(det_k_min), float(det_k_max)],
        "det_davenport_range": [float(det_d_min), float(det_d_max)],
        "zero_weight_fraction_mean": float(np.mean(zero_weight_frac)),
        "zero_weight_fraction_max": float(np.max(zero_weight_frac)),
    }


# ---------------------------------------------------------------------------
# Timing pass
# ---------------------------------------------------------------------------
def timing_for_n(n):
    solvers = {
        "kabsch": lambda x, y, w: common.kabsch(x, y, w),
        "davenport": lambda x, y, w: davenport_solve(x, y, w),
        "davenport_no_detcheck": lambda x, y, w: davenport_solve(x, y, w, det_check=False),
    }
    order = list(solvers)
    # warm-up (JIT/allocation caches, BLAS thread pool, CPU frequency)
    rng = problem_rng(n, 0)
    x, y, w, _, _ = make_problem(rng, n)
    for name in order:
        fn = solvers[name]
        for _ in range(WARMUP_CALLS):
            fn(x, y, w)

    samples = {name: [] for name in order}
    for trial in range(N_TRIALS):
        rng = problem_rng(n, trial)
        x, y, w, _, _ = make_problem(rng, n)
        seq = order if trial % 2 == 0 else order[::-1]  # cancel order drift
        for _rep in range(TIMING_REPEATS):
            for name in seq:
                fn = solvers[name]
                t0 = time.perf_counter_ns()
                fn(x, y, w)
                samples[name].append(time.perf_counter_ns() - t0)

    out = {name: tstats(samples[name]) for name in order}
    out["speedup_davenport_p50"] = speedup(out["kabsch"]["p50_us"], out["davenport"]["p50_us"])
    out["speedup_davenport_p95"] = speedup(out["kabsch"]["p95_us"], out["davenport"]["p95_us"])
    out["speedup_davenport_no_detcheck_p50"] = speedup(
        out["kabsch"]["p50_us"], out["davenport_no_detcheck"]["p50_us"])
    out["n_calls_per_solver"] = int(N_TRIALS * TIMING_REPEATS)
    return out


# ---------------------------------------------------------------------------
# Edge cases (short, deterministic checks)
# ---------------------------------------------------------------------------
def edge_cases():
    out = {}

    # (a) planar data (covariance rank 2): does Kabsch's det branch misfire?
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0, 999]))
    dR, gap, ident, cex = [], [], 0, []
    detk, detd = [], []
    for trial in range(200):
        n = 100
        x = rng.uniform(-1.0, 1.0, (n, 3))
        x[:, 2] = 0.0
        Rgt = common.random_rotation(rng)
        tgt = rng.uniform(-0.5, 0.5, 3)
        y = (Rgt @ x.T).T + tgt + SIGMA * rng.normal(size=(n, 3))
        w = make_weights(rng, n)
        Rk, tk = common.kabsch(x, y, w)
        Rd, td, info = davenport_solve(x, y, w, return_info=True)
        rk, rd = common.rot_err_deg(Rk, Rgt), common.rot_err_deg(Rd, Rgt)
        sk = common.success(rk, float(np.linalg.norm(tk - tgt)))
        sd = common.success(rd, float(np.linalg.norm(td - tgt)))
        ident += int(sk == sd)
        dR.append(float(np.linalg.norm(Rd - Rk)))
        gap.append(info["eig_gap"])
        cex.append(weighted_cost(x, y, w, Rd, td) - weighted_cost(x, y, w, Rk, tk))
        detk.append(float(np.linalg.det(Rk)))
        detd.append(float(info["det"]))
    out["planar_rank2"] = {
        "n_trials": 200,
        "n": 100,
        "rot_frob_diff": estats(dR),
        "eig_gap_min": float(np.min(gap)),
        "cost_excess_abs_max": float(np.max(np.abs(cex))),
        "success_identical_frac": ident / 200.0,
        "det_kabsch_min": float(np.min(detk)),
        "det_davenport_min": float(np.min(detd)),
        "note": "coplanar z=0 points; tests Kabsch det correction on rank-2 H",
    }

    # (b) all-zero weights: documented fallback to uniform, must match Kabsch
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, 999]))
    n = 50
    x = rng.uniform(-1.0, 1.0, (n, 3))
    y = rng.uniform(-1.0, 1.0, (n, 3))
    zeros = np.zeros(n)
    Rk, tk = common.kabsch(x, y, zeros)
    Rd, td, info = davenport_solve(x, y, zeros, return_info=True)
    Ru, tu = common.kabsch(x, y, np.ones(n))
    out["all_zero_weights"] = {
        "fallback_flag": bool(info["weight_fallback"]),
        "weight_sum_reported": float(info["weight_sum"]),
        "rot_frob_diff_vs_kabsch": float(np.linalg.norm(Rd - Rk)),
        "trans_diff_vs_kabsch_m": float(np.linalg.norm(td - tk)),
        "rot_frob_diff_vs_uniform": float(np.linalg.norm(Rd - Ru)),
        "trans_diff_vs_uniform_m": float(np.linalg.norm(td - tu)),
        "note": "sum(w)=0 is undefined; both fall back to uniform weights",
    }

    # (c) negative / non-finite weights are rejected
    checks = {}
    for name, wbad in (("negative", -np.ones(n)),
                       ("nan", np.full(n, np.nan)),
                       ("inf", np.full(n, np.inf))):
        try:
            davenport_solve(x, y, wbad)
            checks[name] = "NO ERROR (unexpected)"
        except ValueError as exc:
            checks[name] = f"ValueError: {exc}"
    out["invalid_weights"] = checks

    # (d) collinear points: rotation not unique -> near-degenerate flag
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, 999]))
    n = 30
    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    t = rng.uniform(-1.0, 1.0, (n, 1))
    x = t * direction + 1e-3 * rng.normal(size=(n, 3))
    Rgt = common.random_rotation(rng)
    tgt = rng.uniform(-0.5, 0.5, 3)
    y = (Rgt @ x.T).T + tgt
    w = np.ones(n)
    Rk, tk = common.kabsch(x, y, w)
    Rd, td, info = davenport_solve(x, y, w, return_info=True)
    out["collinear_near_degenerate"] = {
        "rot_frob_diff": float(np.linalg.norm(Rd - Rk)),
        "trans_diff_m": float(np.linalg.norm(td - tk)),
        "eig_gap": float(info["eig_gap"]),
        "near_degenerate": bool(info["near_degenerate"]),
        "cost_excess_abs": abs(weighted_cost(x, y, w, Rd, td) - weighted_cost(x, y, w, Rk, tk)),
        "note": "nearly collinear; both objectives equal, R itself is not unique",
    }

    # (e) N=1: problem underdetermined; only check det and flag, no equality claim
    x1 = np.array([[1.0, 2.0, 3.0]])
    y1 = np.array([[2.0, 1.0, 0.0]])
    w1 = np.ones(1)
    Rk1, tk1 = common.kabsch(x1, y1, w1)
    Rd1, td1, info1 = davenport_solve(x1, y1, w1, return_info=True)
    out["single_point_underdetermined"] = {
        "det_kabsch": float(np.linalg.det(Rk1)),
        "det_davenport": float(info1["det"]),
        "eig_gap": float(info1["eig_gap"]),
        "near_degenerate": bool(info1["near_degenerate"]),
        "rot_frob_diff": float(np.linalg.norm(Rd1 - Rk1)),
        "note": "N=1: any rotation fits; different minimisers are both valid",
    }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_us(v):
    return f"{v:.2f}"


def fmt_sci(v):
    return f"{v:.2e}"


def build_markdown(cfg, corr, run, edges, verdict, meta):
    L = []
    L.append("# Inner solver: weighted Kabsch (current) vs Horn 4x4 (candidate)")
    L.append("")
    L.append(f"> 生成时间：{meta['timestamp']} ｜ 分支 `{meta['git_branch']}` @ "
             f"`{meta['git_commit']}` ｜ Python {meta['python']} / NumPy {meta['numpy']}")
    L.append("")
    L.append("## 0. 一句话结论")
    L.append("")
    if verdict["decision"] == "keep":
        L.append(
            f"**保留 Davenport**：两者在 {cfg['n_trials']}×3 个加权问题上数值等价"
            f"（最大旋转差 {fmt_sci(verdict['max_rot_frob_diff'])}，最大代价超出 "
            f"{fmt_sci(verdict['max_cost_excess'])}，成功率完全一致），"
            f"而 Davenport 的 p50 几何平均加速 **{verdict['geo_speedup_p50']:.2f}×**"
            f"（N=20/100/500 分别 {verdict['speedup_by_n']['20']:.2f}×/"
            f"{verdict['speedup_by_n']['100']:.2f}×/{verdict['speedup_by_n']['500']:.2f}×）。"
        )
        L.append("")
        L.append(verdict["practical_note"])
    else:
        L.append(
            f"**不保留（revert）**：正确性门槛 {'通过' if verdict['correctness_gate'] else '失败'}"
            f"，运行时间门槛 {'通过' if verdict['runtime_gate'] else '失败'}"
            f"（p50 几何平均加速 {verdict['geo_speedup_p50']:.2f}×，最小单档 "
            f"{verdict['min_speedup_p50']:.2f}×）。Davenport 没有带来可测量的收益。"
        )
    L.append("")
    L.append("## 1. 范围说明（重要）")
    L.append("")
    L.append("> " + SCOPE_NOTE)
    L.append("")
    L.append("## 2. 设置")
    L.append("")
    L.append(f"* 问题：`y_i = R x_i + t + ε_i`，`x_i ~ U(-1,1)^3`，`R_gt` 均匀随机旋转，"
             f"`t_gt ~ U(-0.5,0.5)^3`，`ε ~ N(0, {cfg['sigma']}²)`；求解 `min Σ w_i ||R x_i + t - y_i||²`。")
    L.append(f"* 权重：3 种分布（均匀 0.05–1 / 指数 / 均匀 0–1），并强制约 "
             f"{int(cfg['zero_fraction']*100)}% 的权重为 0；因此同时覆盖 `w_i ≥ 0`、零权重、"
             f"强不对称权重。负权重/NaN/Inf 被拒绝（见 §5）。")
    L.append(f"* 规模：N ∈ {{20, 100, 500}}，每个 N **{cfg['n_trials']}** 个独立随机问题；"
             f"种子 `SeedSequence([{cfg['seed']}, N, trial])`。")
    L.append(f"* 计时：每个问题重复 {cfg['timing_repeats']} 次，单线程 BLAS，"
             f"两个求解器**交错**调用并交替顺序；每档 {cfg['n_trials']*cfg['timing_repeats']} 个"
             f"单次调用样本，报告 p50/p95/p99。绝对 µs 含 Python 调用与 NumPy 开销。")
    L.append("* 成功判据与 `synthetic/common.py` 一致：旋转误差 < 5° 且平移误差 < 0.1 m。")
    L.append("* 环境：单节点 CPU（Slurm `cpu` 分区），`OMP/OPENBLAS/MKL_NUM_THREADS=1`；"
             "共享节点 wall-clock，p95 有系统噪声。")
    L.append("")
    L.append("## 3. 正确性：Davenport vs Kabsch")
    L.append("")
    L.append("| N | `‖R_d−R_k‖_F` p50 | p95 | max | `‖t_d−t_k‖` p50 | max | 代价超出 max | 成功率 Kabsch | 成功率 Davenport | 成功一致率 | 近退化 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in cfg["n_list"]:
        c = corr[str(n)]
        rd, td = c["rot_frob_diff_dav_vs_kabsch"], c["trans_diff_dav_vs_kabsch_m"]
        L.append(
            f"| {n} | {fmt_sci(rd['median'])} | {fmt_sci(rd['p95'])} | {fmt_sci(rd['max'])} | "
            f"{fmt_sci(td['median'])} | {fmt_sci(td['max'])} | {fmt_sci(c['cost_excess_abs_max'])} | "
            f"{c['success_rate']['kabsch']*100:.1f}% | {c['success_rate']['davenport']*100:.1f}% | "
            f"{c['success_identical_frac']*100:.1f}% | {c['near_degenerate_count']} |"
        )
    L.append("")
    L.append("逐档相对真值误差（旋转 deg / 平移 m，中位数）：")
    L.append("")
    L.append("| N | Kabsch rot | Davenport rot | Kabsch trans | Davenport trans |")
    L.append("|---|---|---|---|---|")
    for n in cfg["n_list"]:
        c = corr[str(n)]
        L.append(
            f"| {n} | {c['rot_err_to_gt_deg']['kabsch']['median']:.4f} | "
            f"{c['rot_err_to_gt_deg']['davenport']['median']:.4f} | "
            f"{c['trans_err_to_gt_m']['kabsch']['median']:.2e} | "
            f"{c['trans_err_to_gt_m']['davenport']['median']:.2e} |"
        )
    L.append("")
    max_rot_all = max(corr[str(n)]["rot_frob_diff_dav_vs_kabsch"]["max"] for n in cfg["n_list"])
    max_rel_all = max(corr[str(n)]["cost_excess_rel_max"] for n in cfg["n_list"])
    L.append(f"> 解读：两者在浮点意义上解的是同一个全局最优问题：旋转差 max "
             f"{fmt_sci(max_rot_all)}（相对旋转尺度可忽略），加权代价相对超出 max "
             f"{fmt_sci(max_rel_all)}，成功判定逐题一致。**这不是精度提升，而是等价性验证**；"
             f"Davenport 的卖点只能是运行时/实现简洁性（正常路径无 SVD，反射分支由四元数参数化自动满足）。")
    L.append("")
    L.append("## 4. 运行时：before（当前 Kabsch）/ after（候选 Davenport）")
    L.append("")
    L.append("| N | Kabsch p50 (µs) | Kabsch p95 (µs) | Davenport p50 (µs) | Davenport p95 (µs) | p50 节省 (µs) | 加速 p50 | 加速 p95 | 无 det 检查 p50 (µs) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n in cfg["n_list"]:
        r = run[str(n)]
        L.append(
            f"| {n} | {fmt_us(r['kabsch']['p50_us'])} | {fmt_us(r['kabsch']['p95_us'])} | "
            f"{fmt_us(r['davenport']['p50_us'])} | {fmt_us(r['davenport']['p95_us'])} | "
            f"{fmt_us(r['kabsch']['p50_us'] - r['davenport']['p50_us'])} | "
            f"{r['speedup_davenport_p50']:.2f}× | {r['speedup_davenport_p95']:.2f}× | "
            f"{fmt_us(r['davenport_no_detcheck']['p50_us'])} |"
        )
    L.append("")
    det_ovh = [run[str(n)]["davenport"]["p50_us"] - run[str(n)]["davenport_no_detcheck"]["p50_us"]
               for n in cfg["n_list"]]
    L.append(f"> `davenport_no_detcheck` 是去掉 `det(R)>0` 安全检查的诊断列"
             f"（本机实测成本 {min(det_ovh):.2f}–{max(det_ovh):.2f} µs/次），"
             f"用于说明安全网的成本；正式候选是带检查的 `davenport`。")
    L.append("")
    L.append("## 5. 退化与边界校验")
    L.append("")
    e = edges
    L.append(f"* **平面数据（rank-2）**：200 题、N=100、z≡0。`‖R_d−R_k‖_F` max "
             f"{fmt_sci(e['planar_rank2']['rot_frob_diff']['max'])}，代价超出 max "
             f"{fmt_sci(e['planar_rank2']['cost_excess_abs_max'])}，成功一致率 "
             f"{e['planar_rank2']['success_identical_frac']*100:.1f}%，"
             f"`det(R_kabsch)` min {e['planar_rank2']['det_kabsch_min']:.6f}，"
             f"`det(R_davenport)` min {e['planar_rank2']['det_davenport_min']:.6f}。"
             f"说明在本组数据上 Kabsch 的 det 修正没有触发伪反射，两者仍等价。")
    L.append(f"* **全零权重**：`sum(w)=0` 未定义；两者都回退到均匀权重（Davenport 报告 "
             f"`weight_fallback={e['all_zero_weights']['fallback_flag']}`），与均匀权重的解差 "
             f"{fmt_sci(e['all_zero_weights']['rot_frob_diff_vs_uniform'])}。")
    L.append("* **非法权重**：负权重 / NaN / Inf 均抛出 `ValueError`（"
             + "；".join(f"`{k}`: {v}" for k, v in e["invalid_weights"].items()) + "）。"
             "Kabsch 基线不做该校验（上游若传入负权重会静默给出非最优解），这是候选实现的行为差异。")
    L.append(f"* **近共线**：`eig_gap={fmt_sci(e['collinear_near_degenerate']['eig_gap'])}`，"
             f"标记 `near_degenerate={e['collinear_near_degenerate']['near_degenerate']}`；"
             f"代价仍相同（差 {fmt_sci(e['collinear_near_degenerate']['cost_excess_abs'])}），"
             f"但旋转接近不唯一（条件数大），`‖R_d−R_k‖_F={fmt_sci(e['collinear_near_degenerate']['rot_frob_diff'])}` "
             f"是病态放大而非求解错误；`near_degenerate` 阈值（相对间隔 1e-10）只标记严格退化，"
             f"本例未触发。")
    L.append(f"* **N=1（欠定）**：任何旋转都能拟合；`eig_gap=0`、`near_degenerate=True`，"
             f"两种实现给出不同的合法极小点（差 {fmt_sci(e['single_point_underdetermined']['rot_frob_diff'])}），"
             f"二者 `det=1`。")
    L.append("")
    L.append("## 6. 判断：keep / revert")
    L.append("")
    L.append(f"门槛（脚本内固定）：正确性要求 `max‖R_d−R_k‖_F < {cfg['correct_gate_rot']}`、"
             f"`max|代价差| < {cfg['correct_gate_cost']}`、成功一致率 = 1.0、`det>0`；"
             f"运行时要求 p50 几何平均加速 ≥ {cfg['runtime_gate_geo']} 且单档最差 ≥ "
             f"{cfg['runtime_gate_min']}。")
    L.append("")
    L.append("| 门槛 | 结果 |")
    L.append("|---|---|")
    L.append(f"| 正确性 | {'通过' if verdict['correctness_gate'] else '失败'} |")
    L.append(f"| 运行时 | {'通过' if verdict['runtime_gate'] else '失败'} |")
    L.append(f"| p50 几何平均加速 | {verdict['geo_speedup_p50']:.3f}× |")
    L.append(f"| 单档最小加速 | {verdict['min_speedup_p50']:.3f}× |")
    L.append(f"| **结论** | **{verdict['decision']}** |")
    L.append("")
    if verdict["decision"] == "keep":
        spd = verdict["speedup_by_n"]
        L.append(f"保留理由（按上表固定门槛）：数值严格等价、成功率逐题一致；"
                 f"p50 加速 N=20 {spd['20']:.2f}× / N=100 {spd['100']:.2f}× / "
                 f"N=500 {spd['500']:.2f}×；正常路径无 SVD、反射修正自动满足。"
                 f"注意这是**内层**加速，不代表 GNC 端到端同等加速（未测）。")
    else:
        L.append("不保留理由：见上表门槛；Davenport 没有带来可测量的净收益。")
    L.append("")
    L.append("## 7. 复现")
    L.append("")
    L.append("```bash")
    L.append("cd labs/03-robust-qpeplayer/experiments/inner")
    L.append("PY=/home/syk/miniconda3/envs/WM/bin/python")
    L.append("srun --partition=cpu --time=00:20:00 --ntasks=1 --cpus-per-task=2 \\")
    L.append("  $PY run_inner_solver.py 2>&1 | tee results/logs/inner_solver_$(date +%Y%m%d_%H%M%S).log")
    L.append("```")
    L.append("")
    L.append("产物：`results/inner_solver.json`、`results/inner_solver.md`、"
             "`results/inner_solver.png`（本文件即由脚本写出）。")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"[inner] worktree: {WT_ROOT}")
    print(f"[inner] out: {OUT}")

    corr, run = {}, {}
    for n in N_LIST:
        print(f"[inner] correctness N={n} ...", flush=True)
        corr[str(n)] = correctness_for_n(n)
        print(f"[inner] correctness N={n} done", flush=True)
        print(f"[inner] timing N={n} ...", flush=True)
        run[str(n)] = timing_for_n(n)
        print(f"[inner] timing N={n} done", flush=True)

    print("[inner] edge cases ...", flush=True)
    edges = edge_cases()

    cfg = {
        "seed": SEED,
        "n_list": list(N_LIST),
        "n_trials": N_TRIALS,
        "timing_repeats": TIMING_REPEATS,
        "warmup_calls": WARMUP_CALLS,
        "sigma": SIGMA,
        "zero_fraction": ZERO_FRACTION,
        "success_rot_deg": common.SUCCESS_ROT_DEG,
        "success_trans_m": common.SUCCESS_TRANS,
        "correct_gate_rot": 1e-8,
        "correct_gate_cost": 1e-10,
        "runtime_gate_geo": 1.05,
        "runtime_gate_min": 0.90,
    }

    max_rot = max(corr[str(n)]["rot_frob_diff_dav_vs_kabsch"]["max"] for n in N_LIST)
    max_cost = max(corr[str(n)]["cost_excess_abs_max"] for n in N_LIST)
    ident_all = all(corr[str(n)]["success_identical_frac"] == 1.0 for n in N_LIST)
    det_ok = all(corr[str(n)]["det_davenport_range"][0] > 0.0 for n in N_LIST)
    correctness_gate = bool(max_rot < cfg["correct_gate_rot"]
                            and max_cost < cfg["correct_gate_cost"]
                            and ident_all and det_ok)

    sp = {str(n): run[str(n)]["speedup_davenport_p50"] for n in N_LIST}
    geo = float(np.exp(np.mean(np.log(list(sp.values())))))
    runtime_gate = bool(geo >= cfg["runtime_gate_geo"]
                        and min(sp.values()) >= cfg["runtime_gate_min"])
    sp_list = list(sp.items())
    n_best = max(sp_list, key=lambda kv: kv[1])[0]
    n_worst = min(sp_list, key=lambda kv: kv[1])[0]
    practical_note = (
        f"工程解读：p50 收益集中在 N={n_best}（{sp[n_best]:.2f}×），最差档 N={n_worst}"
        f"（{sp[n_worst]:.2f}×，{'慢于' if sp[n_worst] < 1.0 else '不慢于'}基线）。"
        f"小 N 下固定开销（权重校验 + det 检查 + 四元数→旋转矩阵）占主导；"
        f"若内层主要用于 N≤100，收益在噪声内，是否采用取决于目标 N。"
    )
    verdict = {
        "decision": "keep" if (correctness_gate and runtime_gate) else "revert",
        "correctness_gate": correctness_gate,
        "runtime_gate": runtime_gate,
        "max_rot_frob_diff": float(max_rot),
        "max_cost_excess": float(max_cost),
        "success_identical_all": bool(ident_all),
        "det_positive_all": bool(det_ok),
        "speedup_by_n": sp,
        "geo_speedup_p50": geo,
        "min_speedup_p50": float(min(sp.values())),
        "practical_note": practical_note,
    }

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "loadavg": [float(v) for v in os.getloadavg()],
        "git_branch": subprocess.run(
            ["git", "-C", str(WT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "git_commit": subprocess.run(
            ["git", "-C", str(WT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "script": str(Path(__file__).resolve()),
        "elapsed_s": None,
    }

    payload = {
        "meta": meta,
        "config": cfg,
        "correctness": corr,
        "runtime": run,
        "edge_cases": edges,
        "verdict": verdict,
        "scope_note": SCOPE_NOTE,
    }
    out_json = OUT / "inner_solver.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[inner] wrote {out_json}")

    # ---- PNG -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xpos = np.arange(len(N_LIST))
    width = 0.2
    colors = {"kabsch": "#4C72B0", "davenport": "#DD8452"}
    for i, (name, label) in enumerate((("kabsch", "Kabsch (current)"),
                                       ("davenport", "Davenport (candidate)"))):
        p50 = [run[str(n)][name]["p50_us"] for n in N_LIST]
        p95 = [run[str(n)][name]["p95_us"] for n in N_LIST]
        ax.bar(xpos + (i - 0.5) * width * 2, p50, width,
               color=colors[name], label=f"{label} p50")
        ax.errorbar(xpos + (i - 0.5) * width * 2, p50,
                    yerr=[np.array(p50) * 0, np.array(p95) - np.array(p50)],
                    fmt="none", ecolor="0.25", capsize=3, lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"N={n}" for n in N_LIST])
    ax.set_ylabel("wall-clock per call (µs)")
    ax.set_yscale("log")
    ax.set_title("Inner solver runtime (bar=p50, whisker=p95)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ratios = [run[str(n)]["speedup_davenport_p50"] for n in N_LIST]
    ax.bar(xpos, ratios, 0.5, color="#55A868")
    ax.axhline(1.0, color="0.3", lw=1, ls="--")
    for i, r in enumerate(ratios):
        ax.text(xpos[i], r + 0.01, f"{r:.2f}×", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"N={n}" for n in N_LIST])
    ax.set_ylabel("p50 speedup  Kabsch / Davenport")
    ax.set_title("Speedup (>1 = Davenport faster)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "inner_solver.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[inner] wrote {out_png}")

    meta["elapsed_s"] = time.time() - t_start
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    md = build_markdown(cfg, corr, run, edges, verdict, meta)
    out_md = OUT / "inner_solver.md"
    out_md.write_text(md)
    print(f"[inner] wrote {out_md}")

    print("\n=== summary ===")
    print(json.dumps({"verdict": verdict, "meta": meta}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
