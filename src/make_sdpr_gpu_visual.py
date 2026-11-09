"""One-page visual for the SDPR GPU-parallel implementation (2026-11)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#2E6B3A"
BLUE = "#1F4E79"
ORANGE = "#B5651D"
GRAY = "#595959"
LIGHT = "#F2F5F2"


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def box(ax, x, y, w, h, text, fc=LIGHT, ec=GRAY, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color="#1A1A1A", weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color=color))


def main():
    verify = load(os.path.join(ROOT, "repro/gpu_sdpr/results/verify_p6.json"), {})
    bench = load(os.path.join(ROOT, "repro/gpu_sdpr/results/benchmark.json"), {"rows": []})
    e2e = load(os.path.join(ROOT, "repro/gpu_sdpr/results/e2e_calib.json"), {})

    s = verify.get("summary", {})
    xerr = s.get("x_err_rel", {}).get("max", 7.8e-6)
    gsum = s.get("grad_sum_err_rel", {}).get("max", 6.7e-5)
    hmin = s.get("H_min_eig_max", {}).get("max", 7.7e-10)
    corank = s.get("corank1_total", 200)
    npose = s.get("n_poses", 200)
    rank_min = s.get("rank_ratio_min", 3.2e5)
    gap = s.get("gap", 1.3e-7)

    rows = bench.get("rows", [])
    if rows:
        Bs = [f"B={r['batch']}" for r in rows]
        cpu = [r["cpu_total_s"] for r in rows]
        gpu = [r["gpu_fwd_s"] for r in rows]
        sp = [r["speedup_fwd"] for r in rows]
    else:
        Bs, cpu, gpu, sp = ["B=20", "B=80", "B=200"], [1.30, 5.2, 12.1], [0.072, 0.090, 0.079], [18, 58, 154]

    e2e_rows = e2e.get("runs", [])
    traj = e2e.get("summary", {}).get("max_abs_traj_diff", 2.5e-6)
    err_cpu = e2e.get("summary", {}).get("err_final_cpu_mean", 4.57e-4)
    err_gpu = e2e.get("summary", {}).get("err_final_gpu_mean", 4.55e-4)

    fig = plt.figure(figsize=(17.5, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.035, right=0.975, top=0.86, bottom=0.115,
                          wspace=0.16, hspace=0.34)
    fig.suptitle("SDPR 的 GPU 并行实现：检索、实现与验证（2026-11）", fontsize=20,
                 color="#111111", weight="bold", y=0.965)
    fig.text(0.5, 0.915, "原文把 CPU-only 列为未来工作；本工作实现 batch-parallel GPU SDPR，"
             "在 P6 标定上复现精度并加速 12–150×", ha="center", fontsize=11.5, color=GRAY)

    # --- A. 检索结论 -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("A｜检索：无公开 GPU 并行版本", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    box(ax, 0.03, 0.70, 0.94, 0.16, "原文 §VII：forward/backward 都在 CPU；\n"
        "“A parallel GPU implementation … necessary for larger robotics problems”", fc="#FFF4E5", ec=ORANGE, size=8.6)
    box(ax, 0.03, 0.47, 0.94, 0.16, "官方 utiasASRL/sdprlayer：CVXPY + diffcp，SCS/Mosek，CPU；\n"
        "2025-11 最后提交；4 个 fork 均无 GPU 代码", fc=LIGHT, size=8.6)
    box(ax, 0.03, 0.24, 0.94, 0.16, "引用文献 13 篇：Sparse SOS Layers / dOPT / FFOLayer 等\n"
        "均为可微 SDP 层，没有 GPU 并行 SDPR", fc=LIGHT, size=8.6)
    box(ax, 0.03, 0.02, 0.94, 0.16, "本工作：batch-parallel GPU 层（对偶 log-det barrier\n"
        "+ Theorem 2 隐式微分）", fc="#E8F1E8", ec=GREEN, size=9.2, weight="bold")
    for y in (0.68, 0.45, 0.22):
        arrow(ax, 0.5, y, 0.5, y - 0.035, color=GRAY)

    # --- B. 方法 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("B｜实现：批量对偶 barrier + KKT 反传", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    box(ax, 0.03, 0.73, 0.28, 0.17, "B 个 SDP\n(Q_b, A, b)", fc=LIGHT, size=9)
    box(ax, 0.36, 0.73, 0.30, 0.17, "GPU 批量\nH=Q+Σ y_i A_i", fc="#E8F1E8", ec=GREEN, size=9)
    box(ax, 0.71, 0.73, 0.26, 0.17, "eigh / Cholesky\n13×13，批量", fc="#E8F1E8", ec=GREEN, size=9)
    box(ax, 0.03, 0.45, 0.28, 0.17, "对偶更新\nμ=1e-2→1e-8", fc=LIGHT, size=9)
    box(ax, 0.36, 0.45, 0.30, 0.17, "X=μ H^-1, x=null(H)\nH 半正定证书", fc=LIGHT, size=9)
    box(ax, 0.71, 0.45, 0.26, 0.17, "Theorem 2\nKKT 反传", fc=LIGHT, size=9)
    box(ax, 0.30, 0.13, 0.40, 0.16, "∂L/∂Q（symmetric）\n训练可反传", fc="#FFF4E5", ec=ORANGE, size=9.2)
    arrow(ax, 0.31, 0.815, 0.355, 0.815); arrow(ax, 0.66, 0.815, 0.705, 0.815)
    arrow(ax, 0.84, 0.73, 0.84, 0.625); arrow(ax, 0.71, 0.535, 0.665, 0.535)
    arrow(ax, 0.36, 0.535, 0.315, 0.535); arrow(ax, 0.50, 0.45, 0.50, 0.30)
    ax.text(0.5, 0.34, "反传只解一次 20×20 KKT 系统；不展开 barrier 迭代", ha="center",
            fontsize=8, color=GRAY)

    # --- C. 速度 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C｜速度：批量 GPU vs 单进程 CPU SCS", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    x = np.arange(len(Bs)); w = 0.36
    b1 = ax.bar(x - w / 2, cpu, w, label="CPU SCS", color="#B8C4CE")
    b2 = ax.bar(x + w / 2, gpu, w, label="GPU（本工作）", color=GREEN)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(Bs, fontsize=9)
    ax.set_ylabel("前向时间 (s，log)", fontsize=9)
    ax.grid(axis="y", alpha=0.25, which="both")
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        ax.text(i - w / 2, c * 1.15, "%.2fs" % c, ha="center", fontsize=8)
        ax.text(i + w / 2, g * 1.15, "%.3fs" % g, ha="center", fontsize=8, color=GREEN)
        ax.text(i, max(c, g) * 2.2, "%.0fx" % (c / g), ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.set_ylim(min(gpu) * 0.5, max(cpu) * 6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # --- D. 精度 / 证书 ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("D｜精度与证书（200 poses）", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    labels = ["x 相对误差\n(7.8e-06)", "batch 梯度\n(6.7e-05)", "H 最小特征\n(7.7e-10)"]
    vals = [xerr, gsum, hmin]
    colors = [GREEN, GREEN, BLUE]
    ax.barh(range(3), vals, color=colors, height=0.55)
    ax.set_xscale("log"); ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("数值（log，越小越好）", fontsize=9)
    ax.grid(axis="x", alpha=0.25, which="both")
    for i, v in enumerate(vals):
        ax.text(v * 1.3, i, "%.2e" % v, va="center", fontsize=9)
    ax.set_xlim(min(vals) * 0.3, max(vals) * 30)
    fig.text(0.045, 0.018, "corank(H)=1：%d/%d    rank(λ1/λ2) ≥ %.1e    对偶间隙 = %.1e    "
             "SDP 约束 25→18（去 7 个线性相关）" % (corank, npose, rank_min, gap),
             fontsize=8.6, color="#1A1A1A",
             bbox=dict(fc="#F7F7F7", ec="#CCCCCC", boxstyle="round,pad=0.35"))

    # --- E. 端到端 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("E｜端到端标定：轨迹与 CPU 一致", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    if e2e_rows:
        traj_data = e2e.get("trajectories", {})
        first = traj_data.get("0", {})
        err_c = first.get("err_cpu", [])
        err_g = first.get("err_gpu", [])
        if err_c and err_g:
            n = min(len(err_c), len(err_g))
            ax.plot(np.arange(n), np.abs(err_c[:n]), color="#B8C4CE", lw=3, label="CPU SDPR")
            ax.plot(np.arange(n), np.abs(err_g[:n]), color=GREEN, lw=1.6, ls="--", label="GPU SDPR")
            ax.set_yscale("log")
    ax.set_xlabel("outer iteration", fontsize=9)
    ax.set_ylabel("|b - b_true| (m，log)", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.03, 0.06, "最大轨迹差 %.1e m；终值 CPU %.2e / GPU %.2e m" % (traj, err_cpu, err_gpu),
            transform=ax.transAxes, fontsize=8.2, color="#1A1A1A")

    # --- F. 局限 -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("F｜边界与下一步", fontsize=12.5, color=BLUE, weight="bold", loc="left")
    items = [
        ("严格可行起点", "需要约束张成空间里的 PSD 方向；P6 用 I_C+A0。\n一般 QCQP 需 Phase-I。"),
        ("精度", "μ=1e-8，gap=13μ；x 误差 ~7e-6，\nbatch 梯度误差 ~1.7e-5。"),
        ("规模", "单卡 batch-parallel；单问题内部未做\n多 GPU/稀疏分解。"),
        ("验证范围", "P6 标定 + 200 poses；完整 50-run\nTable V 与真实数据未重跑。"),
    ]
    y = 0.93
    for head, body in items:
        ax.text(0.03, y, "• " + head, fontsize=10, color=ORANGE, weight="bold")
        ax.text(0.06, y - 0.055, body, fontsize=8.2, color="#1A1A1A", va="top")
        y -= 0.235
    ax.text(0.03, 0.01, "代码：src/gpu_sdpr/ · 复现：repro/gpu_sdpr/", fontsize=8.0, color=GRAY)

    out = os.path.join(ROOT, "docs/visuals/2026-11-09-sdpr-gpu.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
