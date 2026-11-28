#!/usr/bin/env python
"""生成四张实验图与一页总览图。"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
DOCS = os.path.join(os.path.dirname(ROOT), "docs", "visuals")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

C0, C1, C2, C3 = "#2E5EAA", "#D1495B", "#00798C", "#EDAE49"


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


def fig0():
    d = load("e0_theorem0.json")
    gaps = [d["choi_lam"]["gap"], d["real_qpep_gap_max"], d["psd_search_40"]["worst_gap"]]
    labels = ["Choi–Lam\n(不定 Q，非 QPEP)", "真实 QPEP 12 例\n(Qm PSD)", "随机 PSD 40 例\n(Q PSD)"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(3), gaps, color=[C1, C0, C2], width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(range(3), labels, fontsize=9)
    ax.set_ylabel("|f* − SDP 下界|", fontsize=10)
    ax.set_title("定理 0 的间隙只出现在非 QPEP 的不定矩阵上", fontsize=11)
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2, g * 1.6, f"{g:.2e}", ha="center", fontsize=9)
    ax.axhline(1e-6, color="gray", ls="--", lw=0.8)
    ax.text(2.3, 1.2e-6, "数值紧性阈值 1e-6", fontsize=8, color="gray", ha="right")
    ax.set_ylim(1e-10, 1)
    fig.savefig(os.path.join(FIG, "fig0_theorem0.png"))
    plt.close(fig)


def fig1():
    d = load("e1_riem.json")
    inst = d["instances"]
    rot = np.array([r["vjp_rot_rel"] for r in inst])
    tr = np.array([r["vjp_trans_rel"] for r in inst])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.semilogy(np.arange(len(rot)), np.maximum(rot, 1e-16), "o-", color=C0, label="旋转 VJP")
    ax.semilogy(np.arange(len(tr)), np.maximum(tr, 1e-16), "s--", color=C1, label="平移链式 VJP")
    ax.axhline(1e-6, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("随机实例", fontsize=9)
    ax.set_ylabel("与中心差分的相对误差", fontsize=9)
    ax.set_title("3×3 隐式 VJP 的精度", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    times = [d["backward_us"], d["local_refine_us"], d["forward_ms"] * 1e3]
    names = ["反向\n3×3", "局部牛顿\n(1 次)", "多起点\n(32 起点)"]
    bars = ax.bar(range(3), times, color=[C0, C2, C3], width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set_ylabel("时间 (µs, log)", fontsize=9)
    ax.set_title("单样本 CPU 计时", fontsize=10)
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t * 1.5, f"{t:.0f}", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig1_riem.png"))
    plt.close(fig)


def fig2():
    d = load("e2_barrier.json")
    rows = d["homotopy"]
    tau = np.array([r["tau"] for r in rows])
    err = np.array([r["grad_err"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.loglog(tau, err, "o-", color=C0, label="实测")
    ref = err[0] * (tau / tau[0])
    ax.loglog(tau, ref, "k--", lw=0.8, label=r"$O(\tau)$ 参考")
    ax.set_xlabel(r"障碍温度 $\tau$", fontsize=10)
    ax.set_ylabel("梯度相对误差", fontsize=10)
    ax.set_title(f"BarrierQPEP 的积极集收敛（斜率 {d['slope']:.3f}）", fontsize=10)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    fig.savefig(os.path.join(FIG, "fig2_barrier.png"))
    plt.close(fig)


def fig3():
    d = load("e3_spec.json")
    rows = d["rows"]
    th = np.array([r["theta"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for key, lab, col, mk in [("Z0", r"$Z_0$（仅能量）", C0, "o"),
                              ("ZML", r"$Z_{\rm ML}$（曲率修正）", C2, "s"),
                              ("true", "真实 $S^3$ 积分", C1, "^"),
                              ("hard", "硬 argmin", C3, "x")]:
        ax.plot(th, [r["fd"][key] for r in rows], marker=mk, color=col, label=lab)
    ax.axvline(0.2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\theta$（0.2 处 Maxwell 交叉）", fontsize=9)
    ax.set_ylabel(r"$d\log Z/d\theta$", fontsize=9)
    ax.set_title("交叉附近仍然光滑 vs 硬 argmin 跳变", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[1]
    sdp = d["sdp"]
    ratios = [s["lambda1_over_lambda2"] for s in sdp]
    ax.bar(range(len(sdp)), ratios, color=[C0, C1, C2], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sdp)), [f"$\\theta$={s['theta']:.2f}" for s in sdp], fontsize=9)
    ax.set_ylabel(r"$\lambda_1/\lambda_2$", fontsize=9)
    ax.set_title("tie 处 SDP 原解秩 2（比值 1.0）", fontsize=10)
    ax.axhline(1e6, color="gray", ls="--", lw=0.8)
    fig.savefig(os.path.join(FIG, "fig3_spec.png"))
    plt.close(fig)


def summary():
    d0, d1, d2, d3 = (load(f"e{i}_{n}.json") for i, n in
                      enumerate(["theorem0", "riem", "barrier", "spec"]))
    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    gaps = [d0["choi_lam"]["gap"], d0["real_qpep_gap_max"], d0["psd_search_40"]["worst_gap"]]
    ax.bar(range(3), gaps, color=[C1, C0, C2], width=0.55)
    ax.set_yscale("log"); ax.set_ylim(1e-10, 1)
    ax.set_xticks(range(3), ["Choi–Lam\n不定 Q", "真实 QPEP\n12 例", "随机 PSD\n40 例"], fontsize=8)
    ax.set_title("A. 定理 0：间隙只在不定的非 QPEP 矩阵上", fontsize=10)
    for i, g in enumerate(gaps):
        ax.text(i, g * 2, f"{g:.1e}", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    rows = d2["homotopy"]
    tau = [r["tau"] for r in rows]; err = [r["grad_err"] for r in rows]
    ax.loglog(tau, err, "o-", color=C0)
    ax.loglog(tau, [err[0] * (t / tau[0]) for t in tau], "k--", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\tau$", fontsize=9); ax.set_ylabel("梯度误差", fontsize=9)
    ax.set_title(f"B. Barrier：斜率 {d2['slope']:.3f}（理论 1）", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    rows = d3["rows"]; th = [r["theta"] for r in rows]
    ax.plot(th, [r["fd"]["Z0"] for r in rows], "o-", color=C0, label=r"$Z_0$")
    ax.plot(th, [r["fd"]["ZML"] for r in rows], "s-", color=C2, label=r"$Z_{\rm ML}$")
    ax.plot(th, [r["fd"]["true"] for r in rows], "^-", color=C1, label="真实积分")
    ax.axvline(0.2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\theta$", fontsize=9); ax.set_ylabel(r"$d\log Z/d\theta$", fontsize=9)
    ax.set_title("C. Spec：$Z_0$ 有偏，$Z_{\\rm ML}$ 接近真实，硬 argmin 跳变", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    txt = (
        "本批结论\n\n"
        "• RiemQPEP：3×3 VJP 与 FD 一致到 1e-8；双覆盖差 0；反向 220 µs\n"
        "• BarrierQPEP：τ→0 斜率 0.997；平方松弛 KKT 奇异确认\n"
        "• SpecQPEP：Z0 梯度与 FD 1e-6；Z0 偏差 4.59，ZML 0.45；tie 秩 2\n\n"
        "关键更正\n"
        "• 真实 QPEP 的 Qm 是 PSD（Schur 补合同），Choi–Lam 不是合法 QPEP\n"
        "• 故“QPEP 必然 Hilbert 非紧”不成立；12 例真实 SDP 间隙 ≤ 4.8e-8\n\n"
        "未覆盖：真实数据集、LibQPEP/GPU、13×13 独立复跑、端到端训练"
    )
    ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=8.6, linespacing=1.5)

    fig.suptitle("QPEP 第二批：构建与预实验一页总览（2026-11-28）", fontsize=13, y=0.99)
    out = os.path.join(DOCS, "2026-11-28-qpep-batch2.png")
    fig.savefig(out, dpi=170)
    fig.savefig(os.path.join(FIG, "summary.png"), dpi=170)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); summary()
