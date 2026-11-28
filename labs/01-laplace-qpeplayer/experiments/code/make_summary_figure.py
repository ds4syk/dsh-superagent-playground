#!/usr/bin/env python3
"""One-page P1 summary figure (all four panels read committed measurements).

A. synthetic 6D crossing: Laplace / MC / hard gradient rel error vs theta (raw_grid.json)
B. fold: mode count, fragile-mode Laplace weight and |dpose| near |theta|~1.3e-3 (robustness.json)
C. real LINEMOD planar-face two-mode cost: Delta J vs flip angle (real/linemod results)
D. SDP: f_13 vs f_10 for the 10 instances (sdp results), y=x reference and the rowdot counterexample
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "results"


def main():
    raw = json.load(open(OUT / "raw_grid.json"))
    rob = json.load(open(OUT / "results" / "robustness.json")) if (OUT / "results").exists() else json.load(open(OUT / "robustness.json"))
    two = json.load(open(EXP / "real" / "linemod" / "results" / "two_mode.json"))
    sdp = json.load(open(EXP / "sdp" / "results" / "sdp_check.json"))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))

    # A
    th = np.array([r["theta"] for r in raw])
    ax[0, 0].semilogy(th, [r["laplace_grad_rel"] for r in raw], "-", lw=1.4, label="Laplace (analytic)")
    ax[0, 0].semilogy(th, [r["mc_grad_rel"] for r in raw], ".", ms=3, label="mode-level MC")
    hv = [r["hard_grad_rel"] if r["hard_grad_rel"] is not None else np.nan for r in raw]
    ax[0, 0].semilogy(th, hv, ".", ms=3, label="hard argmin (N/A at tie)")
    ax[0, 0].set_xlabel("theta"); ax[0, 0].set_ylabel("gradient rel. error vs FD")
    ax[0, 0].set_title("A. synthetic 6D branch crossing (81 points)")
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend(fontsize=8)

    # B
    f = rob["a_fold"]["fine"]
    t = np.array([r["theta"] for r in f])
    K = np.array([r["K"] for r in f])
    dp = np.array([r["dpose_norm"] for r in f])
    wf = np.array([r["w_fragile"] for r in f])
    ax[0, 1].plot(t * 1e3, K, "o-", ms=3, color="C0", label="K (left)")
    ax[0, 1].set_yticks([3, 4]); ax[0, 1].set_ylim(2.7, 4.25)
    ax[0, 1].set_xlabel("theta (x1e-3)"); ax[0, 1].set_ylabel("number of minima K")
    ax[0, 1].set_title("B. mode fold near |theta| ~ 1.3e-3")
    axb = ax[0, 1].twinx()
    axb.semilogy(t * 1e3, dp, "s--", ms=3, color="C3", label="|dpose|")
    axb.semilogy(t * 1e3, wf, "^:", ms=3, color="C2", label="w(fragile)")
    try:
        f6 = json.load(open(OUT / "fold_6d.json"))
        t6 = np.array([r["theta"] for r in f6["rows"] if "reg" in r])
        dp6 = np.array([r["reg"]["dpose_norm"] for r in f6["rows"] if "reg" in r])
        axb.semilogy(t6 * 1e3, dp6, "d-", ms=3, color="C4", label="|dpose| (tau^1/3 reg)")
    except Exception:
        pass
    axb.set_ylabel("|dpose| / weight")
    h1, l1 = ax[0, 1].get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
    ax[0, 1].legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax[0, 1].grid(alpha=0.3)

    # C
    a = two["part_a"]
    ax[1, 0].semilogy([r["flip_deg"] for r in a], [r["dJ"] for r in a], "o", ms=6)
    ax[1, 0].set_xlabel("flip vs GT rotation difference (deg)")
    ax[1, 0].set_ylabel("Delta J (flip - true)")
    ax[1, 0].set_title("C. real LINEMOD: 15/15 frames K=2 (planar face)")
    ax[1, 0].grid(alpha=0.3)

    # D: f*_10 (full 20D) vs f*_13 upstream (17D); full-20D 13x13 equals 10x10
    m = sdp["main"]
    f10 = np.array([r["f_10"] for r in m]); f13 = np.array([r["f_13"] for r in m])
    names = [r["name"] for r in m]
    ax[1, 1].scatter(f10, f13, s=28, label="10 instances (full 20D vs upstream)")
    try:
        six = json.load(open(EXP / "sdp" / "results" / "six_point.json"))
        ax[1, 1].scatter([six["f10"]], [six["f13_up"]], s=70, marker="*", color="C3",
                         label="6-point physical counterexample")
        ax[1, 1].annotate("6-point: 5.415 vs 3.594\n(full 20D 13x13 = 5.415)",
                          (six["f10"], six["f13_up"]), fontsize=7, color="C3",
                          xytext=(6, -18), textcoords="offset points")
    except Exception:
        pass
    lo = min(f10.min(), f13.min()) - 0.2; hi = max(f10.max(), f13.max()) + 0.2
    ax[1, 1].plot([lo, hi], [lo, hi], "k--", lw=1, label="f13 = f10 (tight)")
    for n, x, y in zip(names, f10, f13):
        if n in ("rowdot", "CL_exact"):
            ax[1, 1].annotate(n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1, 1].set_xlabel("f*_10 full 20D (21 constraints)")
    ax[1, 1].set_ylabel("f*_13 upstream (24 constraints, 17D)")
    ax[1, 1].set_title("D. upstream 13x13 is looser; full 20D = 10x10")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out = OUT / "p1_summary.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
