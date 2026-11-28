#!/usr/bin/env python3
"""Two-panel summary figure for the P3 data-coverage work.

A: 3DMatch success vs random-outlier rate, true low-overlap 0.20-0.50 band
   (solid) against the matched >0.80 control (dashed, GNC only).
B: KITTI odometry GNC success for the 8 newly added sequences over
   p = 0 / 0.3 / 0.5 / 0.7 (10 trials/cell).

Run: /home/syk/miniconda3/envs/WM/bin/python make_p3_data_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOW = HERE / "threedmatch_low_overlap" / "results" / "low_overlap.json"
KIT = HERE / "kitti_more" / "results" / "kitti_more.json"


def main():
    low = json.loads(LOW.read_text())
    kit = json.loads(KIT.read_text())
    P = low["config"]["p"]
    tabs = {(t["band"], t["p"], t["method"]): t for t in low["tables"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9), dpi=170)

    colors = {"GNC-GM": "#1f77b4", "RANSAC-1000": "#d62728",
              "IRLS-GM": "#2ca02c", "Kabsch": "#7f7f7f"}
    for m in ["GNC-GM", "RANSAC-1000", "IRLS-GM", "Kabsch"]:
        y = [tabs[("low_20_50", p, m)]["success"] for p in P]
        ax1.plot(P, y, "o-", color=colors[m], label=f"{m} (0.20-0.50)")
    yh = [tabs[("high", p, "GNC-GM")]["success"] for p in P]
    ax1.plot(P, yh, "s--", color=colors["GNC-GM"], alpha=0.55,
             label="GNC-GM (>0.80 control)")
    ax1.set_xlabel("random-outlier rate p")
    ax1.set_ylabel("success (rot<5°, |t|<0.1 m)")
    ax1.set_title("3DMatch: true low overlap (32 pairs) vs high control (14)")
    ax1.set_xticks(P)
    ax1.set_ylim(-0.03, 1.05)
    ax1.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.25)
    ax1.text(0.33, 0.52, "base inlier 0.953 (low) / 0.984 (high)\n"
                         "clean rot 0.47° vs 0.19°",
             transform=ax1.transAxes, fontsize=7, color="#444")

    seqs = kit["config"]["sequences"]
    M = np.array([[next(r for r in kit["per_sequence"][s]["rows"]
                        if r["method"] == "GNC-GM" and abs(r["p"] - p) < 1e-9)["success"]
                   for p in kit["config"]["p"]] for s in seqs])
    im = ax2.imshow(M, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(len(kit["config"]["p"])))
    ax2.set_xticklabels([f"{p:.1f}" for p in kit["config"]["p"]])
    ax2.set_yticks(range(len(seqs)))
    ax2.set_yticklabels([f"seq {s}" for s in seqs])
    ax2.set_xlabel("random-outlier rate p")
    ax2.set_title("KITTI: GNC-GM success, 8 added sequences\n(10 trials/cell)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, label="success")

    fig.tight_layout()
    out = HERE / "p3_data_coverage.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, f"({out.stat().st_size/1e3:.1f} kB)")


if __name__ == "__main__":
    main()
