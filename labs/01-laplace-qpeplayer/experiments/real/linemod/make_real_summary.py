#!/usr/bin/env python3
"""One-page real-data summary: pose-error / ADD-S medians with bootstrap CIs.

Reads the committed JSON files only (no recomputation):
  results/pose_error.json            (P1 hard / Laplace / MC)
  results/classical_baselines.json   (EPnP / SQPNP / IPPE / LM)
  results/theseus_baseline.json      (Theseus LM, GT / random init)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def ci(entry):
    med = entry["median"]
    lo, hi = entry.get("median_ci", [med, med])
    return med, med - lo, hi - med


def main():
    pose = json.load(open(RES / "pose_error.json"))["aggregate"]
    clas = json.load(open(RES / "classical_baselines.json"))["aggregate"]
    thes = json.load(open(RES / "theseus_baseline.json"))["main_summary"]

    rot, adds = [], []
    for name, key, src in [
        ("P1 hard", "hard", "pose"), ("P1 Laplace", "laplace", "pose"), ("P1 MC", "mc", "pose"),
        ("SQPNP", "SQPNP first", "class"), ("IPPE", "IPPE first", "class"),
        ("EPnP", "EPnP", "class"), ("LM (GT init)", "LM GT init", "class"),
    ]:
        e = pose["rot_deg"][key] if src == "pose" else clas[key]["rot"]
        a = pose["adds_m"][key] if src == "pose" else clas[key]["add_s_mm"]
        rot.append((name, *ci(e)))
        adds.append((name, (a["median"] * 1000.0 if src == "pose" else a["median"]),
                     (a["median"] - a.get("median_ci", [a["median"]])[0]) * (1000.0 if src == "pose" else 1.0),
                     (a.get("median_ci", [a["median"], a["median"]])[1] - a["median"]) * (1000.0 if src == "pose" else 1.0)))
    rot.append(("Theseus (GT init)", thes["gt"]["rot_err_deg"]["median"], 0.0, 0.0))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for a, rows, ylab, title in [
        (ax[0], rot, "rotation error (deg, median [95% CI])", "Rotation error vs GT"),
        (ax[1], adds, "ADD-S (mm, median [95% CI])", "ADD-S vs GT (symmetric objects)"),
    ]:
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        err = np.array([[r[2] for r in rows], [r[3] for r in rows]])
        colors = ["C0" if n.startswith("P1") else ("C3" if "Theseus" in n else "C2") for n in names]
        x = np.arange(len(names))
        a.bar(x, vals, yerr=err, capsize=4, color=colors, alpha=0.85)
        a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        a.set_ylabel(ylab); a.set_title(title); a.grid(alpha=0.25, axis="y")
    fig.suptitle("Real LINEMOD eggbox/glue (15 frames): P1 layer vs classical / local baselines", fontsize=10)
    fig.tight_layout()
    out = RES / "real_summary.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()
