#!/usr/bin/env python3
"""Wilson 95% intervals for the headline success rates.

Success cells with 20 trials and 100% have a wide interval; the paper
should not present them as exact.  This script recomputes Wilson intervals
from the measured aggregate JSON and prints a table that can be cited.
"""
from __future__ import annotations
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def add(rows, name, k, n):
    lo, hi = wilson(k, n)
    rows.append(dict(cell=name, k=k, n=n, rate=k / n, wilson_lo=lo, wilson_hi=hi))


def main():
    rows = []
    syn = json.loads((HERE / "synthetic/results/main_table_extended.json").read_text())["rows"]
    for p in [0.5, 0.7]:
        r = next(x for x in syn if x["outlier_type"] == "random" and x["method"] == "GNC-GM"
                 and abs(x["p"] - p) < 1e-9)
        add(rows, f"synthetic random p={p} GNC", int(round(r["success"] * 100)), 100)
    tm = json.loads((HERE / "real/threedmatch/results/threedmatch.json").read_text())["rows"]
    tm += json.loads((HERE / "real/threedmatch_extra/results/extra_scenes.json").read_text())["rows"]
    for p in [0.5, 0.7]:
        rr = [x for x in tm if x["outlier_type"] == "random" and x["method"] == "GNC-GM"
              and abs(x["p"] - p) < 1e-9]
        n = sum(x["n_trials"] for x in rr)
        k = sum(round(x["success"] * x["n_trials"]) for x in rr)
        add(rows, f"3DMatch-4 random p={p} GNC", k, n)
    for p in [0.5, 0.6]:
        rr = [x for x in tm if x["outlier_type"] == "coherent" and x["method"] == "GNC-GM"
              and abs(x["p"] - p) < 1e-9]
        n = sum(x["n_trials"] for x in rr)
        k = sum(round(x["success"] * x["n_trials"]) for x in rr)
        add(rows, f"3DMatch-4 coherent p={p} GNC", k, n)
    kit = json.loads((HERE / "real/kitti/results/kitti.json").read_text())["rows"]
    kit += json.loads((HERE / "real/kitti2/results/kitti2.json").read_text())["rows"]
    for p in [0.5, 0.7]:
        rr = [x for x in kit if x["outlier_type"] == "random" and x["method"] == "GNC-GM"
              and abs(x["p"] - p) < 1e-9]
        n = sum(x["n_trials"] for x in rr)
        k = sum(round(x["success"] * x["n_trials"]) for x in rr)
        add(rows, f"KITTI-2 random p={p} GNC", k, n)
    (OUT / "wilson.json").write_text(json.dumps(rows, indent=2))
    L = ["# Wilson 95% intervals for headline success rates", "",
         "| cell | k/n | rate | 95% Wilson |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cell']} | {r['k']}/{r['n']} | {100*r['rate']:.1f}% | "
                 f"[{100*r['wilson_lo']:.1f}, {100*r['wilson_hi']:.1f}]% |")
    L += ["", "## Reading", "",
          "The lower bound for a 100% cell with n=20 is ~83.9%; with n=40 ~91.2%. "
          "Do not present 20-trial 100% cells as exact 100%."]
    (OUT / "wilson.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
