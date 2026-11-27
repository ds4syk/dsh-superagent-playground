#!/usr/bin/env python3
"""3DMatch: does success depend on the base (clean) inlier fraction?"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REAL = HERE.parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
BINS = [(0.80, 0.90), (0.90, 0.95), (0.95, 1.001)]


def main():
    rows = []
    for path in [REAL / "threedmatch/results/threedmatch_raw.jsonl",
                 REAL / "threedmatch_extra/results/extra_scenes_raw.jsonl"]:
        rows += [json.loads(x) for x in path.read_text().splitlines()]
    out = []
    for lo, hi in BINS:
        for p in [0.0, 0.3, 0.5, 0.7]:
            for m in ["GNC-GM", "RANSAC-1000", "IRLS-GM"]:
                rr = [r for r in rows if lo <= r["base_inlier"] < hi and abs(r["p"] - p) < 1e-9
                      and r["method"] == m]
                if not rr:
                    continue
                out.append(dict(bin=f"[{lo:.2f},{min(hi,1.0):.2f})", p=p, method=m, n=len(rr),
                                success=float(np.mean([r["success"] for r in rr])),
                                rot_median=float(np.median([r["rot"] for r in rr]))))
    (OUT / "base_inlier.json").write_text(json.dumps(out, indent=2))
    L = ["# 3DMatch: success vs base (clean) inlier fraction", "",
         "| base-inlier bin | p | method | n | success | median rot |", "|---|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['bin']} | {r['p']:.1f} | {r['method']} | {r['n']} | "
                 f"{100*r['success']:.0f}% | {r['rot_median']:.2f}° |")
    L += ["", "## Reading", "",
          "The base-inlier fraction is the fraction of GT-NN pairs inside c before injected outliers. "
          "If success is flat across bins, the controlled outlier rate, not base quality, drives failures."]
    (OUT / "base_inlier.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
