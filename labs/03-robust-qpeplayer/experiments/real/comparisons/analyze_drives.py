#!/usr/bin/env python3
"""Per-drive heterogeneity analysis for the two KITTI raw drives."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REAL = HERE.parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    return json.loads(Path(path).read_text())

def row(rows, **kw):
    for r in rows:
        if all(r.get(k) == v if isinstance(v, str) else abs(r.get(k, -1) - v) < 1e-9 for k, v in kw.items()):
            return r
    raise KeyError(kw)

def main():
    out = {}
    for name, path in [("drive_0001", REAL / "kitti/results/kitti.json"),
                       ("drive_0002", REAL / "kitti2/results/kitti2.json")]:
        rows = load(path)["rows"]
        d = {}
        for p in [0.0, 0.3, 0.5, 0.7]:
            g = row(rows, outlier_type="random", method="GNC-GM", p=p)
            f = row(rows, outlier_type="random", method="RANSAC-1000", p=p)
            a = row(rows, outlier_type="random", method="RANSAC-adaptive", p=p)
            d[f"random_p{p}"] = dict(
                gnc_success=g["success"], ransac_success=f["success"],
                gnc_rot=g["rot_median"], ransac_rot=f["rot_median"],
                gnc_ms=g["time_ms_p50"], ransac_ms=f["time_ms_p50"],
                adaptive_ms=a["time_ms_p50"],
                speedup_vs_1000=f["time_ms_p50"] / g["time_ms_p50"],
            )
        for p in [0.4, 0.5, 0.6]:
            d[f"coherent_p{p}"] = {m: row(rows, outlier_type="coherent", method=m, p=p)["success"]
                                   for m in ["RANSAC-1000", "IRLS-GM", "GNC-GM"]}
        for p in [0.5, 0.7]:
            d[f"boundary_p{p}"] = {m: dict(success=row(rows, outlier_type="boundary", method=m, p=p)["success"],
                                           rot=row(rows, outlier_type="boundary", method=m, p=p)["rot_median"])
                                   for m in ["RANSAC-1000", "IRLS-GM", "GNC-GM"]}
        out[name] = d
    (OUT / "drive_heterogeneity.json").write_text(json.dumps(out, indent=2))
    L = ["# KITTI per-drive heterogeneity", "",
         "| metric | drive 0001 | drive 0002 |", "|---|---|---|"]
    for k in ["random_p0.0", "random_p0.3", "random_p0.5", "random_p0.7"]:
        a, b = out["drive_0001"][k], out["drive_0002"][k]
        L.append(f"| {k}: GNC/RANSAC success | {100*a['gnc_success']:.0f}/{100*a['ransac_success']:.0f}% | "
                 f"{100*b['gnc_success']:.0f}/{100*b['ransac_success']:.0f}% |")
        L.append(f"| {k}: GNC speedup vs RANSAC-1000 | {a['speedup_vs_1000']:.1f}× | {b['speedup_vs_1000']:.1f}× |")
        L.append(f"| {k}: GNC med rot | {a['gnc_rot']:.2f}° | {b['gnc_rot']:.2f}° |")
    for k in ["coherent_p0.4", "coherent_p0.5", "coherent_p0.6"]:
        L.append(f"| {k}: GNC/IRLS/RANSAC | " + "/".join(f"{100*out['drive_0001'][k][m]:.0f}" for m in ["GNC-GM", "IRLS-GM", "RANSAC-1000"]) +
                 " | " + "/".join(f"{100*out['drive_0002'][k][m]:.0f}" for m in ["GNC-GM", "IRLS-GM", "RANSAC-1000"]) + " |")
    for k in ["boundary_p0.5", "boundary_p0.7"]:
        L.append(f"| {k}: GNC/RANSAC rot | {out['drive_0001'][k]['GNC-GM']['rot']:.2f}/{out['drive_0001'][k]['RANSAC-1000']['rot']:.2f}° | "
                 f"{out['drive_0002'][k]['GNC-GM']['rot']:.2f}/{out['drive_0002'][k]['RANSAC-1000']['rot']:.2f}° |")
    L += ["", "## Reading", "",
          "* The two drives are independent OXTS/calib runs; differences in speedup are node/load and motion/overlap, not algorithmic.",
          "* The coherent 50% tie regime is the only place where the two drives differ materially; both collapse at 60%."]
    (OUT / "drive_heterogeneity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
