#!/usr/bin/env python3
"""Paired check: SDPR (W_i=I, unit weights) vs Kabsch on the same instances.

SDPR's forward objective with identity inverse-covariance weights and unit
scalar weights is the ordinary least-squares 3D-3D objective, whose global
minimiser is Kabsch.  This script pairs the per-trial records of
``sdpr_forward_raw.jsonl`` (N=100) and ``reference_n20_raw.jsonl`` (Kabsch)
on (p, trial) and writes the max/mean absolute differences.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def main() -> None:
    kab = {}
    for line in open(RES / "reference_n20_raw.jsonl"):
        r = json.loads(line)
        if r["method"] == "Kabsch":
            kab[(r["p"], r["trial"])] = (r["rot_err_deg"], r["trans_err"])
    sdp = {}
    for line in open(RES / "sdpr_forward_raw.jsonl"):
        r = json.loads(line)
        if r["n"] == 100:
            sdp[(r["p"], r["trial"])] = (r["rot_err_deg"], r["trans_err"],
                                         r["ok"])
    keys = sorted(set(kab) & set(sdp))
    assert len(keys) == 80, len(keys)
    assert all(sdp[k][2] for k in keys), "SDPR failure in paired set"
    dr = np.array([abs(sdp[k][0] - kab[k][0]) for k in keys])
    dt = np.array([abs(sdp[k][1] - kab[k][1]) for k in keys])
    out = dict(
        n_pairs=len(keys),
        n=100,
        max_abs_rot_diff_deg=float(dr.max()),
        mean_abs_rot_diff_deg=float(dr.mean()),
        p99_abs_rot_diff_deg=float(np.percentile(dr, 99)),
        max_abs_trans_diff_m=float(dt.max()),
        mean_abs_trans_diff_m=float(dt.mean()),
        note="SDPR W_i=I, weights=1 is the same LS objective as Kabsch; the "
             "differences are SCS solve tolerance, not a modelling gap.",
    )
    with open(RES / "sdpr_kabsch_equiv.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
