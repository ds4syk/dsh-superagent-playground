#!/usr/bin/env python3
"""P1-Q3 robustness: 40 projective roots across more random instances.

Runs the total-degree homotopy of ``roots/run_rootcount_fast.py`` on
10 random symmetric M + the instance M(0) + one real planar-frame M, and
reports the projective-root count, real/complex split, local-minima count,
residuals and runtime.  Read-only w.r.t. the existing results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_rootcount_fast as R  # noqa: E402
import numpy as np  # noqa: E402
from instance import Instance  # noqa: E402

OUT = HERE.parent / "results" / "rootcount_many.json"


def main():
    I = Instance(seed=20261127)
    cases = []
    for s in range(1, 11):
        M = R.random_symmetric_M(s)
        t0 = time.time()
        r = R.run_case(f"random_seed{s}", M, I, chart_seed=7)
        cases.append({"name": r.get("name", f"random_seed{s}"),
                      "n_projective": r["n_projective_roots"],
                      "n_affine": r["n_affine_roots"],
                      "n_real": r["n_real_projective"],
                      "n_complex": r["n_complex_projective"],
                      "n_minima": r["minima"]["n_minima"],
                      "max_chart_residual": r["max_chart_residual"],
                      "runtime_s": time.time() - t0})
        print(cases[-1], flush=True)

    r = R.run_case("instance_M_theta0", I.M(0.0), I, chart_seed=7)
    cases.append({"name": "instance_M_theta0", "n_projective": r["n_projective_roots"],
                  "n_affine": r["n_affine_roots"], "n_real": r["n_real_projective"],
                  "n_complex": r["n_complex_projective"], "n_minima": r["minima"]["n_minima"],
                  "max_chart_residual": r["max_chart_residual"],
                  "runtime_s": None})
    proj = [c["n_projective"] for c in cases]
    out = {"n_cases": len(cases),
           "all_40": all(p == 40 for p in proj),
           "n_40": sum(1 for p in proj if p == 40),
           "max_chart_residual": max(c["max_chart_residual"] for c in cases),
           "cases": cases,
           "command": "python roots/run_rootcount_many.py"}
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, "| all 40:", out["all_40"], "| n40:", out["n_40"], flush=True)


if __name__ == "__main__":
    main()
