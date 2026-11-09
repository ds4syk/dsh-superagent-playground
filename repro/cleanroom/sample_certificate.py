"""Sample SDP certificate spectra across many problem instances.

The main experiment stored only aggregate diagnostics per iteration, and the
"second eigenvalue of the certificate" diagnostic was originally computed
with the wrong index (second *largest* instead of second *smallest*).  This
small script re-solves a sample of the calibration problems and records the
certificate spectrum correctly, so that the tightness figure and the report
use a trustworthy statistic.

Output: ``repro/results_clean/certificate_sample.json``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bilevel  # noqa: E402
import sdp_pose  # noqa: E402


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=5)
    ap.add_argument("--noise-mode", default="independent")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir or os.path.join(here, "..", "results_clean"))
    os.makedirs(out_dir, exist_ok=True)

    cfg = bilevel.ExpConfig(n_batch=20, noise_mode=args.noise_mode)
    cam = bilevel.make_camera(cfg)
    solver = sdp_pose.SDPPoseSolver(solver=cfg.sdp_solver, eps=cfg.sdp_eps)
    rows = []
    for ds in range(args.datasets):
        data = bilevel.make_dataset(cfg, ds)
        m = data["r_l"]
        for k in range(cfg.n_batch):
            for delta in (0.0, 0.002, -0.002):
                b = cfg.b_true + cfg.b_offset + delta
                meas, W = cam.inverse(data["pixels"][k], b)
                Q = sdp_pose.build_objective_matrix(m, meas, W)
                sc = sdp_pose.objective_scale(Q)
                rec = solver.solve(sdp_pose.scale_objective(Q, sc))
                e = np.linalg.eigvalsh(rec["H"])
                scale = max(abs(e[-1]), 1e-300)
                rows.append(
                    dict(
                        dataset=ds,
                        pose=k,
                        b=b,
                        r_tight=rec["r_tight"],
                        corank_1e6=int(np.sum(np.abs(e) < 1e-6 * scale)),
                        corank_1e9=int(np.sum(np.abs(e) < 1e-9 * scale)),
                        eig2_ratio=float(abs(e[1]) / scale),
                        H_min_eig=float(e[0]),
                        Hx_res=float(rec["Hx_res"]),
                        duality_gap=float(abs(rec["primal_obj"] - rec["dual_obj"])),
                    )
                )
        print(f"dataset {ds+1}/{args.datasets} done ({len(rows)} solves)")

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                    median=float(np.median(v)), min=float(np.min(v)),
                    max=float(np.max(v)),
                    frac_one=float(np.mean(v == 1)) if key.startswith("corank") else None)

    summary = dict(
        n_solves=len(rows),
        noise_mode=args.noise_mode,
        config=dict(b_true=cfg.b_true, b_offset=cfg.b_offset,
                    datasets=args.datasets, n_batch=cfg.n_batch),
        r_tight=agg("r_tight"),
        corank_1e6=agg("corank_1e6"),
        corank_1e9=agg("corank_1e9"),
        eig2_ratio=agg("eig2_ratio"),
        H_min_eig=agg("H_min_eig"),
        Hx_res=agg("Hx_res"),
        duality_gap=agg("duality_gap"),
        rows=rows,
    )
    path = os.path.join(out_dir, "certificate_sample.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
