"""Fig. 3b experiment: convergence of the local solver from 100 random initialisations.

The paper's Fig. 3b shows 100 random initialisations of Theseus' Gauss-Newton
solver for one stereo-localisation problem (one ground-truth pose).  Roughly
60% of them converge to a poor local minimum (red) and the rest to the global
minimum (green).

Here we repeat that experiment with the clean-room local solver:

* ``n_datasets`` independent problem instances (pose + noise realisations),
* 100 random initialisations each, drawn exactly as in the paper (positions
  uniform on the 3 m sphere around the board, optical axis pointing at the
  board centre, random rotation about the optical axis),
* the local solver is run with the paper's Gauss-Newton configuration
  (step size 0.2, 500 iterations, relative tolerance 1e-6) followed by a
  tight polish to a stationary point,
* a solution is classified as a *local minimum* if its inner cost exceeds
  the best cost by more than 5 (the threshold used by the upstream script),
  and as a *failure* if the returned point is not stationary.

Outputs: ``repro/results_clean/local_minima.json`` and ``local_minima.npz``
(poses for the 3-D figure), plus a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bilevel  # noqa: E402
import data_gen  # noqa: E402
import implicit_diff as idiff  # noqa: E402
import local_solver as ls  # noqa: E402
import sdp_pose  # noqa: E402


def run_dataset(cfg, dataset_idx, n_inits=100, verbose=False):
    rng = np.random.default_rng(1234 + dataset_idx)
    cam = bilevel.make_camera(cfg)
    data = data_gen.get_calibration_data(
        n_batch=1,
        board_dims=cfg.board_dims,
        n_squares=cfg.n_squares,
        radius=cfg.radius,
        beta=cfg.beta,
        alpha=cfg.alpha,
        cam=cam,
        b_true=cfg.b_true,
        rng=rng,
        noise_mode=cfg.noise_mode,
    )
    m = data["r_l"]
    pixels = data["pixels"][0]
    C_gt = data["C_p0s"][0]
    r_gt = data["r_p0s"][0]
    meas, W = cam.inverse(pixels, cfg.b_true)

    # global optimum from the SDP relaxation (reference)
    Q = sdp_pose.build_objective_matrix(m, meas, W)
    solver = sdp_pose.SDPPoseSolver(
        solver=cfg.sdp_solver, eps=cfg.sdp_eps, max_iters=cfg.sdp_max_iters
    )
    rec = solver.solve(sdp_pose.scale_objective(Q, sdp_pose.objective_scale(Q)))
    C_sdp, r_sdp = idiff.pose_from_x(rec["x"])
    e = (meas - C_sdp @ m + (C_sdp @ r_sdp)[:, None]).T
    cost_global = float(np.einsum("nij,ni,nj->", W, e, e))

    r_init, C_init = data_gen.get_random_poses(
        radius=cfg.radius, n_batch=n_inits, rng=rng
    )
    costs = np.zeros(n_inits)
    stationary = np.zeros(n_inits, dtype=bool)
    pos_err = np.zeros(n_inits)
    rot_err = np.zeros(n_inits)
    Cs = np.zeros((n_inits, 3, 3))
    rs = np.zeros((n_inits, 3))
    for i in range(n_inits):
        res = ls.solve_pose_local(
            m, meas, W, C_init[i], r_init[i],
            max_iter=cfg.gn_max_iter, step_size=cfg.gn_step_size,
            abs_tol=cfg.gn_abs_tol, rel_tol=cfg.gn_rel_tol,
        )
        res = ls.solve_pose_local(
            m, meas, W, res["C"], res["r_p"], max_iter=cfg.gn_polish_iter,
            step_size=1.0, abs_tol=1e-14, rel_tol=1e-14,
        )
        costs[i] = res["cost"]
        stationary[i] = res["stationary"]
        pos_err[i] = np.linalg.norm(res["r_p"] - r_gt.reshape(3))
        rot_err[i] = np.linalg.norm(res["C"].T @ C_gt - np.eye(3))
        Cs[i] = res["C"]
        rs[i] = res["r_p"]
        if verbose and (i + 1) % 20 == 0:
            print(f"    init {i+1}/{n_inits}")

    # classification (upstream threshold: |cost - cost_min| > 5)
    cost_min = max(cost_global, costs.min())
    is_local = np.abs(costs - cost_min) > 5.0
    is_failure = ~stationary
    # "global" = converged to the global optimum (cost close to the SDP cost)
    is_global = (np.abs(costs - cost_global) < 5.0) & stationary
    out = dict(
        dataset=dataset_idx,
        n_inits=n_inits,
        cost_global=cost_global,
        cost_min=float(cost_min),
        n_local=int(np.sum(is_local)),
        n_global=int(np.sum(is_global)),
        n_failure=int(np.sum(is_failure)),
        frac_local=float(np.mean(is_local)),
        frac_global=float(np.mean(is_global)),
        frac_failure=float(np.mean(is_failure)),
        cost_local_min=float(costs.min()),
        cost_local_max=float(costs.max()),
    )
    return out, dict(
        r_init=r_init, C_init=C_init, r_final=rs, C_final=Cs,
        costs=costs, is_local=is_local, is_global=is_global,
        is_failure=is_failure, C_gt=C_gt, r_gt=r_gt, m=m,
        cost_global=cost_global,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=3)
    ap.add_argument("--inits", type=int, default=100)
    ap.add_argument("--noise-mode", default="independent")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir or os.path.join(here, "..", "results_clean"))
    os.makedirs(out_dir, exist_ok=True)

    cfg = bilevel.ExpConfig(n_batch=1, noise_mode=args.noise_mode)
    summary = []
    for d in range(args.datasets):
        print(f"dataset {d}: {args.inits} random initialisations ...", flush=True)
        out, raw = run_dataset(cfg, d, n_inits=args.inits, verbose=True)
        summary.append(out)
        print(f"    local minima {out['n_local']}/{out['n_inits']} "
              f"({100*out['frac_local']:.1f}%), global {out['n_global']}, "
              f"failures {out['n_failure']}")
        np.savez(
            os.path.join(out_dir, f"local_minima_dataset{d}_{args.noise_mode}.npz"),
            **{k: np.asarray(v) for k, v in raw.items()},
        )
    # aggregate
    tot_inits = sum(s["n_inits"] for s in summary)
    agg = dict(
        datasets=summary,
        n_inits_total=tot_inits,
        frac_local_mean=float(np.mean([s["frac_local"] for s in summary])),
        frac_global_mean=float(np.mean([s["frac_global"] for s in summary])),
        frac_failure_mean=float(np.mean([s["frac_failure"] for s in summary])),
        n_local_total=sum(s["n_local"] for s in summary),
        n_global_total=sum(s["n_global"] for s in summary),
        n_failure_total=sum(s["n_failure"] for s in summary),
        noise_mode=args.noise_mode,
        config=dict(cfg.__dict__),
    )
    with open(os.path.join(out_dir, f"local_minima_{args.noise_mode}.json"), "w") as f:
        json.dump(agg, f, indent=2, default=float)
    with open(os.path.join(out_dir, f"local_minima_{args.noise_mode}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_inits", "n_local", "n_global", "n_failure",
                    "frac_local", "frac_global", "frac_failure", "cost_global"])
        for s in summary:
            w.writerow([s["dataset"], s["n_inits"], s["n_local"], s["n_global"],
                        s["n_failure"], s["frac_local"], s["frac_global"],
                        s["frac_failure"], s["cost_global"]])
    print(json.dumps({k: v for k, v in agg.items()
                      if k not in ("datasets", "config")}, indent=2))
    print(f"wrote {out_dir}/local_minima_{args.noise_mode}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
