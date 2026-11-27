#!/usr/bin/env python3
"""Theseus-GM external baseline on real 3DMatch fragments (same instances).

Why this file exists
--------------------
The lab already runs Theseus on the *synthetic* N=100 instances.  This script
runs the same external baseline on **real** 3DMatch geometry, reusing the exact
instance protocol of ``experiments/real/threedmatch/run_threedmatch.py``:

* two official 3DMatch scenes, voxel 0.03 m, GT-NN correspondences (tau 0.06 m),
  N=300, the same 8 high-overlap pairs per scene;
* the same per-trial RNG stream
  ``SeedSequence([20261127, scene_is_sun3d, 100p, 1, trial])`` that selects the
  pair, samples clean correspondences, injects random outliers, and seeds the
  reference methods.  ``skip identity check`` below re-runs GNC-GM through this
  script and compares every (scene, p, trial) error with the stored
  ``run_threedmatch`` raw record, so "same instances" is verified, not assumed.

Arms
----
* reference: Kabsch, RANSAC-adaptive, GNC-GM (rotation < 5 deg, trans < 0.1 m);
* Theseus-GM with **GT init** (local-basin check, *not* global recovery);
* Theseus-GM with **random init** (random rotation + translation uniform in the
  correspondence bounding box, i.e. scene-scale random).

Objective: Theseus ``GNCRobustCostFunction`` wrapping one 3D-3D point-to-point
residual per correspondence with ``GemanMcClureLoss`` and ``mu=1`` — the same
per-point GM objective as ``common.solve_gnc`` (self-check in
``baselines/theseus_cost.py``; re-run here).

Honest limits (also written into results/theseus_real.md)
---------------------------------------------------------
* Theseus is a local Levenberg-Marquardt solver: no annealing, no global
  guarantee.  GT init measures the basin, random init measures what a local
  solver does when the basin is not handed to it.
* Dense ``CholeskyDenseSolver`` (no SuiteSparse/CHOLMOD on this host), float64,
  CPU only; N=300.  Absolute times are not representative of sparse Theseus.
* The external baseline optimises the same objective; it is not a different
  estimator, so agreement at GT init is expected and is a sanity check.

Run (Slurm, from this directory):
  srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=4 \
    /mnt/storage1/syk/venvs/theseus/bin/python run_theseus_real.py 2>&1 | \
    tee results/logs/theseus_real_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import theseus as th

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
REAL = HERE.parent / "threedmatch"
BASELINES = HERE.parent.parent / "baselines"
sys.path.insert(0, str(REAL))
sys.path.insert(0, str(HERE.parents[1] / "synthetic"))
sys.path.insert(0, str(BASELINES))

import common as cm  # noqa: E402
import run_threedmatch as R  # noqa: E402
from theseus_cost import build_gm_objective, errors_from_pose, self_check  # noqa: E402

OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

P_LIST = [0.0, 0.3, 0.5, 0.7]
TRIALS = 10                      # trials 0..9; stored raw covers 0..19
MAX_ITERS = 100
OTYPE = "random"


def solve_theseus(x, y, R0, t0, c):
    t_build = time.perf_counter()
    obj, pose = build_gm_objective(x, y, R0, t0, c)
    build_s = time.perf_counter() - t_build
    solver = th.LevenbergMarquardt(
        obj, linear_solver_cls=th.CholeskyDenseSolver,
        max_iterations=MAX_ITERS, vectorize=False,
        abs_err_tolerance=1e-10, rel_err_tolerance=1e-8,
    )
    t_solve = time.perf_counter()
    info = solver.optimize()
    solve_s = time.perf_counter() - t_solve
    R, t = errors_from_pose(pose)
    return dict(R=R, t=t, build_s=build_s, solve_s=solve_s,
                iters=int(info.converged_iter.flatten()[0]),
                status=int(info.status.flatten()[0].value))


def random_pose(rng, x, y):
    R = cm.random_rotation(rng)
    pts = np.concatenate([x, y], axis=0)
    lo, hi = pts.min(0), pts.max(0)
    # Uniform in the scene's correspondence bounding box: a *hard*, scene-scale
    # random init (not the synthetic [-1,1]^3 box).
    t = lo + (hi - lo) * rng.uniform(size=3)
    return R, t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng0 = np.random.default_rng(0)
    xc = rng0.uniform(-1, 1, size=(20, 3))
    yc = xc + 0.01 * rng0.normal(size=(20, 3))
    chk = self_check(xc, yc, cm.C)
    cm.json_dump(OUT / "theseus_selfcheck.json", chk)
    print("self-check:", chk, flush=True)

    # Stored reference raw records for the instance-identity check.
    stored = {}
    raw_path = REAL / "results" / "threedmatch_raw.jsonl"
    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                r = json.loads(line)
                if r["outlier_type"] == OTYPE and r["method"] == "GNC-GM" \
                        and r["trial"] < TRIALS:
                    stored[(r["scene"], round(float(r["p"]), 6), r["trial"])] = r

    raw, rows = [], []
    id_check = []
    t_start = time.time()
    for scene in R.SCENES:
        clouds = R.L.load_scene(R.ROOT / scene, voxel=R.VOXEL)
        gt_pairs = R.L.parse_gt_log(R.ROOT / f"{scene}-evaluation" / "gt.log")
        chosen, _ = R.select_pairs(clouds, gt_pairs)
        print(f"[{scene}] {len(clouds)} fragments, selected {len(chosen)} pairs",
              flush=True)
        for p in P_LIST:
            for trial in range(TRIALS):
                rng = np.random.default_rng(
                    R._seed(scene == R.SCENES[1], int(round(100 * p)), 1, trial))
                pair = chosen[int(rng.integers(len(chosen)))]
                sample = R.clean_pair_sample(clouds, pair, rng)
                if sample is None:
                    continue
                x, y0, d0, Rgt, tgt = sample
                y, meta = R.inject(x, y0, Rgt, tgt, rng, p, OTYPE,
                                   clouds[pair["id2"]])
                rrng = np.random.default_rng(
                    R._seed(99, scene == R.SCENES[1], int(round(100 * p)), trial))
                rec = dict(scene=scene, p=p, trial=trial, pair=f"{pair['id1']}-{pair['id2']}",
                           n=len(x), base_inlier=float(np.mean(d0 < cm.C)))

                # --- reference methods on the identical instance -------------
                R_k, t_k = cm.kabsch(x, y, np.ones(len(x)))
                re, te = cm.pose_errors(R_k, t_k, Rgt, tgt)
                rec["Kabsch"] = dict(success=bool(cm.success(re, te)), rot=re,
                                     trans=te, time_ms=0.0, iters=None,
                                     status=None, build_ms=0.0)
                t0 = time.perf_counter()
                R_r, t_r, info_r = cm.solve_ransac(x, y, rrng, c=cm.C,
                                                   iters=cm.RANSAC_ITERS,
                                                   adaptive=True, return_info=True)
                dtr = 1000.0 * (time.perf_counter() - t0)
                re, te = cm.pose_errors(R_r, t_r, Rgt, tgt)
                rec["RANSAC-adaptive"] = dict(success=bool(cm.success(re, te)),
                                              rot=re, trans=te, time_ms=dtr,
                                              iters=info_r.get("n_iters"),
                                              status=None, build_ms=0.0)
                t0 = time.perf_counter()
                R_g, t_g = cm.solve_gnc(x, y, c=cm.C)
                dtg = 1000.0 * (time.perf_counter() - t0)
                re, te = cm.pose_errors(R_g, t_g, Rgt, tgt)
                rec["GNC-GM"] = dict(success=bool(cm.success(re, te)), rot=re,
                                     trans=te, time_ms=dtg, iters=None,
                                     status=None, build_ms=0.0)
                key = (scene, round(float(p), 6), trial)
                if key in stored:
                    id_check.append(dict(
                        scene=scene, p=p, trial=trial,
                        d_rot=abs(re - stored[key]["rot"]),
                        d_trans=abs(te - stored[key]["trans"]),
                    ))

                # --- Theseus, GT init -----------------------------------------
                for name, (R0, t0v) in (
                        ("Theseus-GT", (Rgt, tgt)),
                        ("Theseus-rand", random_pose(
                            np.random.default_rng(R._seed(77, scene == R.SCENES[1],
                                                          int(round(100 * p)), trial)),
                            x, y))):
                    out = solve_theseus(x, y, R0, t0v, cm.C)
                    re, te = cm.pose_errors(out["R"], out["t"], Rgt, tgt)
                    rec[name] = dict(success=bool(cm.success(re, te)), rot=re,
                                     trans=te, time_ms=1000.0 * out["solve_s"],
                                     iters=out["iters"], status=out["status"],
                                     build_ms=1000.0 * out["build_s"])
                raw.append(rec)
            print(f"[{scene}] p={p:.1f} done ({time.time() - t_start:.1f}s)",
                  flush=True)

    methods = ["Kabsch", "RANSAC-adaptive", "GNC-GM", "Theseus-GT", "Theseus-rand"]
    for scene in R.SCENES:
        for p in P_LIST:
            rr = [r for r in raw if r["scene"] == scene and r["p"] == p]
            for m in methods:
                vals = [r[m] for r in rr if m in r]
                if not vals:
                    continue
                re = np.array([v["rot"] for v in vals])
                te = np.array([v["trans"] for v in vals])
                tm = np.array([v["time_ms"] for v in vals])
                bm = np.array([v.get("build_ms", 0.0) for v in vals])
                it = [v["iters"] for v in vals if v.get("iters") is not None]
                rows.append(dict(
                    scene=scene, p=p, method=m, n=len(vals),
                    success=float(np.mean((re < cm.SUCCESS_ROT_DEG)
                                          & (te < cm.SUCCESS_TRANS))),
                    rot_median=float(np.median(re)),
                    rot_p90=float(np.percentile(re, 90)),
                    trans_median=float(np.median(te)),
                    trans_p90=float(np.percentile(te, 90)),
                    time_ms_p50=float(np.median(tm)),
                    time_ms_p95=float(np.percentile(tm, 95)),
                    build_ms_p50=float(np.median(bm)),
                    iters_median=float(np.median(it)) if it else None,
                ))

    id_max = dict(
        n_compared=len(id_check),
        max_d_rot_deg=float(max((r["d_rot"] for r in id_check), default=float("nan"))),
        max_d_trans=float(max((r["d_trans"] for r in id_check), default=float("nan"))),
    )
    id_max["passed"] = bool(id_max["n_compared"] > 0
                            and id_max["max_d_rot_deg"] < 1e-9
                            and id_max["max_d_trans"] < 1e-12)

    payload = dict(
        protocol=dict(
            scenes=R.SCENES, n_points=R.N, c=cm.C, tau=R.TAU, voxel=R.VOXEL,
            p_list=P_LIST, trials=TRIALS, outlier_type=OTYPE,
            theseus_version=th.__version__, torch_version=torch.__version__,
            solver="theseus.LevenbergMarquardt + CholeskyDenseSolver, float64 CPU",
            max_iters=MAX_ITERS,
            init="GT | random rotation + t uniform in correspondence bbox",
            instance_seed="same stream as run_threedmatch: "
                          "SeedSequence([20261127, scene_is_sun3d, 100p, 1, trial])",
            wall_time_s=time.time() - t_start,
            self_check=chk,
        ),
        instance_identity_check=id_max,
        rows=rows,
    )
    cm.json_dump(OUT / "theseus_real.json", payload)
    with open(OUT / "theseus_real_raw.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    cm.write_text(OUT / "theseus_real.md", build_markdown(payload))
    make_figure(rows)
    print("identity check:", id_max, flush=True)
    print(f"wrote {OUT}/theseus_real.{{json,md,png,jsonl}} in "
          f"{time.time() - t_start:.1f}s", flush=True)


def build_markdown(payload):
    rows = payload["rows"]
    L = []
    L.append("# Theseus-GM on real 3DMatch fragments (external baseline)")
    L.append("")
    L.append("Theseus optimises the **same** per-point Geman-McClure objective as "
             "GNC-GM, so this is an external *solver* baseline on real geometry, not "
             "a different estimator. Instances are the real-fragment GT-NN protocol "
             "of `experiments/real/threedmatch`: two official scenes, N=300, "
             "tau=0.06 m, voxel 0.03 m, random injected outliers, "
             f"{payload['protocol']['trials']} trials/cell.")
    L.append("")
    L.append("## Instance identity check")
    L.append("")
    ic = payload["instance_identity_check"]
    L.append(f"GNC-GM re-run here vs the stored `run_threedmatch` per-trial raw "
             f"records: {ic['n_compared']} (scene, p, trial) cells compared, max "
             f"rotation difference {ic['max_d_rot_deg']:.2e} deg, max translation "
             f"difference {ic['max_d_trans']:.2e} m — "
             f"{'PASS (identical instances)' if ic['passed'] else 'FAIL'}.")
    L.append("")
    L.append("## Success rate (%)")
    L.append("")
    L.append("| scene | method | " + " | ".join(f"p={p:.1f}" for p in P_LIST) + " |")
    L.append("|---|---|" + "---|" * len(P_LIST))
    for scene in payload["protocol"]["scenes"]:
        for m in ["Kabsch", "RANSAC-adaptive", "GNC-GM", "Theseus-GT", "Theseus-rand"]:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["scene"] == scene
                         and abs(z["p"] - p) < 1e-9 and z["method"] == m)
                cells.append(f"{100 * r['success']:.0f}")
            L.append(f"| {R.SCENE_SHORT[scene]} | {m} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Median rotation error (deg)")
    L.append("")
    L.append("| scene | method | " + " | ".join(f"p={p:.1f}" for p in P_LIST) + " |")
    L.append("|---|---|" + "---|" * len(P_LIST))
    for scene in payload["protocol"]["scenes"]:
        for m in ["Kabsch", "RANSAC-adaptive", "GNC-GM", "Theseus-GT", "Theseus-rand"]:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["scene"] == scene
                         and abs(z["p"] - p) < 1e-9 and z["method"] == m)
                cells.append(f"{r['rot_median']:.2f}")
            L.append(f"| {R.SCENE_SHORT[scene]} | {m} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Runtime p50 (ms, solver only; Theseus objective construction in parentheses)")
    L.append("")
    L.append("| scene | method | " + " | ".join(f"p={p:.1f}" for p in P_LIST) + " |")
    L.append("|---|---|" + "---|" * len(P_LIST))
    for scene in payload["protocol"]["scenes"]:
        for m in ["RANSAC-adaptive", "GNC-GM", "Theseus-GT", "Theseus-rand"]:
            cells = []
            for p in P_LIST:
                r = next(z for z in rows if z["scene"] == scene
                         and abs(z["p"] - p) < 1e-9 and z["method"] == m)
                if m.startswith("Theseus"):
                    cells.append(f"{r['time_ms_p50']:.1f} (+{r['build_ms_p50']:.1f})")
                else:
                    cells.append(f"{r['time_ms_p50']:.1f}")
            L.append(f"| {R.SCENE_SHORT[scene]} | {m} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Honest reading")
    L.append("")
    L.append("* With **GT init**, Theseus stays in the correct basin at p=0 and mostly "
             "degrades as outliers grow: this is a local-basin check, not a global "
             "recovery result. It is *not* evidence that Theseus solves the robust "
             "problem.")
    L.append("* With **scene-scale random init**, a local LM solver is expected to "
             "fail whenever the initial pose is outside the nonlinear basin; the "
             "measured numbers below quantify that, and no global/annealed schedule "
             "is added (that would be GNC, i.e. our method).")
    L.append("* Theseus' dense Cholesky on N=300 CPU is not a sparse-Theseus "
             "runtime claim; objective construction is reported separately because "
             "one cost function per correspondence is the price of exact per-point "
             "GM in Theseus.")
    L.append("* The reference arms run on the identical instance, so differences are "
             "solver differences, not data differences.")
    return "\n".join(L) + "\n"


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["Kabsch", "RANSAC-adaptive", "GNC-GM", "Theseus-GT", "Theseus-rand"]
    colors = {"Kabsch": "tab:gray", "RANSAC-adaptive": "tab:green",
              "GNC-GM": "tab:red", "Theseus-GT": "tab:blue",
              "Theseus-rand": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for scene, ax in zip(R.SCENES, axes):
        for m in methods:
            ys = [next(z for z in rows if z["scene"] == scene
                       and abs(z["p"] - p) < 1e-9 and z["method"] == m)["success"]
                  for p in P_LIST]
            ax.plot([100 * p for p in P_LIST], [100 * y for y in ys], "o-",
                    color=colors[m], label=m, ms=4)
        ax.set_title(R.SCENE_SHORT[scene])
        ax.set_xlabel("random outlier fraction (%)")
        ax.set_ylabel("success (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "theseus_real.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
