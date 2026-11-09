"""Resource / timing benchmark for the clean-room calibration experiment.

Each method is run in its own child process so that the peak resident set
size (``ru_maxrss``) can be attributed to that method.  A short warm-up run
is performed to measure the one-off CVXPY canonicalisation cost of the SDP
layer separately from the steady-state solve time.

Outputs
-------
``repro/results_clean/benchmark.csv``   one row per method
``repro/results_clean/benchmark.json``  machine-readable details

Usage
-----
    python benchmark.py --runs 3 --batch 20 --noise-mode independent
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METHODS = ["sdpr", "local_gt", "local_rnd"]


# ----------------------------------------------------------------------
# child process: run trials and report
# ----------------------------------------------------------------------
def child_worker(args):
    import bilevel
    import data_gen
    import implicit_diff as idiff
    import local_solver as ls
    import sdp_pose
    from camera_model import StereoCamera

    cfg = bilevel.ExpConfig(
        n_batch=args.batch,
        n_runs=args.runs,
        noise_mode=args.noise_mode,
        seed0=args.seed0,
        max_iter=args.max_iter,
    )
    rss0_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"method": args.method, "runs": [],
           "rss_after_import_mb": rss0_kb / 1024.0}

    # ---- one-off SDP layer setup cost (forward canonicalisation) -----
    if args.method == "sdpr":
        data = bilevel.make_dataset(cfg, 0)
        cam = bilevel.make_camera(cfg)
        b = cfg.b_true + cfg.b_offset
        m = data["r_l"]
        meas, W = cam.inverse(data["pixels"][0], b)
        Q = sdp_pose.build_objective_matrix(m, meas, W)
        sc = sdp_pose.objective_scale(Q)
        t0 = time.perf_counter()
        solver = sdp_pose.SDPPoseSolver(solver=cfg.sdp_solver, eps=cfg.sdp_eps)
        rec = solver.solve(sdp_pose.scale_objective(Q, sc))
        t_cold = time.perf_counter() - t0
        # warm solve / backward timings
        qs = sdp_pose.scale_objective(Q, sc)
        t0 = time.perf_counter()
        for _ in range(20):
            rec = solver.solve(qs)
        t_warm = (time.perf_counter() - t0) / 20
        loss, grad_x = idiff.loss_from_x_and_grad(rec["x"], data["C_p0s"][0],
                                                  data["r_p0s"][0])
        dQ = bilevel._dQ_db_complex_step(cam, data["pixels"][0], m, b) * sc
        dQ[0, 0] = 0.0
        G_r, _ = solver.reduced_gradient_rows(rec["x"])
        t0 = time.perf_counter()
        for _ in range(50):
            idiff.qcqp_backward(rec["H"], G_r, rec["x"], dQ, grad_x)
        t_bwd = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        for _ in range(20):
            bilevel._dQ_db_complex_step(cam, data["pixels"][0], m, b)
        t_dq = (time.perf_counter() - t0) / 20
        Qbuild = sdp_pose.build_objective_matrix  # noqa: F841
        t0 = time.perf_counter()
        for _ in range(20):
            Qbuild(m, meas, W)
        t_qbuild = (time.perf_counter() - t0) / 20
        out["setup"] = dict(
            t_cold_first_solve_s=t_cold,
            t_warm_solve_s=t_warm,
            t_backward_s=t_bwd,
            t_qbuild_s=t_qbuild,
            t_dQ_db_complex_step_s=t_dq,
            n_vars=sdp_pose.N_VARS,
            n_equality_constraints=len(solver.A_list) + 1,
            X_dim=sdp_pose.N_VARS,
            r_tight=rec["r_tight"],
            corank_H=rec["corank_H"],
            corank_H_strict=rec["corank_H_strict"],
            H_min_eig=rec["H_min_eig"],
            eig2_eig1=rec["eig2_eig1"],
            Hx_res=rec["Hx_res"],
            primal=rec["primal_obj"],
            dual=rec["dual_obj"],
        )
        # ---- certificate / tightness statistics over 20 poses --------
        stats = dict(r_tight=[], corank=[], corank_strict=[], H_min_eig=[],
                     eig2_eig1=[], Hx_res=[], t_solve=[], gap=[])
        for k in range(cfg.n_batch):
            meas_k, W_k = cam.inverse(data["pixels"][k], b)
            Qk = sdp_pose.build_objective_matrix(m, meas_k, W_k)
            sck = sdp_pose.objective_scale(Qk)
            rec_k = solver.solve(sdp_pose.scale_objective(Qk, sck))
            stats["r_tight"].append(rec_k["r_tight"])
            stats["corank"].append(rec_k["corank_H"])
            stats["corank_strict"].append(rec_k["corank_H_strict"])
            stats["H_min_eig"].append(rec_k["H_min_eig"])
            stats["eig2_eig1"].append(rec_k["eig2_eig1"])
            stats["Hx_res"].append(rec_k["Hx_res"])
            stats["t_solve"].append(rec_k["t_solve"])
            stats["gap"].append(abs(rec_k["primal_obj"] - rec_k["dual_obj"]))
        out["setup"]["certificate_stats"] = {
            k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                    min=float(np.min(v)), max=float(np.max(v)))
            for k, v in stats.items()
        }
        out["setup"]["frac_rank1"] = float(np.mean(np.array(stats["r_tight"]) > 1e5))
        out["setup"]["frac_corank1"] = float(np.mean(np.array(stats["corank"]) == 1))
        out["setup"]["frac_corank1_strict"] = float(
            np.mean(np.array(stats["corank_strict"]) == 1))
        # per-pose forward/backward timing inside a real trial (from diagnostics)
        del solver

    for run in range(args.runs):
        data = bilevel.make_dataset(cfg, run)
        t0 = time.perf_counter()
        h = bilevel.run_trial(data, args.method, cfg)
        wall = time.perf_counter() - t0
        rec = dict(
            run=run,
            wall_time=wall,
            n_iter=h["n_iter"],
            final_err=float(h["err"][-1]),
            final_loss=float(h["loss"][-1]),
            status=h["status"],
            mean_iter_time=float(np.mean(h["time_iter"])),
            max_iter_time=float(np.max(h["time_iter"])),
        )
        agg = {}
        if h["diag"]:
            for key in h["diag"][0]:
                vals = np.concatenate([np.ravel(d[key]) for d in h["diag"]])
                agg[key] = dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                                max=float(np.max(vals)), total=float(np.sum(vals)))
        rec["diag"] = agg
        out["runs"].append(rec)

    out["rss_peak_self_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("BENCH_JSON_BEGIN")
    print(json.dumps(out))
    print("BENCH_JSON_END")
    return 0


# ----------------------------------------------------------------------
# parent: spawn children, collect peak RSS
# ----------------------------------------------------------------------
def run_child(args_list):
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker"] + args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid, status, ru = os.wait4(proc.pid, 0)
    stdout, stderr = proc.communicate()
    maxrss_mb = ru.ru_maxrss / 1024.0  # Linux: kilobytes
    payload = None
    if "BENCH_JSON_BEGIN" in stdout:
        s = stdout.split("BENCH_JSON_BEGIN", 1)[1].split("BENCH_JSON_END", 1)[0]
        payload = json.loads(s)
    else:
        sys.stderr.write(stderr[-4000:])
        raise RuntimeError(f"child failed (status={status}): {stdout[-2000:]}")
    # prefer the value reported by the child itself (per-process maximum);
    # the wait4 rusage can be dominated by the interpreter/import footprint
    payload["peak_rss_mb"] = float(payload.get("rss_peak_self_mb", maxrss_mb))
    payload["peak_rss_wait4_mb"] = maxrss_mb
    payload["exit_status"] = status
    return payload


def cpu_info():
    model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return dict(
        cpu_model=model,
        n_cores=os.cpu_count(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS", "unset"),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    )


def env_versions():
    import cvxpy
    import numpy
    import scipy
    import torch

    out = dict(
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        cvxpy=cvxpy.__version__,
        torch=torch.__version__,
        torch_cuda=torch.cuda.is_available(),
    )
    try:
        import scs

        out["scs"] = getattr(scs, "__version__", "?")
    except ImportError:
        out["scs"] = "missing"
    try:
        import clarabel

        out["clarabel"] = getattr(clarabel, "__version__", "?")
    except ImportError:
        out["clarabel"] = "missing"
    return out


def gpu_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--method", default="sdpr")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--noise-mode", default="independent")
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if args.worker:
        return child_worker(args)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(args.out_dir or os.path.join(here, "..", "results_clean"))
    os.makedirs(out_dir, exist_ok=True)

    env = dict(cpu=cpu_info(), versions=env_versions(), gpu=gpu_info())
    print(json.dumps(env, indent=2))
    print(f"benchmark: runs={args.runs}, batch={args.batch}, noise={args.noise_mode}")

    results = {}
    for method in args.methods:
        print(f"--- benchmarking {method} ...", flush=True)
        payload = run_child([
            "--method", method,
            "--runs", str(args.runs),
            "--batch", str(args.batch),
            "--noise-mode", args.noise_mode,
            "--seed0", str(args.seed0),
            "--max-iter", str(args.max_iter),
        ])
        results[method] = payload
        runs = payload["runs"]
        print(f"    peak RSS {payload['peak_rss_mb']:.0f} MB, "
              f"wall/run {np.mean([r['wall_time'] for r in runs]):.2f}s, "
              f"iters {np.mean([r['n_iter'] for r in runs]):.1f}")

    # ---- table ----------------------------------------------------------
    rows = []
    for method, payload in results.items():
        runs = payload["runs"]
        wall = np.array([r["wall_time"] for r in runs])
        iters = np.array([r["n_iter"] for r in runs])
        it_time = np.array([r["mean_iter_time"] for r in runs])
        row = dict(
            method=method,
            n_runs=len(runs),
            n_batch=args.batch,
            noise_mode=args.noise_mode,
            wall_time_per_run_mean_s=float(np.mean(wall)),
            wall_time_per_run_std_s=float(np.std(wall)),
            wall_time_per_run_median_s=float(np.median(wall)),
            outer_iter_mean_s=float(np.mean(it_time)),
            n_iter_mean=float(np.mean(iters)),
            peak_rss_mb=float(payload["peak_rss_mb"]),
            peak_rss_wait4_mb=float(payload.get("peak_rss_wait4_mb", float("nan"))),
            rss_after_import_mb=float(payload.get("rss_after_import_mb", float("nan"))),
            n_failed_runs=int(sum(r["status"] == "diverged" for r in runs)),
        )
        # forward / backward timings from the per-iteration diagnostics
        def _m(key, field="mean"):
            vals = [r["diag"][key][field] for r in runs if key in r["diag"]]
            return float(np.mean(vals)) if vals else float("nan")

        row.update(
            fwd_solve_ms=1e3 * _m("t_solve"),
            backward_ms=1e3 * _m("t_backward"),
            r_tight_median=_m("r_tight", "median"),
            r_tight_min_median=float(
                min([r["diag"]["r_tight"]["median"] for r in runs
                     if "r_tight" in r["diag"]], default=float("nan"))),
            frac_rank1=float(np.mean([r["diag"]["r_tight"]["median"] > 1e5
                                      for r in runs if "r_tight" in r["diag"]])) if "r_tight" in runs[0]["diag"] else float("nan"),
            corank_H_max=_m("corank_H", "max"),
            corank_H_strict_max=_m("corank_H_strict", "max"),
            H_min_eig_median=_m("H_min_eig", "median"),
            eig2_eig1_median=_m("eig2_eig1", "median"),
            n_retry_total=_m("n_retry", "total") if "n_retry" in runs[0]["diag"] else 0.0,
            fallback_H_total=_m("fallback_H", "total") if "fallback_H" in runs[0]["diag"] else 0.0,
            gn_iter_mean=_m("n_gn_iter") if "n_gn_iter" in runs[0]["diag"] else float("nan"),
            natural_step_max=_m("natural_step", "max") if "natural_step" in runs[0]["diag"] else float("nan"),
            cond_max=_m("cond", "max") if "cond" in runs[0]["diag"] else float("nan"),
        )
        setup = payload.get("setup", {})
        cs = setup.get("certificate_stats", {})
        if cs:
            row["cert_frac_rank1"] = setup.get("frac_rank1", float("nan"))
            row["cert_frac_corank1"] = setup.get("frac_corank1", float("nan"))
            row["cert_frac_corank1_strict"] = setup.get("frac_corank1_strict", float("nan"))
            row["cert_eig2_eig1_median"] = cs["eig2_eig1"]["median"]
            row["cert_eig2_eig1_max"] = cs["eig2_eig1"]["max"]
            row["cert_H_min_eig_min"] = cs["H_min_eig"]["min"]
            row["cert_Hx_res_max"] = cs["Hx_res"]["max"]
            row["cert_duality_gap_max"] = cs["gap"]["max"]
            row["cert_solve_ms_median"] = 1e3 * cs["t_solve"]["median"]
        row.update(
            cold_first_solve_s=setup.get("t_cold_first_solve_s", float("nan")),
            warm_solve_s=setup.get("t_warm_solve_s", float("nan")),
            backward_s=setup.get("t_backward_s", float("nan")),
            q_build_s=setup.get("t_qbuild_s", float("nan")),
            dQ_db_s=setup.get("t_dQ_db_complex_step_s", float("nan")),
            sdp_n_vars=setup.get("n_vars", ""),
            sdp_n_equality_constraints=setup.get("n_equality_constraints", ""),
            sdp_X_dim=setup.get("X_dim", ""),
        )
        rows.append(row)

    import csv

    path = os.path.join(out_dir, "benchmark.csv")
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(dict(environment=env, config=vars(args), results=results,
                       table=rows), f, indent=2, default=float)
    print(f"\nwrote {path}")
    for r in rows:
        print({k: r[k] for k in (
            "method", "wall_time_per_run_mean_s", "outer_iter_mean_s",
            "fwd_solve_ms", "backward_ms", "peak_rss_mb", "r_tight_median",
            "frac_rank1", "corank_H_strict_max", "n_retry_total") if k in r})
    return 0


if __name__ == "__main__":
    sys.exit(main())
