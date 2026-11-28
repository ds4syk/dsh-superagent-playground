#!/usr/bin/env python3
"""GPU probe: eager-dynamic vs static-shape vs CUDA-graph 4x4 eigen GNC-GM.

Protocol (matches ``experiments/systems/run_batch.py`` semantics):

* batches are built with the P3 synthetic generator (``p=0.3`` random
  outliers, fixed seed), then transposed to the ``(B,3,N)`` torch layout;
* the NumPy batched reference is ``systems/batch_gnc.batch_solve_gnc(...,
  method="eigh")`` -- the same algorithm as the paper tables;
* four torch execution paths are timed on the same inputs:
  ``dynamic`` (eager active-set ``batch_gnc_torch.solve_gnc_t``),
  ``static-eigh`` / ``static-jacobi`` (fixed-shape masked loop, cuSOLVER or
  hand-written Jacobi 4x4), and the CUDA graph of each static solver;
* correctness is reported against the NumPy reference (max |dR|_F,
  max |dt|, per-instance ``n_mu`` equality) and against the ground truth
  (rotation degrees, translation metres, success = rot < 5 deg and
  |t| < 0.1 m), the paper's success criterion;
* float32 is run only to check stability; a dtype is flagged ``stable`` if
  the static-eigh solve matches the NumPy ``n_mu`` schedule on every
  instance, max |dR|_F < 1e-3, and the success count differs by <= 1
  instance.  Timings of an unstable dtype are kept but labelled.

Everything is written to ``results/qpep_gpu.{json,md,png}`` plus a log.
Run under Slurm:

  srun --partition=gpu --gres=gpu:1 --time=00:20:00 --ntasks=1 \
    --cpus-per-task=2 /home/syk/miniconda3/envs/WM/bin/python probe.py
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "systems"))
sys.path.insert(0, str(HERE.parent / "synthetic"))

import common as cm  # noqa: E402
import batch_gnc as bg  # noqa: E402

BASE_SEED = 20261127
P_OUT = 0.3


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", str(P3), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_batch(b, n, p, trial=0):
    """(B,N,3) float64 model/observation arrays and ground-truth poses."""
    ss = np.random.SeedSequence([BASE_SEED, 11, int(b), int(n),
                                 int(round(100 * p)), int(trial)])
    xs = np.empty((b, n, 3))
    ys = np.empty((b, n, 3))
    Rgts = np.empty((b, 3, 3))
    tgts = np.empty((b, 3))
    for i, child in enumerate(ss.spawn(b)):
        x, y, Rgt, tgt = cm.make_instance(np.random.default_rng(child), p, n)
        xs[i] = x
        ys[i] = y
        Rgts[i] = Rgt
        tgts[i] = tgt
    return xs, ys, Rgts, tgts


def angle_deg(Ra, Rb):
    d = np.einsum("bij,bij->b", Ra, Rb)
    d = np.clip((d - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(d))


def summarize(R, t, Rgts, tgts):
    rot = angle_deg(R, Rgts)
    tra = np.linalg.norm(t - tgts, axis=1)
    return dict(rot_med_deg=float(np.median(rot)),
                rot_p95_deg=float(np.percentile(rot, 95)),
                trans_med_m=float(np.median(tra)),
                trans_p95_m=float(np.percentile(tra, 95)),
                success=float(np.mean((rot < 5.0) & (tra < 0.1))))


def bench(fn, repeats, warmup=3, cuda=True):
    """Wall-clock per call with a synchronize before/after each repeat."""
    import torch
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(1000.0 * (time.perf_counter() - t0))
    ts = np.asarray(ts)
    return dict(p50_ms=float(np.median(ts)),
                p95_ms=float(np.percentile(ts, 95)),
                mean_ms=float(ts.mean()), min_ms=float(ts.min()),
                n_repeats=int(len(ts)))


def maxdiff(Ra, ta, Rb, tb):
    dR = np.abs(Ra - Rb).reshape(len(Ra), -1).max(axis=1)
    dT = np.abs(ta - tb).max(axis=1)
    return float(dR.max()), float(dT.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--B", default="1,8,32,128,512")
    ap.add_argument("--N", default="100,500")
    ap.add_argument("--dtypes", default="float64,float32")
    ap.add_argument("--p", type=float, default=P_OUT)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--repeats-numpy", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--force-cpu", action="store_true",
                    help="debug only: run the torch paths on CPU, no graphs")
    args = ap.parse_args()

    import torch
    import gpu_solver as gs

    dev_name = "cpu" if args.force_cpu else "cuda"
    if dev_name == "cuda" and not torch.cuda.is_available():
        print("FATAL: CUDA is not available on this allocation", file=sys.stderr)
        return 3
    device = torch.device(dev_name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = dict(
        device=dev_name,
        device_name=(torch.cuda.get_device_name(0) if dev_name == "cuda" else
                     platform.processor()),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        numpy=np.__version__,
        python=sys.version.split()[0],
        platform=platform.platform(),
        git_head=git_head(),
        seed=BASE_SEED, p=args.p,
        protocol="GNC-GM c=0.05 gamma=1.4 max_mu=40 polish=5, "
                 "random outliers, success rot<5deg & |t|<0.1m",
        timings="sync-bracketed wall clock, includes graph input copies; "
                "one call = full GNC solve",
        repeats=args.repeats, warmup=args.warmup,
    )
    rows = []
    errors = []

    for n in [int(v) for v in args.N.split(",")]:
        for b in [int(v) for v in args.B.split(",")]:
            print(f"\n=== B={b} N={n} p={args.p}", flush=True)
            xs, ys, Rgts, tgts = make_batch(b, n, args.p)

            # --- NumPy reference -------------------------------------------------
            t0 = time.perf_counter()
            Rref, tref, inf = bg.batch_solve_gnc(
                xs, ys, method="eigh", return_info=True)
            numpy_ms = 1000.0 * (time.perf_counter() - t0)
            # a few more repeats only for small batches (CPU reference is not
            # the object of study; it is the correctness oracle)
            if b <= 32:
                reps = []
                for _ in range(args.repeats_numpy):
                    t0 = time.perf_counter()
                    bg.batch_solve_gnc(xs, ys, method="eigh")
                    reps.append(1000.0 * (time.perf_counter() - t0))
                numpy_ms = float(np.median(reps))
            ref_summ = summarize(Rref, tref, Rgts, tgts)
            ref_nmu = np.asarray(inf["n_mu"], dtype=np.int64)
            rows.append(dict(b=b, n=n, dtype="float64", method="numpy-ref",
                             p50_ms=numpy_ms, p95_ms=None,
                             correct=True, **ref_summ))
            print(f"  numpy-ref {numpy_ms:8.2f} ms  rot {ref_summ['rot_med_deg']:.4f} deg"
                  f"  succ {ref_summ['success']:.2f}", flush=True)

            xT64 = np.ascontiguousarray(xs.transpose(0, 2, 1))
            yT64 = np.ascontiguousarray(ys.transpose(0, 2, 1))

            for dtype_name in args.dtypes.split(","):
                td = getattr(torch, dtype_name)
                xT = torch.tensor(xT64, dtype=td, device=device)
                yT = torch.tensor(yT64, dtype=td, device=device)
                common = dict(b=b, n=n, dtype=dtype_name)
                dtype_rows = []

                def add(method, R, t, info, timing=None, correct_extra=None,
                        failed=None):
                    if failed is not None:
                        row = dict(common, method=method, failed=failed)
                        rows.append(row)
                        dtype_rows.append(row)
                        print(f"  {method:16s} FAILED: {failed}", flush=True)
                        return
                    dR, dT = maxdiff(R, t, Rref, tref)
                    nmu = np.asarray(info["n_mu"].cpu().numpy(), dtype=np.int64)
                    nmu_match = float(np.mean(nmu == ref_nmu))
                    summ = summarize(R, t, Rgts, tgts)
                    row = dict(common, method=method,
                               p50_ms=timing and timing["p50_ms"],
                               p95_ms=timing and timing["p95_ms"],
                               mean_ms=timing and timing["mean_ms"],
                               min_ms=timing and timing["min_ms"],
                               n_repeats=timing and timing["n_repeats"],
                               maxdR_fro=dR, maxdT=dT,
                               n_mu_match=nmu_match,
                               success_delta=summ["success"] - ref_summ["success"],
                               **summ)
                    if correct_extra:
                        row.update(correct_extra)
                    rows.append(row)
                    dtype_rows.append(row)
                    p50 = f"{timing['p50_ms']:8.2f}" if timing else "     n/a"
                    print(f"  {method:16s} p50={p50} ms  maxdR={dR:.2e} "
                          f"maxdT={dT:.2e} nmu={nmu_match:.2f} "
                          f"succ={summ['success']:.2f}", flush=True)
                    return row

                # --- eager dynamic ----------------------------------------------
                try:
                    with torch.inference_mode():
                        Rd, td_, infd = gs.solve_gnc_t(xT, yT)
                    timing = bench(lambda: gs.solve_gnc_t(xT, yT), args.repeats,
                                   args.warmup, cuda=(dev_name == "cuda"))
                    add("dynamic-eager", Rd.cpu().double().numpy(),
                        td_.cpu().double().numpy(), infd, timing)
                except Exception as exc:
                    add("dynamic-eager", None, None, None,
                        failed=f"{type(exc).__name__}: {str(exc)[:400]}")

                # --- static + graph --------------------------------------------
                for eigen in ("eigh", "jacobi"):
                    name = f"static-{eigen}"
                    try:
                        with torch.inference_mode():
                            Rs, ts_, infs = gs.solve_gnc_static(
                                xT, yT, eigen=eigen)
                        timing = bench(
                            lambda e=eigen: gs.solve_gnc_static(xT, yT, eigen=e),
                            args.repeats, args.warmup, cuda=(dev_name == "cuda"))
                        add(name, Rs.cpu().double().numpy(),
                            ts_.cpu().double().numpy(), infs, timing)
                    except Exception as exc:
                        add(name, None, None, None,
                            failed=f"{type(exc).__name__}: {str(exc)[:400]}")
                        continue

                    if dev_name != "cuda":
                        continue
                    gname = f"graph-{eigen}"
                    try:
                        g = gs.CudaGraphSolver(
                            lambda a, b_, e=eigen: gs.solve_gnc_static(
                                a, b_, eigen=e), xT, yT, warmup=2)
                        g.replay(xT, yT)
                        timing = bench(lambda: g.replay(xT, yT), args.repeats, 2,
                                       cuda=True)
                        Rg, tg_, infg = g.replay(xT, yT)
                        add(gname, Rg.cpu().double().numpy(),
                            tg_.cpu().double().numpy(), infg, timing,
                            correct_extra=dict(graph_capture=True))
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {str(exc)[:800]}"
                        add(gname, None, None, None, failed=msg)
                        errors.append(dict(b=b, n=n, dtype=dtype_name,
                                           method=gname, error=msg))

                # stability flag on the main static path (float32 check)
                stat = next((r for r in dtype_rows
                             if r["method"] == "static-eigh"
                             and "failed" not in r), None)
                if stat is not None:
                    stable = (stat["n_mu_match"] == 1.0
                              and stat["maxdR_fro"] < 1e-3
                              and abs(stat["success_delta"]) <= 1.0 / b)
                    for r in dtype_rows:
                        r["dtype_stable"] = bool(stable)
                    print(f"  dtype {dtype_name}: stable={stable}", flush=True)

                del xT, yT
                if dev_name == "cuda":
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ save
    result = dict(meta=meta, rows=rows, errors=errors,
                  b_values=[int(v) for v in args.B.split(",")],
                  n_values=[int(v) for v in args.N.split(",")],
                  dtypes=args.dtypes.split(","))
    (out / "qpep_gpu.json").write_text(json.dumps(result, indent=2))
    write_md(result, out / "qpep_gpu.md")
    try:
        write_png(result, out / "qpep_gpu.png")
    except Exception as exc:  # figure is auxiliary
        print(f"PNG failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        result["errors"].append(dict(stage="png", error=str(exc)))
        (out / "qpep_gpu.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out/'qpep_gpu.json'} / .md / .png")
    return 0


def _fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def write_md(result, path):
    meta = result["meta"]
    L = ["# QPEP-GPU probe: eager vs static vs CUDA graph (4x4 eigen GNC)",
         "",
         f"- device: **{meta['device']}** ({meta['device_name']})",
         f"- torch {meta['torch']} / CUDA {meta['cuda']} / NumPy {meta['numpy']}",
         f"- git HEAD `{meta['git_head']}`",
         f"- protocol: {meta['protocol']}",
         f"- timings: {meta['timings']}; {meta['repeats']} repeats, "
         f"{meta['warmup']} warmup",
         "", "## Correctness vs NumPy batched reference", "",
         "| B | N | dtype | method | p50 ms | p95 ms | max|dR|_F | max|dt| | n_mu match | success |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        if "failed" in r:
            L.append(f"| {r['b']} | {r['n']} | {r['dtype']} | {r['method']} | "
                     f"FAILED | | | | | |")
            continue
        if r["method"] == "numpy-ref":
            L.append(f"| {r['b']} | {r['n']} | {r['dtype']} | {r['method']} | "
                     f"{_fmt(r.get('p50_ms'))} | n/a | - | - | - | "
                     f"{r['success']:.2f} |")
            continue
        L.append(f"| {r['b']} | {r['n']} | {r['dtype']} | {r['method']} | "
                 f"{_fmt(r.get('p50_ms'))} | {_fmt(r.get('p95_ms'))} | "
                 f"{r['maxdR_fro']:.2e} | {r['maxdT']:.2e} | "
                 f"{r['n_mu_match']:.2f} | {r['success']:.2f} |")
    if result["errors"]:
        L += ["", "## Failures (exact error)", ""]
        for e in result["errors"]:
            L.append(f"- B={e.get('b')} N={e.get('n')} {e.get('dtype')} "
                     f"`{e.get('method')}`: `{e.get('error')}`")
    L += ["", "## Read this", "",
          "- `dynamic-eager` is the existing active-set loop; it cannot be "
          "CUDA-graphed because its tensor shapes change.",
          "- `static-*` carries every instance through the full schedule with "
          "a frozen mask, so shapes are static; finished instances are "
          "recomputed (no FLOP saving).",
          "- `graph-*` timing includes copying the batch into the static "
          "input buffers.",
          "- A `stable=false` dtype is reported for completeness but should "
          "not be used for the paper's claims.",
          "- NumPy reference is CPU float64 and is the correctness oracle, "
          "not a GPU baseline."]
    path.write_text("\n".join(L) + "\n")


def write_png(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["rows"]
            if "failed" not in r and r["method"] not in ("numpy-ref",)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"dynamic-eager": "C0", "static-eigh": "C1",
              "static-jacobi": "C2", "graph-eigh": "C3",
              "graph-jacobi": "C4"}
    for ax, n in zip(axs[0], result["n_values"]):
        for method in colors:
            pts = [(r["b"], r["p50_ms"]) for r in rows
                   if r["n"] == n and r["dtype"] == "float64"
                   and r["method"] == method and r.get("p50_ms")]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=method,
                        color=colors[method])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size B")
        ax.set_ylabel("p50 latency (ms)")
        ax.set_title(f"N={n}, float64")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    ax = axs[1][0]
    for method in ("static-eigh", "graph-eigh", "graph-jacobi"):
        pts = [(r["b"], r["p50_ms"]) for r in rows
               if r["method"] == method and r["dtype"] == "float64"]
        if pts:
            ax.scatter(*zip(*pts), label=method, color=colors[method], s=25)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("all N, float64")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axs[1][1]
    for dtype, marker in (("float64", "o"), ("float32", "x")):
        pts = [(r["b"], max(r["maxdR_fro"], 1e-18)) for r in rows
               if r["method"] in ("static-eigh", "dynamic-eager")
               and r["dtype"] == dtype]
        if pts:
            ax.scatter(*zip(*pts), label=f"{dtype} static/dynamic",
                       marker=marker, s=25)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("max |dR|_F vs NumPy")
    ax.set_title("correctness")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("QPEP-GPU: eager dynamic vs static 4x4 eigen vs CUDA graph")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
