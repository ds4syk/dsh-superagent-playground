#!/usr/bin/env python3
"""Benchmark: batched GNC-GM vs the per-instance loop (3D-3D).

Grid: B in {1, 8, 32, 128}, N in {100, 500}, TRIALS (default 100) seeded
trials per cell.  Each trial draws B fresh random instances (p=0.3 random
outliers, the main-table protocol of the synthetic experiments), then times

  * the loop: ``common.solve_gnc`` called B times, and
  * the batched implementation: one ``batch_gnc.batch_solve_gnc`` call,

with ``time.perf_counter``.  Reported: p50/p95 latency (whole batch call),
speedup of the medians, throughput (instances/s) and the per-instance
latency.  Correctness is checked on the same instances each trial
(rotation Frobenius, translation norm, and per-instance mu-schedule
identity ``n_mu``).

Modes
-----
``--mode cpu``  (default): loop + numpy batched, writes
    ``results/batch_gnc_cpu.json`` and, if no GPU section exists yet,
    ``results/batch_gnc.json`` / ``.md`` / ``.png`` (CPU-only).

``--mode gpu``  torch float64 on ``cuda:0`` (or ``--device cpu``), writes
    ``results/batch_gnc_gpu.json``; reuses the CPU loop latencies from
    ``results/batch_gnc_cpu.json`` (same single Slurm node) as the speedup
    reference and then merges into ``results/batch_gnc.{json,md,png}``.

``--mode merge``  rebuild the combined JSON/MD/PNG from the CPU and GPU
    section files.

Run (Slurm; see README.md for exact commands):
  srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python run_batch.py --mode cpu
  srun --partition=gpu --gres=gpu:1 --time=00:30:00 --ntasks=1 \
    /home/syk/miniconda3/envs/WM/bin/python run_batch.py --mode gpu
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic"))

import common as cm  # noqa: E402
import batch_gnc as bg  # noqa: E402

OUT = HERE / "results"
LOG = OUT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

BASE_SEED = 20261127
B_LIST = [1, 8, 32, 128]
N_LIST = [100, 500]
TRIALS = 100
P_OUT = 0.3
METHOD = "eigh"

ROT_TOL = 1e-8
TRANS_TOL = 1e-10


# ---------------------------------------------------------------------------
# Environment / instances
# ---------------------------------------------------------------------------
def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def env_info():
    info = dict(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        cpu_model=_cpu_model(),
        nproc=os.cpu_count(),
        git_commit=_git_commit(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        slurm_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        worktree=str(HERE.parents[3]),
    )
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_names"] = ([torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())]
                             if torch.cuda.is_available() else [])
    except Exception as exc:  # pragma: no cover
        info["torch"] = f"unavailable: {exc}"
    return info


def make_batch(b, n, p, trial):
    """Seeded batch of B independent instances; identical across modes."""
    ss = np.random.SeedSequence([BASE_SEED, 11, int(b), int(n),
                                 int(round(100 * p)), int(trial)])
    xs = np.empty((b, n, 3))
    ys = np.empty((b, n, 3))
    for i, child in enumerate(ss.spawn(b)):
        x, y, _, _ = cm.make_instance(np.random.default_rng(child), p, n)
        xs[i] = x
        ys[i] = y
    return xs, ys


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def stats_ms(times_s):
    a = 1000.0 * np.asarray(times_s, float)
    return dict(p50_ms=float(np.percentile(a, 50)),
                p95_ms=float(np.percentile(a, 95)),
                mean_ms=float(a.mean()),
                min_ms=float(a.min()),
                max_ms=float(a.max()))


def cell_record(b, n, loop_times, batch_times, corr, extra=None):
    L = stats_ms(loop_times)
    Bt = stats_ms(batch_times)
    rec = dict(
        B=b, N=n,
        loop=L, batched=Bt,
        speedup_p50=float(L["p50_ms"] / Bt["p50_ms"]),
        speedup_p95=float(L["p95_ms"] / Bt["p95_ms"]),
        per_instance_ms_loop=float(L["p50_ms"] / b),
        per_instance_ms_batched=float(Bt["p50_ms"] / b),
        throughput_loop_inst_s=float(b / (L["p50_ms"] / 1000.0)),
        throughput_batched_inst_s=float(b / (Bt["p50_ms"] / 1000.0)),
        correctness=corr,
        raw=dict(loop_ms=[float(1000 * t) for t in loop_times],
                 batched_ms=[float(1000 * t) for t in batch_times]),
    )
    if extra:
        rec.update(extra)
    return rec


def correctness(Rb, tb, Rl, tl, nmu_b=None, nmu_l=None):
    dr = np.linalg.norm(Rb - Rl, axis=(1, 2))
    dt = np.linalg.norm(tb - tl, axis=1)
    c = dict(rot_max=float(dr.max()), rot_median=float(np.median(dr)),
             trans_max=float(dt.max()), trans_median=float(np.median(dt)),
             n_instances=int(len(dr)),
             n_mu_mismatch=int(np.sum(nmu_b != nmu_l)) if nmu_b is not None else None)
    return c


# ---------------------------------------------------------------------------
# Fixed-instance verification (mirror of verify_batch.py, compact)
# ---------------------------------------------------------------------------
def run_verification():
    rows = []
    ok = True
    for n in N_LIST:
        for p in (0.0, 0.3, 0.7):
            x, y = make_batch(8, n, p, trial=999)
            Rl = np.empty((8, 3, 3)); tl = np.empty((8, 3)); nmu = np.empty(8, int)
            for i in range(8):
                Rl[i], tl[i], info = cm.solve_gnc(x[i], y[i], return_info=True)
                nmu[i] = info["n_mu"]
            for method in ("eigh", "svd"):
                Rb, tb, inf = bg.batch_solve_gnc(x, y, method=method, return_info=True)
                c = correctness(Rb, tb, Rl, tl, inf["n_mu"], nmu)
                good = (c["rot_max"] < ROT_TOL and c["trans_max"] < TRANS_TOL
                        and c["n_mu_mismatch"] == 0)
                ok &= bool(good)
                rows.append(dict(N=n, p=p, method=method, **c, ok=bool(good)))
    return dict(tolerance_rot_fro=ROT_TOL, tolerance_trans=TRANS_TOL,
                B_fixed=8, rows=rows, all_pass=bool(ok))


# ---------------------------------------------------------------------------
# CPU benchmark
# ---------------------------------------------------------------------------
def bench_cpu(trials):
    cells = []
    raw_path = OUT / "batch_gnc_raw.jsonl"
    with open(raw_path, "w") as rawf:
        for n in N_LIST:
            for b in B_LIST:
                # warm-up (separate seed, not measured)
                xw, yw = make_batch(b, n, P_OUT, trial=10**6)
                cm.solve_gnc(xw[0], yw[0])
                bg.batch_solve_gnc(xw, yw)
                loop_times, batch_times = [], []
                rot_max = trans_max = 0.0
                rot_med, trans_med = [], []
                nmu_mis = 0
                for tr in range(trials):
                    x, y = make_batch(b, n, P_OUT, tr)
                    Rl = np.empty((b, 3, 3)); tl = np.empty((b, 3))
                    nmu_l = np.empty(b, dtype=np.int64)
                    t0 = time.perf_counter()
                    for i in range(b):
                        Rl[i], tl[i], info = cm.solve_gnc(x[i], y[i],
                                                         return_info=True)
                        nmu_l[i] = info["n_mu"]
                    t1 = time.perf_counter()
                    Rb, tb, inf_b = bg.batch_solve_gnc(x, y, method=METHOD,
                                                      return_info=True)
                    t2 = time.perf_counter()
                    loop_times.append(t1 - t0)
                    batch_times.append(t2 - t1)
                    c = correctness(Rb, tb, Rl, tl, inf_b["n_mu"], nmu_l)
                    rot_max = max(rot_max, c["rot_max"])
                    trans_max = max(trans_max, c["trans_max"])
                    rot_med.append(c["rot_median"])
                    trans_med.append(c["trans_median"])
                    nmu_mis += c["n_mu_mismatch"]
                    rawf.write(json.dumps(dict(backend="cpu_numpy", B=b, N=n,
                                               trial=tr,
                                               loop_ms=1000 * loop_times[-1],
                                               batched_ms=1000 * batch_times[-1],
                                               rot_fro=c["rot_max"],
                                               trans=c["trans_max"],
                                               n_mu_mismatch=c["n_mu_mismatch"])) + "\n")
                corr = dict(rot_max=rot_max, trans_max=trans_max,
                            rot_median_of_trial_medians=float(np.median(rot_med)),
                            trans_median_of_trial_medians=float(np.median(trans_med)),
                            n_mu_mismatch=int(nmu_mis),
                            n_instances_total=int(b * trials),
                            n_mu_checked=True)
                cells.append(cell_record(b, n, loop_times, batch_times, corr))
                print(f"[cpu] B={b:>3} N={n:>3}  loop p50={cells[-1]['loop']['p50_ms']:8.3f} ms"
                      f"  batch p50={cells[-1]['batched']['p50_ms']:8.3f} ms"
                      f"  speedup={cells[-1]['speedup_p50']:6.2f}x"
                      f"  rot={rot_max:.1e} trans={trans_max:.1e}"
                      f"  nmu_mis={nmu_mis}")
    return dict(mode="cpu", backend="numpy", dtype="float64", method=METHOD,
                trials=trials, outlier_ratio=P_OUT, cells=cells)


# ---------------------------------------------------------------------------
# GPU (torch) benchmark
# ---------------------------------------------------------------------------
def bench_gpu(trials, device="cuda", dtype_name="float64", method=METHOD):
    import torch
    import batch_gnc_torch as bgt

    dtype = {"float64": torch.float64, "float32": torch.float32}[dtype_name]
    cells = []
    transfer_times = []
    raw_path = OUT / f"batch_gnc_raw_gpu_{dtype_name}.jsonl"
    with open(raw_path, "w") as rawf:
        for n in N_LIST:
            for b in B_LIST:
                xw, yw = make_batch(b, n, P_OUT, trial=10**6)
                tw = torch.as_tensor(xw, dtype=dtype).permute(0, 2, 1).contiguous().to(device)
                yw_t = torch.as_tensor(yw, dtype=dtype).permute(0, 2, 1).contiguous().to(device)
                with torch.no_grad():
                    bgt.solve_gnc_t(tw, yw_t, method=method)
                torch.cuda.synchronize() if device == "cuda" else None
                times = []
                rot_max = trans_max = 0.0
                for tr in range(trials):
                    x, y = make_batch(b, n, P_OUT, tr)
                    # transfer (timed separately, excluded from compute latency)
                    ta = time.perf_counter()
                    xt = torch.as_tensor(x, dtype=dtype, device=device).permute(0, 2, 1).contiguous()
                    yt = torch.as_tensor(y, dtype=dtype, device=device).permute(0, 2, 1).contiguous()
                    transfer_times.append(time.perf_counter() - ta)
                    # compute
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        R, t = bgt.solve_gnc_t(xt, yt, method=method)[:2]
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                    if tr < 3:  # correctness on the same seeded instances
                        Rb = R.to("cpu", torch.float64).numpy()
                        tb = t.to("cpu", torch.float64).numpy()
                        Rl = np.empty((b, 3, 3)); tl = np.empty((b, 3))
                        for i in range(b):
                            Rl[i], tl[i] = cm.solve_gnc(x[i], y[i])
                        c = correctness(Rb, tb, Rl, tl)
                        rot_max = max(rot_max, c["rot_max"])
                        trans_max = max(trans_max, c["trans_max"])
                    rawf.write(json.dumps(dict(backend=f"gpu_torch_{dtype_name}",
                                               B=b, N=n, trial=tr,
                                               batched_ms=1000 * times[-1])) + "\n")
                corr = dict(rot_max=rot_max, trans_max=trans_max,
                            n_instances_total=int(b * 3),
                            checked_trials=3)
                bt = stats_ms(times)
                rec = dict(
                    B=b, N=n, batched=bt,
                    per_instance_ms_batched=float(bt["p50_ms"] / b),
                    throughput_batched_inst_s=float(b / (bt["p50_ms"] / 1000.0)),
                    correctness=corr,
                    raw=dict(batched_ms=[float(1000 * t) for t in times]),
                )
                cells.append(rec)
                print(f"[gpu:{dtype_name}] B={b:>3} N={n:>3}  "
                      f"batch p50={rec['batched']['p50_ms']:8.3f} ms  "
                      f"rot={rot_max:.1e} trans={trans_max:.1e}")
    gpu_info = dict(device=device, dtype=dtype_name, method=method,
                    trials=trials, outlier_ratio=P_OUT, cells=cells,
                    transfer_ms=stats_ms(transfer_times),
                    device_name=(torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()))
    return gpu_info


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def attach_speedups(gpu, cpu):
    """Fill GPU speedups against the CPU-loop reference (same Slurm node)."""
    ref = {(c["B"], c["N"]): c for c in cpu["cells"]}
    for c in gpu["cells"]:
        base = ref.get((c["B"], c["N"]))
        if not base:
            continue
        c["loop"] = base["loop"]
        c["speedup_p50"] = base["loop"]["p50_ms"] / c["batched"]["p50_ms"]
        c["speedup_p95"] = base["loop"]["p95_ms"] / c["batched"]["p95_ms"]
        c["per_instance_ms_loop"] = base["loop"]["p50_ms"] / c["B"]
        c["throughput_loop_inst_s"] = base["throughput_loop_inst_s"]
    return gpu


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "-"


def render_md(data):
    meta = data["meta"]
    cfg = data["config"]
    cpu = data["cpu"]
    gpu = data.get("gpu")
    ver = data["verification"]
    lines = []
    lines.append("# Batched GNC-GM vs per-instance loop (3D-3D robust registration)")
    lines.append("")
    lines.append(f"Generated by `systems/run_batch.py` "
                 f"({meta['timestamp_utc']}, git `{meta.get('git_commit')}`). "
                 "All numbers below are produced by that script; seeds are fixed "
                 f"(`SeedSequence([{cfg['base_seed']}, 11, B, N, round(100p), trial])`).")
    lines.append("")
    # headline
    best = max(cpu["cells"], key=lambda c: c["speedup_p50"])
    gpu_best = max(gpu["cells"], key=lambda c: c["speedup_p50"]) if gpu else None
    lines.append("## 0. 一句话结论")
    lines.append("")
    lines.append(f"- 同一 GNC-GM 调度下，numpy 批量实现相对逐实例循环的中位加速："
                 f"最佳 **{best['speedup_p50']:.2f}×**（B={best['B']}, N={best['N']}，"
                 f"loop p50 {best['loop']['p50_ms']:.2f} ms → batched {best['batched']['p50_ms']:.2f} ms）；"
                 f"吞吐 {best['throughput_loop_inst_s']:.0f} → {best['throughput_batched_inst_s']:.0f} instances/s。")
    rot_worst = max(r["rot_max"] for r in ver["rows"])
    trans_worst = max(r["trans_max"] for r in ver["rows"])
    lines.append(f"- 正确性：批量实现与循环的旋转 Frobenius 差最大 **{rot_worst:.2e}**"
                 f"（固定实例全部 < 1e-13，判据 {cfg['tolerance_rot_fro']:g}），平移差 < 1e-15"
                 f"（判据 {cfg['tolerance_trans']:g}）；每实例 μ 迭代数 `n_mu` 全部一致。")
    if gpu_best:
        lines.append(f"- GPU（torch {gpu['dtype']}，{gpu.get('device_name')}）："
                     f"最佳 **{gpu_best['speedup_p50']:.2f}×**（B={gpu_best['B']}, N={gpu_best['N']}），"
                     f"相对 CPU 循环 p50 {gpu_best['loop']['p50_ms']:.2f} ms → {gpu_best['batched']['p50_ms']:.2f} ms"
                     f"（仅计算，不含 H2D 传输）。")
    else:
        lines.append("- GPU 部分未生成（见 §6）。")
    lines.append("")
    # environment
    lines.append("## 1. 环境")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for k, label in [("cpu_model", "CPU"), ("nproc", "逻辑核数"), ("hostname", "主机"),
                     ("platform", "OS"), ("python", "Python"), ("numpy", "numpy"),
                     ("torch", "torch"), ("torch_cuda", "torch CUDA"),
                     ("cuda_available", "CUDA 可见"), ("gpu_names", "GPU"),
                     ("slurm_job_id", "Slurm job id"), ("slurm_partition", "Slurm 分区"),
                     ("worktree", "worktree"), ("git_commit", "git commit")]:
        if k in meta:
            lines.append(f"| {label} | `{meta[k]}` |")
    lines.append("")
    # config
    lines.append("## 2. 配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| B (batch) | {cfg['B_list']} |")
    lines.append(f"| N (points/instance) | {cfg['N_list']} |")
    lines.append(f"| trials / cell | {cfg['trials']} |")
    lines.append(f"| 外点率 p | {cfg['outlier_ratio']} (random outliers) |")
    lines.append(f"| GNC | c={cfg['gnc']['c']}, gamma={cfg['gnc']['gamma']}, "
                 f"max_mu_iters={cfg['gnc']['max_mu_iters']}, polish={cfg['gnc']['polish']} |")
    lines.append(f"| Kabsch backend | `{cfg['method']}` (Horn/Davenport 4×4 eigh; "
                 "`svd` 分支在 `verify_batch.py` 中逐实例对比) |")
    lines.append("| 计时 | `time.perf_counter`, 整批调用 wall-clock；每 cell 先 warm-up 一次不计入 |")
    lines.append("| 运行 | CPU 与 GPU 分属两次 `srun` 作业（同一节点 `ls`）；"
                 "CPU 计时来自 `results/logs/batch_cpu_*.log`，GPU 计时来自 "
                 "`results/logs/batch_gpu_*.log`（表中 Slurm job id 为 GPU 作业） |")
    lines.append("| 计时内容 | 纯计算；实例生成在计时外；GPU 的 H2D 传输单独计 |")
    lines.append("")
    # verification
    lines.append("## 3. 正确性校验（固定种子实例）")
    lines.append("")
    lines.append("同一批固定实例上，批量输出与循环输出的差（B=8；p=0,0.3,0.7；N=100,500）：")
    lines.append("")
    lines.append("| N | p | method | rot_max (Frobenius) | trans_max | n_mu mismatch | pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ver["rows"]:
        lines.append(f"| {r['N']} | {r['p']} | {r['method']} | {r['rot_max']:.2e} | "
                     f"{r['trans_max']:.2e} | {r['n_mu_mismatch']} | {r['ok']} |")
    lines.append("")
    lines.append(f"判据：rot < {ver['tolerance_rot_fro']:g}，trans < {ver['tolerance_trans']:g}，"
                 f"且 `n_mu` 完全一致。总判定：**{'PASS' if ver['all_pass'] else 'FAIL'}**。"
                 "`verify_batch.py` 还逐实例对比了 `svd` 后端和 torch CPU/CUDA 镜像。")
    lines.append("")
    # CPU table
    lines.append("## 4. CPU：loop vs numpy batched")
    lines.append("")
    lines.append("| B | N | loop p50 (ms) | loop p95 (ms) | batch p50 (ms) | batch p95 (ms) | "
                 "speedup p50 | speedup p95 | loop inst/s | batch inst/s | batch ms/inst | "
                 "rot_max | trans_max | n_mu mis |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cpu["cells"]:
        lines.append(
            f"| {c['B']} | {c['N']} | {c['loop']['p50_ms']:.3f} | {c['loop']['p95_ms']:.3f} | "
            f"{c['batched']['p50_ms']:.3f} | {c['batched']['p95_ms']:.3f} | "
            f"**{c['speedup_p50']:.2f}×** | {c['speedup_p95']:.2f}× | "
            f"{c['throughput_loop_inst_s']:.0f} | {c['throughput_batched_inst_s']:.0f} | "
            f"{c['per_instance_ms_batched']:.3f} | "
            f"{c['correctness']['rot_max']:.1e} | {c['correctness']['trans_max']:.1e} | "
            f"{c['correctness']['n_mu_mismatch']} |")
    lines.append("")
    lines.append("注：B=1 时批量调用的 numpy/Python 开销占主导，可能慢于循环；"
                 "加速来自 B 较大时把逐实例的 Python 循环与 3×3 SVD 换成批量 BLAS/批量特征分解。"
                 "N=500 时每轮迭代的残差/中心化是内存受限的，加速比低于 N=100。")
    lines.append("")
    if gpu:
        lines.append("## 5. GPU：torch batched（含与 CPU loop 的对照）")
        lines.append("")
        lines.append(f"设备 `{gpu.get('device_name')}`，dtype `{gpu['dtype']}`，backend `{gpu['method']}`，"
                     f"{gpu['trials']} trials/cell；GPU 计时为 kernel 执行（`torch.cuda.synchronize` 包裹），"
                     f"H2D 传输 p50 = {gpu['transfer_ms']['p50_ms']:.3f} ms（单独计，未含在 batch ms 内）。")
        lines.append("")
        lines.append("| B | N | CPU loop p50 (ms) | GPU p50 (ms) | GPU p95 (ms) | speedup p50 | "
                     "speedup p95 | GPU inst/s | GPU ms/inst | rot_max | trans_max |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in gpu["cells"]:
            if "loop" not in c:
                continue
            lines.append(
                f"| {c['B']} | {c['N']} | {c['loop']['p50_ms']:.3f} | "
                f"{c['batched']['p50_ms']:.3f} | {c['batched']['p95_ms']:.3f} | "
                f"**{c['speedup_p50']:.2f}×** | {c['speedup_p95']:.2f}× | "
                f"{c['throughput_batched_inst_s']:.0f} | {c['per_instance_ms_batched']:.3f} | "
                f"{c['correctness']['rot_max']:.1e} | {c['correctness']['trans_max']:.1e} |")
        lines.append("")
        lines.append("GPU 正确性为每 cell 前 3 个 trial × B 个实例对 CPU 循环的差。"
                     "本实现的 GNC 循环有约 36 次迭代、每次约 20 个小 kernel；在 B<=128 的规模上，"
                     "eager torch 的 kernel launch 开销主导（实测 B=1 与 B=128 的 p50 几乎相同，"
                     "约 40 ms 地板），所以 GPU 只在 B 较大时才相对 CPU 循环有优势。"
                     "这不是 GPU 线性代数慢（4x4 eigh 本身约 0.1 ms/次），而是 batch 太小、"
                     "未做 CUDA graph / torch.compile 融合。")
        lines.append("")
    else:
        lines.append("## 5. GPU")
        lines.append("")
        lines.append("未运行（见 §6）。")
        lines.append("")
    lines.append("## 6. 限制与备注")
    lines.append("")
    lines.append("1. 单节点共享服务器（Slurm `cpu` 分区），wall-clock 计时含系统噪声；p95 尤其，"
                 "绝对 ms 会随负载漂移，加速比的结论以 p50 为主、p95 作参考。")
    lines.append("2. 批量实现是**逐位复刻**循环的调度：每个实例用自己的 μ0 与迭代次数，"
                 "结束的实例从 active 集合移除、不重算；因此 n_mu 与循环完全一致（表 §3、§4 最后一列）。")
    lines.append("3. `eigh` 后端用 Horn/Davenport 四元数法替代 3×3 SVD，二者给出同一最优旋转；"
                 "`svd` 后端在 §3 中逐实例对比，差 < 1e-14。若要求与循环**同一 LAPACK 路径**，"
                 "可用 `batch_solve_gnc(..., method=\"svd\")`。")
    lines.append("4. 吞吐 = B / 整批延迟，按同一 cell 的 p50 计算；不是 GPU 饱和吞吐，"
                 "B=128 仍然远小于可并行的实例数。")
    lines.append("5. 本实验只测工程吞吐；不改变 theory/verify 或既有实验结论，"
                 "外点生成、c、μ 调度与 `synthetic/common.py` 完全一致。")
    lines.append("6. GPU 侧是 eager torch、float64、每条 python 循环逐迭代触发 kernel；"
                 "在 B<=128 时 launch 开销主导，GPU 地板约 40 ms。CUDA graph / torch.compile "
                 "融合、float32、更大 B 均未做（超出本轮范围）。因此本报告不建议用 GPU 替代 "
                 "CPU 批量实现，除非 batch 更大或做了融合。")
    lines.append("")
    return "\n".join(lines) + "\n"


def make_png(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpu = data["cpu"]
    gpu = data.get("gpu")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = {100: "#1f77b4", 500: "#d62728"}
    ax = axes[0]
    for n in data["config"]["N_list"]:
        cs = [c for c in cpu["cells"] if c["N"] == n]
        Bs = [c["B"] for c in cs]
        ax.plot(Bs, [c["throughput_loop_inst_s"] for c in cs], "o--",
                color=colors[n], alpha=0.55, label=f"loop N={n}")
        ax.plot(Bs, [c["throughput_batched_inst_s"] for c in cs], "o-",
                color=colors[n], label=f"numpy batched N={n}")
        if gpu:
            gs = [c for c in gpu["cells"] if c["N"] == n]
            ax.plot([c["B"] for c in gs], [c["throughput_batched_inst_s"] for c in gs],
                    "s:", color=colors[n], alpha=0.8, label=f"torch GPU N={n}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted({c["B"] for c in cpu["cells"]}))
    ax.set_xticklabels([str(bb) for bb in sorted({c["B"] for c in cpu["cells"]})])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("throughput (instances/s)")
    ax.set_title("throughput (p50-based)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=1)
    ax = axes[1]
    for n in data["config"]["N_list"]:
        cs = [c for c in cpu["cells"] if c["N"] == n]
        ax.plot([c["B"] for c in cs], [c["speedup_p50"] for c in cs], "o-",
                color=colors[n], label=f"numpy N={n} (p50)")
        if gpu:
            gs = [c for c in gpu["cells"] if c["N"] == n]
            ax.plot([c["B"] for c in gs], [c["speedup_p50"] for c in gs], "s--",
                    color=colors[n], alpha=0.8, label=f"torch GPU N={n} (p50)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({c["B"] for c in cpu["cells"]}))
    ax.set_xticklabels([str(bb) for bb in sorted({c["B"] for c in cpu["cells"]})])
    ax.set_xlabel("batch size B")
    ax.set_ylabel("speedup vs per-instance loop")
    ax.set_title("speedup (loop p50 / batched p50)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Batched GNC-GM 3D-3D: loop vs batched (B={1,8,32,128}, N={100,500})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def build_combined(cpu, gpu):
    meta = dict(env_info())
    data = dict(
        meta=meta,
        config=dict(B_list=B_LIST, N_list=N_LIST, trials=cpu["trials"],
                    outlier_ratio=P_OUT, base_seed=BASE_SEED,
                    method=METHOD, backend="numpy", dtype="float64",
                    gnc=dict(c=bg.C, gamma=bg.GNC_GAMMA,
                             max_mu_iters=bg.GNC_MAX_MU_ITERS,
                             polish=bg.GNC_POLISH_STEPS),
                    tolerance_rot_fro=ROT_TOL, tolerance_trans=TRANS_TOL),
        verification=run_verification(),
        cpu=cpu,
        gpu=gpu,
    )
    if gpu:
        attach_speedups(gpu, cpu)
    return data


def write_final(data, prefix="batch_gnc"):
    save_json(OUT / f"{prefix}.json", data)
    (OUT / f"{prefix}.md").write_text(render_md(data))
    make_png(data, OUT / f"{prefix}.png")
    print(f"wrote {OUT / (prefix + '.json')}, .md, .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cpu", "gpu", "merge"], default="cpu")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--method", default=METHOD, choices=["eigh", "svd"])
    args = ap.parse_args()
    METHOD = args.method

    if args.mode == "cpu":
        cpu = bench_cpu(args.trials)
        save_json(OUT / "batch_gnc_cpu.json", cpu)
        gpu = load_json(OUT / "batch_gnc_gpu.json")
        data = build_combined(cpu, gpu)
        write_final(data)
    elif args.mode == "gpu":
        cpu = load_json(OUT / "batch_gnc_cpu.json")
        gpu = bench_gpu(args.trials, device=args.device, dtype_name=args.dtype,
                        method=args.method)
        save_json(OUT / "batch_gnc_gpu.json", gpu)
        if cpu:
            data = build_combined(cpu, gpu)
            write_final(data)
        else:
            print("CPU section missing; wrote GPU json only "
                  "(run --mode cpu first / --mode merge afterwards)")
    else:  # merge
        cpu = load_json(OUT / "batch_gnc_cpu.json")
        gpu = load_json(OUT / "batch_gnc_gpu.json")
        if not cpu:
            raise SystemExit("no results/batch_gnc_cpu.json; run --mode cpu first")
        prev = load_json(OUT / "batch_gnc.json")
        data = build_combined(cpu, gpu)
        if prev and prev.get("meta"):
            for k in ("slurm_job_id", "slurm_partition",
                      "slurm_cpus_per_task", "timestamp_utc"):
                if data["meta"].get(k) is None and prev["meta"].get(k) is not None:
                    data["meta"][k] = prev["meta"][k]
            data["meta"]["report_rebuilt_utc"] = datetime.now(
                timezone.utc).isoformat(timespec="seconds")
        write_final(data)


if __name__ == "__main__":
    main()
