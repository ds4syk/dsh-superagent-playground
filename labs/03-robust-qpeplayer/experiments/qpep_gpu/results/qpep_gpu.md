# QPEP-GPU probe: eager vs static vs CUDA graph (4x4 eigen GNC)

> **STATUS: GPU measurement BLOCKED; this file is the CPU validation fallback.**
> Node `ls` has `Gres=gpu:3` and all three RTX 5090s were allocated to other
> Slurm jobs (3151/3094/3086, user cyr) for the whole session; `squeue` shows
> this probe's job 3171 was PENDING with `StartTime=2026-11-28T21:34` (a duplicate
> job 3170 from the main workspace is also PENDING) and was cancelled after
> recording `scontrol show node/job` in `logs/slurm_gpu_3171_state.txt`. No CUDA graph was
> captured, so there is **no GPU p50/p95 and no exact graph error** in this
> file. The tables below are `--force-cpu` validation of the same code path:
> NumPy reference, eager dynamic, static-eigh and static-jacobi agree to
> ~1e-16 (float64) and ~1e-7 (float32), with identical `n_mu`. The CUDA paths
> remain unmeasured; do not cite these numbers as GPU timings.

- device: **cpu** (x86_64)
- torch 2.11.0+cu128 / CUDA 12.8 / NumPy 2.2.6
- git HEAD `17d7ffcc30204bdd3dee8f29a8eaf24f739a97d3`
- protocol: GNC-GM c=0.05 gamma=1.4 max_mu=40 polish=5, random outliers, success rot<5deg & |t|<0.1m
- timings: sync-bracketed wall clock, includes graph input copies; one call = full GNC solve; 5 repeats, 3 warmup

## Correctness vs NumPy batched reference

| B | N | dtype | method | p50 ms | p95 ms | max|dR|_F | max|dt| | n_mu match | success |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | float64 | numpy-ref | 3.06 | n/a | - | - | - | 1.00 |
| 1 | 100 | float64 | dynamic-eager | 13.95 | 14.05 | 1.11e-15 | 8.33e-17 | 1.00 | 1.00 |
| 1 | 100 | float64 | static-eigh | 16.35 | 16.59 | 1.11e-15 | 8.33e-17 | 1.00 | 1.00 |
| 1 | 100 | float64 | static-jacobi | 100.19 | 101.32 | 3.89e-16 | 5.55e-17 | 1.00 | 1.00 |
| 1 | 100 | float32 | dynamic-eager | 14.47 | 14.57 | 1.50e-07 | 3.09e-08 | 1.00 | 1.00 |
| 1 | 100 | float32 | static-eigh | 17.29 | 19.44 | 1.50e-07 | 3.09e-08 | 1.00 | 1.00 |
| 1 | 100 | float32 | static-jacobi | 104.67 | 105.13 | 2.82e-07 | 3.54e-08 | 1.00 | 1.00 |
| 8 | 100 | float64 | numpy-ref | 4.44 | n/a | - | - | - | 1.00 |
| 8 | 100 | float64 | dynamic-eager | 24.64 | 24.68 | 9.71e-16 | 3.33e-16 | 1.00 | 1.00 |
| 8 | 100 | float64 | static-eigh | 28.14 | 28.19 | 9.71e-16 | 3.33e-16 | 1.00 | 1.00 |
| 8 | 100 | float64 | static-jacobi | 166.14 | 166.33 | 7.49e-16 | 3.33e-16 | 1.00 | 1.00 |
| 8 | 100 | float32 | dynamic-eager | 25.09 | 25.21 | 2.81e-07 | 1.15e-07 | 1.00 | 1.00 |
| 8 | 100 | float32 | static-eigh | 28.76 | 28.78 | 2.81e-07 | 1.15e-07 | 1.00 | 1.00 |
| 8 | 100 | float32 | static-jacobi | 171.69 | 172.18 | 2.10e-07 | 9.53e-08 | 1.00 | 1.00 |
| 32 | 100 | float64 | numpy-ref | 8.04 | n/a | - | - | - | 1.00 |
| 32 | 100 | float64 | dynamic-eager | 32.08 | 32.26 | 9.44e-16 | 2.78e-16 | 1.00 | 1.00 |
| 32 | 100 | float64 | static-eigh | 35.25 | 35.33 | 9.44e-16 | 2.78e-16 | 1.00 | 1.00 |
| 32 | 100 | float64 | static-jacobi | 186.70 | 186.95 | 9.99e-16 | 4.44e-16 | 1.00 | 1.00 |
| 32 | 100 | float32 | dynamic-eager | 31.07 | 31.11 | 4.66e-07 | 1.67e-07 | 1.00 | 1.00 |
| 32 | 100 | float32 | static-eigh | 34.06 | 34.13 | 4.66e-07 | 1.67e-07 | 1.00 | 1.00 |
| 32 | 100 | float32 | static-jacobi | 192.38 | 192.65 | 2.59e-07 | 2.09e-07 | 1.00 | 1.00 |
| 128 | 100 | float64 | numpy-ref | 26.47 | n/a | - | - | - | 1.00 |
| 128 | 100 | float64 | dynamic-eager | 49.72 | 51.16 | 1.28e-15 | 5.55e-16 | 1.00 | 1.00 |
| 128 | 100 | float64 | static-eigh | 56.33 | 56.46 | 1.28e-15 | 5.55e-16 | 1.00 | 1.00 |
| 128 | 100 | float64 | static-jacobi | 201.59 | 203.18 | 1.25e-15 | 4.44e-16 | 1.00 | 1.00 |
| 128 | 100 | float32 | dynamic-eager | 34.86 | 35.13 | 4.34e-07 | 2.01e-07 | 1.00 | 1.00 |
| 128 | 100 | float32 | static-eigh | 51.66 | 52.39 | 4.34e-07 | 2.01e-07 | 1.00 | 1.00 |
| 128 | 100 | float32 | static-jacobi | 198.76 | 201.99 | 3.86e-07 | 2.28e-07 | 1.00 | 1.00 |
| 512 | 100 | float64 | numpy-ref | 158.17 | n/a | - | - | - | 1.00 |
| 512 | 100 | float64 | dynamic-eager | 91.49 | 94.94 | 1.64e-15 | 7.77e-16 | 1.00 | 1.00 |
| 512 | 100 | float64 | static-eigh | 102.50 | 105.84 | 1.64e-15 | 7.77e-16 | 1.00 | 1.00 |
| 512 | 100 | float64 | static-jacobi | 472.64 | 658.67 | 1.44e-15 | 5.00e-16 | 1.00 | 1.00 |
| 512 | 100 | float32 | dynamic-eager | 89.10 | 89.23 | 8.34e-07 | 2.31e-07 | 1.00 | 1.00 |
| 512 | 100 | float32 | static-eigh | 105.96 | 106.97 | 8.34e-07 | 2.31e-07 | 1.00 | 1.00 |
| 512 | 100 | float32 | static-jacobi | 460.68 | 639.61 | 4.13e-07 | 2.29e-07 | 1.00 | 1.00 |
| 1 | 500 | float64 | numpy-ref | 3.47 | n/a | - | - | - | 1.00 |
| 1 | 500 | float64 | dynamic-eager | 15.08 | 15.34 | 7.22e-16 | 1.11e-16 | 1.00 | 1.00 |
| 1 | 500 | float64 | static-eigh | 16.77 | 17.66 | 7.22e-16 | 1.11e-16 | 1.00 | 1.00 |
| 1 | 500 | float64 | static-jacobi | 101.05 | 102.34 | 3.89e-16 | 1.39e-17 | 1.00 | 1.00 |
| 1 | 500 | float32 | dynamic-eager | 15.47 | 15.61 | 1.60e-07 | 4.22e-08 | 1.00 | 1.00 |
| 1 | 500 | float32 | static-eigh | 17.36 | 17.55 | 1.60e-07 | 4.22e-08 | 1.00 | 1.00 |
| 1 | 500 | float32 | static-jacobi | 104.44 | 105.05 | 8.37e-08 | 1.24e-08 | 1.00 | 1.00 |
| 8 | 500 | float64 | numpy-ref | 6.74 | n/a | - | - | - | 1.00 |
| 8 | 500 | float64 | dynamic-eager | 30.98 | 31.10 | 7.22e-16 | 3.89e-16 | 1.00 | 1.00 |
| 8 | 500 | float64 | static-eigh | 33.21 | 33.85 | 7.22e-16 | 3.89e-16 | 1.00 | 1.00 |
| 8 | 500 | float64 | static-jacobi | 172.15 | 280.52 | 7.77e-16 | 2.50e-16 | 1.00 | 1.00 |
| 8 | 500 | float32 | dynamic-eager | 34.70 | 38.06 | 2.75e-07 | 9.57e-08 | 1.00 | 1.00 |
| 8 | 500 | float32 | static-eigh | 41.10 | 42.95 | 2.75e-07 | 9.57e-08 | 1.00 | 1.00 |
| 8 | 500 | float32 | static-jacobi | 118.39 | 136.77 | 2.16e-07 | 1.61e-07 | 1.00 | 1.00 |
| 32 | 500 | float64 | numpy-ref | 23.11 | n/a | - | - | - | 1.00 |
| 32 | 500 | float64 | dynamic-eager | 33.47 | 35.08 | 9.99e-16 | 3.33e-16 | 1.00 | 1.00 |
| 32 | 500 | float64 | static-eigh | 37.64 | 44.77 | 9.99e-16 | 3.33e-16 | 1.00 | 1.00 |
| 32 | 500 | float64 | static-jacobi | 145.43 | 163.58 | 8.50e-16 | 3.33e-16 | 1.00 | 1.00 |
| 32 | 500 | float32 | dynamic-eager | 30.14 | 30.31 | 4.28e-07 | 1.31e-07 | 1.00 | 1.00 |
| 32 | 500 | float32 | static-eigh | 32.29 | 34.63 | 4.28e-07 | 1.31e-07 | 1.00 | 1.00 |
| 32 | 500 | float32 | static-jacobi | 141.97 | 150.29 | 4.11e-07 | 1.84e-07 | 1.00 | 1.00 |
| 128 | 500 | float64 | numpy-ref | 87.48 | n/a | - | - | - | 1.00 |
| 128 | 500 | float64 | dynamic-eager | 85.30 | 92.37 | 1.17e-15 | 4.44e-16 | 1.00 | 1.00 |
| 128 | 500 | float64 | static-eigh | 72.52 | 76.22 | 1.17e-15 | 4.44e-16 | 1.00 | 1.00 |
| 128 | 500 | float64 | static-jacobi | 239.45 | 295.80 | 1.17e-15 | 4.44e-16 | 1.00 | 1.00 |
| 128 | 500 | float32 | dynamic-eager | 57.43 | 58.00 | 4.29e-07 | 2.28e-07 | 1.00 | 1.00 |
| 128 | 500 | float32 | static-eigh | 65.06 | 65.32 | 4.29e-07 | 2.28e-07 | 1.00 | 1.00 |
| 128 | 500 | float32 | static-jacobi | 221.01 | 285.08 | 3.34e-07 | 1.84e-07 | 1.00 | 1.00 |
| 512 | 500 | float64 | numpy-ref | 712.19 | n/a | - | - | - | 1.00 |
| 512 | 500 | float64 | dynamic-eager | 257.07 | 382.24 | 1.53e-15 | 5.55e-16 | 1.00 | 1.00 |
| 512 | 500 | float64 | static-eigh | 219.55 | 225.65 | 1.53e-15 | 5.55e-16 | 1.00 | 1.00 |
| 512 | 500 | float64 | static-jacobi | 572.25 | 655.79 | 1.25e-15 | 6.11e-16 | 1.00 | 1.00 |
| 512 | 500 | float32 | dynamic-eager | 120.23 | 120.83 | 8.16e-07 | 2.61e-07 | 1.00 | 1.00 |
| 512 | 500 | float32 | static-eigh | 132.01 | 132.15 | 8.16e-07 | 2.61e-07 | 1.00 | 1.00 |
| 512 | 500 | float32 | static-jacobi | 508.78 | 537.82 | 3.68e-07 | 2.19e-07 | 1.00 | 1.00 |

## Read this

- `dynamic-eager` is the existing active-set loop; it cannot be CUDA-graphed because its tensor shapes change.
- `static-*` carries every instance through the full schedule with a frozen mask, so shapes are static; finished instances are recomputed (no FLOP saving).
- `graph-*` timing includes copying the batch into the static input buffers.
- A `stable=false` dtype is reported for completeness but should not be used for the paper's claims.
- NumPy reference is CPU float64 and is the correctness oracle, not a GPU baseline.
