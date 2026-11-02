# Reproduce on the lab node

Everything below runs as user `syk` on `ls` (172.19.108.185), no sudo, no Docker.

## 1. Python environment

The environment `mcppo` is a clone of the existing `mc` environment (patched
MineRL 0.4.4 + Java 8 + Mesa/Xvfb deps) plus PyTorch:

```bash
export PATH=$HOME/miniconda3/bin:$PATH
conda create -y -p /mnt/storage1/syk/dreamer/envs/mcppo \
  --clone /mnt/storage1/syk/dreamer/envs/mc
/mnt/storage1/syk/dreamer/envs/mcppo/bin/pip install torch tensorboard pytest imageio imageio-ffmpeg
```

The patched MineRL wheel (`minerl_mirror-0.4.4-cp311`) contains Malmo,
Minecraft 1.11.2 and the assets, so no network is needed at run time.

## 2. Stage the game tree on NVMe

`/mnt/storage1` is a busy HDD; each Minecraft instance copies the 0.7 GB tree
into `TMPDIR` on first reset. Stage once:

```bash
bash scripts/prepare_mc.sh   # rsync to /tmp/mcppo/Minecraft
```

## 3. Smoke test (2 envs, 128 samples)

```bash
sbatch scripts/dev/smoke_train.sbatch     # logs/smoke-train-*.out
```

## 4. Training

```bash
RUN=/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0 sbatch scripts/train_diamond.sbatch
```

`--resume` is on by default, so resubmitting the script continues from
`ckpt/latest.pt`. The job writes `metrics.jsonl`, `episodes.jsonl`, `tb/`,
`ckpt/` and `status.log` (5-minute heartbeat from `scripts/dev/watch_run.sh`).

## 5. Curves / evaluation

```bash
python -m mcppo.plot_curves --run <run_dir>
python -m mcppo.evaluate --run <run_dir> --max-steps 4000 --record eval.mp4
python scripts/dev/summarize.py <run_dir>
```

## 6. Knowledge base

```bash
bash scripts/kb_publish.sh    # Feishu folder + pages + DOCX export
```

## Expected numbers (6 envs / 10 CPU)

| Quantity | Value |
|-|-|
| first reset (6 worlds) | 2–3 min |
| per-env tick rate | 7–12 tick/s |
| samples per update | 6,144 |
| collect / learn per update | ~125 s / ~27 s |
| end-to-end | ~40 samples/s |

If the tick rate collapses below ~4/s, check `nvidia-smi`/`squeue` for competing
jobs first; the bottleneck is Java/CPU, not the PPO update.
