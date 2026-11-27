#!/bin/bash
# Phase-4 GPU probe (2026-11-28). 4x4 eigen GNC only; NOT a 40-root QPEP solver.
# Submit from labs/03-robust-qpeplayer/experiments/qpep_gpu/:
#   sbatch sbatch_probe_phase4.sbatch
#SBATCH --job-name=p3gpu-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/slurm_probe_phase4_%j.log
#SBATCH --error=results/logs/slurm_probe_phase4_%j.err

set -euo pipefail
cd "$(dirname "$0")"
PY=/home/syk/miniconda3/envs/WM/bin/python
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv || true
$PY probe.py
echo "end=$(date -Is)"
