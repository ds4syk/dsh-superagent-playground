#!/bin/bash
#SBATCH --job-name=sdpr-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/storage1/syk/logs/sdpr-gpu-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source /home/syk/miniconda3/etc/profile.d/conda.sh
conda activate jepa
export PYTHONUNBUFFERED=1
python repro/gpu_sdpr/verify_p6.py --seeds 10
python repro/gpu_sdpr/benchmark.py --batches 20 80 200
python repro/gpu_sdpr/e2e_calib.py --runs 3 --iters 30
