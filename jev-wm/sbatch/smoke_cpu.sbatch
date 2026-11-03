#!/bin/bash
#SBATCH --job-name=jev-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/storage1/syk/jev-wm/logs/%x-%j.out
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jev
cd /mnt/storage1/syk/.superagent-work/jamba-world-model-fdd0e082a6b1/jev-wm
export PYTHONPATH=.
python scripts/smoke_cpu.py
