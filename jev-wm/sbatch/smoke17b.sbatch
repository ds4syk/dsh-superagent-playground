#!/bin/bash
#SBATCH --job-name=jev-sm17b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/storage1/syk/jev-wm/logs/%x-%j.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jev
cd /mnt/storage1/syk/.superagent-work/jamba-world-model-fdd0e082a6b1/jev-wm
export PYTHONPATH=.
export HF_HOME=/mnt/storage1/syk/jev-wm/hf
nvidia-smi --query-gpu=index,name,memory.used --format=csv
TACIT_PATH=/mnt/storage1/syk/jev-wm/models/Tacit-1.7B python scripts/smoke_tacit.py
