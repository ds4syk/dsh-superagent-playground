#!/bin/bash
#SBATCH --job-name=jev-exp1-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage1/syk/jev-wm/logs/%x-%j.out
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jev
cd /mnt/storage1/syk/.superagent-work/jamba-world-model-fdd0e082a6b1/jev-wm
export PYTHONPATH=. HF_HOME=/mnt/storage1/syk/jev-wm/hf
python scripts/exp1.py --backend decider --model-path /mnt/storage1/syk/jev-wm/models/Qwen3-4B-Instruct-2507 \
  --level raw --limit 400 --rollouts 40 --out results
