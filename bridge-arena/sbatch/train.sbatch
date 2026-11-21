#!/bin/bash
#SBATCH --job-name=bridge-arena-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.log
set -e
cd /mnt/storage1/syk/.superagent-work/bridge-sim-models-98ea85c61f10/bridge-arena
export PATH=/home/syk/miniconda3/envs/jev/bin:$PATH
python scripts/train.py --data results/dataset_l1.npz --out results/checkpoints --epochs 400
