#!/bin/bash
#SBATCH --job-name=bridge-arena-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/eval_%j.log
set -e
cd /mnt/storage1/syk/.superagent-work/bridge-sim-models-98ea85c61f10/bridge-arena
export PATH=/home/syk/miniconda3/envs/jev/bin:$PATH
python scripts/evaluate.py --n 512 --workers 16
