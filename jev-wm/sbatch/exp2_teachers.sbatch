#!/bin/bash
#SBATCH --job-name=jev-exp2-teach
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/storage1/syk/jev-wm/logs/%x-%j.out
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jev
cd /mnt/storage1/syk/.superagent-work/jamba-world-model-fdd0e082a6b1/jev-wm
export PYTHONPATH=. HF_HOME=/mnt/storage1/syk/jev-wm/hf
python scripts/exp2_teachers.py --model-path /mnt/storage1/syk/jev-wm/models/Tacit-4B \
  --teachers choice,score,wm --worlds 24 --starts-per-world 4 --max-steps 20 \
  --labels-out data/labels_wm.jsonl --label-teacher wm --label-worlds 40 --label-starts 3 \
  --out results
