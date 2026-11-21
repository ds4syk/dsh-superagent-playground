#!/bin/bash
#SBATCH --job-name=aimath-blind
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --chdir=/mnt/storage1/syk/.superagent-work/ai-math-survey-repro-3877d01b7b12/repro/openai-10

set -euo pipefail
cd /mnt/storage1/syk/.superagent-work/ai-math-survey-repro-3877d01b7b12/repro/openai-10
mkdir -p logs
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python3 run_sessions.py --workers 8 "$@"
