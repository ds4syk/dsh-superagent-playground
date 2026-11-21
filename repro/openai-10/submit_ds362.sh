#!/bin/bash
#SBATCH --job-name=aimath-ds362
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-ds362-%j.out
#SBATCH --error=logs/slurm-ds362-%j.err
#SBATCH --chdir=/mnt/storage1/syk/.superagent-work/ai-math-survey-repro-3877d01b7b12/repro/openai-10
set -euo pipefail
cd /mnt/storage1/syk/.superagent-work/ai-math-survey-repro-3877d01b7b12/repro/openai-10
mkdir -p logs
export PYTHONUNBUFFERED=1
python3 run_sessions.py --models deepseek --problems 362_vlasov_maxwell --force --workers 1
