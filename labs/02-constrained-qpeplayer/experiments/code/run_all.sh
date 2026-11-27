#!/usr/bin/env bash
# Constrained-QPEPLayer 合成桌面实验 —— 一键复现。
# 用法：cd labs/02-constrained-qpeplayer/experiments && bash code/run_all.sh
# 环境：/home/syk/miniconda3/envs/sdpr/bin/python（numpy 1.26 / scipy 1.13 / cvxpy 1.5 / matplotlib）
# 计算全部走 Slurm（单节点 cpu 分区，单任务）。
set -euo pipefail
PY=/home/syk/miniconda3/envs/sdpr/bin/python
RUN="srun --partition=cpu --time=00:40:00 --ntasks=1"

echo "[1/4] code validation"
$RUN "$PY" code/validate_code.py 2>&1 | tee results/validation.log

echo "[2/4] main experiment (50 trials/cell)"
$RUN "$PY" code/run_main.py 2>&1 | tee results/run_main.log

echo "[3/4] four-legged table / FB limit tables"
$RUN "$PY" code/run_fb_table.py 2>&1 | tee results/fb_table.log

echo "[4/4] tables + figures"
$RUN "$PY" code/make_results.py 2>&1 | tee results/make_results.log
