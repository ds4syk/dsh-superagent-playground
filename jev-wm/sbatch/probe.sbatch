#!/bin/bash
#SBATCH --job-name=jev-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/storage1/syk/jev-wm/logs/%x-%j.out
hostname; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
