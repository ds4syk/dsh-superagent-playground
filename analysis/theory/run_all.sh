#!/usr/bin/env bash
# Reproduce all theory numerics for notes/theory_Q1_Q3.md (SDPRLayers, TRO 2025).
# Usage:  cd analysis/theory && bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=/home/syk/miniconda3/envs/WM/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

mkdir -p results figures

echo "=== exp1: certificate / tightness / corank / SOSC (paper Sec. VI-A) ==="
"$PY" exp1_certificate.py

echo "=== exp2: Thm.2 Jacobian vs finite differences / CIFT / dual non-uniqueness ==="
"$PY" exp2_jacobian.py

echo "=== exp3: redundant constraints, singular M, pseudo-inverse formula Mr^+ ==="
"$PY" exp3_redundancy.py

echo
echo "done. outputs:"
ls -1 results/*.json figures/*.png
