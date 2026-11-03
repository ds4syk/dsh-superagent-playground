#!/bin/bash
# Reproducible environment for the SDPRLayers stereo-calibration reproduction.
#
# Target: conda env `sdpr` (python 3.11), CPU only.
#   /home/syk/miniconda3/bin/conda create -n sdpr python=3.11 -y
#   bash repro/upstream/install_env.sh
#
# Design notes / hard-won facts (see repro/upstream/REPORT.md for the full story):
#  * torch CPU wheels: the experiment's SDPs are tiny (13x13), no GPU needed;
#    RTX 5090 is sm_120 and would require cu128 builds. torch 2.5.1 matches
#    upstream requirements.txt (2.5.1+cu118) -> use 2.5.1+cpu.
#  * numpy MUST stay 1.26.3: poly_matrix / diffcp / cvxpylayers are 2023-2024
#    code, and diffcp's C++ extension here is compiled against numpy 1.26.
#    Installing torch/torchvision/cvxpy later can silently bump numpy to 2.x,
#    so numpy+scipy are force-reinstalled at the end.
#  * cvxpy must be <=1.5.x: cvxpy 1.6+ needs numpy>=2, which breaks diffcp.
#  * mosek is intentionally NOT installed (no license). Two upstream modules
#    import it at module scope; patch 001 makes those imports optional.
#  * sparseqr needs SuiteSparseQR headers (absent, no sudo). It was an unused
#    import in sdprlayers; patch 003 makes it optional.
#  * theseus-ai needs scikit-sparse (CHOLMOD) at import time. That library is
#    unavailable here, so Theseus is optional: repro/upstream/stubs/sksparse
#    satisfies the import and the dense Gauss-Newton solver is unaffected.
set -x
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_THREADING_LAYER=GNU
P=/home/syk/miniconda3/envs/sdpr/bin/python
ROOT=/home/syk/labs/SDPR
REQ=$ROOT/code/sdprlayer

$P -m pip install --upgrade pip

# 1) Scientific stack (pins chosen for numpy 1.26 compatibility)
$P -m pip install "numpy==1.26.3" "scipy==1.13.1" "matplotlib==3.8.4" pytest tqdm \
    "scikit-image==0.22.0" "scikit-learn==1.4.2" "pandas==2.2.2" \
    "opencv-python-headless==4.10.0.84" kornia asrl-pylgmath

# 2) CPU-only torch (may pull numpy 2.x; fixed again in step 3)
$P -m pip install "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# 3) Re-pin numpy/scipy that torch/torchvision may have bumped
$P -m pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.3" "scipy==1.13.1"

# 4) Conic solvers + build deps for diffcp (C++ extension)
$P -m pip install pybind11 scs ecos clarabel threadpoolctl cffi "cvxpy==1.5.4"
$P -m pip install sparseqr || echo "WARN: sparseqr unavailable (patch 003 makes it optional)"

# 5) Local packages in dependency order; --no-build-isolation so that diffcp
#    compiles against the installed numpy/pybind11 (its pyproject pins
#    numpy<1.24 for py3.11, which only affects an isolated build env).
$P -m pip install -e $REQ/src/submodules/poly_matrix --no-deps
$P -m pip install -e $REQ/src/submodules/cert_tools --no-deps
$P -m pip install -e $REQ/src/submodules/diffcp --no-build-isolation --no-deps
$P -m pip install -e $REQ/src/submodules/cvxpylayers --no-deps
$P -m pip install -e $REQ --no-deps --no-build-isolation

# 6) OPTIONAL: Theseus local-solver baseline. Needs torchlie/torchkin and
#    (import-time only) sksparse. Install theseus itself --no-deps because its
#    sdist setup.py needs torch at build time and its dependency scikit-sparse
#    cannot be built without SuiteSparse. If this block fails, the ASCII/JSON
#    results of the SDPR runs are unaffected; the Theseus baseline can be run
#    with repro/upstream/local_solver.py (GN fallback) instead.
$P -m pip install torchlie torchkin semantic-version --no-deps || true
$P -m pip install "theseus-ai==0.2.1" --no-deps --no-build-isolation || \
    echo "WARN: theseus-ai unavailable; use repro/upstream/local_solver.py (GN fallback)"

# 7) Final state
$P -m pip list
