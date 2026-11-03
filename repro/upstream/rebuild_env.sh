#!/bin/bash
set -x
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_THREADING_LAYER=GNU
CONDA=/home/syk/miniconda3/bin/conda
ENV=sdpr
REQ=/home/syk/labs/SDPR/code/sdprlayer
$CONDA env remove -n $ENV -y
$CONDA create -n $ENV -y python=3.11
P=/home/syk/miniconda3/envs/$ENV/bin/python
$P -m pip install --upgrade pip
$P -m pip install "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --index-url https://download.pytorch.org/whl/cpu
$P -m pip install "scipy==1.13.1" "matplotlib==3.8.4" pytest tqdm "scikit-image==0.22.0" "scikit-learn==1.4.2" "pandas==2.2.2" opencv-python-headless kornia asrl-pylgmath
$P -m pip install "numpy==1.26.3"
$P -m pip install pybind11 scs ecos clarabel threadpoolctl cffi "cvxpy==1.7.5"
$P -m pip install -e $REQ/src/submodules/poly_matrix --no-deps
$P -m pip install -e $REQ/src/submodules/cert_tools --no-deps
$P -m pip install -e $REQ/src/submodules/diffcp --no-build-isolation --no-deps
$P -m pip install -e $REQ/src/submodules/cvxpylayers --no-deps
$P -m pip install -e $REQ --no-deps --no-build-isolation
echo "=== VERIFY ==="
$P -c "import numpy; print('numpy', numpy.__version__)"
$P -c "import scipy; print('scipy', scipy.__version__)"
$P -c "import torch; print('torch', torch.__version__, torch.tensor([1.0]).numpy())"
$P -c "import cvxpy; print('cvxpy', cvxpy.__version__)"
$P -c "import diffcp, cvxpylayers, poly_matrix; print('diffcp/cvxpylayers/poly_matrix OK')"
echo "=== DONE ==="
