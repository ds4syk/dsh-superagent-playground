"""Minimal *import-only* stub for ``sksparse`` (scikit-sparse).

Why this exists
---------------
``theseus-ai==0.2.1`` imports ``sksparse.cholmod.Factor`` at package import time
(``theseus/optimizer/autograd/cholmod_sparse_autograd.py``), even when the user
only ever asks for a dense Gauss-Newton solve:

    from sksparse.cholmod import Factor as CholeskyDecomposition

``scikit-sparse`` needs the SuiteSparse CHOLMOD native library (headers were not
available on this machine and the conda-forge build forced a NumPy 2 upgrade
that breaks diffcp/cvxpylayers, so it cannot be installed here).

This stub satisfies the import so that Theseus' *dense* optimizers work.  Any
attempt to actually use the CHOLMOD sparse factorisation raises a clear error
instead of silently returning wrong numbers.
"""

__all__ = ["cholmod"]
