"""Compatibility shims required to run theseus-ai 0.2.1 on this machine.

Two independent problems, both environmental (no repo change needed):

1. ``sksparse`` (scikit-sparse / SuiteSparse CHOLMOD) is not installable here:
   the conda-forge build forces NumPy 2 (which breaks diffcp/cvxpylayers) and the
   pip sdist needs CHOLMOD headers that are absent.  theseus imports sksparse at
   package-import time even for dense solves; ``repro/upstream/stubs/sksparse``
   satisfies that import (and raises if CHOLMOD is actually used).  Theseus'
   default Gauss-Newton linear solver is ``CholeskyDenseSolver``, so no CHOLMOD
   code path is exercised.

2. theseus 0.2.1's ``NonlinearOptimizer._merge_infos`` calls
   ``torch.where(cond[:, None], sol, sol)`` which relied on NumPy-style
   broadcasting that PyTorch >= 2.4 no longer accepts for (B,1) vs (B,*,*)
   tensors ("The size of tensor a (20) must match the size of tensor b (3)").
   We re-implement ``_merge_infos`` verbatim with the condition reshaped to
   ``(B, *([1]*(sol.ndim-1)))``, which is the intended semantics.

Call ``install()`` before building the Theseus layer.
"""

import numpy as np
import torch

from theseus.optimizer.nonlinear.nonlinear_optimizer import (
    BackwardMode,
    NonlinearOptimizer,
    NonlinearOptimizerInfo,
    NonlinearOptimizerStatus,
)


def _merge_infos(self, grad_loop_info, num_no_grad_iters, num_grad_iters, info):
    total_iters = num_no_grad_iters + num_grad_iters
    info_idx = slice(num_no_grad_iters + 1, total_iters + 1)
    grad_info_idx = slice(1, num_grad_iters + 1)
    if info.err_history is not None:
        info.err_history[:, info_idx] = grad_loop_info.err_history[:, grad_info_idx]
    if info.state_history is not None:
        for var in self.objective.optim_vars.values():
            info.state_history[var.name][
                ..., info_idx
            ] = grad_loop_info.state_history[var.name][..., grad_info_idx]

    if info.best_solution is not None:
        best_solution = {}
        best_err_no_grad = info.best_err
        best_err_grad = grad_loop_info.best_err
        idx_no_grad = (best_err_no_grad < best_err_grad).cpu().view(-1, 1)
        best_err = torch.minimum(best_err_no_grad, best_err_grad)
        for var_name in info.best_solution:
            sol_no_grad = info.best_solution[var_name]
            sol_grad = grad_loop_info.best_solution[var_name]
            # only change vs upstream: reshape the (B,1) condition to the
            # solution's trailing dimensions (torch>=2.4 broadcasting rules)
            cond = idx_no_grad.view(
                -1, *([1] * (sol_no_grad.ndim - 1))
            )
            best_solution[var_name] = torch.where(cond, sol_no_grad, sol_grad)
        info.best_solution = best_solution
        info.best_err = best_err

    M = info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
    assert np.all(
        (grad_loop_info.status[M] == NonlinearOptimizerStatus.MAX_ITERATIONS)
        | (grad_loop_info.status[M] == NonlinearOptimizerStatus.CONVERGED)
    )
    info.status[M] = grad_loop_info.status[M]
    info.converged_iter[M] = info.converged_iter[M] + grad_loop_info.converged_iter[M]
    info.converged_iter[
        M & (grad_loop_info.status == NonlinearOptimizerStatus.MAX_ITERATIONS)
    ] = -1


def install():
    import theseus  # noqa: F401  (import early: needs the sksparse stub on path)

    NonlinearOptimizer._merge_infos = _merge_infos
    return theseus


if __name__ == "__main__":
    install()
    print("theseus compat shims installed")
