"""PyTorch layer: batched GPU SDPR forward + implicit-differentiation backward."""
from __future__ import annotations

import torch

from .barrier import BarrierSolver

__all__ = ["SDPRLayer"]


class _SDPRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, solver, y0):
        with torch.no_grad():
            out = solver.solve(Q, y0=y0)
        solver.last_out = out
        ctx.solver = solver
        ctx.save_for_backward(out["x"], out["H"])
        return out["X"], out["x"]

    @staticmethod
    def backward(ctx, grad_X, grad_x):
        x, H = ctx.saved_tensors
        if grad_x is None:
            grad_x = torch.zeros_like(x)
        if grad_X is not None:
            # X = x x^T  =>  dL/dx = (grad_X + grad_X^T) x
            grad_x = grad_x + torch.einsum(
                "bij,bj->bi", grad_X + grad_X.transpose(1, 2), x)
        grad_Q, _ = ctx.solver.gradient(x, H, grad_x)
        return grad_Q, None, None


class SDPRLayer(torch.nn.Module):
    """Differentiable batched SDP relaxation layer.

    Example
    -------
    >>> layer = SDPRLayer(A, b, c0)
    >>> X, x = layer(Q)             # Q: (B, 13, 13), requires_grad optional
    >>> loss = x.sum()
    >>> loss.backward()             # populates Q.grad
    """

    def __init__(self, A, b, start_coeff, device="cuda", dtype=torch.float64,
                 mu0=1e-2, mu_final=1e-8, **kwargs):
        super().__init__()
        self.solver = BarrierSolver(A, b, start_coeff, device=device, dtype=dtype,
                                    mu0=mu0, mu_final=mu_final, **kwargs)
        self._y = None
        self.last_out = None

    def forward(self, Q, warm_start=False):
        y0 = None
        if warm_start and self._y is not None and self._y.shape[0] == Q.shape[0]:
            y0 = self._y
        X, x = _SDPRFunction.apply(Q, self.solver, y0)
        self._y = self.solver.last_out["y"].detach()
        self.last_out = self.solver.last_out
        return X, x

    def certificate(self):
        if self.last_out is None:
            raise RuntimeError("call the layer first")
        return self.solver.certificate(self.last_out)
