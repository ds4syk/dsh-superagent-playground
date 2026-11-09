"""Batched dual log-det barrier solver for Shor SDPs (torch / CUDA).

Problem
-------
For a batch of independent SDPs

    min_X  <Q_b, X>   s.t.  <A_i, X> = b_i,  i = 0..m-1,   X >= 0,

this module solves the dual

    max_y  -b^T y     s.t.  H(y) = Q_b + sum_i y_i A_i >= 0

with the log-det barrier

    min_y  b^T y - mu logdet H(y).

At the barrier optimum the primal estimate is  X = mu H^{-1} >= 0,
<A_i, X> = b_i, and the duality gap is  n mu.  All linear algebra
(eigendecomposition, Cholesky, small dense solves) is batched over the
leading dimension, so a batch of SDPs is processed in parallel on the GPU.

The implementation follows the classical dual barrier method for SDP
(Vandenberghe & Boyd) and is deliberately self-contained: no CVXPY, SCS or
Mosek call is made on the GPU path.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["BarrierSolver"]


class BarrierSolver:
    """Batched dual barrier solver.

    Parameters
    ----------
    A : array_like, shape (m, n, n)
        Symmetric constraint matrices (including the homogenising constraint).
    b : array_like, shape (m,)
        Right-hand side, e.g. ``[0, ..., 0, 1]`` for a homogenised QCQP.
    start_coeff : array_like, shape (m,)
        Coefficients of a PSD matrix ``P = sum_i c_i A_i`` that is positive
        definite on the subspace spanned by the variables that matter.  A
        sufficiently large multiple of ``P`` is added to ``Q`` to obtain a
        strictly feasible starting point ``H > 0``.
    mu0, mu_final : float
        Initial and final barrier parameter.  Accuracy scales with
        ``mu_final`` (duality gap ``n * mu_final``).
    """

    def __init__(self, A, b, start_coeff, device="cuda", dtype=torch.float64,
                 mu0=1e-2, mu_final=1e-8, shrink=0.1, newton_tol=1e-10,
                 max_newton=25, armijo=1e-4, verbose=False):
        self.A = torch.as_tensor(np.asarray(A), dtype=dtype, device=device)
        self.b = torch.as_tensor(np.asarray(b), dtype=dtype, device=device)
        self.c0 = torch.as_tensor(np.asarray(start_coeff), dtype=dtype, device=device)
        self.P = torch.einsum("i,iab->ab", self.c0, self.A)
        self.device, self.dtype = device, dtype
        self.mu0, self.mu_final, self.shrink = mu0, mu_final, shrink
        self.newton_tol, self.max_newton, self.armijo = newton_tol, max_newton, armijo
        self.verbose = verbose
        self.n = self.A.shape[1]
        self.m = self.A.shape[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _sym(H):
        return 0.5 * (H + H.transpose(-1, -2))

    def _H(self, Q, y):
        return self._sym(Q + torch.einsum("ni,iab->nab", y, self.A))

    def _start(self, Q):
        B = Q.shape[0]
        s = torch.ones(B, dtype=Q.dtype, device=Q.device)
        for _ in range(60):
            H = self._sym(Q + s[:, None, None] * self.P)
            bad = torch.linalg.eigvalsh(H)[:, 0] <= 1e-7
            if not bool(bad.any()):
                return s[:, None] * self.c0[None, :], H
            s = torch.where(bad, s * 2.0, s)
        raise RuntimeError("no strictly feasible start found")

    @staticmethod
    def _chol_logdet(H):
        L, info = torch.linalg.cholesky_ex(H)
        ok = info == 0
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.log(torch.clamp(diag, min=1e-300)).sum(-1)
        return ok, logdet

    # ------------------------------------------------------------------
    def solve(self, Q, y0=None):
        """Solve a batch of SDPs.

        Parameters
        ----------
        Q : (B, n, n) tensor
            Batch of symmetric cost matrices.
        y0 : (B, m) tensor, optional
            Warm start for the dual variables.  Requires the same batch size
            as ``Q``.

        Returns
        -------
        dict with ``X`` (B,n,n), ``x`` (B,n), ``H`` (B,n,n), ``y`` (B,m),
        ``mu``, ``gap`` and per-stage ``stats``.
        """
        Q = torch.as_tensor(Q, dtype=self.dtype, device=self.device)
        B, n, _ = Q.shape
        if y0 is None:
            y, H = self._start(Q)
        else:
            y = torch.as_tensor(y0, dtype=self.dtype, device=self.device)
            if y.shape != (B, self.m):
                raise ValueError("y0 has the wrong shape for this batch")
            H = self._H(Q, y)

        mus = []
        mu = self.mu0
        while mu > self.mu_final * (1 + 1e-9):
            mus.append(mu)
            mu *= self.shrink
        mus.append(self.mu_final)

        stats = []
        for mu in mus:
            logdet_H = torch.linalg.slogdet(H)[1]
            F = self.b @ y.T - mu * logdet_H
            it = 0
            for it in range(self.max_newton):
                w, V = torch.linalg.eigh(H)
                if bool((w[:, 0] <= 1e-13 * torch.clamp(w[:, -1], min=1.0)).any()):
                    break
                invw = 1.0 / w
                Hi = self._sym((V * invw.unsqueeze(1)) @ V.transpose(1, 2))
                gF = self.b[None, :] - mu * torch.einsum("iab,nab->ni", self.A, Hi)
                L = V * torch.sqrt(invw).unsqueeze(1)
                Lt = L.transpose(1, 2)
                Bm = torch.einsum("bap,ipq,bqc->biac", Lt, self.A, L)
                G = mu * torch.einsum("biac,bkac->bik", Bm, Bm)
                G = self._sym(G) + 1e-14 * torch.eye(self.m, dtype=Q.dtype, device=Q.device)
                d = -torch.linalg.solve(G, gF.unsqueeze(-1)).squeeze(-1)

                alpha = torch.ones((), dtype=Q.dtype, device=Q.device)
                accepted = False
                for _ in range(12):
                    yn = y + alpha * d
                    Hn = self._H(Q, yn)
                    ok, ld = self._chol_logdet(Hn)
                    Fn = self.b @ yn.T - mu * ld
                    if bool((ok & (Fn < F + self.armijo * alpha * (gF * d).sum(1))).all()):
                        accepted = True
                        break
                    alpha = alpha * 0.5
                if not accepted:
                    break
                y, H, F = yn, Hn, Fn
                if (float(torch.linalg.norm(gF, dim=1).max()) < self.newton_tol
                        and float(alpha) == 1.0):
                    break
            stats.append((mu, it + 1, float(torch.linalg.norm(gF, dim=1).max())))
            if self.verbose:
                print("  mu %.1e it %2d |g| %.2e" % (mu, it + 1, stats[-1][2]), flush=True)

        w, V = torch.linalg.eigh(H)
        Hi = self._sym((V * (1.0 / w).unsqueeze(1)) @ V.transpose(1, 2))
        X = mu * Hi
        x = V[:, :, 0] / V[:, 0:1, 0]
        return dict(X=X, x=x, H=H, y=y, mu=mu, gap=n * mu, stats=stats)

    # ------------------------------------------------------------------
    def gradient(self, x, H, grad_x):
        """Reverse-mode gradient of a loss through the QCQP KKT conditions.

        Parameters
        ----------
        x : (B, n) primal vector solution (homogenising entry normalised to 1)
        H : (B, n, n) certificate matrix
        grad_x : (B, n) gradient of the loss with respect to ``x``

        Returns
        -------
        grad_Q : (B, n, n) symmetric gradient with respect to ``Q``
        y : (B, n+r) KKT adjoint (diagnostic)
        """
        B, n = x.shape
        # Reduced constraint Jacobian used by SDPR Theorem 2: the six
        # orthogonality constraints and the homogenising constraint.
        Grows = torch.cat([self.A[:6], self.A[-1:]], dim=0)
        G = torch.einsum("ria,na->nri", Grows, x)
        r = G.shape[1]
        M = torch.zeros(B, n + r, n + r, dtype=x.dtype, device=x.device)
        M[:, :n, :n] = H
        M[:, :n, n:] = G.transpose(1, 2)
        M[:, n:, :n] = G
        rhs = torch.zeros(B, n + r, dtype=x.dtype, device=x.device)
        rhs[:, :n] = grad_x
        y = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        y1 = y[:, :n]
        grad_Q = -0.5 * (x.unsqueeze(2) * y1.unsqueeze(1)
                         + y1.unsqueeze(2) * x.unsqueeze(1))
        return grad_Q, y

    # ------------------------------------------------------------------
    def certificate(self, out):
        """Basic tightness/certificate diagnostics for a solved batch."""
        H, X, x = out["H"], out["X"], out["x"]
        wH = torch.linalg.eigvalsh(H)
        wX = torch.linalg.eigvalsh(X)
        ratio = wX[:, -1] / torch.clamp(wX[:, -2], min=1e-300)
        return dict(
            H_min_eig=wH[:, 0].detach().cpu(),
            corank1=(wH[:, 1] > 1e-6).sum().item(),
            rank_ratio=ratio.detach().cpu(),
            Hx_res=torch.linalg.norm(torch.einsum("bij,bj->bi", H, x), dim=1).detach().cpu(),
            gap=float(out["gap"]),
        )
