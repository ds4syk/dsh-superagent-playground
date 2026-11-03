"""Rotation synchronization as a homogeneous QCQP + Shor SDP (SDPRLayer).

Problem
-------
Given relative-rotation measurements ``R_ij ~= R_i R_j^T`` with weights
``w_ij`` and a prior pulling ``R_0`` to the identity, solve

    min_{R_0..R_{N-1} in SO(3)}  sum_ij w_ij ||R_ij - R_i R_j^T||_F^2
                                 + w_0 ||R_0 - I||_F^2 .

After homogenisation ``x = [h, vec(R_0), ..., vec(R_{N-1})]`` (column-major
``vec``) and quadratic constraints encoding ``R_i in SO(3)``, this is a
homogeneous QCQP of the form used by SDPRLayers:

    min_x x^T Q_theta x   s.t.  x^T A_k x = 0,  x^T A_0 x = 1.

The Shor relaxation drops ``rank(X)=1`` and is solved with CVXPY.  When the
relaxation is tight the leading eigenvector of ``X`` is the globally optimal
``x``, and the SDP duals give the certificate matrix

    H = Q + sum_k lambda_k A_k + lambda_0 A_0 >= 0,   H x = 0.

The backward pass is the SDPRLayer implicit-differentiation rule: solve the
reduced KKT system with ``H`` and the independent constraint gradients, then
``dL/dQ = -sym(y x^T)``.  The layer is therefore a drop-in
``torch.autograd.Function`` whose input is the cost matrix ``Q`` produced by an
upstream network.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch

__all__ = [
    "RotSyncProblem",
    "SDPRunner",
    "SDPRLayer",
    "solve_sdp",
    "extract_rotations",
    "project_so3",
    "reduced_gradient_rows",
]


# ---------------------------------------------------------------------------
# construction of the polynomial problem
# ---------------------------------------------------------------------------


class RotSyncProblem:
    """Build ``Q`` and the constraint matrices for ``N`` rotations."""

    def __init__(self, n_nodes: int, redundant: bool = True):
        self.N = int(n_nodes)
        self.n = 1 + 9 * self.N
        self.redundant = bool(redundant)
        self.A_list = self._constraint_matrices()
        self.A0 = np.zeros((self.n, self.n))
        self.A0[0, 0] = 1.0

    def node_slice(self, i: int) -> slice:
        return slice(1 + 9 * i, 1 + 9 * (i + 1))

    def _constraint_matrices(self) -> list[np.ndarray]:
        mats: list[np.ndarray] = []
        n = self.n
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
        for i in range(self.N):
            base = 1 + 9 * i
            # 6 orthogonality constraints: c_a . c_b = delta_ab h^2
            for a in range(3):
                for b in range(a, 3):
                    A = np.zeros((n, n))
                    for k in range(3):
                        p, q = base + 3 * a + k, base + 3 * b + k
                        if a == b:
                            A[p, q] += 1.0
                        else:
                            A[p, q] += 0.5
                            A[q, p] += 0.5
                    if a == b:
                        A[0, 0] -= 1.0
                    mats.append(A)
            # 9 handedness constraints: c_a = c_b x c_c (cyclic)
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                for l in range(3):
                    A = np.zeros((n, n))
                    idx = base + 3 * a + l
                    A[0, idx] += 0.5  # + h c_a[l]
                    A[idx, 0] += 0.5
                    for m in range(3):
                        for nn in range(3):
                            coef = -eps[l, m, nn]
                            if abs(coef) < 1e-12:
                                continue
                            p, q = base + 3 * b + m, base + 3 * c + nn
                            A[p, q] += 0.5 * coef
                            A[q, p] += 0.5 * coef
                    mats.append(A)
            # 9 redundant row/column norm equalities (tightening)
            if self.redundant:
                for a in range(3):
                    for b in range(3):
                        A = np.zeros((n, n))
                        for k in range(3):
                            A[base + 3 * a + k, base + 3 * a + k] += 1.0
                            A[base + 3 * k + b, base + 3 * k + b] -= 1.0
                        mats.append(A)
        return mats

    # -- cost ------------------------------------------------------------
    def build_Q(
        self,
        meas: np.ndarray,
        weights: np.ndarray,
        edges: list[tuple[int, int]],
        prior_weight: float = 10.0,
    ) -> np.ndarray:
        """Cost matrix for ``meas[e] = R_ij``, ``weights[e] = w_ij`` (numpy)."""
        Q = np.zeros((self.n, self.n), dtype=np.float64)
        eye3 = np.eye(3)
        for (i, j), Rm, w in zip(edges, meas, weights):
            w = float(w)
            if w <= 0:
                continue
            si, sj = self.node_slice(i), self.node_slice(j)
            Q[si, si] += 2.0 * w * np.eye(9)
            Q[sj, sj] += 2.0 * w * np.eye(9)
            cross = -w * np.kron(eye3, Rm)
            Q[si, sj] += cross
            Q[sj, si] += cross.T
        for k in range(3):
            idx = 1 + 3 * k + k
            Q[0, idx] += -prior_weight
            Q[idx, 0] += -prior_weight
        return Q


# ---------------------------------------------------------------------------
# SO(3) helpers
# ---------------------------------------------------------------------------


def project_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def extract_rotations(x: np.ndarray, n_nodes: int) -> np.ndarray:
    Rs = []
    for i in range(n_nodes):
        c = np.asarray(x[1 + 9 * i : 1 + 9 * (i + 1)], dtype=np.float64)
        Rs.append(project_so3(c.reshape(3, 3, order="F")))
    return np.stack(Rs)


def reduced_gradient_rows(G: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Indices of a maximal independent subset of the rows of ``G``."""
    if G.size == 0:
        return np.zeros(0, dtype=int)
    _, R, piv = sla.qr(G.T, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return np.zeros(0, dtype=int)
    rank = int(np.sum(diag > tol * max(1.0, diag[0])))
    return np.asarray(piv[:rank], dtype=int)


# ---------------------------------------------------------------------------
# SDP solve + certificate
# ---------------------------------------------------------------------------


class SDPRunner:
    """A compiled CVXPY SDP with ``Q`` as a parameter (reused across calls)."""

    def __init__(
        self,
        problem: RotSyncProblem,
        solver: str = "SCS",
        eps: float = 1e-7,
        verbose: bool = False,
    ):
        self.problem = problem
        self.solver = solver
        self.eps = eps
        self.verbose = verbose
        n = problem.n
        self.Q = cp.Parameter((n, n), value=np.zeros((n, n)))
        self.X = cp.Variable((n, n), symmetric=True)
        self.cons = [cp.trace(sp.csr_matrix(A) @ self.X) == 0.0 for A in problem.A_list]
        self.cons.append(cp.trace(sp.csr_matrix(problem.A0) @ self.X) == 1.0)
        self.cons.append(self.X >> 0)
        self.prob = cp.Problem(cp.Minimize(cp.trace(self.Q @ self.X)), self.cons)

    def solve(self, Q: np.ndarray) -> dict:
        Qsym = 0.5 * (Q + Q.T)
        self.Q.value = Qsym
        t0 = time.time()
        if self.solver == "SCS":
            self.prob.solve(solver=cp.SCS, eps=self.eps, max_iters=200_000, verbose=self.verbose)
        elif self.solver == "CLARABEL":
            self.prob.solve(solver=cp.CLARABEL, verbose=self.verbose)
        else:
            self.prob.solve(solver=self.solver, verbose=self.verbose)
        solve_time = time.time() - t0
        if self.X.value is None:
            raise RuntimeError(f"SDP failed: status={self.prob.status}")
        Xv = np.asarray(self.X.value, dtype=np.float64)
        w, V = np.linalg.eigh(0.5 * (Xv + Xv.T))
        w = np.maximum(w, 0.0)
        x = V[:, -1] * np.sqrt(max(w[-1], 1e-30))
        if x[0] < 0:
            x = -x
        h = x[0] if abs(x[0]) >= 1e-12 else 1e-12
        x = x / h
        # certificate from the equality-constraint duals (try both sign conventions)
        lam = np.array([float(c.dual_value) for c in self.cons[:-2]], dtype=np.float64)
        lam0 = float(self.cons[-2].dual_value)
        cands = []
        for sign in (+1.0, -1.0):
            H = Qsym + sign * lam0 * self.problem.A0
            for l, A in zip(sign * lam, self.problem.A_list):
                H += l * A
            H = 0.5 * (H + H.T)
            eigs = np.linalg.eigvalsh(H)
            cands.append((eigs[0], np.linalg.norm(H @ x), H, eigs))
        cands.sort(key=lambda t: (-t[0], t[1]))
        _, hx_norm, H, eigs = cands[0]
        rank_gap = float(w[-2] / w[-1]) if len(w) > 1 and w[-1] > 0 else 0.0
        return dict(
            X=Xv,
            x=x,
            H=H,
            eig_min=float(eigs[0]),
            eig_max=float(eigs[-1]),
            hx_norm=float(hx_norm),
            rank_gap=rank_gap,
            solve_time=solve_time,
            status=str(self.prob.status),
            objective=float(self.prob.value),
        )


def solve_sdp(
    Q: np.ndarray,
    problem: RotSyncProblem,
    solver: str = "SCS",
    eps: float = 1e-7,
    runner: SDPRunner | None = None,
) -> dict:
    if runner is None:
        runner = SDPRunner(problem, solver=solver, eps=eps)
    return runner.solve(Q)


# ---------------------------------------------------------------------------
# torch layer
# ---------------------------------------------------------------------------


class _SDPRSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor):
        problem = _CURRENT_PROBLEM[0]
        runner = _CURRENT_RUNNER[0]
        out = runner.solve(Q.detach().cpu().numpy())
        x = torch.as_tensor(out["x"], dtype=Q.dtype, device=Q.device)
        G = np.stack([A @ out["x"] for A in problem.A_list] + [problem.A0 @ out["x"]])
        rows = reduced_gradient_rows(G)
        rows = np.unique(np.append(rows, len(problem.A_list)))
        ctx.save_for_backward(x)
        ctx.H = out["H"]
        ctx.G = G[rows]
        return x, out["eig_min"], out["hx_norm"], out["rank_gap"]

    @staticmethod
    def backward(ctx, grad_x, *args):
        (x,) = ctx.saved_tensors
        x_np = x.detach().cpu().numpy()
        g = grad_x.detach().cpu().numpy()
        H, G = ctx.H, ctx.G
        n, r = H.shape[0], G.shape[0]
        M = np.zeros((n + r, n + r))
        M[:n, :n] = H
        M[:n, n:] = G.T
        M[n:, :n] = G
        rhs = np.zeros(n + r)
        rhs[:n] = g
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        y = sol[:n]
        grad_Q = -0.5 * (np.outer(y, x_np) + np.outer(x_np, y))
        return torch.as_tensor(grad_Q, dtype=grad_x.dtype, device=grad_x.device)


_CURRENT_PROBLEM: list[RotSyncProblem] = [RotSyncProblem(8)]
_CURRENT_RUNNER: list[SDPRunner] = [SDPRunner(_CURRENT_PROBLEM[0])]


def SDPRLayer(Q: torch.Tensor, problem: RotSyncProblem, runner: SDPRunner | None = None):
    """Differentiable SDPR rotation-synchronization layer (input: cost matrix)."""
    _CURRENT_PROBLEM[0] = problem
    if runner is None:
        if _CURRENT_RUNNER[0].problem is not problem:
            runner = SDPRunner(problem)
        else:
            runner = _CURRENT_RUNNER[0]
    _CURRENT_RUNNER[0] = runner
    x, eig_min, hx_norm, rank_gap = _SDPRSolve.apply(Q)
    diag = {"eig_min": eig_min, "hx_norm": hx_norm, "rank_gap": rank_gap}
    return x, diag
