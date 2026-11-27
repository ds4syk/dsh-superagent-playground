#!/usr/bin/env python3
"""Numerical check of the Riemannian Hessian of the robust pose cost on S^3 x R^3.

For  F(z) = sum_i rho(e_i(z)),  z=(q,t),  e_i = ||R(q) p_i + t - y_i||_{W_i}^2,
the doc's formula (Euclidean Hessians, projected gradients)

    H_doc = B^T [ sum_i rho'(e_i) H_i^E + rho''(e_i) g_i^E (g_i^E)^T ] B

is compared with

    H_corr = H_doc - lambda_rob * diag(I_3, 0_3),
    lambda_rob = q^T g_q = sum_i rho'(e_i) lambda_i,   lambda_i = q^T g_{i,q},

and with a finite-difference Hessian obtained from geodesic second differences.
It also checks lambda_i = 4 u_i^T W_i r_i  (u_i = R(q) p_i), which is *not* 4 e_i.

Run:  /home/syk/miniconda3/bin/python riemann_hessian_check.py
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(7)


# ----------------------------- GM kernel ---------------------------------
C = 1.0
C2 = C * C
C4 = C2 * C2


def rho(e):
    return C2 * e / (C2 + e)


def drho(e):
    return C4 / (C2 + e) ** 2


def d2rho(e):
    return -2.0 * C4 / (C2 + e) ** 3


# ----------------------------- geometry ----------------------------------
def Rmat(q):
    """Homogeneous (quadratic) quaternion rotation map; R(q) for ||q||=1."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_A(p):
    """Constant symmetric A_k with (R(q)p)_k = q^T A_k q."""
    E = np.eye(4)

    def uk(q):
        return (Rmat(q) @ p)

    A = np.zeros((3, 4, 4))
    for k in range(3):
        for i in range(4):
            A[k, i, i] = uk(E[i])[k]
        for i in range(4):
            for j in range(i + 1, 4):
                val = (uk(E[i] + E[j])[k] - uk(E[i])[k] - uk(E[j])[k]) / 2.0
                A[k, i, j] = A[k, j, i] = val
    return A


def tangent_basis(q):
    """4x3 orthonormal basis of the tangent space at q in S^3."""
    _, _, vh = np.linalg.svd(q.reshape(1, 4))
    return vh[1:].T


def exp_map(q, t, xi, Bq):
    """Geodesic exponential on S^3 x R^3; xi in R^6 (tangent coordinates)."""
    xq = Bq @ xi[:3]
    xt = xi[3:]
    nq = np.linalg.norm(xq)
    if nq < 1e-14:
        q2 = q.copy()
    else:
        q2 = np.cos(nq) * q + np.sin(nq) * xq / nq
    return q2, t + xt


# ----------------------------- problem -----------------------------------
def make_problem(N=6):
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    t_true = np.array([0.10, -0.10, 0.20])
    p = rng.normal(size=(N, 3))
    y = np.array([Rmat(q_true) @ p[i] + t_true + 0.10 * rng.normal(size=3) for i in range(N)])
    W = []
    for _ in range(N):
        A = rng.normal(size=(3, 3))
        W.append(A.T @ A + 0.15 * np.eye(3))
    q0 = rng.normal(size=4)
    q0 /= np.linalg.norm(q0)
    t0 = np.array([0.15, -0.12, 0.22])
    return p, y, W, q0, t0


def residual_terms(p, y, W, q, t):
    """Return e_i, Euclidean g_i (7), Euclidean H_i (7x7), u_i, r_i, J_i."""
    N = len(p)
    e = np.zeros(N)
    g = np.zeros((N, 7))
    H = np.zeros((N, 7, 7))
    u = np.zeros((N, 3))
    r = np.zeros((N, 3))
    J = np.zeros((N, 3, 4))
    for i in range(N):
        A = build_A(p[i])
        ui = np.array([q @ A[k] @ q for k in range(3)])
        Ji = np.array([2.0 * A[k] @ q for k in range(3)])  # 3x4
        ri = ui + t - y[i]
        Wi = W[i]
        ei = ri @ Wi @ ri
        gq = 2.0 * Ji.T @ Wi @ ri
        gt = 2.0 * Wi @ ri
        Hqq = 2.0 * Ji.T @ Wi @ Ji + 4.0 * sum((Wi @ ri)[k] * A[k] for k in range(3))
        Hqt = 2.0 * Ji.T @ Wi
        Htt = 2.0 * Wi
        Hi = np.zeros((7, 7))
        Hi[:4, :4] = Hqq
        Hi[:4, 4:] = Hqt
        Hi[4:, :4] = Hqt.T
        Hi[4:, 4:] = Htt
        e[i] = ei
        g[i, :4] = gq
        g[i, 4:] = gt
        H[i] = Hi
        u[i] = ui
        r[i] = ri
        J[i] = Ji
    return e, g, H, u, r, J


def Fval(p, y, W, q, t):
    e, *_ = residual_terms(p, y, W, q, t)
    return float(np.sum(rho(e)))


def fd_hessian(p, y, W, q, t, Bq, h=1e-4):
    """Hessian in tangent coordinates via geodesic second differences."""
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)
    f0 = Fval(p, y, W, q, t)
    Q = np.zeros((6, 6))

    def qform(xi):
        qp, tp = exp_map(q, t, h * xi, Bq)
        qm, tm = exp_map(q, t, -h * xi, Bq)
        return (Fval(p, y, W, qp, tp) - 2.0 * f0 + Fval(p, y, W, qm, tm)) / (h * h)

    for j in range(6):
        for k in range(j, 6):
            ej = np.zeros(6)
            ek = np.zeros(6)
            ej[j] = 1.0
            ek[k] = 1.0
            Q[j, k] = Q[k, j] = (qform(ej + ek) - qform(ej - ek)) / 4.0
    return Q, Bfull


def main():
    out = []
    out.append("=== Riemannian Hessian check on S^3 x R^3 (3D-3D, GM kernel) ===")
    N = 6
    p, y, W, q0, t0 = make_problem(N)
    e, g, Hi, u, r, J = residual_terms(p, y, W, q0, t0)
    out.append(f"N = {N}, c = {C}")
    out.append(f"q0 = {q0}, t0 = {t0}")
    out.append(f"e_i = {e}")
    out.append(f"rho'(e_i) = {np.array([drho(x) for x in e])}")

    gtot = np.sum(np.array([drho(e[i]) for i in range(N)])[:, None] * g, axis=0)
    Htot = np.zeros((7, 7))
    for i in range(N):
        Htot += drho(e[i]) * Hi[i] + d2rho(e[i]) * np.outer(g[i], g[i])

    Bq = tangent_basis(q0)
    Bfull = np.zeros((7, 6))
    Bfull[:4, :3] = Bq
    Bfull[4:, 3:] = np.eye(3)

    H_doc = Bfull.T @ Htot @ Bfull
    lam_rob = float(q0 @ gtot[:4])
    D6 = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    H_corr = H_doc - lam_rob * D6

    # alternative route: sum of Riemannian Hessians of e_i
    H_alt = np.zeros((6, 6))
    for i in range(N):
        lam_i = float(q0 @ g[i, :4])
        Hess_ei = Bfull.T @ Hi[i] @ Bfull - lam_i * D6
        grad_ei = Bfull.T @ g[i]
        H_alt += drho(e[i]) * Hess_ei + d2rho(e[i]) * np.outer(grad_ei, grad_ei)

    H_fd, _ = fd_hessian(p, y, W, q0, t0, Bq, h=1e-4)

    def relerr(A, B):
        return np.linalg.norm(A - B) / max(1.0, np.linalg.norm(B))

    out.append("")
    out.append(f"lambda_rob = {lam_rob:+.6f}")
    out.append(f"||H_doc  - H_fd||_F / ||H_fd||_F = {relerr(H_doc,  H_fd):.3e}")
    out.append(f"||H_corr - H_fd||_F / ||H_fd||_F = {relerr(H_corr, H_fd):.3e}")
    out.append(f"||H_alt  - H_corr||_F            = {np.linalg.norm(H_alt-H_corr):.3e}")
    out.append(f"eig(H_fd)   = {np.linalg.eigvalsh(H_fd)}")
    out.append(f"eig(H_doc)  = {np.linalg.eigvalsh(H_doc)}")
    out.append(f"eig(H_corr) = {np.linalg.eigvalsh(H_corr)}")

    # lambda_i checks
    lam_i = np.array([q0 @ g[i, :4] for i in range(N)])
    lam_u = np.array([4.0 * u[i] @ W[i] @ r[i] for i in range(N)])
    out.append("")
    out.append("lambda_i = q^T g_{i,q}  vs  4 u_i^T W_i r_i  vs  4 e_i:")
    for i in range(N):
        out.append(f"  i={i}: e={e[i]:+.6f}  lam={lam_i[i]:+.6f}  "
                   f"4uWr={lam_u[i]:+.6f}  4e={4*e[i]:+.6f}  "
                   f"|lam-4uWr|={abs(lam_i[i]-lam_u[i]):.2e}")
    out.append(f"max |lambda_i - 4u_i^T W_i r_i| = {np.max(np.abs(lam_i-lam_u)):.3e}")
    out.append(f"max |lambda_i - 4 e_i|         = {np.max(np.abs(lam_i-4*e)):.3e}")

    # trivial constant-on-sphere test: f(q)=0.5||q||^2
    out.append("")
    out.append("[constant test] f(q)=0.5||q||^2 restricted to S^3")
    Bq2 = tangent_basis(q0)
    H_doc_const = Bq2.T @ np.eye(4) @ Bq2
    lam_const = float(q0 @ q0)
    H_corr_const = H_doc_const - lam_const * np.eye(3)
    out.append(f"  eig(H_doc_const)  = {np.linalg.eigvalsh(H_doc_const)}  (should be O(1), wrong)")
    out.append(f"  eig(H_corr_const) = {np.linalg.eigvalsh(H_corr_const)}  (should be 0)")

    text = "\n".join(out)
    print(text)
    with open(HERE / "riemann_hessian_check_output.txt", "w") as f:
        f.write(text + "\n")

    # figure: error heatmaps + eigenvalues
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
        vmax = max(np.max(np.abs(H_doc - H_fd)), np.max(np.abs(H_corr - H_fd)), 1e-9)
        for k, (M, title) in enumerate(
            [(H_doc - H_fd, "H_doc - H_fd"), (H_corr - H_fd, "H_corr - H_fd")]
        ):
            im = ax[k].imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax[k].set_title(title)
            fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)
        ax[2].bar(
            ["H_doc", "H_corr"],
            [relerr(H_doc, H_fd), relerr(H_corr, H_fd)],
            color=["tab:red", "tab:blue"],
        )
        ax[2].set_yscale("log")
        ax[2].set_ylabel("relative Frobenius error vs FD")
        ax[2].set_title("Hessian check")
        fig.tight_layout()
        fig.savefig(HERE / "riemann_hessian_check.png", dpi=160)
        print("saved", HERE / "riemann_hessian_check.png")
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
