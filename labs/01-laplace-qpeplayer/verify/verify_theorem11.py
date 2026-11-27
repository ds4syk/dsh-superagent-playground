#!/usr/bin/env python
"""Minimal numerical sanity check for Theorem 1.1 and the Laplace constant.

Checks
------
1. Tangent basis E(q): q^T E = 0, E^T E = I_3, and E(-q) = -E(q).
2. Euler identity at a stationary point: q^T grad J = 4 J (J is quartic).
3. Riemannian Hessian H = E^T (grad^2 J - 4 J I) E vs. a finite-difference
   Hessian of the pullback J(q(u)) with q(u) = (q + E u)/||q + E u||.
4. Implicit derivative dq*/dtheta = -E H^{-1} E^T (2 Jv^T M1 v) vs. central
   finite differences of the nearby local minimiser.
5. Antipodal invariance: H(-q) = H(q), dq*/dtheta(-q) = -dq*/dtheta(q),
   dR/dtheta invariant.
6. Translation: t*(q) = -G^{-1} D v, dt*/dtheta formula vs. finite differences.
7. Laplace integral on S^3: I(tau) = int exp(-J/tau) dq near one minimum vs.
   (2 pi tau)^{3/2}/sqrt(det H); fit the relative error order in tau.

This is a *sanity check on random instances*, not a proof and not a benchmark.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_theorem11_output.json")
rng = np.random.default_rng(20261127)
res = {}

SQ = np.sqrt(2.0)


def v2(q):
    return np.array([q[0]**2, q[1]**2, q[2]**2, q[3]**2,
                     SQ*q[0]*q[1], SQ*q[0]*q[2], SQ*q[0]*q[3],
                     SQ*q[1]*q[2], SQ*q[1]*q[3], SQ*q[2]*q[3]])


def Jv(q):
    return np.array([
        [2*q[0], 0, 0, 0],
        [0, 2*q[1], 0, 0],
        [0, 0, 2*q[2], 0],
        [0, 0, 0, 2*q[3]],
        [SQ*q[1], SQ*q[0], 0, 0],
        [SQ*q[2], 0, SQ*q[0], 0],
        [SQ*q[3], 0, 0, SQ*q[0]],
        [0, SQ*q[2], SQ*q[1], 0],
        [0, SQ*q[3], 0, SQ*q[1]],
        [0, 0, SQ*q[3], SQ*q[2]],
    ])


def Hv():
    """Hv[a] = d^2 v_a / dq dq (10 x 4 x 4)."""
    H = np.zeros((10, 4, 4))
    for i in range(4):
        H[i, i, i] = 2.0
    pairs = [(0, 1, 4), (0, 2, 5), (0, 3, 6), (1, 2, 7), (1, 3, 8), (2, 3, 9)]
    for i, j, a in pairs:
        H[a, i, j] = H[a, j, i] = SQ
    return H


HV = Hv()


def E(q):
    q0, q1, q2, q3 = q
    return np.array([
        [-q1, -q2, -q3],
        [q0, -q3, q2],
        [q3, q0, -q1],
        [-q2, q1, q0],
    ])


def Jfun(q, M):
    vv = v2(q)
    return float(vv @ M @ vv)


def grad(q, M):
    return 2 * Jv(q).T @ (M @ v2(q))


def hess(q, M):
    J = Jv(q)
    vv = v2(q)
    H = 2 * J.T @ M @ J
    Mv = M @ vv
    for a in range(10):
        if Mv[a] != 0:
            H += 2 * Mv[a] * HV[a]
    return H


def Hsphere(q, M):
    Eq = E(q)
    J = Jfun(q, M)
    return Eq.T @ (hess(q, M) - 4*J*np.eye(4)) @ Eq


def find_min(M, nstarts=120):
    best, bx = np.inf, None
    cons = [{"type": "eq", "fun": lambda z: z @ z - 1.0,
             "jac": lambda z: 2*z}]
    for _ in range(nstarts):
        s = rng.normal(size=4)
        s /= np.linalg.norm(s)
        r = minimize(lambda z: Jfun(z, M), s, jac=lambda z: grad(z, M),
                     method="SLSQP", constraints=cons,
                     options={"ftol": 1e-14, "maxiter": 2000})
        z = r.x / np.linalg.norm(r.x)
        val = Jfun(z, M)
        if val < best:
            best, bx = val, z
    return bx, best


def Rmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3],
    ])


# random cost matrix: PSD + a little indefinite part (still has isolated minima)
A = rng.normal(size=(10, 10))
M = A @ A.T
M += 0.5 * (rng.normal(size=(10, 10)))
M = 0.5 * (M + M.T)
M1 = rng.normal(size=(10, 10)); M1 = 0.5*(M1 + M1.T)

qstar, fstar = find_min(M)
qstar = qstar / np.linalg.norm(qstar)
g = grad(qstar, M)
Eq = E(qstar)
Hs = Hsphere(qstar, M)
res["qstar"] = qstar.tolist()
res["fstar"] = fstar
res["E_orthonormal_err"] = float(np.linalg.norm(Eq.T @ Eq - np.eye(3)))
res["E_orthogonal_to_q_err"] = float(abs(qstar @ Eq).max())
res["E_antipodal_err"] = float(np.linalg.norm(E(-qstar) + Eq))
res["projected_grad_norm"] = float(np.linalg.norm(Eq.T @ g))
res["euler_identity_abs_err"] = float(abs(qstar @ g - 4*fstar))
res["H_sphere_symmetric_err"] = float(np.linalg.norm(Hs - Hs.T))
res["H_sphere_eigs"] = sorted(np.linalg.eigvalsh(Hs).tolist())

# finite-difference Hessian of the pulled-back objective
def pullback(u, M):
    z = qstar + Eq @ u
    return Jfun(z / np.linalg.norm(z), M)

h = 1e-5
Hfd = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ei = np.eye(3)[i]; ej = np.eye(3)[j]
        Hfd[i, j] = (pullback(h*ei + h*ej, M) - pullback(h*ei - h*ej, M)
                     - pullback(-h*ei + h*ej, M) + pullback(-h*ei - h*ej, M)) / (4*h*h)
res["H_fd_vs_analytic_relerr"] = float(np.linalg.norm(Hfd - Hs) / np.linalg.norm(Hs))
res["H_fd_eigs"] = sorted(np.linalg.eigvalsh(Hfd).tolist())

# implicit derivative dq*/dtheta
def local_min_theta(M, M1, theta, q0):
    E0 = E(q0)
    Mt = M + theta * M1

    def q_of(u):
        z = q0 + E0 @ u
        r = np.linalg.norm(z)
        return z / r, r

    def obj(u):
        qq, _ = q_of(u)
        return Jfun(qq, Mt)

    def jac(u):
        qq, r = q_of(u)
        dqdu = (E0 - np.outer(qq, qq @ E0)) / r
        return dqdu.T @ grad(qq, Mt)

    r = minimize(obj, np.zeros(3), jac=jac, method="BFGS",
                 options={"gtol": 1e-14, "maxiter": 5000})
    u = r.x
    z, _ = q_of(u)
    return z, u

eps = 1e-4
qp, up = local_min_theta(M, M1, eps, qstar)
qm, um = local_min_theta(M, M1, -eps, qstar)
dq_fd = (qp - qm) / (2*eps)
dq_an = -Eq @ np.linalg.solve(Hs, Eq.T @ (2 * Jv(qstar).T @ (M1 @ v2(qstar))))
res["dq_implicit_vs_fd_rel"] = float(np.linalg.norm(dq_fd - dq_an) /
                                     max(np.linalg.norm(dq_an), 1e-14))
res["dq_fd"] = dq_fd.tolist()
res["dq_analytic"] = dq_an.tolist()

# antipodal: same H, opposite q-derivative, same R-derivative
Hm = Hsphere(-qstar, M)
res["H_antipodal_rel_err"] = float(np.linalg.norm(Hm - Hs) / np.linalg.norm(Hs))
dq_an_neg = -E(-qstar) @ np.linalg.solve(Hm, E(-qstar).T @ (2 * Jv(-qstar).T @ (M1 @ v2(-qstar))))
res["dq_antipodal_relation_rel"] = float(np.linalg.norm(dq_an_neg + dq_an) /
                                         max(np.linalg.norm(dq_an), 1e-14))
dR_pos = 2 * np.array([0.0])  # placeholder replaced below
# dR/dtheta via the chain rule dR = sum_a (dR/dq_a) dq_a ; compare +q and -q
def dR_dq(q):
    """Jacobian of vec(R(q)) w.r.t. q, shape (9,4)."""
    J = np.zeros((9, 4)); h = 1e-6
    for k in range(4):
        e = np.zeros(4); e[k] = h
        J[:, k] = (Rmat(q+e) - Rmat(q-e)).reshape(-1) / (2*h)
    return J
dR_pos = dR_dq(qstar) @ dq_an
dR_neg = dR_dq(-qstar) @ dq_an_neg
res["dR_antipodal_rel_err"] = float(np.linalg.norm(dR_pos - dR_neg) /
                                    max(np.linalg.norm(dR_pos), 1e-14))

# translation block
G0 = np.diag([2.0, 3.0, 4.0]) + 0.1*rng.normal(size=(3, 3))
G0 = G0 @ G0.T
G1 = 0.1*rng.normal(size=(3, 3)); G1 = 0.5*(G1 + G1.T)
D0 = 0.3*rng.normal(size=(3, 10))
D1 = 0.1*rng.normal(size=(3, 10))
def tstar(q, theta):
    G = G0 + theta*G1
    D = D0 + theta*D1
    return -np.linalg.solve(G, D @ v2(q))
t_p = tstar(qp, eps); t_m = tstar(qm, -eps)
dt_fd = (t_p - t_m) / (2*eps)
G = G0; D = D0
dt_an = -np.linalg.solve(G, G1 @ tstar(qstar, 0.0) + D1 @ v2(qstar)
                         + D @ Jv(qstar) @ dq_an)
res["dt_implicit_vs_fd_rel"] = float(np.linalg.norm(dt_fd - dt_an) /
                                     max(np.linalg.norm(dt_an), 1e-14))

# Laplace integral on S^3 near qstar, and O(tau) error order
def laplace_ratio(tau, nsamp=400000):
    """Importance-sampling estimate of int exp(-J/tau) dq over the chart
    q(u) = normalize(qstar + E u), divided by (2 pi tau)^{3/2}/sqrt(det H)."""
    L = np.linalg.cholesky(Hs)          # Hs = L L^T
    cov = tau * np.linalg.inv(Hs)
    u = rng.multivariate_normal(np.zeros(3), cov, size=nsamp)
    z = qstar[None, :] + u @ Eq.T
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    vals = np.array([Jfun(zz, M) for zz in z])
    # Gaussian proposal density
    sign, logdet = np.linalg.slogdet(cov)
    quad = np.einsum("ij,jk,ik->i", u, np.linalg.inv(cov), u)
    logp = -0.5*(3*np.log(2*np.pi) + logdet + quad)
    w = np.exp(-vals/tau - logp)
    I = np.mean(w)
    lap = np.exp(-fstar/tau) * (2*np.pi*tau)**1.5 / np.sqrt(np.linalg.det(Hs))
    return I, lap, I/lap - 1.0

taus = [0.2, 0.1, 0.05, 0.025]
errs = []
for tau in taus:
    I, lap, rel = laplace_ratio(tau)
    errs.append(rel)
    print(f"tau={tau}: I={I:.6e} laplace={lap:.6e} rel_err={rel:.3e}")
res["laplace_taus"] = taus
res["laplace_rel_errors"] = errs
# fit slope of |rel err| vs tau in log-log
sl = np.polyfit(np.log(taus), np.log(np.abs(errs)), 1)[0]
res["laplace_error_order_slope"] = float(sl)
res["laplace_constant_note"] = ("chart integral on S^3; local dR=8dq per sheet and the double cover give "
                                "int_SO3 = 4 int_S3, i.e. a factor 8 per SO(3) mode")

print(json.dumps(res, indent=2, default=str))
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, default=str)
print("wrote", OUT)
