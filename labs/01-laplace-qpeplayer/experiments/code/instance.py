#!/usr/bin/env python
"""Synthetic two-mode QPEP instance with an exactly controlled branch crossing.

Construction (all in the v2 / 10x10 QPEP parameterisation)
-----------------------------------------------------------
* q_A = e0 = (1,0,0,0) (identity rotation) and q_B = e3 = (0,0,0,1)
  (180 deg rotation about z) are the two symmetric poses.
* g(q) = q (x) k is the body-frame 180 deg symmetry; on v2 it acts through an
  orthogonal 10x10 matrix P (v2(g q) = P v2(q)).  A P-invariant M_sym makes the
  two poses exactly degenerate: J_A(0) = J_B(0), i.e. the branch crossing sits
  exactly at theta = 0.
* M(0) = M_sym = P-average of  -alpha(e0 e0^T + e3 e3^T)
                              + beta (e0+e3)(e0+e3)^T + eta * R_sym
  where R_sym is a fixed random P-averaged symmetric matrix.  The first two
  terms are a double well on q0^2,q3^2 with an explicit barrier; R_sym makes the
  wells generic (non-degenerate Hessian, slightly displaced minima).
* M(theta) = M_sym + theta * M_break, M_break random and NOT P-invariant, scaled
  so that d(J_B - J_A)/dtheta|_0 = SLOPE_TARGET > 0.

The translation block uses fixed (theta-independent) G, D:
    t*(q) = -G^{-1} D v2(q),     J_red(q) = v2^T M v2.
"""
from __future__ import annotations

import numpy as np

import qpep_core as core

ALPHA = 1.0
BETA = 0.6
ETA = 0.35
SLOPE_TARGET = 2.0
D_SCALE = 0.30


# ------------------------------------------------------------------ helpers

def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def right_mult_matrix(p):
    """4x4 matrix L with L q = q (x) p (quaternion right multiplication)."""
    p0, p1, p2, p3 = p
    return np.array([
        [p0, -p1, -p2, -p3],
        [p1, p0, -p3, p2],
        [p2, p3, p0, -p1],
        [p3, -p2, p1, p0],
    ])


def P_of_right_mult(p, rng):
    """10x10 orthogonal matrix with P v2(q) = v2(q (x) p).  Verified numerically."""
    L = right_mult_matrix(p)
    Q = rng.normal(size=(10, 4))
    Q = _normalize(Q)
    V = core.v2(Q)                 # (10,10)  rows = v2(q_i)
    Vp = core.v2(Q @ L.T)          # v2(L q_i)
    P = np.linalg.solve(V, Vp)    # P @ V = Vp  ->  P v2(q_i) = v2(L q_i)
    return P


# ------------------------------------------------------------------ instance

class Instance:
    def __init__(self, seed=20261127, alpha=ALPHA, beta=BETA, eta=ETA,
                 slope_target=SLOPE_TARGET, d_scale=D_SCALE, n_starts=512,
                 strict=False):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.seed = seed
        self.params = dict(seed=seed, alpha=alpha, beta=beta, eta=eta,
                           slope_target=slope_target, d_scale=d_scale)

        k = np.array([0.0, 0.0, 0.0, 1.0])
        self.P = P_of_right_mult(k, rng)
        # verify orthogonality and v2-equivariance on random q
        Q = _normalize(rng.normal(size=(64, 4)))
        assert np.allclose(self.P @ self.P.T, np.eye(10), atol=1e-12)
        assert np.allclose(core.v2(Q @ right_mult_matrix(k).T), core.v2(Q) @ self.P.T,
                           atol=1e-12)

        e0 = np.array([1.0, 0.0, 0.0, 0.0])
        e3 = np.array([0.0, 0.0, 0.0, 1.0])
        u0, u3 = core.v2(e0), core.v2(e3)

        R = rng.normal(size=(10, 10))
        R = 0.5 * (R + R.T)
        R = 0.5 * (R + self.P @ R @ self.P.T)
        M_core = (-alpha * (np.outer(u0, u0) + np.outer(u3, u3))
                  + beta * np.outer(u0 + u3, u0 + u3))
        self.M_sym = 0.5 * (M_core + self.P @ M_core @ self.P.T) + eta * R
        self.M_sym = 0.5 * (self.M_sym + self.M_sym.T)

        # translation model: t* = -G^{-1} D v2
        G = np.diag([2.0, 3.0, 4.0]) + 0.05 * rng.normal(size=(3, 3))
        G = 0.5 * (G + G.T)
        G = G @ G.T / 4.0 + np.eye(3)
        D = d_scale * rng.normal(size=(3, 10))
        self.G, self.D = G, D
        self.GinvD = -np.linalg.solve(G, D)          # t* = GinvD @ v2

        # modes at theta = 0 and the break direction
        self.modes0 = self.find_modes(self.M_sym, n_starts=n_starts, polish=True)
        modes0 = self.modes0
        if strict and len(modes0['q']) != 2:
            raise RuntimeError(f"expected 2 minima, got {len(modes0['q'])}")
        if len(modes0['q']) < 2:
            raise RuntimeError("fewer than 2 minima")
        v = core.v2(modes0['q'])
        order = np.argsort([np.linalg.norm(qq - e0) for qq in modes0['q']])[:2]
        vA, vB = v[order[0]], v[order[1]]

        # symmetry-breaking direction: flips the depth of the two wells,
        # M_break = c * (u0 u0^T - u3 u3^T);  P M_break P^T = -M_break, so
        # M(-theta) = P M(theta) P^T  and  Delta J(-theta) = -Delta J(theta).
        Rb = np.outer(u0, u0) - np.outer(u3, u3)
        slope_raw = vB @ Rb @ vB - vA @ Rb @ vA
        self.M_break = Rb * (slope_target / slope_raw)
        self.slope0 = slope_target

        # label modes at theta=0
        self.qA0 = modes0['q'][order[0]]
        self.qB0 = modes0['q'][order[1]]

    # -------------------------------------------------------------- cost
    def M(self, theta):
        return self.M_sym + theta * self.M_break

    def dM(self):
        return self.M_break

    def J(self, q, theta=0.0):
        return core.Jval(q, self.M(theta))

    def pose(self, q):
        return core.pose_vector(q, self.GinvD)

    # -------------------------------------------------------------- mode finding
    def find_modes(self, M, n_starts=512, starts=None, iters=400, tol=1e-13,
                   polish=True):
        """Batched damped Newton on S^3; returns distinct minima (up to +-q)."""
        rng = np.random.default_rng(self.seed + 7)
        if starts is None:
            S = _normalize(rng.normal(size=(n_starts, 4)))
        else:
            S = _normalize(np.asarray(starts, float))
        q = S.copy()
        Jc = core.Jval(q, M)
        eye = np.eye(3)
        for _ in range(iters):
            g = core.proj_grad(q, M)
            gn = np.linalg.norm(g, axis=1)
            alive = gn > tol
            if not np.any(alive):
                break
            H = core.H_sphere(q, M)
            w, V = np.linalg.eigh(H)
            # Levenberg-Marquardt shift so that the step is always a descent dir
            wmax = np.maximum(w[:, -1], 1e-12)
            shift = np.maximum(0.0, 0.05 * wmax - w[:, 0])
            Hs = H + shift[:, None, None] * eye
            step = -np.linalg.solve(Hs, g[:, :, None])[:, :, 0]
            sn = np.linalg.norm(step, axis=1)
            step = step * np.minimum(1.0, 1.5 / np.maximum(sn, 1e-300))[:, None]
            alpha = np.ones(len(q))
            accepted = np.zeros(len(q), bool)
            Jnew = Jc.copy()
            qnew = q.copy()
            for _ls in range(30):
                idx = ~accepted
                if not np.any(idx):
                    break
                cand = q[idx] + alpha[idx, None] * np.einsum(
                    'nij,nj->ni', core.E(q[idx]), step[idx])
                cand = _normalize(cand)
                Jcand = core.Jval(cand, M)
                ok = Jcand <= Jc[idx] + 1e-15
                take = idx.copy()
                take[idx] = ok
                qnew[take] = cand[ok]
                Jnew[take] = Jcand[ok]
                accepted[take] = True
                alpha[idx] = alpha[idx] * 0.5
                del cand, Jcand
            if not np.all(accepted):
                # points that could not decrease: they are (numerically) stationary
                stuck = ~accepted
                qnew[stuck] = q[stuck]
                Jnew[stuck] = Jc[stuck]
            q, Jc = qnew, Jnew
        if polish:
            for _ in range(60):
                g = core.proj_grad(q, M)
                if np.max(np.linalg.norm(g, axis=1)) < 1e-14:
                    break
                H = core.H_sphere(q, M)
                w, _ = np.linalg.eigh(H)
                shift = np.maximum(0.0, 0.02 * np.maximum(w[:, -1], 1e-12) - w[:, 0])
                step = -np.linalg.solve(H + shift[:, None, None] * eye, g[:, :, None])[:, :, 0]
                q = _normalize(q + np.einsum('nij,nj->ni', core.E(q), step))
        return self._classify(q, Jc, M)

    def _classify(self, q, Jc, M):
        # dedupe by SO(3) distance (|q.q'|), keep lowest J
        order = np.argsort(Jc)
        q = q[order]
        Jc = Jc[order]
        keep = []
        for i in range(len(q)):
            dup = False
            for j in keep:
                c = abs(float(q[i] @ q[j]))
                if 2.0 * np.arccos(min(1.0, c)) < 1e-5:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        qk = q[keep]
        Jk = Jc[keep]
        gk = core.proj_grad(qk, M)
        Hk = core.H_sphere(qk, M)
        w = np.linalg.eigvalsh(Hk)
        mins = []
        for i in range(len(qk)):
            if np.linalg.norm(gk[i]) > 1e-8:
                continue
            if w[i, 0] <= 1e-8:
                continue
            # canonical sign: first nonzero component positive
            qq = qk[i].copy()
            nz = np.flatnonzero(np.abs(qq) > 1e-12)
            if len(nz) and qq[nz[0]] < 0:
                qq = -qq
            mins.append(dict(q=qq, J=float(Jk[i]), H=Hk[i],
                             grad_norm=float(np.linalg.norm(gk[i])),
                             H_eig=w[i].tolist(),
                             detH=float(np.linalg.det(Hk[i])),
                             logdetH=float(np.log(np.linalg.det(Hk[i])))))
        mins.sort(key=lambda d: d['J'])
        # all stationary points (for reporting K)
        stat = []
        for i in range(len(qk)):
            stat.append(dict(q=qk[i].copy(), J=float(Jk[i]),
                             grad_norm=float(np.linalg.norm(gk[i])),
                             H_eig=w[i].tolist(),
                             kind=('min' if (w[i, 0] > 1e-8 and
                                             np.linalg.norm(gk[i]) < 1e-8)
                                   else ('saddle' if np.linalg.norm(gk[i]) < 1e-8
                                         else 'notconv'))))
        n_total = len(qk)
        return dict(q=np.array([m['q'] for m in mins]),
                    J=np.array([m['J'] for m in mins]),
                    H=np.array([m['H'] for m in mins]),
                    detH=np.array([m['detH'] for m in mins]),
                    logdetH=np.array([m['logdetH'] for m in mins]),
                    grad_norm=np.array([m['grad_norm'] for m in mins]),
                    H_eig=np.array([m['H_eig'] for m in mins]),
                    n_starts=n_total, stationary=stat)

    # -------------------------------------------------------------- branch continuation
    def branch(self, q0, theta, iters=60):
        """Projected Newton continuation of one minimum from theta=0 to theta."""
        M = self.M(theta)
        q = np.asarray(q0, float).copy()
        for _ in range(iters):
            g = core.proj_grad(q[None], M)[0]
            if np.linalg.norm(g) < 1e-14:
                break
            H = core.H_sphere(q[None], M)[0]
            step = -np.linalg.solve(H, g)
            q = q + core.E(q[None])[0] @ step
            q = q / np.linalg.norm(q)
        return q

    # -------------------------------------------------------------- crossing analysis
    def delta_J(self, theta, qA0=None, qB0=None):
        qA0 = self.qA0 if qA0 is None else qA0
        qB0 = self.qB0 if qB0 is None else qB0
        qA = self.branch(qA0, theta)
        qB = self.branch(qB0, theta)
        return float(core.Jval(qB, self.M(theta)) - core.Jval(qA, self.M(theta))), qA, qB
