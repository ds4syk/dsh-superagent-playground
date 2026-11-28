"""QPEP 问题的构造：矩阵加权 3D-3D、随机实例、受控双峰实例。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import algebra as A


@dataclass
class QPEPProblem:
    """min_{q∈S^3} f(q) = m2(q)^T Qm m2(q)，以及平移闭式解。"""

    Qm: np.ndarray
    Qx: np.ndarray
    Mtt: np.ndarray
    MRt: np.ndarray
    bt: np.ndarray
    p: np.ndarray | None = None
    y: np.ndarray | None = None
    W: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ---- rotation-only cost
    def f(self, q):
        return A.fval(q, self.Qm)

    def grad_f(self, q):
        return A.grad(q, self.Qm)

    def hess_f(self, q):
        return A.hess(q, self.Qm)

    # ---- translation
    def t_star(self, q):
        r = A.R_of_q(q).reshape(-1)
        return -np.linalg.solve(self.Mtt, self.MRt.T @ r + self.bt)

    def pose(self, q):
        return A.R_of_q(q), self.t_star(q)

    def full_cost(self, q, t):
        R = A.R_of_q(q)
        res = self.y - (self.p @ R.T) - t
        return float(np.einsum('ni,nij,nj->', res, self.W, res))

    def dt_dq(self, q):
        """t*(q) 对 q 的雅可比 dt*/dq ∈ R^{3x4}。"""
        dR = _dvecR_dq(q)                      # (9,4)
        return -np.linalg.solve(self.Mtt, self.MRt.T) @ dR


def _dvecR_dq(q):
    """d vec_row(R(q)) / d q ∈ R^{9x4}（数值稳定解析式）。"""
    q0, q1, q2, q3 = q
    return np.array([
        [2*q0, 2*q1, -2*q2, -2*q3],
        [-2*q3, 2*q2, 2*q1, -2*q0],
        [2*q2, 2*q3, 2*q0, 2*q1],
        [2*q3, 2*q2, 2*q1, 2*q0],
        [2*q0, -2*q1, 2*q2, -2*q3],
        [-2*q1, -2*q0, 2*q3, 2*q2],
        [-2*q2, 2*q3, -2*q0, 2*q1],
        [2*q1, 2*q0, 2*q3, 2*q2],
        [2*q0, -2*q1, -2*q2, 2*q3],
    ], dtype=float)


QPEPProblem.dvecR_dq = staticmethod(_dvecR_dq)


def build_from_points(p, y, W):
    """由 3D-3D 对应点构造 QPEP 的缩减代价。

    min_{R,t} Σ_k (y_k - R p_k - t)^T W_k (y_k - R p_k - t)
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    W = np.asarray(W, float)
    N = p.shape[0]
    M_RR = np.zeros((9, 9))
    M_Rt = np.zeros((9, 3))
    b_R = np.zeros(9)
    M_tt = np.zeros((3, 3))
    b_t = np.zeros(3)
    c0 = 0.0
    for k in range(N):
        Ak = np.kron(np.eye(3), p[k][None, :])
        M_RR += Ak.T @ W[k] @ Ak
        M_Rt += Ak.T @ W[k]
        b_R += -Ak.T @ W[k] @ y[k]
        M_tt += W[k]
        b_t += -W[k] @ y[k]
        c0 += y[k] @ W[k] @ y[k]
    Mtt_inv = np.linalg.inv(M_tt)
    Qx = np.zeros((10, 10))
    Qx[:9, :9] = M_RR - M_Rt @ Mtt_inv @ M_Rt.T
    tr = b_R - M_Rt @ Mtt_inv @ b_t
    Qx[:9, 9] = tr
    Qx[9, :9] = tr
    Qx[9, 9] = c0 - b_t @ Mtt_inv @ b_t
    Qm = A.T.T @ Qx @ A.T
    Qm = 0.5 * (Qm + Qm.T)
    return QPEPProblem(Qm=Qm, Qx=Qx, Mtt=M_tt, MRt=M_Rt, bt=b_t,
                       p=p, y=y, W=W)


def random_instance(seed=0, N=10, noise=0.02, anisotropy=0.5):
    """随机各向异性矩阵加权 3D-3D 实例。"""
    rng = np.random.default_rng(seed)
    from scipy.spatial.transform import Rotation as Rot
    p = rng.normal(size=(N, 3))
    R_true = Rot.random(random_state=rng).as_matrix()
    t_true = rng.normal(size=3) * 0.5
    y = (R_true @ p.T).T + t_true + rng.normal(size=(N, 3)) * noise
    W = []
    for _ in range(N):
        S = rng.normal(size=(3, 3))
        Wk = S @ S.T + anisotropy * np.eye(3)
        W.append(Wk)
    W = np.array(W)
    prob = build_from_points(p, y, W)
    prob.meta.update(dict(seed=seed, N=N, noise=noise, anisotropy=anisotropy,
                          R_true=R_true, t_true=t_true,
                          q_true=_q_from_R(R_true)))
    return prob


def _q_from_R(R):
    """旋转矩阵 -> 单位四元数 (scalar-first, 正分支)。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
             0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    return q / np.linalg.norm(q)


def two_mode_psd(seed=20261128):
    """构造一个 PSD、具 180° 对称双极小值的 QPEP 型代价 Qm(θ)。

    Q0 由随机 PSD 矩阵关于右乘 k 的 10x10 正交作用 P 做平均得到：
    Q0 = 1/2 (A + P^T A P)，因此 f(q k) = f(q)，两个极小简并。
    对称破缺 B 取另一个 PSD 方向，Q(θ) = Q0 + θ B（仍 PSD）。
    """
    rng = np.random.default_rng(seed)
    # 右乘 k 的四元数矩阵与 10x10 正交作用 P
    Lk = np.array([[0, 0, 0, -1],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=float)
    Q = rng.normal(size=(10, 4))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    V = A.m2(Q)
    Vp = A.m2(Q @ Lk.T)
    P = np.linalg.solve(V, Vp)
    assert np.allclose(P @ P.T, np.eye(10), atol=1e-10)

    S = rng.normal(size=(10, 10))
    Apsd = S @ S.T
    Q0 = 0.5 * (Apsd + P.T @ Apsd @ P)
    S2 = rng.normal(size=(10, 10))
    B = S2 @ S2.T
    B = 0.5 * (B + B.T)
    # 用 θ 控制两个对称极小之间的能差
    return dict(Q0=Q0, B=B, P=P)


def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])
