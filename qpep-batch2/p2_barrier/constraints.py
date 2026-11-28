"""把物理不等式写成 m2(q) 的线性约束：g_j(q) = c_j^T m2(q) <= 0。"""
from __future__ import annotations

import numpy as np

from core import algebra as A


def tilt_constraint(phi_max_deg):
    """姿态倾角安全锥：旋转角 <= phi_max，等价 q0^2 >= cos^2(phi/2)||q||^2。"""
    c = np.cos(np.deg2rad(phi_max_deg) / 2.0) ** 2
    e0 = np.zeros(10); e0[0] = 1.0
    return c * A.U_NORM - e0


def depth_constraints(prob, zmin=0.0, idx=None):
    """相机正深度/手性约束 z_k(q) >= zmin，写成 c_k^T m2 <= 0。

    z_k(q) = e3^T (R(q) p_k + t*(q))，t* 由 Schur 闭式给出；
    在 ||q||=1 上 z_k 是 m2 的线性函数（常数项用 ||q||^2 齐次化）。
    """
    p = prob.p
    if idx is None:
        idx = range(p.shape[0])
    Mtt_inv = np.linalg.inv(prob.Mtt)
    t_slope = -Mtt_inv @ prob.MRt.T           # (3,9)
    t_const = -Mtt_inv @ prob.bt              # (3,)
    e3 = np.array([0.0, 0.0, 1.0])
    cs = []
    for k in idx:
        Ak = np.kron(np.eye(3), p[k][None, :])   # 3x9, Ak r = R p_k
        a = e3 @ (Ak + t_slope)                  # (9,)
        const = float(e3 @ t_const)
        c = zmin * A.U_NORM - A.T[:9].T @ a - const * A.U_NORM
        cs.append(c)
    return cs


def depth_value(prob, q, k):
    R = A.R_of_q(q)
    t = prob.t_star(q)
    return float((R @ prob.p[k] + t)[2])


def all_constraints(prob, phi_max_deg=40.0, zmin=0.0, depth_idx=None):
    cs = [tilt_constraint(phi_max_deg)]
    names = [f"tilt<={phi_max_deg}deg"]
    if depth_idx is not None:
        for i, c in enumerate(depth_constraints(prob, zmin, depth_idx)):
            cs.append(c)
            names.append(f"depth[{depth_idx[i]}]>={zmin}")
    return np.array(cs), names


def gvals(q, cs):
    v = A.m2(q)
    return cs @ v
