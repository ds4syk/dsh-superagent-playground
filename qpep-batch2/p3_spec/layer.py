"""Prototype 3: SpecQPEP-Layer — 代数临界点谱 + Morse–Laplace 配分函数。"""
from __future__ import annotations

import numpy as np

from core import algebra as A, modes


# 受控双峰实例：qA=e0, qB=e3 为两个简并极小，theta 控制能差。
uA = np.zeros(10); uA[0] = 1.0
uB = np.zeros(10); uB[3] = 1.0
Q_BASE = np.eye(10) - np.outer(uA, uA) - np.outer(uB, uB)
Q_A_TERM = 0.2 * np.outer(uA, uA)


def two_mode_Q(theta):
    return Q_BASE + Q_A_TERM + theta * np.outer(uB, uB)


def two_mode_data():
    qA = np.array([1.0, 0.0, 0.0, 0.0])
    qB = np.array([0.0, 0.0, 0.0, 1.0])
    return qA, qB


def spectrum(Qm, beta, n_starts=24, seed=0, extra=None):
    """返回极小点谱、Boltzmann/Laplace 权重、Z0、ZML。"""
    res = modes.find_minima(Qm, n_starts=n_starts, seed=seed, extra_starts=extra)
    q = res["q"]
    f = res["f"]
    H = np.array([A.riemann_hess(qq, Qm) for qq in q])
    logdet = np.array([np.linalg.slogdet(Hk)[1] for Hk in H])
    # Z0
    logw0 = -f / beta
    m0 = logw0.max()
    Z0 = float(np.exp(m0) * np.sum(np.exp(logw0 - m0)))
    pi0 = np.exp(logw0 - m0)
    pi0 /= pi0.sum()
    # Laplace
    logwL = -f / beta - 0.5 * logdet
    mL = logwL.max()
    ZML = float(np.exp(mL) * np.sum(np.exp(logwL - mL)))
    piL = np.exp(logwL - mL)
    piL /= piL.sum()
    return dict(q=q, f=f, H=H, logdet=logdet, Z0=Z0, ZML=ZML,
                pi0=pi0, piL=piL)


def loss_Z0(spec, f_gt, beta):
    return f_gt / beta + np.log(spec["Z0"])


def loss_ZML(spec, f_gt, beta):
    return f_gt / beta + np.log(spec["ZML"])


def grad_Z0_theta(spec, B, beta):
    """ddθ log Z0 的包络公式：−(1/β) Σ π_k f_k'，f_k' = m2(q_k)^T B m2(q_k)。"""
    fp = np.array([A.m2(q) @ B @ A.m2(q) for q in spec["q"]])
    return -float(np.sum(spec["pi0"] * fp)) / beta


def grad_ZML_shape_only(spec, B, beta):
    """只对 Boltzmann 因子求导（丢掉 ∂ logdet H_k 项）的“朴素”梯度。"""
    fp = np.array([A.m2(q) @ B @ A.m2(q) for q in spec["q"]])
    return -float(np.sum(spec["piL"] * fp)) / beta


def Z_true(theta, beta, quats):
    """用固定样本（common random numbers）估计 S^3 上的配分函数积分。"""
    Qm = two_mode_Q(theta)
    v = A.m2(quats)
    vals = np.einsum('ni,ij,nj->n', v, Qm, v)
    w = np.exp(-vals / beta)
    return float(np.mean(w))
