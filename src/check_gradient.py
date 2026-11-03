"""Finite-difference check of the SDPR implicit gradient (well-conditioned N=4).

This is the test quoted in the report: the analytic Jacobian of the SDPR layer
matches central differences to ~1e-7 on a small, well-conditioned rotation
synchronization problem.  On the N=8 learned problems the KKT system is much
more ill-conditioned and the same check degrades (see report §4.5).
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import build_Q_torch, geodesic_loss, rotation_6d_to_matrix, solve_rotations  # noqa: E402
from panorama import trajectory  # noqa: E402
from rotsync import RotSyncProblem, SDPRunner  # noqa: E402


def exp_so3(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def main():
    torch.set_num_threads(4)
    N = 4
    prob = RotSyncProblem(N, redundant=True)
    runner = SDPRunner(prob, solver="SCS", eps=1e-8)
    rng = np.random.default_rng(3)
    R_gt = torch.tensor(trajectory(N, rng), dtype=torch.float64)
    edges = [(i, i + 1) for i in range(N - 1)] + [(i, i + 2) for i in range(N - 2)]
    m6 = torch.zeros(len(edges), 6, dtype=torch.float64)
    for e, (i, j) in enumerate(edges):
        Rm = R_gt[i].numpy() @ R_gt[j].numpy().T
        Rm = Rm @ exp_so3(np.deg2rad(1.5) * rng.standard_normal(3))
        Rm = torch.tensor(Rm, dtype=torch.float64)
        m6[e] = torch.cat([Rm[:, 0], Rm[:, 1]])
    m6 = m6.clone().requires_grad_(True)

    def loss_of(m):
        Q = build_Q_torch(prob, rotation_6d_to_matrix(m), torch.ones(len(edges), dtype=torch.float64), edges, 10.0)
        Rs, _ = solve_rotations(prob, Q, runner)
        return geodesic_loss(Rs, R_gt)

    loss = loss_of(m6)
    loss.backward()
    g = m6.grad.clone()
    eps = 1e-5
    for e, k in [(0, 0), (0, 5)]:
        mp, mm = m6.detach().clone(), m6.detach().clone()
        mp[e, k] += eps
        mm[e, k] -= eps
        fd = (float(loss_of(mp)) - float(loss_of(mm))) / (2 * eps)
        print(f"coord ({e},{k}): analytic={float(g[e,k]):.10f} fd={fd:.10f} "
              f"absdiff={abs(float(g[e,k])-fd):.2e}")


if __name__ == "__main__":
    main()
