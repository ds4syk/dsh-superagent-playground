#!/usr/bin/env python3
"""Theseus cost function for the same weighted 3D-3D / GM objective.

The theseus experiment must optimise *exactly* the objective used by the
GNC-GM reference implementation in ``experiments/synthetic/common.py``:

    Phi(T) = sum_i rho_c( || R x_i + t - y_i ||^2 ),
    rho_c(e) = c^2 e / (c^2 + e)        (Geman-McClure, c = 0.05).

Theseus' ``GemanMcClureLoss`` with ``mu = 1`` is

    rho(e; radius=mu*c^2) = mu*radius*e / (mu*radius + e) = c^2 e /(c^2+e),

so we wrap the point-to-point error in ``GNCRobustCostFunction`` with a
fixed ``gnc_control_val = 1`` and ``log_loss_radius = log(c^2)``.

The residual is vectorised over the N correspondences (batch = N), with a
single SE3 optimisation variable of batch 1.  Theseus broadcasts a batch-1
variable against a batch-N cost; the normal equations are summed over the
batch by the linear solver.

Local-coordinate convention (verified numerically by ``self_check``):
``T <- T @ Exp(delta)`` with ``delta = (v, omega)`` (translation first,
rotation second, matching theseus/torchlie), so

    d r_i / d delta = [ R ,  -R [x_i]_x ].

The analytic Jacobian ``jacobians()`` returns that, and ``self_check``
compares it against central differences so that a silent convention error
cannot masquerade as a "Theseus result".
"""
from __future__ import annotations

import numpy as np
import torch

import theseus as th
from theseus.core import CostFunction, ScaleCostWeight, Variable

# Theseus defaults to float32; the reference implementation is float64 and
# the analytic Jacobian self-check is only meaningful at float64.
torch.set_default_dtype(torch.float64)


def skew_batch(x: torch.Tensor) -> torch.Tensor:
    """(N,3) -> (N,3,3) with skew(x) @ v = x cross v."""
    z = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([z[:, 0], -x[:, 2], x[:, 1]], dim=-1),
            torch.stack([x[:, 2], z[:, 1], -x[:, 0]], dim=-1),
            torch.stack([-x[:, 1], x[:, 0], z[:, 2]], dim=-1),
        ],
        dim=-2,
    )


class PointToPoint3D(CostFunction):
    """r_i = R x_i + t - y_i, dim 3, batch = number of correspondences."""

    def __init__(self, pose: th.SE3, x: torch.Tensor, y: torch.Tensor,
                 name: str = "p2p3d"):
        super().__init__(ScaleCostWeight(1.0), name=name)
        self.pose = pose
        self.register_optim_var("pose")
        self.x = Variable(tensor=x)
        self.y = Variable(tensor=y)
        self.register_aux_var("x")
        self.register_aux_var("y")

    def dim(self) -> int:
        return 3

    def error(self) -> torch.Tensor:
        # batch = correspondences: pose has batch 1 and is broadcast.
        R = self.pose.rotation().tensor          # (1,3,3)
        t = self.pose.translation().tensor       # (1,3)
        u = torch.matmul(R, self.x.tensor.unsqueeze(-1)).squeeze(-1) + t
        return u - self.y.tensor                 # (N,3)

    def _copy_impl(self, new_name=None):
        return PointToPoint3D(
            self.pose.copy(), self.x.tensor.clone(), self.y.tensor.clone(),
            name=new_name or self.name,
        )

    def jacobians(self):
        R = self.pose.rotation().tensor                       # (1,3,3)
        x = self.x.tensor                                      # (N,3)
        S = skew_batch(x)                                      # (N,3,3)
        J = torch.zeros(x.shape[0], 3, 6, dtype=R.dtype, device=R.device)
        J[..., :3] = R                                         # d r / d v = R
        J[..., 3:] = -torch.matmul(R, S)                       # d r / d w = -R[x]_x
        return [J], self.error()


def gm_robust_cost(pose: th.SE3, x: torch.Tensor, y: torch.Tensor, c: float,
                   name: str = "p2p3d_gm"):
    """Point-to-point cost wrapped in Theseus' GM loss (mu = 1, exact GM)."""
    base = PointToPoint3D(pose, x, y, name=name + "_base")
    log_radius = Variable(tensor=torch.tensor([[float(np.log(c * c))]], dtype=torch.float64))
    mu = Variable(tensor=torch.tensor([[1.0]], dtype=torch.float64))
    return th.GNCRobustCostFunction(
        base, th.GemanMcClureLoss, log_loss_radius=log_radius,
        gnc_control_val=mu, name=name,
    )


def build_gm_objective(x: np.ndarray, y: np.ndarray,
                       R0: np.ndarray, t0: np.ndarray, c: float):
    """Objective with one GM-wrapped 3D-3D residual per correspondence.

    Each residual is a separate cost function of batch 1 so that the GM
    weight is applied *per correspondence* (a single 3N-dim residual would
    apply one weight to the whole vector).  All costs share one SE3 variable.
    """
    pose = th.SE3(tensor=se3_from_Rt(R0, t0))
    obj = th.Objective(dtype=torch.float64)
    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y)
    for i in range(x.shape[0]):
        obj.add(gm_robust_cost(pose, x_t[i:i + 1], y_t[i:i + 1], c,
                               name=f"gm_{i:04d}"))
    obj.update()
    return obj, pose


def se3_from_Rt(R: np.ndarray, t: np.ndarray) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.as_tensor(R)
    T[:3, 3] = torch.as_tensor(t)
    return T[:3, :].unsqueeze(0)


def errors_from_pose(pose: th.SE3):
    T = pose.tensor[0].detach().cpu().numpy()
    return T[:3, :3].copy(), T[:3, 3].copy()


def self_check(x: np.ndarray, y: np.ndarray, c: float, seed: int = 0) -> dict:
    """Finite-difference check of the analytic Jacobian and the GM cost value."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    R = Q * np.sign(np.linalg.det(Q))
    t = rng.uniform(-0.5, 0.5, size=3)
    base = th.SE3(tensor=se3_from_Rt(R, t))
    pose = th.SE3(tensor=se3_from_Rt(R, t))
    cost = PointToPoint3D(pose, torch.as_tensor(x), torch.as_tensor(y))
    J, err = cost.jacobians()
    J = J[0].detach().numpy()                     # (N,3,6)
    h = 1e-6
    Jfd = np.zeros_like(J)
    for k in range(6):
        d = np.zeros((1, 6))
        d[0, k] = h
        pose.update(base.retract(torch.as_tensor(d)).tensor)
        ep = cost.error().detach().numpy()
        d[0, k] = -h
        pose.update(base.retract(torch.as_tensor(d)).tensor)
        em = cost.error().detach().numpy()
        # restore base pose
        pose.update(se3_from_Rt(R, t))
        Jfd[:, :, k] = (ep - em) / (2 * h)
    jac_rel = float(np.linalg.norm(J - Jfd) / max(np.linalg.norm(Jfd), 1e-12))

    # cost value of the wrapped GM objective vs. the numpy reference
    robust = gm_robust_cost(th.SE3(tensor=se3_from_Rt(R, t)),
                            torch.as_tensor(x), torch.as_tensor(y), c)
    # weighted_error() has shape (N,3) with ||h_i||^2 = rho_i (theseus contract)
    val_th = float((robust.error().detach() ** 2).sum())
    r = (R @ x.T).T + t - y
    e = np.sum(r * r, axis=1)
    c2 = c * c
    val_np = float(np.sum(c2 * e / (c2 + e)))
    return dict(jac_rel_err=jac_rel, cost_rel_err=abs(val_th - val_np) / max(abs(val_np), 1e-12),
                cost_theseus=val_th, cost_numpy=val_np)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(20, 3))
    y = x + 0.01 * rng.normal(size=(20, 3))
    print(self_check(x, y, 0.05))
