"""Synthetic QPEP instances (algebraic PnP and point-to-plane)."""
import numpy as np
from .poly import q2R


def normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def random_rotation(rng):
    q = rng.normal(size=4)
    return normalize(q)


def make_pnp(n=8, noise=0.0, seed=0, zmin=3.0, spread=1.0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)) * spread
    p[:, 2] += zmin
    q = random_rotation(rng)
    R = q2R(q)
    t = rng.normal(size=3) * 0.3
    Xc = (R @ p.T).T + t
    uv = Xc[:, :2] / Xc[:, 2:3]
    uv = uv + noise * rng.normal(size=uv.shape)
    return dict(p=p, uv=uv, R=R, t=t, q=q)


def make_planar_pnp(n=8, noise=0.0, seed=0):
    """Points on a plane: weak/poor conditioning and symmetry-prone."""
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 3))
    p[:, :2] = rng.normal(size=(n, 2))
    p[:, 2] = 0.02 * rng.normal(size=n)
    p[:, 2] += 4.0
    q = random_rotation(rng)
    R = q2R(q)
    t = rng.normal(size=3) * 0.2
    Xc = (R @ p.T).T + t
    uv = Xc[:, :2] / Xc[:, 2:3]
    uv = uv + noise * rng.normal(size=uv.shape)
    return dict(p=p, uv=uv, R=R, t=t, q=q)


def make_symmetric_pnp(noise=0.0, seed=0, tilt=0.0):
    """A point set with a near-180-degree rotational ambiguity.

    The points are symmetric under a rotation about the z axis; a small tilt
    breaks the tie, so sweeping tilt crosses a global-optimum switch.
    """
    rng = np.random.default_rng(seed)
    base = np.array([
        [1.0, 0.2, 4.0], [-1.0, -0.2, 4.0],
        [0.2, 1.0, 4.2], [-0.2, -1.0, 4.2],
        [0.8, -0.8, 3.8], [-0.8, 0.8, 3.8],
    ])
    # duplicate with a small deterministic perturbation scaled by tilt
    p = base.copy()
    p[:, 0] += tilt * np.array([1, -1, 1, -1, 1, -1])
    q = normalize(np.array([np.cos(0.3), np.sin(0.3), 0.1, -0.05]))
    R = q2R(q)
    t = np.array([0.1, -0.05, 0.2])
    Xc = (R @ p.T).T + t
    uv = Xc[:, :2] / Xc[:, 2:3]
    uv = uv + noise * rng.normal(size=uv.shape)
    return dict(p=p, uv=uv, R=R, t=t, q=q)


def pose_error(R1, t1, R2, t2):
    dR = R1.T @ R2
    ang = np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    return float(ang), float(np.linalg.norm(t1 - t2))
