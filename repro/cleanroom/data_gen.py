"""Synthetic calibration data for the stereo-localisation example (Sec. VI-B).

Reproduces the data-generation protocol of the paper (Fig. 3a and the text of
Sec. VI-B1) independently of the upstream code:

* a planar checkerboard-like feature grid: ``8 x 8`` equally spaced points
  filling a ``1.0 m x 1.0 m`` rectangle centred at the origin,
* ground-truth camera poses on a sphere of radius ``radius = 3 m`` around the
  grid centre, inside a cone of half-angle ``beta = 45 deg`` (the paper's
  "90 degree cone"), oriented so that the grid centre is inside a ``90 deg``
  field of view (camera z axis within ``alpha = 45 deg`` of the direction to
  the grid centre, plus a random azimuthal orientation),
* pixel measurements from the projective stereo camera model with Gaussian
  pixel noise (Eq. 27/28).

The camera-frame point corresponding to world point ``m`` is

    y = C (m - r_p),        C in SO(3), r_p = camera position in world,

i.e. ``C`` maps world coordinates to camera coordinates.  This matches the
upstream convention (``r_l_inC = C_p0 @ (r_l - r_p)``).
"""

from __future__ import annotations

import numpy as np

from camera_model import StereoCamera, exp_so3, skew

__all__ = [
    "make_board",
    "get_poses_in_cone",
    "get_random_poses",
    "get_calibration_data",
]


def make_board(
    width: float = 1.0,
    height: float = 1.0,
    n_squares=(8, 8),
) -> np.ndarray:
    """Feature grid in the world frame (planar, in the y-z plane, x = 0).

    Returns
    -------
    (3, N) array with ``N = n_squares[0] * n_squares[1]``.
    """
    sq = np.array([width, height], dtype=float) / np.array(n_squares, dtype=float)
    pts = []
    for i in range(n_squares[0]):
        for j in range(n_squares[1]):
            pts.append([0.0, i * sq[0], j * sq[1]])
    r_l = np.array(pts, dtype=float).T
    return r_l - r_l.mean(axis=1, keepdims=True)


def get_poses_in_cone(
    radius: float = 3.0,
    beta: float = np.pi / 4.0,
    alpha: float = np.pi / 4.0,
    n_batch: int = 20,
    rng: np.random.Generator | None = None,
):
    """Ground-truth poses as in the paper's Fig. 3a.

    Positions are uniformly distributed on the sphere of radius ``radius``
    inside a cone of half-angle ``beta`` around the +x axis (the cone centre
    is arbitrary since the board is isotropic).  Orientations have their
    optical (z) axis pointing at the grid centre exactly, with a random
    azimuth, and are then perturbed by a random rotation about the camera x
    axis by up to ``alpha`` (the paper's "alpha = 90 deg FOV" condition).

    Returns
    -------
    r_p0s : (n_batch, 3, 1) camera positions (world frame)
    C_p0s : (n_batch, 3, 3) rotations world -> camera
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    polar = 2.0 * np.pi * rng.random(n_batch)
    azimuth = beta * (2.0 * rng.random(n_batch) - 1.0)
    r_p0s = []
    C_p0s = []
    for i in range(n_batch):
        Cz = exp_so3(np.array([0.0, 0.0, azimuth[i]]))
        Cx = exp_so3(np.array([polar[i], 0.0, 0.0]))
        C = Cx @ Cz
        r = C[:, [0]] * radius  # (3,1) position on the sphere
        r_p0s.append(r)
        # camera z axis points at the grid centre (origin)
        z = -r / np.linalg.norm(r)
        y = rng.standard_normal((3, 1))
        y = y - (y.T @ z) * z
        y = y / np.linalg.norm(y)
        x = -skew(z.reshape(-1)) @ y
        C_pose = np.hstack([x, y, z]).T  # rows are the camera axes in world
        # field-of-view perturbation: rotation about the camera x axis
        delta = (2.0 * rng.random() - 1.0) * alpha
        C_pose = exp_so3(np.array([delta, 0.0, 0.0])) @ C_pose
        C_p0s.append(C_pose)
    return np.stack(r_p0s), np.stack(C_p0s)


def get_random_poses(
    radius: float = 3.0,
    n_batch: int = 1,
    rng: np.random.Generator | None = None,
):
    """Random initialisations used by the local solver (paper's Fig. 3b).

    Translations are uniform on the sphere of radius ``radius`` around the
    grid centre; the camera z axis points at the grid centre and the
    remaining rotation about the optical axis is random.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    r_p0s, C_p0s = [], []
    for _ in range(n_batch):
        r_ = rng.random((3, 1)) - 0.5
        r = radius * r_ / np.linalg.norm(r_)
        r_p0s.append(r)
        z = -r / np.linalg.norm(r)
        y = rng.standard_normal((3, 1))
        y = y - (y.T @ z) * z
        y = y / np.linalg.norm(y)
        x = -skew(z.reshape(-1)) @ y
        C_p0s.append(np.hstack([x, y, z]).T)
    return np.stack(r_p0s), np.stack(C_p0s)


def get_calibration_data(
    n_batch: int = 20,
    board_dims=(1.0, 1.0),
    n_squares=(8, 8),
    radius: float = 3.0,
    beta: float = np.pi / 4.0,
    alpha: float = np.pi / 4.0,
    cam: StereoCamera | None = None,
    b_true: float = 0.24,
    rng: np.random.Generator | None = None,
    noise_mode: str = "independent",
):
    """Generate one full problem instance (20 poses x 64 features).

    Returns
    -------
    dict with keys
        ``r_p0s`` (20,3,1) GT positions, ``C_p0s`` (20,3,3) GT rotations,
        ``r_l`` (3,64) landmarks, ``pixels`` (20,4,64) noisy pixel
        measurements, ``pixels_clean`` (20,4,64) noise-free pixels.
    """
    cam = cam if cam is not None else StereoCamera()
    r_l = make_board(*board_dims, n_squares=n_squares)
    r_p0s, C_p0s = get_poses_in_cone(
        radius=radius, beta=beta, alpha=alpha, n_batch=n_batch, rng=rng
    )
    pixels, pixels_clean = [], []
    for i in range(n_batch):
        p_in_c = C_p0s[i] @ (r_l - r_p0s[i])
        assert np.all(p_in_c[2] > 0), "feature behind the camera"
        clean = cam.project(p_in_c, b_true)
        noisy = cam.add_pixel_noise(
            clean, rng, noise_mode=noise_mode, b=b_true, z=p_in_c[2]
        )
        pixels.append(noisy)
        pixels_clean.append(clean)
    return dict(
        r_p0s=r_p0s,
        C_p0s=C_p0s,
        r_l=r_l,
        pixels=np.stack(pixels),
        pixels_clean=np.stack(pixels_clean),
    )
