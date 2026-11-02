"""Clean-room stereo camera model for Section VI-B of Holmes et al. (TRO 2025).

This module implements, from scratch and without importing the upstream
``sdprlayer`` package:

* the forward (projective) stereo camera model used to synthesise pixel
  measurements,
* the differentiable inverse stereo camera model of Eq. (27),

      m_k = (b / d_k) * [ (u_k - c_u),
                          (v_k - c_v) * f_u / f_v,
                          f_u ]^T ,

* the first-order propagation of the pixel noise covariance to the Euclidean
  measurement covariance Sigma_k (Eq. (28) of the paper / the derivation in
  Holmes et al., "On Semidefinite Relaxations for Matrix-Weighted
  State-Estimation Problems in Robotics"), and the inverse covariance weights
  W_k = Sigma_k^{-1} used by Problem (P6).

Conventions
-----------
* A 3-D point in the camera frame is ``p = (x, y, z)^T`` with ``z > 0``.
* Pixel measurements are ``(u_l, v_l, u_r, v_r)^T`` for the left and right
  cameras of a rectified stereo rig with baseline ``b``.
* The disparity is ``d = u_l - u_r``.
* The left camera model is the pinhole model with focal lengths
  ``(f_u, f_v)`` and principal point ``(c_u, c_v)``; the right camera is the
  same camera translated by ``-b`` along its x axis.

All functions accept real *or complex* arguments (numpy dtypes are
propagated), which lets us differentiate the whole measurement model to
machine precision with the complex-step method.
"""

from __future__ import annotations

import numpy as np

__all__ = ["StereoCamera", "skew", "exp_so3", "log_so3"]


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(v) @ x == cross(v, x)``."""
    v = np.asarray(v).reshape(3)
    S = np.zeros((3, 3), dtype=v.dtype)
    S[0, 1] = -v[2]
    S[1, 0] = v[2]
    S[0, 2] = v[1]
    S[2, 0] = -v[1]
    S[1, 2] = -v[0]
    S[2, 1] = v[0]
    return S


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map (Rodrigues' formula)."""
    phi = np.asarray(phi).reshape(3)
    theta2 = float(np.real(np.vdot(phi, phi)))
    theta = np.sqrt(theta2)
    K = skew(phi)
    if theta < 1e-8:
        # 2nd order Taylor expansion; exact to machine precision here
        return np.eye(3, dtype=phi.dtype) + K + 0.5 * K @ K
    return (
        np.eye(3, dtype=phi.dtype)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / theta2) * K @ K
    )


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map (inverse of :func:`exp_so3`)."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    theta = np.arccos(c)
    if theta < 1e-8:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # near pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        v = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(v))
        v = v * np.sign(A[:, k] + 1e-300)
        v = v / np.linalg.norm(v)
        return v * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (theta / (2.0 * np.sin(theta))) * w


class StereoCamera:
    """Rectified stereo camera with a pinhole left camera.

    Parameters
    ----------
    fu, fv : float
        Focal lengths (pixels).
    cu, cv : float
        Principal point (pixels).
    sigma_u, sigma_v : float
        Pixel noise standard deviations (pixels) in the horizontal and
        vertical image directions.
    """

    def __init__(
        self,
        fu: float = 484.5,
        fv: float = 484.5,
        cu: float = 0.0,
        cv: float = 0.0,
        sigma_u: float = 0.5,
        sigma_v: float = 0.5,
    ) -> None:
        self.fu = fu
        self.fv = fv
        self.cu = cu
        self.cv = cv
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

    # ------------------------------------------------------------------
    # forward model: 3-D point in camera frame -> pixel measurements
    # ------------------------------------------------------------------
    def project(self, p_in_c: np.ndarray, b: float) -> np.ndarray:
        """Noise-free pixel measurements ``(u_l, v_l, u_r, v_r)``.

        Parameters
        ----------
        p_in_c : (3, N) array
            Points expressed in the left camera frame.
        b : float
            Stereo baseline.
        """
        p_in_c = np.asarray(p_in_c)
        x, y, z = p_in_c[0], p_in_c[1], p_in_c[2]
        u_l = self.fu * x / z + self.cu
        v_l = self.fv * y / z + self.cv
        u_r = self.fu * (x - b) / z + self.cu
        v_r = self.fv * y / z + self.cv
        return np.stack([u_l, v_l, u_r, v_r], axis=0)

    def add_pixel_noise(
        self,
        pixels: np.ndarray,
        rng: np.random.Generator,
        noise_mode: str = "independent",
        b: float = 0.24,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Add Gaussian pixel noise.

        ``noise_mode='independent'`` (paper-consistent): the left and right
        image measurements are perturbed independently, so that
        ``var(u_l) = var(u_r) = sigma_u^2`` and ``var(d) = 2 sigma_u^2``,
        matching the pixel covariance used to build ``W_k``.

        ``noise_mode='upstream'``: replicate the generator in the released
        code (``stereo_tuner.StereoCamera.forward``), where the noisy left
        coordinate is reused for the right one (``u_r = u_l - f_u b / z +
        sigma_u n``).  Kept only for cross-checking against the upstream
        reproduction; it is *inconsistent* with the estimator covariance.
        """
        pixels = np.array(pixels, dtype=float)
        N = pixels.shape[1]
        if noise_mode == "independent":
            n = rng.standard_normal((4, N))
            pixels[0] += self.sigma_u * n[0]
            pixels[1] += self.sigma_v * n[1]
            pixels[2] += self.sigma_u * n[2]
            pixels[3] += self.sigma_v * n[3]
        elif noise_mode == "upstream":
            n = rng.standard_normal((4, N))
            pixels[0] += self.sigma_u * n[0]
            pixels[1] += self.sigma_v * n[1]
            pixels[2] = pixels[0] - self.fu * b / z + self.sigma_u * n[2]
            pixels[3] = pixels[1] + self.sigma_v * n[3]
        else:
            raise ValueError(f"unknown noise_mode {noise_mode!r}")
        return pixels

    # ------------------------------------------------------------------
    # inverse model (Eq. 27) and its Jacobian
    # ------------------------------------------------------------------
    def inverse(self, pixels: np.ndarray, b: float):
        """Eq. (27): pixels ``(u_l, v_l, u_r, v_r)`` -> Euclidean measurement.

        Returns
        -------
        meas : (3, N) array
            Euclidean measurements ``m_k``.
        W : (N, 3, 3) array
            Inverse covariance (matrix) weights ``W_k = Sigma_k^{-1}``.
        """
        pixels = np.asarray(pixels)
        u_l, v_l, u_r, _v_r = pixels[0], pixels[1], pixels[2], pixels[3]
        d = u_l - u_r
        ratio = b / d
        x = (u_l - self.cu) * ratio
        y = (v_l - self.cv) * ratio * (self.fu / self.fv)
        z = ratio * self.fu
        meas = np.stack([x, y, z], axis=0)
        G = self.inverse_jacobian(pixels, b, meas=meas)
        Sigma_pix = self.pixel_covariance()
        # Cov = G Sigma_pix G^T, batched over the N points
        Cov = np.einsum("nij,jk,nlk->nil", G, Sigma_pix, G)
        W = np.linalg.inv(Cov)
        W = 0.5 * (W + np.swapaxes(W, -1, -2))
        return meas, W

    def inverse_jacobian(
        self, pixels: np.ndarray, b: float, meas: np.ndarray | None = None
    ) -> np.ndarray:
        """Jacobian ``G_k = dm_k / d(u_l, v_l, d)`` of the inverse model.

        With ``r = b / d``, ``m = (r (u-c_u), r (v-c_v) f_u/f_v, r f_u)`` so
        that ``z = r f_u`` and

            G = [[ z/f_u,     0,  -x/d ],
                 [     0, z/f_v,  -y/d ],
                 [     0,     0,  -z/d ]].
        """
        pixels = np.asarray(pixels)
        u_l, v_l, u_r = pixels[0], pixels[1], pixels[2]
        d = u_l - u_r
        if meas is None:
            meas, _ = self.inverse_meas_only(pixels, b)
        x, y, z = meas[0], meas[1], meas[2]
        N = pixels.shape[1]
        G = np.zeros((N, 3, 3), dtype=np.result_type(pixels, b, float))
        G[:, 0, 0] = z / self.fu
        G[:, 1, 1] = z / self.fv
        G[:, 0, 2] = -x / d
        G[:, 1, 2] = -y / d
        G[:, 2, 2] = -z / d
        return G

    def inverse_meas_only(self, pixels: np.ndarray, b: float):
        """Eq. (27) without computing the covariance (used internally)."""
        pixels = np.asarray(pixels)
        u_l, v_l, u_r = pixels[0], pixels[1], pixels[2]
        d = u_l - u_r
        ratio = b / d
        x = (u_l - self.cu) * ratio
        y = (v_l - self.cv) * ratio * (self.fu / self.fv)
        z = ratio * self.fu
        return np.stack([x, y, z], axis=0), d

    def pixel_covariance(self) -> np.ndarray:
        """Covariance of ``(u_l, v_l, d)``, with ``d = u_l - u_r``.

        Left/right horizontal noise is independent with std ``sigma_u``, hence
        ``var(d) = 2 sigma_u^2`` and ``cov(u_l, d) = sigma_u^2``.
        """
        S = np.zeros((3, 3))
        S[0, 0] = self.sigma_u**2
        S[1, 1] = self.sigma_v**2
        S[2, 2] = 2.0 * self.sigma_u**2
        S[0, 2] = self.sigma_u**2
        S[2, 0] = self.sigma_u**2
        return S

    # ------------------------------------------------------------------
    # convenience: measurement Jacobian w.r.t. the baseline b
    # ------------------------------------------------------------------
    def inverse_dmeas_db(self, pixels: np.ndarray, b: float) -> np.ndarray:
        """``dm_k / db`` (analytic).  Robust to any noise model."""
        pixels = np.asarray(pixels)
        u_l, v_l, u_r = pixels[0], pixels[1], pixels[2]
        d = u_l - u_r
        J = np.zeros((pixels.shape[1], 3, 1), dtype=np.result_type(pixels, b, float))
        J[:, 0, 0] = (u_l - self.cu) / d
        J[:, 1, 0] = (v_l - self.cv) / d * (self.fu / self.fv)
        J[:, 2, 0] = self.fu / d
        return J


def rotate_points(C: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply rotation ``C`` to the columns of ``p``."""
    return C @ p
