"""Synthetic rotating-camera sequences on an equirectangular panorama.

The "world" is a large textured wall modelled as an equirectangular
environment map at infinity.  A camera rotating in place therefore sees
exactly the perspective view obtained by sampling the environment map along
the rotated rays, so the ground-truth camera rotation is exact and no
rendering engine is needed.

This gives a controlled benchmark for *rotation synchronization*: the
downstream quantity is a set of camera orientations, whose relative rotations
are the measurements produced by a learned head on top of V-JEPA 2 features.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# small SO(3) helpers
# ---------------------------------------------------------------------------


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between rotation matrices (batch aware)."""
    Ra = np.asarray(Ra, dtype=np.float64)
    Rb = np.asarray(Rb, dtype=np.float64)
    tr = np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ---------------------------------------------------------------------------
# panorama construction
# ---------------------------------------------------------------------------


def _crop_black_border(img: np.ndarray, thr: int = 6) -> np.ndarray:
    gray = img.mean(axis=2)
    rows = np.where(gray.mean(axis=1) > thr)[0]
    cols = np.where(gray.mean(axis=0) > thr)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _read_video_frames(path: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 1), n + 2)[1:-1].astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            out.append(_crop_black_border(frame))
    cap.release()
    return out


def build_panorama(
    video_root: str | None = None,
    seed: int = 0,
    size: tuple[int, int] = (2048, 1024),
    n_tiles: int = 16,
    image_path: str | None = None,
) -> np.ndarray:
    """Build a 2:1 equirectangular panorama.

    If ``image_path`` is given, the image is resized to ``size`` and used
    directly (a coherent natural scene).  Otherwise frames from UCF101 are
    tiled into a collage (fallback for a fully local, license-clean source).
    """
    W, H = size
    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if video_root is None:
        raise ValueError("either video_root or image_path is required")
    rng = np.random.default_rng(seed)
    videos = sorted(glob.glob(os.path.join(video_root, "*", "*.avi")))
    if not videos:
        raise FileNotFoundError(f"no avi files under {video_root}")
    cols = int(np.ceil(np.sqrt(n_tiles * W / H)))
    rows = int(np.ceil(n_tiles / cols))
    tw, th = W // cols, H // rows
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    # Sample enough distinct frames to fill every cell; unused cells repeat.
    chosen = rng.choice(len(videos), size=rows * cols, replace=len(videos) < rows * cols)
    for k, vi in enumerate(chosen):
        frames = _read_video_frames(videos[int(vi)], 1, rng)
        if not frames:
            continue
        img = frames[0]
        # keep aspect ratio: resize short side then centre crop
        ih, iw = img.shape[:2]
        s = max(tw / iw, th / ih)
        img = cv2.resize(img, (max(1, int(round(iw * s))), max(1, int(round(ih * s)))))
        y0 = max(0, (img.shape[0] - th) // 2)
        x0 = max(0, (img.shape[1] - tw) // 2)
        img = img[y0 : y0 + th, x0 : x0 + tw]
        r, c = divmod(k, cols)
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = img
    # Slight blur hides hard seams without destroying texture.
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    return canvas


# ---------------------------------------------------------------------------
# equirectangular sampler
# ---------------------------------------------------------------------------

_FOV_DEG = 70.0
_OUT = 256


def _ray_grid(out: int = _OUT, fov_deg: float = _FOV_DEG):
    f = (out / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = cy = out / 2.0
    us, vs = np.meshgrid(np.arange(out, dtype=np.float32), np.arange(out, dtype=np.float32))
    rays = np.stack([(us - cx) / f, (vs - cy) / f, np.ones_like(us)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.astype(np.float64)


_RAYS = _ray_grid()


def render_view(pano: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    """Render a perspective view for camera orientation ``R_wc`` (world->cam)."""
    H, W = pano.shape[:2]
    # world rays: d_world = R_wc^T d_cam
    d = _RAYS @ R_wc  # (out,out,3) @ (3,3) -> rows are R_wc^T d_cam
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1.0, 1.0))
    px = (lon / (2 * np.pi) + 0.5) * W
    py = (0.5 - lat / np.pi) * H
    map_x = px.astype(np.float32)
    map_y = py.astype(np.float32)
    out = cv2.remap(
        pano,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return out


# ---------------------------------------------------------------------------
# sequence generation
# ---------------------------------------------------------------------------

N_NODES = 8
N_FRAMES = 2 * N_NODES


def trajectory(nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth rotation trajectory with R(0) = I.

    Yaw sweeps up to +/-45 deg, pitch +/-14 deg, roll +/-6 deg, with a random
    number of half-cycles.  The camera therefore keeps substantial overlap
    between adjacent nodes while accumulating a large global rotation.
    """
    s = np.linspace(0.0, 1.0, nodes)
    half = rng.uniform(0.6, 1.4)
    phase = rng.uniform(0.0, 0.3)
    yaw = np.deg2rad(rng.uniform(30.0, 45.0)) * np.sin(np.pi * half * s + phase)
    pitch = np.deg2rad(rng.uniform(3.0, 8.0)) * np.sin(
        np.pi * half * s + phase + rng.uniform(0.4, 1.2)
    )
    roll = np.deg2rad(rng.uniform(2.0, 4.0)) * np.sin(
        2 * np.pi * s + rng.uniform(0.0, 2 * np.pi)
    )
    Rs = np.stack([rot_z(r).dot(rot_y(p)).dot(rot_x(y)) for y, p, r in zip(yaw, pitch, roll)])
    # Fix the gauge: orientation at node 0 is the identity.
    R0 = Rs[0]
    return np.einsum("ij,njk->nik", R0.T, Rs)


def make_sequence(
    pano: np.ndarray,
    rng: np.random.Generator,
    nodes: int = N_NODES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames[2*nodes,256,256,3] uint8, R_gt[nodes,3,3])``.

    Node ``t`` corresponds to frames ``2t`` and ``2t+1``; its ground-truth
    orientation is the rotation at frame ``2t``.
    """
    R_nodes = trajectory(nodes, rng)  # (nodes,3,3)
    frames = []
    for k in range(2 * nodes):
        if k % 2 == 0:
            R = R_nodes[k // 2]
        else:
            # half-way between consecutive node orientations
            a, b = k // 2, min(k // 2 + 1, nodes - 1)
            R = _slerp(R_nodes[a], R_nodes[b], 0.5)
        frames.append(render_view(pano, R))
    return np.stack(frames).astype(np.uint8), R_nodes


def _so3_log(R: np.ndarray) -> np.ndarray:
    # only used by a debug branch above; keep a minimal implementation
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-8:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _slerp(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation between two rotations (small-angle safe)."""
    dR = Ra.T @ Rb
    c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-8:
        return Ra.copy()
    axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(a))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rt = np.eye(3) + np.sin(a * t) * K + (1 - np.cos(a * t)) * (K @ K)
    return Ra @ Rt


def build_dataset(
    pano: np.ndarray,
    n_seq: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frames, rots = [], []
    for _ in range(n_seq):
        f, R = make_sequence(pano, rng)
        frames.append(f)
        rots.append(R)
    return np.stack(frames), np.stack(rots)
