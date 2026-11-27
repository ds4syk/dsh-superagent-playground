#!/usr/bin/env python3
"""3DMatch scene-fragment loader and real-correspondence builder.

Data
----
The 3DMatch geometric-registration benchmark distributes each scene as a
zip of TSDF-fused fragments (``cloud_bin_*.ply``) plus an evaluation zip
with ``gt.log`` (pair id1, id2, number of inliers, 4x4 transform).  The
transform in ``gt.log`` maps points of fragment ``id1`` into the frame of
fragment ``id2`` (verified empirically by
``verify_gt_direction.py``; do not assume the opposite).

This module deliberately keeps the pipeline small and dependency-light:
``plyfile`` (present in the WM env) + ``scipy.spatial.cKDTree``.  No
Open3D, no FCGF, no GPU.  Two correspondence modes are supported:

``gt_nn``
    Sample source points, map them with the ground-truth transform, and
    take the nearest target point.  This is a *real-overlap* geometry
    test (real scanner noise, occlusion, sampling) but the base matches
    are oracle correspondences; it isolates the robust outer layer.
``descriptor``
    A hand-crafted local descriptor (see ``local_descriptor``) matched by
    nearest neighbour with mutual-consistency filtering.  This is a real
    front-end stand-in (no ground truth at match time) and is reported
    separately; it is *not* FCGF/Predator.

All coordinates stay in metres.  The benchmark fragments are in metres;
voxel sizes and thresholds are therefore quoted in metres.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# PLY / scene loading
# ---------------------------------------------------------------------------
def read_ply(path, stride=1):
    """Read xyz from a binary or ascii PLY, optionally striding points."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([np.asarray(v["x"], dtype=np.float64),
                    np.asarray(v["y"], dtype=np.float64),
                    np.asarray(v["z"], dtype=np.float64)], axis=1)
    if stride > 1:
        xyz = xyz[::stride]
    return xyz


def voxel_downsample(points, voxel):
    """Deterministic voxel-grid downsample (mean per occupied voxel)."""
    if len(points) == 0:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int64)
    # unique rows + inverse, then mean per cell (stable and deterministic)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float64)
    cnt = np.bincount(inv)
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=points[:, k]) / cnt
    return out


def load_scene(scene_dir, voxel=0.03, cache_dir=None):
    """Load and cache voxel-downsampled fragments of one scene.

    Returns ``dict(bin_id -> (M,3) float64)``.  Cache files are keyed by
    the voxel size so changing the resolution never silently reuses stale
    clouds.
    """
    scene_dir = Path(scene_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else scene_dir.parent / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{scene_dir.name}_v{voxel:.3f}"
    cache = cache_dir / f"{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {int(k): z[k] for k in z.files}
    clouds = {}
    for p in sorted(scene_dir.glob("cloud_bin_*.ply")):
        bid = int(p.stem.split("_")[-1])
        xyz = read_ply(p, stride=1)
        clouds[bid] = voxel_downsample(xyz, voxel)
    np.savez_compressed(cache, **{str(k): v for k, v in clouds.items()})
    return clouds


# ---------------------------------------------------------------------------
# gt.log parsing
# ---------------------------------------------------------------------------
def parse_gt_log(path):
    """Parse 3DMatch ``gt.log`` into ``[(id1, id2, n, T), ...]``."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) != 3:
            raise ValueError(f"bad gt.log header at line {i + 1}: {lines[i]!r}")
        id1, id2, n = int(head[0]), int(head[1]), int(head[2])
        rows = []
        for r in range(4):
            vals = [float(x) for x in lines[i + 1 + r].split()]
            if len(vals) != 4:
                raise ValueError(f"bad gt.log matrix row at line {i + 2 + r}")
            rows.append(vals)
        T = np.asarray(rows, dtype=np.float64)
        pairs.append((id1, id2, n, T))
        i += 5
    return pairs


# ---------------------------------------------------------------------------
# Correspondence construction
# ---------------------------------------------------------------------------
def gt_nn_correspondences(cloud_i, cloud_j, T, n=400, rng=None, tau=0.06):
    """Oracle nearest-neighbour correspondences on real overlapping geometry.

    ``T`` must map source cloud ``cloud_i`` into target cloud ``cloud_j``.
    A source point is kept only if its mapped location has a target point
    within ``tau`` metres.  We oversample by 4x because real fragment
    overlap is only partial; the returned set has at most ``n`` pairs and
    the residual is the real scan/occlusion/sampling residual.
    """
    rng = np.random.default_rng(rng)
    n_probe = min(len(cloud_i), max(n, 4 * n))
    idx = rng.choice(len(cloud_i), n_probe, replace=False) if n_probe < len(cloud_i) else np.arange(len(cloud_i))
    x = cloud_i[idx]
    xin = (T[:3, :3] @ x.T).T + T[:3, 3]
    tree = cKDTree(cloud_j)
    d, j = tree.query(xin, k=1, workers=-1)
    keep = d < tau
    if keep.sum() < 3:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    x, y, d = x[keep], cloud_j[j[keep]], d[keep]
    if len(x) > n:
        take = rng.choice(len(x), n, replace=False)
        x, y, d = x[take], y[take], d[take]
    return x, y, d


def _normals(points, tree, k=16):
    """Local PCA normals; sign is arbitrary but consistent within a cloud."""
    k = min(k, len(points))
    _, idx = tree.query(points, k=k, workers=-1)
    nb = points[idx]                       # (M,k,3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb) / max(k - 1, 1)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                      # smallest-eigenvalue direction


def local_descriptor(points, normals, tree, k=12, bins=8, radius=0.12):
    """A compact FPFH-like descriptor, enough for a CPU-only real front-end.

    For each point we collect neighbours inside ``radius`` and build:
      * 8-bin histogram of ``log(1 + d/0.01)`` (distance profile),
      * 8-bin histogram of the angle between the neighbour offset and the
        point normal (normal-consistency profile),
      * 8-bin histogram of the neighbour-normal / point-normal angle,
      * mean distance, normalised by radius.

    The descriptor is intentionally simple.  It is a stand-in for a learned
    descriptor, not a claim of FCGF-level matching; the paper must label it
    as a hand-crafted front-end.
    """
    M = len(points)
    desc = np.zeros((M, bins * 3 + 1), dtype=np.float32)
    for i in range(M):
        idx = tree.query_ball_point(points[i], radius)
        if len(idx) <= 1:
            continue
        off = points[idx] - points[i]
        d = np.linalg.norm(off, axis=1)
        ok = d > 1e-9
        off, d = off[ok], d[ok]
        idx = np.asarray(idx)[ok]
        if len(d) == 0:
            continue
        u = off / d[:, None]
        c = np.abs(u @ normals[i])
        nang = np.abs(np.sum(normals[idx] * normals[i], axis=1))
        h1, _ = np.histogram(np.log1p(d / 0.01), bins=bins, range=(0.0, 6.0))
        h2, _ = np.histogram(c, bins=bins, range=(0.0, 1.0))
        h3, _ = np.histogram(nang, bins=bins, range=(0.0, 1.0))
        h = np.concatenate([h1, h2, h3]).astype(np.float32)
        h /= max(len(d), 1)
        desc[i, :bins * 3] = h
        desc[i, -1] = float(np.mean(d) / radius)
    return desc


def descriptor_correspondences(cloud_i, cloud_j, n=400, rng=None,
                               voxel=0.03, radius=0.12, mutual=True,
                               max_dist=0.10):
    """Descriptor nearest-neighbour matches with optional mutual check.

    Returns ``(x, y, residual, src_idx, dst_idx)``.  No ground truth is
    used in matching; callers can label correctness with the GT transform.
    """
    rng = np.random.default_rng(rng)
    # Use a dense-ish subset for descriptors, then sample n matches.
    ti = cKDTree(cloud_i)
    tj = cKDTree(cloud_j)
    ni = _normals(cloud_i, ti)
    nj = _normals(cloud_j, tj)
    di = local_descriptor(cloud_i, ni, ti, radius=radius)
    dj = local_descriptor(cloud_j, nj, tj, radius=radius)
    # Normalise descriptor columns for cosine matching.
    di = di / np.maximum(np.linalg.norm(di, axis=1, keepdims=True), 1e-9)
    dj = dj / np.maximum(np.linalg.norm(dj, axis=1, keepdims=True), 1e-9)
    # For speed, restrict source points to a random subset when huge.
    if len(cloud_i) > 6000:
        sidx = rng.choice(len(cloud_i), 6000, replace=False)
    else:
        sidx = np.arange(len(cloud_i))
    sim = di[sidx] @ dj.T
    j = np.argmax(sim, axis=1)
    if mutual:
        i_best = np.argmax(dj @ di[sidx].T, axis=1)
        keep = i_best[j] == np.arange(len(j))
        sidx, j = sidx[keep], j[keep]
    d = np.linalg.norm(cloud_i[sidx] - cloud_j[j], axis=1)
    keep = d < max_dist
    sidx, j, d = sidx[keep], j[keep], d[keep]
    if len(sidx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), sidx, j
    if len(sidx) > n:
        take = rng.choice(len(sidx), n, replace=False)
        sidx, j, d = sidx[take], j[take], d[take]
    return cloud_i[sidx], cloud_j[j], d, sidx, j


# ---------------------------------------------------------------------------
# Small CLI helpers
# ---------------------------------------------------------------------------
def scene_paths(root, scene):
    root = Path(root)
    return root / scene, root / f"{scene}-evaluation" / "gt.log"


def summarize_scene(clouds, pairs, tau=0.06, n=600, seed=0, max_pairs=12):
    rng = np.random.default_rng(seed)
    out = []
    for id1, id2, n_gt, T in pairs:
        if id1 not in clouds or id2 not in clouds:
            continue
        x, y, d = gt_nn_correspondences(clouds[id1], clouds[id2], T,
                                        n=n, rng=rng, tau=tau)
        out.append(dict(id1=id1, id2=id2, n_gt=n_gt, n_pairs=len(x),
                        overlap=float(len(x) / max(n, 1)),
                        med_res=float(np.median(d)) if len(d) else None,
                        p90_res=float(np.percentile(d, 90)) if len(d) else None,
                        T=T.tolist()))
        if len(out) >= max_pairs and max_pairs > 0:
            break
    return out
