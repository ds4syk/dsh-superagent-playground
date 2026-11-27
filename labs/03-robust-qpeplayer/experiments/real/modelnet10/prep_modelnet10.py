#!/usr/bin/env python3
"""Preprocess ModelNet10 OFF meshes into 2048-point clouds (test split).

ModelNet10 ships triangle meshes in OFF format, not the 2048-point HDF5
clouds that ModelNet40 provides.  To keep the registration experiment
comparable we resample every mesh with the same recipe used for the
ModelNet40 release: 2048 points drawn uniformly *by surface area* from the
triangle soup (area-weighted triangle choice + uniform barycentric
coordinates), then cached as a single .npz.

Only the official ``test`` split is used, mirroring the ModelNet40 runner
which reads ``ply_data_test*.h5``.

Usage
-----
    python prep_modelnet10.py \
        --root /mnt/storage1/syk/data/modelnet10/ModelNet10 \
        --split test \
        --out  /mnt/storage1/syk/data/modelnet10/modelnet10_test_2048.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_POINTS = 2048


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an OFF file -> (vertices [V,3], triangular faces [F,3])."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    # Locate the OFF magic line (some files start with a comment or 'OFF' alone).
    start = 0
    for i, line in enumerate(lines[:5]):
        if line.strip().upper().startswith("OFF"):
            start = i
            break
    header = lines[start].strip()
    rest = header[3:].strip()  # tokens after 'OFF' on the same line, if any
    body = lines[start + 1:]
    if rest:
        counts = [int(x) for x in rest.split()[:2]]
    else:
        counts = None
        while body and not body[0].strip():
            body.pop(0)
        parts = body[0].split()
        counts = [int(parts[0]), int(parts[1])]
        body = body[1:]
    nv, nf = counts
    verts = np.empty((nv, 3), dtype=np.float64)
    for i in range(nv):
        p = body[i].split()
        verts[i] = (float(p[0]), float(p[1]), float(p[2]))
    faces = []
    for i in range(nv, nv + nf):
        p = body[i].split()
        k = int(p[0])
        idx = [int(v) for v in p[1:1 + k]]
        # fan-triangulate polygons (ModelNet10 is triangles, but be safe)
        for j in range(1, k - 1):
            faces.append((idx[0], idx[j], idx[j + 1]))
    return verts, np.asarray(faces, dtype=np.int64)


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    tri = verts[faces]                                     # [F,3,3]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.clip(area, 0.0, None)
    if area.sum() <= 0:
        raise ValueError("degenerate mesh")
    prob = area / area.sum()
    pick = rng.choice(len(faces), size=n, p=prob)
    u = rng.random((n, 2))
    s = np.sqrt(u[:, 0])[:, None]
    t = u[:, 1][:, None]
    p = ((1.0 - s) * tri[pick, 0]
         + s * (1.0 - t) * tri[pick, 1]
         + s * t * tri[pick, 2])
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(args.root.glob(f"*/{args.split}/*.off"))
    if not files:
        raise SystemExit(f"no OFF files under {args.root} split={args.split}")
    cats = sorted({p.parts[-3] for p in files})
    clouds, labels, names = [], [], []
    for k, f in enumerate(files):
        v, fc = read_off(f)
        clouds.append(sample_surface(v, fc, N_POINTS, rng))
        labels.append(f.parts[-3])
        names.append(f.stem)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    data = np.stack(clouds).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, data=data, labels=np.asarray(labels),
        names=np.asarray(names), split=args.split,
        source=str(args.root), n_points=N_POINTS, seed=args.seed)
    meta = dict(n_objects=len(files), split=args.split, categories=cats,
                points_per_object=N_POINTS, seed=args.seed,
                roots=[p.parts[-3] + "/" + p.stem for p in files])
    (args.out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} shape={data.shape} categories={cats}")


if __name__ == "__main__":
    main()
