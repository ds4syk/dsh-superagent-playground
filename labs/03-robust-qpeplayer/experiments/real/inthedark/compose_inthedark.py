#!/usr/bin/env python3
"""Compose per-pair GT relative poses for a UTIAS In-The-Dark **repeat** run.

Why this exists
---------------
The official archive layout (verified against the real zips, see README) is:

  * a teach run (run_000000) ships ``transforms_temporal.txt``: consecutive
    teach vertices, ``T`` per row;
  * a repeat run ships only ``transforms_spatial.txt``: one row per repeat
    vertex, mapping it to the teach vertex that place-recognition + ICP
    associated it with.

So a repeat run has no direct per-pair pose.  But for consecutive repeat
vertices i, i+1 with teach matches j_i, j_{i+1} the chain closes:

    T_{rep_{i+1} <- rep_i}
        = T_{rep_{i+1} <- teach_{j_{i+1}}}
        . T_{teach_{j_{i+1}} <- teach_{j_i}}
        . T_{teach_{j_i} <- rep_i}

with ``T_{rep <- teach} = inv(T_{teach <- rep})`` from the spatial file and the
teach leg chained through the teach temporal file (gap 1/2/3 links).

Two readings of the row convention are possible (``ba``: p_b = T_ba p_a, i.e.
the matrix maps *from* the first vertex *to* the second; ``ab``: the transpose
reading).  ``compose_pairs(convention=...)`` implements both; the convention is
selected on a calibration split by stereo reprojection residual
(``calibrate_convention.py``), never by the pairs that are reported.

Pairs whose two repeat vertices match the *same* teach vertex (teach delta 0) are
still exact: both were registered into the same teach frame j, so the relative
pose is simply ``inv(M_{i+1}) M_i``.  Measured on real data: 0.75-1.15 px median
stereo reprojection error, 97-100% of matches below 2 px.

Coordinate frame: the published transforms are between **vehicle** frames.  The
stereo front-end works in the **camera** frame, so ``to_camera`` conjugates with
the published ``transform_camera_vehicle.txt`` (``T_cam = E T_veh inv(E)`` with
E = T_camera_vehicle, i.e. E maps vehicle coords into camera coords).  This is
the only reading that reproduces the images: 0.25-0.93 px median residual and
98-99% of matches under 2 px, versus 10-17 px without conjugation and 4-6 px for
the transposed row convention.

Fetching the teach metadata without the 905 MB download: ``fetch_zip_members.py``
range-extracts ``transforms_temporal.txt`` (315 KB) from the remote zip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import load_inthedark as L


def load_teach_temporal(teach_dir):
    """-> {(run, j_from, j_to): T} for gap-1 teach links, plus raw dict."""
    p = Path(teach_dir) / "transforms_temporal.txt"
    raw = L.read_transforms(p)
    links = {}
    for (ia, ja, ib, jb), T in raw.items():
        if ia == ib and jb == ja + 1:
            links[jb] = T  # T_{ja <- jb} under 'ba' reading (handled by caller)
    return raw, links


def load_spatial(run_dir):
    """-> ({(run, v): (teach_run, teach_v, T_raw)}, all_rows)."""
    p = Path(run_dir) / "transforms_spatial.txt"
    raw = L.read_transforms(p)
    out = {}
    for (ia, ja, ib, jb), T in raw.items():
        out.setdefault((ia, ja), []).append((ib, jb, T))
    multi = {k: v for k, v in out.items() if len(v) > 1}
    return out, raw, multi


def teach_chain(links, j0, j1, convention):
    """T_{teach_{j1} <- teach_{j0}} by walking gap-1 links (both readings)."""
    if j1 == j0:
        return np.eye(4), 0
    step = 1 if j1 > j0 else -1
    T_acc = np.eye(4)
    n = 0
    j = j0
    while j != j1:
        nxt = j + step
        T = links.get(nxt)  # row (0, j, 0, nxt)
        if T is None:
            return None, n
        if convention == "ba":      # row = T_{nxt <- j}
            leg = T
        elif convention == "ab":    # row = T_{j <- nxt}  => invert
            leg = np.linalg.inv(T)
        else:
            raise ValueError(convention)
        T_acc = leg @ T_acc
        j = nxt
        n += 1
    return T_acc, n


def compose_pairs(run_dir, teach_dir, convention="ba", max_teach_jump=4):
    """Per-pair composed GT for consecutive repeat vertices.

    Returns dict:
      run, convention, n_vertices,
      pairs: {(va, vb): T (p_vb = T p_va)},   # only resolvable pairs
      skipped: {(va, vb): reason},
      diag: per-pair teach ids / jumps.
    """
    spat, raw_rows, multi = load_spatial(run_dir)
    teach_raw, links = load_teach_temporal(teach_dir)
    run_no = int(Path(run_dir).name.split("_")[-1])

    match = {}
    for (r, v), lst in spat.items():
        if r == run_no:
            t_run, t_v, T = lst[0]
            match[v] = (t_run, t_v, T)
    if not match:
        raise SystemExit(f"no spatial rows for run {run_no} in {run_dir}")

    verts = sorted(match)
    pairs, skipped, diag = {}, {}, []
    for va, vb in zip(verts, verts[1:]):
        if vb != va + 1:
            skipped[(va, vb)] = "non-consecutive vertex ids"
            continue
        (ta, ja, M_a), (tb, jb, M_b) = match[va], match[vb]
        if (ta, tb) != (0, 0):
            skipped[(va, vb)] = f"teach runs differ ({ta},{tb})"
            continue
        if abs(jb - ja) > max_teach_jump:
            skipped[(va, vb)] = f"teach jump {jb - ja} > {max_teach_jump}"
            continue
        if jb == ja:
            # Both frames were registered into the *same* teach frame j, so
            # p_rep_{vb} = inv(M_b) M_a p_rep_{va} exactly -- no teach leg needed.
            # (Verified on real data: 0.75-1.15 px median stereo reprojection,
            # 97-100% of matches < 2 px; see verify_inthedark.py.)
            T_teach, n_links = np.eye(4), 0
        else:
            T_teach, n_links = teach_chain(links, ja, jb, convention)
            if T_teach is None:
                skipped[(va, vb)] = f"teach chain {ja}->{jb} has a hole"
                continue
        if convention == "ba":
            # M = T_{teach <- rep}; T_teach = T_{teach_jb <- teach_ja}
            T = np.linalg.inv(M_b) @ T_teach @ M_a
        else:
            # M = T_{rep <- teach} (transpose reading); T_teach = T_{jb<-ja} as built
            T = M_b @ np.linalg.inv(T_teach) @ np.linalg.inv(M_a)
        pairs[(va, vb)] = T
        diag.append(dict(va=va, vb=vb, teach_a=ja, teach_b=jb, jump=jb - ja,
                         n_links=n_links,
                         t_norm=float(np.linalg.norm(T[:3, 3])),
                         rot_deg=float(np.degrees(np.arccos(
                             np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))))
    return dict(run=Path(run_dir).name, convention=convention,
                n_vertices=len(verts), n_spatial_rows=len(raw_rows),
                n_teach_links=len(links), pairs=pairs, skipped=skipped,
                diag=diag, multi_match=len(multi))


def trajectory(composed):
    """Integrate composed consecutive poses -> per-vertex pose w.r.t. vertex 0.

    p_0 = T_k p_k, so T_k = T_{0 <- k}; chaining: T_{0<-k} = T_{0<-k-1} . T_{k-1<-k}
    with T_{k-1<-k} = inv(T_{k<-k-1}).
    """
    verts = sorted({v for pair in composed["pairs"] for v in pair})
    if not verts:
        return {}
    T = {verts[0]: np.eye(4)}
    for (va, vb), T_ba in composed["pairs"].items():
        if va not in T:
            continue
        T[vb] = T[va] @ np.linalg.inv(T_ba)
    return T


def read_matrix(path):
    """Parse a whitespace/comma separated 16-value row into a 4x4."""
    vals = [float(x) for x in Path(path).read_text().replace(",", " ").split()]
    if len(vals) != 16:
        raise ValueError(f"{path}: expected 16 values, got {len(vals)}")
    return np.array(vals, float).reshape(4, 4)


def load_camera_vehicle(data_root):
    """E = T_camera_vehicle from the official transform_camera_vehicle.txt."""
    return read_matrix(Path(data_root) / "transform_camera_vehicle.txt")


def to_camera(T_veh, E):
    """Vehicle-frame relative pose -> camera-frame relative pose."""
    return E @ T_veh @ np.linalg.inv(E)


def gt_pairs(run_dir, teach_dir, data_root, convention="ba"):
    """Composed GT in the camera frame, ready for the stereo front-end.

    Returns (pairs, skipped, meta): pairs[(va, vb)] = T with p_cam_vb = T p_cam_va.
    """
    comp = compose_pairs(run_dir, teach_dir, convention=convention)
    E = load_camera_vehicle(data_root)
    pairs = {k: to_camera(T, E) for k, T in comp["pairs"].items()}
    meta = dict(comp={k: v for k, v in comp.items()
                      if k not in ("pairs", "skipped", "diag")},
                n_pairs=len(pairs), n_skipped=len(comp["skipped"]),
                camera_vehicle=E.tolist(), convention=convention)
    return pairs, comp["skipped"], meta
