#!/usr/bin/env python3
"""Shared helpers for the full-set and overlap-bin 3DMatch experiments.

Only *new* code lives here; scene loading and GT parsing are imported from the
parent ``experiments/real/threedmatch/load_3dmatch.py`` (never edited).

Why a dedicated overlap estimator
---------------------------------
The parent helper ``gt_nn_correspondences`` samples ``max(n, 4n)`` source
points, keeps those with target NN distance < tau, then caps the returned set
at ``n``.  Dividing the *capped* count by the probe count therefore reports
``min(1, 4 * true_overlap)``: every pair with >=25% true overlap reads as
100%.  The parent tables are unaffected (they only need high-overlap pairs and
rank within the saturated set by median residual), but a low/mid/high overlap
study needs the uncapped fraction.  ``overlap_fraction`` below computes it
without any cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "threedmatch"
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(SYN))
import load_3dmatch as L  # noqa: E402,F401  (re-exported for the runners)
import common as cm  # noqa: E402,F401

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")

SCENES = [
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-hotel_uc-scan3",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
]
SCENE_SHORT = {
    "7-scenes-redkitchen": "redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1": "sun3d-home_at",
    "sun3d-hotel_uc-scan3": "sun3d-hotel_uc",
    "sun3d-home_md-home_md_scan9_2012_sep_30": "sun3d-home_md",
    "sun3d-hotel_umd-maryland_hotel1": "sun3d-hotel1",
    "sun3d-hotel_umd-maryland_hotel3": "sun3d-hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2": "sun3d-studyroom",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika": "sun3d-mit_lab_hj",
}
NEW_SCENES = SCENES[4:]

VOXEL = 0.03
TAU = 0.06
N = 300
C = 0.05


def overlap_fraction(src, dst, T_src_to_dst, rng, n_probe=2000, tau=TAU):
    """Uncapped fraction of sampled ``src`` points with a ``dst`` NN < tau.

    ``T_src_to_dst`` must map ``src`` into the ``dst`` frame (i.e. ``inv`` of
    the stored gt.log transform, which maps id2->id1).
    """
    idx = rng.choice(len(src), min(n_probe, len(src)), replace=False)
    q = (T_src_to_dst[:3, :3] @ src[idx].T).T + T_src_to_dst[:3, 3]
    d, _ = cKDTree(dst).query(q, k=1, workers=-1)
    keep = d < tau
    return dict(overlap=float(keep.mean()),
                med_res=float(np.median(d[keep])) if keep.any() else None,
                p90_res=float(np.percentile(d[keep], 90)) if keep.any() else None,
                n_kept=int(keep.sum()), n_probe=int(len(idx)),
                n_src=int(len(src)), n_dst=int(len(dst)))


def scan_scene_overlaps(clouds, gt_pairs, n_probe=2000, tau=TAU, seed=0):
    """GT-NN overlap for every pair listed in gt.log of one scene."""
    rng = np.random.default_rng(seed)
    rows = []
    for id1, id2, n_gt, T in gt_pairs:
        if id1 not in clouds or id2 not in clouds:
            continue
        Ti = np.linalg.inv(T)  # stored maps id2 -> id1
        s = overlap_fraction(clouds[id1], clouds[id2], Ti, rng,
                             n_probe=n_probe, tau=tau)
        rows.append(dict(id1=id1, id2=id2, n_gt=n_gt, **s))
    return rows


def overlap_bin(v, low=0.5, high=0.8):
    if v < low:
        return "low"
    if v <= high:
        return "mid"
    return "high"


def pick_bin_pairs(rows, per_bin=3):
    """Pick up to ``per_bin`` pairs per bin spread over the bin's overlap range."""
    out = {}
    for b in ["low", "mid", "high"]:
        sel = sorted([r for r in rows if r["bin"] == b], key=lambda r: r["overlap"])
        if not sel:
            out[b] = []
            continue
        if len(sel) <= per_bin:
            out[b] = sel
        else:
            pos = np.linspace(0, len(sel) - 1, per_bin).round().astype(int)
            out[b] = [sel[i] for i in sorted(set(pos.tolist()))]
    return out


def project_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def clean_pair_sample(clouds, pair, rng, tau=TAU, n=N):
    """N clean GT-NN correspondences for one fragment pair.

    ``pair`` must carry ``Tinv`` (src->dst).  When the pair yields fewer than
    ``n`` valid correspondences they are resampled with replacement; this
    happens for pairs with true overlap below ~``n / (4 n)`` = 25%, and the
    raw ``n_valid`` is recorded by the caller.
    """
    x, y, d = L.gt_nn_correspondences(clouds[pair["id1"]], clouds[pair["id2"]],
                                      pair["Tinv"], n=n, rng=rng, tau=tau)
    if len(x) < 3:
        return None
    n_valid = len(x)
    if len(x) < n:
        take = rng.choice(len(x), n, replace=True)
        x, y, d = x[take], y[take], d[take]
    return dict(x=x, y=y, d=d, n_valid=n_valid,
                Rgt=project_rotation(pair["Tinv"][:3, :3]),
                tgt=pair["Tinv"][:3, 3].copy())


def inject_random(x, y, rng, p, cloud_dst):
    y = y.copy()
    n_out = int(round(p * len(x)))
    if n_out > 0:
        idx = rng.choice(len(x), n_out, replace=False)
        y[idx] = cloud_dst[rng.choice(len(cloud_dst), n_out, replace=True)]
    return y


def make_methods(c=C):
    def kabsch_m(x, y, rng):
        return (*cm.kabsch(x, y, np.ones(len(x))), {})

    def ransac_fixed(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=c, iters=cm.RANSAC_ITERS,
                               adaptive=False, return_info=True)

    def ransac_adapt(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=c, iters=cm.RANSAC_ITERS,
                               adaptive=True, return_info=True)

    def irls(x, y, rng):
        return (*cm.solve_irls(x, y, c=c), {})

    def gnc(x, y, rng):
        return (*cm.solve_gnc(x, y, c=c), {})

    return {"Kabsch": kabsch_m, "RANSAC-1000": ransac_fixed,
            "RANSAC-adaptive": ransac_adapt, "IRLS-GM": irls, "GNC-GM": gnc}
