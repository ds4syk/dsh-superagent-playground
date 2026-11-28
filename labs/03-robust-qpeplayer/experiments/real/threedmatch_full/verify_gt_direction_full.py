#!/usr/bin/env python3
"""Verify the ``gt.log`` transform direction on all eight official 3DMatch scenes.

The parent pipelines established that the stored 4x4 maps fragment ``id2``
into the frame of ``id1`` on four scenes.  Direction is a per-file
convention, so this script re-runs the same check on *all* scenes used in the
full-set table, including the four new ones
(``sun3d-hotel_umd-maryland_hotel1/3``, ``sun3d-mit_76_studyroom``,
``sun3d-mit_lab_hj``).

Method (identical to ``experiments/real/threedmatch_extra/verify_gt_direction_extra.py``)
----------------------------------------------------------------------------------------
For the first 12 pairs of each scene, sample 3000 source points from
fragment ``id1``, map them with the stored transform ``T`` and with
``inv(T)``, and take the nearest neighbour in fragment ``id2``.  The correct
direction gives a much smaller median distance.  We record the per-pair
medians and the scene-level medians so the inversion used by the runner is
auditable.

Output: ``results/verify_gt_direction_full.json``
Run:
  /home/syk/miniconda3/envs/WM/bin/python verify_gt_direction_full.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "threedmatch"))  # reuse, never edit
import load_3dmatch as L  # noqa: E402

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

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
N_SAMPLE = 3000
TAU = 0.06


def nn_stats(src, dst, T, rng, n=N_SAMPLE, tau=TAU):
    idx = rng.choice(len(src), min(n, len(src)), replace=False)
    p = src[idx]
    q = (T[:3, :3] @ p.T).T + T[:3, 3]
    d, _ = cKDTree(dst).query(q, k=1, workers=-1)
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                frac_lt_tau=float(np.mean(d < tau)), n=len(d))


def main():
    report = {"tau": TAU, "n_sample": N_SAMPLE, "scenes": {}}
    for scene in SCENES:
        scene_dir = ROOT / scene
        gt_path = ROOT / f"{scene}-evaluation" / "gt.log"
        clouds = L.load_scene(scene_dir, voxel=0.03)
        pairs = L.parse_gt_log(gt_path)
        rng = np.random.default_rng(20261127)
        rows = []
        for id1, id2, n_gt, T in pairs[:12]:
            if id1 not in clouds or id2 not in clouds:
                continue
            fwd = nn_stats(clouds[id1], clouds[id2], T, rng)
            inv = nn_stats(clouds[id1], clouds[id2], np.linalg.inv(T), rng)
            rows.append(dict(id1=id1, id2=id2, n_gt=n_gt, forward=fwd, inverse=inv,
                             direction="i->j" if fwd["median"] < inv["median"] else "j->i"))
        med_f = float(np.median([r["forward"]["median"] for r in rows]))
        med_i = float(np.median([r["inverse"]["median"] for r in rows]))
        dirs = [r["direction"] for r in rows]
        report["scenes"][scene] = dict(
            n_fragments=len(clouds), n_pairs=len(pairs),
            forward_median=med_f, inverse_median=med_i,
            frac_pairs_j_to_i=float(np.mean([d == "j->i" for d in dirs])),
            rows=rows,
        )
        print(f"{scene}: fragments={len(clouds)} pairs={len(pairs)} "
              f"forward median NN={med_f:.4f} m, inverse={med_i:.4f} m, "
              f"j->i on {100 * np.mean([d == 'j->i' for d in dirs]):.0f}% of pairs",
              flush=True)
    (OUT / "verify_gt_direction_full.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "verify_gt_direction_full.json")


if __name__ == "__main__":
    main()
