#!/usr/bin/env python3
"""Verify the direction of ``gt.log`` transforms on the two extra 3DMatch scenes.

The parent pipeline established empirically (redkitchen, sun3d-home_at) that
the 4x4 in ``gt.log`` maps fragment ``id2`` into the frame of ``id1``; the
runner therefore inverts it.  Direction is a per-file convention, so this
script repeats the check on the two *additional* scenes before they enter the
robust-registration table.

For the first 12 pairs of each scene we map sampled source points with both
the stored transform and its inverse and report nearest-neighbour distance to
the target cloud.  The correct direction yields a much smaller median.

Run (Slurm):
  srun --partition=cpu --time=00:15:00 --ntasks=1 --cpus-per-task=4 \
    /home/syk/miniconda3/envs/WM/bin/python verify_gt_direction_extra.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_3dmatch as L  # noqa: E402

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = ["sun3d-hotel_uc-scan3", "sun3d-home_md-home_md_scan9_2012_sep_30"]
N_SAMPLE = 3000
TAU = 0.06


def nn_stats(src, dst, T, n=N_SAMPLE, seed=0, tau=TAU):
    rng = np.random.default_rng(seed)
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
        rows = []
        for id1, id2, n_gt, T in pairs[:12]:
            if id1 not in clouds or id2 not in clouds:
                continue
            fwd = nn_stats(clouds[id1], clouds[id2], T)
            inv = nn_stats(clouds[id1], clouds[id2], np.linalg.inv(T))
            rows.append(dict(id1=id1, id2=id2, n_gt=n_gt,
                             forward=fwd, inverse=inv,
                             direction="i->j" if fwd["median"] < inv["median"] else "j->i"))
        med_f = float(np.median([r["forward"]["median"] for r in rows]))
        med_i = float(np.median([r["inverse"]["median"] for r in rows]))
        report["scenes"][scene] = dict(n_fragments=len(clouds), n_pairs=len(pairs),
                                       forward_median=med_f, inverse_median=med_i,
                                       rows=rows)
        print(f"{scene}: fragments={len(clouds)} pairs={len(pairs)} "
              f"forward median NN={med_f:.4f} m, inverse={med_i:.4f} m, "
              f"frac<{TAU} (inverse)={np.mean([r['inverse']['frac_lt_tau'] for r in rows]):.3f}",
              flush=True)
    (OUT / "verify_gt_direction_extra.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "verify_gt_direction_extra.json")


if __name__ == "__main__":
    main()
