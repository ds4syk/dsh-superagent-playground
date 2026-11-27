#!/usr/bin/env python3
"""3DMatch real descriptor front-end audit (no GT leakage in matching).

This is the honest "real front-end" companion to ``run_threedmatch.py``.
It computes Open3D FPFH descriptors on the voxel-downsampled fragments,
matches them by mutual nearest neighbour, keeps the top descriptor
candidates, applies a pairwise-distance consistency filter, and finally
keeps the best N by graph degree.  Ground truth is used ONLY to evaluate
the resulting matches and pose, never to build them (except for pair
selection, which is the standard benchmark protocol).

The purpose is not to claim FCGF-level matching.  It answers: if the
front-end is a plain hand-crafted FPFH matcher with a realistic natural
inlier ratio, can the robust layer still recover the pose?  The answer is
reported per scene and compared with the oracle GT-NN table.

Requires the isolated Open3D venv:
  /mnt/storage1/syk/venvs/open3d/bin/python run_descriptor.py

Run (Slurm):
  srun --partition=cpu --time=00:40:00 --ntasks=1 --cpus-per-task=2 \
    /mnt/storage1/syk/venvs/open3d/bin/python run_descriptor.py 2>&1 | \
    tee results/logs/descriptor_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SYN))
import load_3dmatch as L  # noqa: E402
import common as cm  # noqa: E402

ROOT = Path("/mnt/storage1/syk/data/3dmatch_fragments")
CACHE = ROOT / "_cache" / "fpfh"
OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SCENES = ["7-scenes-redkitchen", "sun3d-home_at-home_at_scan1_2013_jan_1"]
SCENE_SHORT = {"7-scenes-redkitchen": "redkitchen",
               "sun3d-home_at-home_at_scan1_2013_jan_1": "sun3d-home_at"}
N = 300
C = 0.05
TAU = 0.06
VOXEL = 0.03
FPFH_RADIUS = 0.20
N_PROBE = 1500
N_TRIALS = 10
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]


def project_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def fpfh(points, radius=FPFH_RADIUS):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * radius,
                                                             max_nn=50))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    F = np.asarray(feat.data).T.astype(np.float32)
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    return F


def get_fpfh(scene, bid, cloud):
    path = CACHE / f"{scene}_bin{bid}_r{FPFH_RADIUS:.2f}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    F = fpfh(cloud)
    np.save(path, F)
    print(f"  FPFH {scene}/{bid}: {len(cloud)} pts in {time.time()-t0:.1f}s", flush=True)
    return F


def descriptor_pairs(clouds, id1, id2, rng, n=N):
    """Return real FPFH matches after mutual NN + geometric consistency."""
    A = get_fpfh(SCENES_CURRENT, id1, clouds[id1])
    B = get_fpfh(SCENES_CURRENT, id2, clouds[id2])
    # Mutual nearest neighbours in descriptor space.
    sim = A @ B.T
    j = np.argmax(sim, axis=1)
    i_best = np.argmax(sim, axis=0)
    i = np.flatnonzero(i_best[j] == np.arange(len(j)))
    j = j[i]
    if len(i) < 6:
        return None
    s = sim[i, j]
    # Keep a descriptor-similarity shortlist, then filter by geometric
    # consistency (pairwise distances must be preserved).  This is the
    # standard "graph consistency" trick; no ground truth is used.
    m = min(N_PROBE, len(i))
    take = np.argsort(s)[-m:]
    i, j = i[take], j[take]
    x_all, y_all = clouds[id1][i], clouds[id2][j]
    Dx = np.linalg.norm(x_all[:, None, :] - x_all[None, :, :], axis=2)
    Dy = np.linalg.norm(y_all[:, None, :] - y_all[None, :, :], axis=2)
    comp = np.abs(Dx - Dy) < 0.08
    np.fill_diagonal(comp, False)
    deg = comp.sum(1)
    top = np.argsort(deg)[-min(3 * n, len(deg)):]
    # Randomise among near-best candidates so trials are not identical.
    top = top[np.argsort(deg[top] + rng.normal(scale=0.05 * max(deg[top].max(), 1), size=len(top)))]
    chosen = top[-n:]
    x, y = x_all[chosen], y_all[chosen]
    err = np.linalg.norm((project_rotation(TGT[0]) @ x.T).T + TGT[1] - y, axis=1)
    return x, y, err, i[chosen], j[chosen]


def make_methods():
    def kabsch_m(x, y, rng):
        return (*cm.kabsch(x, y, np.ones(len(x))), {})

    def ransac_fixed(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=cm.RANSAC_ITERS,
                               adaptive=False, return_info=True)

    def ransac_adapt(x, y, rng):
        return cm.solve_ransac(x, y, rng, c=C, iters=cm.RANSAC_ITERS,
                               adaptive=True, return_info=True)

    def irls(x, y, rng):
        return (*cm.solve_irls(x, y, c=C), {})

    def gnc(x, y, rng):
        return (*cm.solve_gnc(x, y, c=C), {})

    return {"Kabsch": kabsch_m, "RANSAC-1000": ransac_fixed,
            "RANSAC-adaptive": ransac_adapt, "IRLS-GM": irls, "GNC-GM": gnc}


def main():
    global SCENES_CURRENT, TGT
    methods = make_methods()
    all_rows, all_raw = [], []
    for scene in SCENES:
        SCENES_CURRENT = scene
        scene_dir = ROOT / scene
        gt_path = ROOT / f"{scene}-evaluation" / "gt.log"
        clouds = L.load_scene(scene_dir, voxel=VOXEL)
        gt_pairs = L.parse_gt_log(gt_path)
        # Use the same high-overlap pair selection as the oracle table.
        chosen, _ = __import__("run_threedmatch").select_pairs(clouds, gt_pairs)
        chosen = chosen[:6]
        print(f"[{scene}] descriptor audit on {len(chosen)} pairs", flush=True)
        for pair in chosen:
            Ti = pair["Tinv"]
            TGT = (project_rotation(Ti[:3, :3]), Ti[:3, 3])
            for trial in range(N_TRIALS):
                rng = np.random.default_rng([20261127, 7, SCENES.index(scene),
                                             int(pair["id1"]), trial])
                got = descriptor_pairs(clouds, pair["id1"], pair["id2"], rng)
                if got is None:
                    continue
                x, y, err, si, sj = got
                inlier = float(np.mean(err < C))
                for name in METHODS:
                    rrng = np.random.default_rng([20261127, 8, SCENES.index(scene),
                                                  int(pair["id1"]), trial])
                    t0 = time.perf_counter()
                    R, t, info = methods[name](x, y, rrng)
                    dt = time.perf_counter() - t0
                    re, te = cm.pose_errors(R, t, *TGT)
                    rec = dict(scene=scene, pair=f"{pair['id1']}-{pair['id2']}",
                               trial=trial, method=name, n=len(x),
                               natural_inlier=inlier, success=cm.success(re, te),
                               rot=re, trans=te, time_ms=1000.0 * dt,
                               ransac_iters=info.get("n_iters"),
                               ransac_inliers=info.get("best_count"))
                    all_raw.append(rec)
        # aggregate over pairs/trials
        for name in METHODS:
            recs = [r for r in all_raw if r["scene"] == scene and r["method"] == name]
            re = np.array([r["rot"] for r in recs])
            te = np.array([r["trans"] for r in recs])
            tt = np.array([r["time_ms"] for r in recs])
            all_rows.append(dict(
                scene=scene, method=name, n=len(recs),
                mean_natural_inlier=float(np.mean([r["natural_inlier"] for r in recs])),
                success=float(np.mean([r["success"] for r in recs])),
                rot_median=float(np.median(re)),
                rot_p90=float(np.percentile(re, 90)),
                trans_median=float(np.median(te)),
                time_ms_p50=float(np.median(tt)),
                time_ms_p95=float(np.percentile(tt, 95)),
            ))
        print(f"[{scene}] done", flush=True)

    cm.json_dump(OUT / "descriptor.json", dict(
        config=dict(scenes=SCENES, n=N, c=C, tau=TAU, voxel=VOXEL,
                    fpfh_radius=FPFH_RADIUS, n_probe=N_PROBE,
                    n_trials=N_TRIALS, methods=METHODS,
                    note="FPFH mutual NN + pairwise-distance consistency; GT only for eval"),
        rows=all_rows))
    with open(OUT / "descriptor_raw.jsonl", "w") as f:
        for r in all_raw:
            f.write(json.dumps(r) + "\n")
    # Markdown
    Ls = ["# 3DMatch real FPFH front-end audit", "",
          "Open3D FPFH mutual-NN matches + pairwise-distance consistency, no GT in matching. "
          f"N={N}, c={C} m, {N_TRIALS} trials/pair, 6 high-overlap pairs/scene. "
          "GT is used only for evaluation.", "",
          "| scene | method | mean natural inlier | success | median rot (deg) | p90 rot | median trans (m) | p50 (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        Ls.append(f"| {SCENE_SHORT[r['scene']]} | {r['method']} | {100*r['mean_natural_inlier']:.1f}% | "
                  f"{100*r['success']:.0f}% | {r['rot_median']:.2f} | {r['rot_p90']:.2f} | "
                  f"{r['trans_median']:.4f} | {r['time_ms_p50']:.2f} |")
    Ls += ["", "## Caveats", "",
           "* This is a hand-crafted FPFH front-end, not FCGF/Predator; it is deliberately "
           "weak and is reported as a reality check, not as a learned-descriptor result.",
           "* The pairwise-distance consistency filter uses no GT; it does add a strong "
           "geometric prior, so the natural inlier fraction is higher than raw FPFH.",
           "* The oracle GT-NN table (`threedmatch.md`) is the controlled outlier experiment; "
           "this table shows what happens when a real front-end supplies the matches."]
    (OUT / "descriptor.md").write_text("\n".join(Ls) + "\n")
    print("wrote descriptor.json/md")


if __name__ == "__main__":
    main()
