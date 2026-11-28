#!/usr/bin/env python3
"""In-The-Dark driver: same front-end + robust layer as the KITTI fallback.

GT protocol (established from the real archives, see README section 1):
repeat-run ``transforms_spatial.txt`` (repeat vertex -> teach vertex, relative
pose) + teach-run ``transforms_temporal.txt`` (consecutive teach links) compose
into per-pair poses; ``transform_camera_vehicle.txt`` conjugates them from the
vehicle frame into the stereo camera frame.  See ``compose_inthedark.py`` and
``verify_inthedark.py``.

Front-end (label): hand-written SIFT + StereoSGBM dense disparity (default) or
sparse left-right triangulation -- the *same* code as
``../kitti_stereo/stereo_vo.py``.  Not the SDPR VGG/UNet front-end.

Usage:
  $PY run_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 \
      --start 200 --pairs 15 --fe dense --out results/inthedark_dense.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "kitti_stereo"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KS))
sys.path.insert(0, str(HERE.parents[1] / "synthetic"))
import common as cm  # noqa: E402
import compose_inthedark as C  # noqa: E402
import load_inthedark as L  # noqa: E402
import run_kitti_stereo as RK  # noqa: E402
import stereo_vo as S  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/utias_in_the_dark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teach", default=str(DATA / "teach_meta"))
    ap.add_argument("--data-root", default=str(DATA))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--detector", default="sift")
    ap.add_argument("--fe", choices=["dense", "sparse"], default="dense")
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--out", default=str(HERE / "results" / "inthedark.json"))
    a = ap.parse_args()
    d = L.run_dir(a.run)
    pairs, skipped, meta = C.gt_pairs(d, a.teach, a.data_root)
    P_L, P_R = L.projection_matrices()
    order = sorted(pairs)
    sel = order[a.start:a.start + a.pairs]
    if not sel:
        raise SystemExit("BLOCKED: no composed GT pairs in this window")
    records, rows = [], []
    t_start = time.perf_counter()
    for va, vb in sel:
        T_gt = pairs[(va, vb)]
        fa = L.load_frame(d, va, P_L, P_R)
        fb = L.load_frame(d, vb, P_L, P_R)
        t0 = time.perf_counter()
        if a.fe == "dense":
            res = S.correspondences_dense(fa, fb, kind=a.detector,
                                          nfeatures=a.nfeatures, z_max=60.0)
        else:
            res = S.correspondences(fa, fb, kind=a.detector,
                                    nfeatures=a.nfeatures, z_max=60.0)
        fe_ms = 1e3 * (time.perf_counter() - t0)
        x, y = res["x"], res["y"]
        if len(x) < 10:
            rows.append(dict(pair=f"{va}->{vb}", n=int(len(x)), skipped=True))
            continue
        e = np.linalg.norm((T_gt[:3, :3] @ x.T).T + T_gt[:3, 3] - y, axis=1)
        inl = e < a.c
        methods = RK.run_methods(x, y, c=a.c)
        floor = None
        if inl.sum() >= 6:
            Rf, tf_ = cm.kabsch(x[inl], y[inl], np.ones(int(inl.sum())))
            floor = RK.evaluate(Rf, tf_, T_gt[:3, :3], T_gt[:3, 3])
        row = dict(pair=f"{va}->{vb}", n=int(len(x)), inlier_rate=float(inl.mean()),
                   med_resid_inlier=float(np.median(e[inl])) if inl.any() else None,
                   med_resid_outlier=float(np.median(e[~inl])) if (~inl).any() else None,
                   med_depth_m=float(np.median(x[:, 2])),
                   gt_rot_deg=float(np.degrees(np.arccos(np.clip(
                       (np.trace(T_gt[:3, :3]) - 1) / 2, -1, 1)))),
                   gt_trans=float(np.linalg.norm(T_gt[:3, 3])), floor=floor,
                   fe_ms=fe_ms,
                   methods={m: dict(**RK.evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                    time_ms=v["time_ms"]) for m, v in methods.items()})
        rows.append(row)
        for m, v in methods.items():
            records.append(dict(pair=row["pair"], n=row["n"], inlier_rate=row["inlier_rate"],
                                method=m, c=a.c, fe=a.fe,
                                **RK.evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                time_ms=v["time_ms"]))
        g = row["methods"]["GNC-GM"]
        print(f"{va}->{vb} n={row['n']} inl={inl.mean():.1%} depth={row['med_depth_m']:.1f}m "
              f"GNC rot={g['rot_deg']:.3f} trans={g['trans_m']:.3f} fe={fe_ms:.0f}ms",
              flush=True)
    wall = time.perf_counter() - t_start
    out = dict(dataset="UTIAS-In-The-Dark", run=str(d), teach=str(a.teach),
               protocol=meta, c=a.c, detector=a.detector, fe=a.fe,
               front_end="hand-written SIFT + StereoSGBM/sparse stereo (stereo_vo.py)",
               pairs=rows, records=records, wall_s=wall, n_pairs=len(sel),
               skipped_pairs=len(skipped))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out} ({len(rows)} pairs, {wall:.1f}s wall)")


if __name__ == "__main__":
    main()
