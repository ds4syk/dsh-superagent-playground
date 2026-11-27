#!/usr/bin/env python3
"""In-The-Dark driver: same front-end + robust layer as the KITTI fallback.

NOT yet run on real In-The-Dark data: the archives are >= 590 MB and the FTP
link measured ~14 KB/s (README.md section 1), so this driver is *ready but
unexercised*.  It shares ``stereo_vo`` and the robust-layer code of
``../kitti_stereo/run_kitti_stereo.py``; only the dataset adapter differs.

Usage:
  $PY run_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 \
      --pairs 30 --out results/inthedark.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "kitti_stereo"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KS))
sys.path.insert(0, str(HERE.parents[1] / "synthetic"))
import common as cm  # noqa: E402
import load_inthedark as L  # noqa: E402
import run_kitti_stereo as RK  # noqa: E402
import stereo_vo as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--detector", default="sift")
    ap.add_argument("--out", default=str(HERE / "results" / "inthedark.json"))
    a = ap.parse_args()
    d = L.run_dir(a.run)
    tf = next(d.glob("transforms_*.txt"), None)
    ts = L.read_timestamps(d / "timestamps_images.txt")
    if tf is None:
        raise SystemExit(f"BLOCKED: no transforms_*.txt in {d}")
    T = L.read_transforms(tf)
    run_no = int(d.name.split("_")[-1])
    P_L, P_R = L.projection_matrices()
    verts = sorted({int(k) for k in ts})
    pairs = list(zip(verts, verts[1:]))[: a.pairs]
    records, rows = [], []
    for va, vb in pairs:
        T_gt = L.relative_pose(T, run_no, va, run_no, vb)
        fa = L.load_frame(d, va, P_L, P_R)
        fb = L.load_frame(d, vb, P_L, P_R)
        res = S.correspondences_dense(fa, fb, kind=a.detector, nfeatures=4000, z_max=60.0)
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
                   gt_rot_deg=float(cm.rot_err_deg(T_gt[:3, :3], np.eye(3))),
                   gt_trans=float(np.linalg.norm(T_gt[:3, 3])), floor=floor,
                   methods={m: dict(**RK.evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                    time_ms=v["time_ms"]) for m, v in methods.items()})
        rows.append(row)
        for m, v in methods.items():
            records.append(dict(pair=row["pair"], n=row["n"], inlier_rate=row["inlier_rate"],
                                method=m, c=a.c,
                                **RK.evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                time_ms=v["time_ms"]))
        g = row["methods"]["GNC-GM"]
        print(f"{va}->{vb} n={len(x)} inl={inl.mean():.1%} "
              f"GNC rot={g['rot_deg']:.3f} trans={g['trans_m']:.3f}", flush=True)
    out = dict(dataset="UTIAS-In-The-Dark", run=str(d), c=a.c, detector=a.detector,
               pairs=rows, records=records)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
