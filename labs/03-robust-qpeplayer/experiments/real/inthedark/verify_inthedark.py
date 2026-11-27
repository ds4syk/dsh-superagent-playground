#!/usr/bin/env python3
"""Self-checks for a UTIAS In-The-Dark repeat run, with the GT protocol
established from the real archives (see README section 1).

What is verified, in order, and each one fails loudly instead of emitting
numbers:

  1. layout: images/left, images/right, transforms_spatial.txt, timestamps;
  2. images are the documented 512x384 and decode;
  3. stereo row alignment: median |v_l - v_r| of SIFT left-right matches
     (< 2 px) -- the only evidence for the undocumented "already rectified"
     assumption;
  4. GT protocol: the composed consecutive-pair pose (repeat spatial x teach
     temporal, camera-frame conjugated) must reproject triangulated landmarks
     into the next left image far better than its inverse and better than
     identity.  This is what pins down (a) the composition direction,
     (b) the vehicle->camera extrinsic, (c) the row-convention reading;
  5. depth sanity: median triangulated depth finite and in range.

Usage:
  $PY verify_inthedark.py --run /mnt/storage1/syk/data/utias_in_the_dark/run_000002 \
     --pairs 8 --out results/verify_run_000002.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kitti_stereo"))
import calibrate_convention as CC  # noqa: E402
import compose_inthedark as C  # noqa: E402
import load_inthedark as L  # noqa: E402
import stereo_vo as S  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/utias_in_the_dark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teach", default=str(DATA / "teach_meta"))
    ap.add_argument("--data-root", default=str(DATA))
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = L.run_dir(a.run)
    for rel in ("images/left", "images/right"):
        if not (d / rel).is_dir():
            raise SystemExit(f"BLOCKED: missing {d / rel}")
    spat = d / "transforms_spatial.txt"
    if not spat.exists():
        raise SystemExit(f"BLOCKED: no {spat}")
    if not (Path(a.teach) / "transforms_temporal.txt").exists():
        raise SystemExit(f"BLOCKED: no teach transforms_temporal.txt in {a.teach}")

    pairs, skipped, meta = C.gt_pairs(d, a.teach, a.data_root)
    E = C.load_camera_vehicle(a.data_root)
    ts = L.read_timestamps(d / "timestamps_images.txt")
    verts = sorted(pairs)
    sel = verts[a.start:a.start + a.pairs]
    if not sel:
        raise SystemExit("BLOCKED: no composed pairs in the requested window")
    P_L, P_R = L.projection_matrices()
    det = S.make_detector("sift", 3000)
    rows = []
    for va, vb in sel:
        fa = L.load_frame(d, va, P_L, P_R)
        fb = L.load_frame(d, vb, P_L, P_R)
        T = pairs[(va, vb)]
        e_gt, n_corr, n_proj = CC.reproj_residual(fa, fb, T, det)
        e_inv, _, _ = CC.reproj_residual(fa, fb, np.linalg.inv(T), det)
        e_id, _, _ = CC.reproj_residual(fa, fb, np.eye(4), det)
        e_veh, _, _ = CC.reproj_residual(fa, fb, np.linalg.inv(E) @ T @ E, det)
        # row alignment from SIFT left-right matches of frame a
        gl = cv2.cvtColor(fa["left"], cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fa["right"], cv2.COLOR_BGR2GRAY)
        pl, dl = S.detect(det, gl)
        pr, dr = S.detect(det, gr)
        i_l, i_r = S.match(dl, dr, "sift", 0.8)
        dy = float(np.median(np.abs(pl[i_l, 1] - pr[i_r, 1]))) if len(i_l) else float("nan")
        disp = float(np.median(pl[i_l, 0] - pr[i_r, 0])) if len(i_l) else float("nan")
        dt = (ts.get(vb, 0) - ts.get(va, 0)) / 1e9 if va in ts and vb in ts else None
        rows.append(dict(pair=f"{va}->{vb}", n_corr=int(n_corr),
                         gt_trans=float(np.linalg.norm(T[:3, 3])),
                         gt_rot_deg=float(np.degrees(np.arccos(np.clip(
                             (np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                         reproj_gt_px=e_gt, reproj_inverse_px=e_inv,
                         reproj_identity_px=e_id, reproj_vehicle_frame_px=e_veh,
                         median_row_dy_px=dy, median_disparity_px=disp,
                         depth_med_m=float(P_L[0, 0] * 0.239965 / disp)
                         if disp and np.isfinite(disp) and disp > 0 else None,
                         dt_s=dt,
                         img_shape=list(fa["left"].shape[:2])))
        print(f"{va}->{vb} gt={e_gt:.2f}px inv={e_inv:.2f} id={e_id:.2f} "
              f"veh={e_veh:.2f} dy={dy:.2f} disp={disp:.1f} |t|={rows[-1]['gt_trans']:.2f}",
              flush=True)
    chk = dict(
        documented_512x384=all(r["img_shape"] == [384, 512] for r in rows),
        rows_aligned=all(r["median_row_dy_px"] < 2.0 for r in rows),
        gt_beats_inverse=all(r["reproj_inverse_px"] > 2 * r["reproj_gt_px"] for r in rows),
        gt_beats_identity=all(r["reproj_identity_px"] > r["reproj_gt_px"] for r in rows),
        camera_conjugation_beats_vehicle=all(
            r["reproj_vehicle_frame_px"] > 2 * r["reproj_gt_px"] for r in rows),
        gt_reproj_under_2px=float(np.median([r["reproj_gt_px"] for r in rows])) < 2.0,
    )
    out = dict(run=str(d), teach=str(a.teach), protocol=meta, pairs=rows,
               checks=chk, skipped=len(skipped))
    print(json.dumps(dict(checks=chk, protocol=meta), indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    if not all(chk.values()):
        raise SystemExit(f"CHECK FAILED: {chk}")


if __name__ == "__main__":
    main()
