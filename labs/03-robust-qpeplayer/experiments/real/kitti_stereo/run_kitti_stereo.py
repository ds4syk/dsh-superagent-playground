#!/usr/bin/env python3
"""P3 fallback: real KITTI raw stereo -> 3D-3D correspondences -> robust layer.

Why this fallback: every official UTIAS In-The-Dark archive is >= 590 MB and
the direct FTP link measured ~14 KB/s (see ``experiments/real/inthedark/
README.md``), so no run could be downloaded inside the session budget.  KITTI
raw drive 0001 is already on the data disk and provides what this experiment
needs: real rectified stereo images, real calibration, real ego-motion ground
truth, and a real hand-crafted visual front-end with genuine mismatch
outliers -- **not** GT-sampled correspondences and **not** the SDPR VGG/UNet
front-end.

Pipeline
--------
1. Rectified stereo pair (image_02 / image_03; KITTI already rectified):
   SIFT or ORB keypoints, sub-pixel corner refinement.
   * ``dense``  : 3D per keypoint from StereoSGBM dense disparity,
   * ``sparse`` : left-right keypoint matching + analytic triangulation.
   No GT is used in matching or triangulation.
2. Temporal link: match left-image descriptors of frame i and frame j; keep
   pairs where *both* endpoints have a valid 3D point -> (x_i, y_j).
3. GT label: e = || T_{j<-i} x_i - y_j ||, GT camera-02 rectified relative
   pose from OXTS + calib_velo_to_cam + R_rect_02.  Inlier iff e < c_label.
4. Robust outer layer (Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM /
   GNC-GM) on exactly those real correspondences: natural outlier rate, no
   injected outliers, no trials (one deterministic instance per frame pair).
   A GT-inlier-only Kabsch ("floor") bounds what any robust method can reach.

Run:
  srun --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=2 \
    /mnt/storage1/syk/venvs/opencv/bin/python run_kitti_stereo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
SYN = HERE.parents[1] / "synthetic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SYN))
sys.path.insert(0, str(HERE.parent / "kitti"))
import common as cm  # noqa: E402
import stereo_vo as S  # noqa: E402
import load_kitti as K  # noqa: E402

KITTI = Path("/mnt/storage1/syk/data/kitti_raw/2011_09_26")
DRIVE = KITTI / "2011_09_26_drive_0001_sync"
OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

C_ROBUST = 0.5         # m, default robust threshold
Z_MAX = 60.0           # m, stereo range cap
SUCCESS_ROT = 5.0      # deg
SUCCESS_TRANS = 0.1    # m, primary criterion
SUCCESS_TRANS_LOOSE = 0.25  # m, justified relaxed criterion (11% of 2.2 m motion)
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
PAIRS = ([("adj", i, i + 1) for i in range(0, 60, 3)] +
         [("gap3", i, i + 3) for i in range(0, 30, 3)])
# (front-end, detector, label/robust threshold)
CONFIGS = [("sparse", "sift", 0.5), ("dense", "sift", 0.5),
           ("dense", "orb", 0.5), ("dense", "sift", 0.3),
           ("dense", "sift", 1.0)]


# ---------------------------------------------------------------------------
# Calibration / GT poses
# ---------------------------------------------------------------------------
def read_cam_calib(path=KITTI / "calib_cam_to_cam.txt"):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        key, val = line.split(":", 1)
        if key in ("calib_time", "corner_dist"):
            continue
        out[key.strip()] = np.array([float(x) for x in val.split()])
    return out


def rect_cam_poses():
    """World pose of the rectified camera-02 frame for every frame."""
    cam = read_cam_calib()
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam["R_rect_02"].reshape(3, 3)
    T_cam0_velo = K.read_calib_rigid(KITTI / "calib_velo_to_cam.txt")
    T_velo_imu = K.read_calib_rigid(KITTI / "calib_imu_to_velo.txt")
    T_cam0_imu = T_cam0_velo @ T_velo_imu
    T_rect_imu = R_rect @ T_cam0_imu
    T_imu_rect = np.linalg.inv(T_rect_imu)
    oxts = K.load_oxts_poses()
    poses = [T_w @ T_imu_rect for T_w in oxts]
    return poses, cam["P_rect_02"].reshape(3, 4), cam["P_rect_03"].reshape(3, 4)


def load_frame(idx, P_L, P_R):
    left = cv2.imread(str(DRIVE / "image_02" / "data" / f"{idx:010d}.png"))
    right = cv2.imread(str(DRIVE / "image_03" / "data" / f"{idx:010d}.png"))
    if left is None or right is None:
        raise FileNotFoundError(f"missing stereo frame {idx}")
    return {"left": left, "right": right, "P_L": P_L, "P_R": P_R}


def build_pair(idx_i, idx_j, P_L, P_R, fe, kind, nfeatures=4000):
    fa = load_frame(idx_i, P_L, P_R)
    fb = load_frame(idx_j, P_L, P_R)
    t0 = time.perf_counter()
    if fe == "dense":
        res = S.correspondences_dense(fa, fb, kind=kind, nfeatures=nfeatures,
                                      temporal_ratio=0.8, z_max=Z_MAX)
    else:
        res = S.correspondences(fa, fb, kind=kind, nfeatures=nfeatures,
                                stereo_ratio=0.8, temporal_ratio=0.8,
                                z_max=Z_MAX, min_disp=1.0, dy_max=2.0)
    return res, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Robust layer
# ---------------------------------------------------------------------------
def run_methods(x, y, c=C_ROBUST, seed=20261127):
    rng = np.random.default_rng(seed)
    out = {}
    t0 = time.perf_counter()
    R0, t0_ = cm.kabsch(x, y, np.ones(len(x)))
    out["Kabsch"] = dict(R=R0, t=t0_, time_ms=1e3 * (time.perf_counter() - t0), extra={})
    for name, fn in (
        ("RANSAC-1000", lambda: cm.solve_ransac(x, y, rng, c=c, iters=1000,
                                                adaptive=False, return_info=True)),
        ("RANSAC-adaptive", lambda: cm.solve_ransac(x, y, rng, c=c, iters=1000,
                                                    adaptive=True, return_info=True)),
        ("IRLS-GM", lambda: cm.solve_irls(x, y, c=c)),
        ("GNC-GM", lambda: cm.solve_gnc(x, y, c=c)),
    ):
        ts = time.perf_counter()
        r = fn()
        dt = 1e3 * (time.perf_counter() - ts)
        if name.startswith("RANSAC"):
            R, t, info = r
            out[name] = dict(R=R, t=t, time_ms=dt, extra=info)
        else:
            R, t = r
            out[name] = dict(R=R, t=t, time_ms=dt, extra={})
    return out


def evaluate(R, t, Rgt, tgt):
    re, te = cm.pose_errors(R, t, Rgt, tgt)
    return dict(rot_deg=float(re), trans_m=float(te),
                success=bool(re < SUCCESS_ROT and te < SUCCESS_TRANS),
                success_loose=bool(re < SUCCESS_ROT and te < SUCCESS_TRANS_LOOSE))


def main(configs=CONFIGS, pairs=list(PAIRS), write=True, tag="main"):
    poses, P_L, P_R = rect_cam_poses()
    records, pair_rows = [], []
    corr_cache = {}
    for fe, kind, c in configs:
        for name, i, j in pairs:
            key = (fe, kind, i, j)
            if key not in corr_cache:
                corr_cache[key] = build_pair(i, j, P_L, P_R, fe, kind)
            res, dt_fe = corr_cache[key]
            x, y = res["x"], res["y"]
            T_gt = np.linalg.inv(poses[j]) @ poses[i]
            if len(x) < 10:
                pair_rows.append(dict(frontend=fe, detector=kind, c_label=c,
                                      pair=f"{name}:{i}->{j}", kind=name, i=i, j=j,
                                      n=len(x), skipped=True, frontend_time_s=dt_fe))
                continue
            e = np.linalg.norm((T_gt[:3, :3] @ x.T).T + T_gt[:3, 3] - y, axis=1)
            T_inv = np.linalg.inv(T_gt)
            e_inv = np.linalg.norm((T_inv[:3, :3] @ x.T).T + T_inv[:3, 3] - y, axis=1)
            inl = e < c
            methods = run_methods(x, y, c=c)
            if inl.sum() >= 6:
                Rf, tf = cm.kabsch(x[inl], y[inl], np.ones(int(inl.sum())))
                floor = evaluate(Rf, tf, T_gt[:3, :3], T_gt[:3, 3])
            else:
                floor = dict(rot_deg=float("nan"), trans_m=float("nan"),
                             success=False, success_loose=False)
            row = dict(frontend=fe, detector=kind, c_label=c, pair=f"{name}:{i}->{j}",
                       kind=name, i=i, j=j, n_kp_L=res["n_kp_L"],
                       n_stereo=res["n_stereo"], n_temporal=int(res["n_temporal"]),
                       n=len(x), inlier_rate=float(inl.mean()),
                       n_inlier=int(inl.sum()),
                       med_resid_inlier=float(np.median(e[inl])) if inl.any() else None,
                       med_resid_outlier=float(np.median(e[~inl])) if (~inl).any() else None,
                       med_resid_all=float(np.median(e)),
                       med_resid_inverse=float(np.median(e_inv)),
                       med_depth_inlier=float(np.median(x[inl, 2])) if inl.any() else None,
                       med_depth_all=float(np.median(x[:, 2])),
                       gt_trans=float(np.linalg.norm(T_gt[:3, 3])),
                       gt_rot_deg=float(cm.rot_err_deg(T_gt[:3, :3], np.eye(3))),
                       frontend_time_s=dt_fe, floor_inlier_kabsch=floor,
                       methods={m: dict(**evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                        time_ms=v["time_ms"], extra=v["extra"])
                                for m, v in methods.items()})
            pair_rows.append(row)
            for m, v in methods.items():
                records.append(dict(frontend=fe, detector=kind, c_label=c,
                                    pair=row["pair"], n=len(x),
                                    inlier_rate=float(inl.mean()), method=m, c=c,
                                    **evaluate(v["R"], v["t"], T_gt[:3, :3], T_gt[:3, 3]),
                                    time_ms=v["time_ms"]))
            g = evaluate(methods["GNC-GM"]["R"], methods["GNC-GM"]["t"],
                         T_gt[:3, :3], T_gt[:3, 3])
            print(f"[{fe}/{kind}/c={c}] {row['pair']:<12} n={len(x):4d} "
                  f"inl={inl.mean():5.0%} d={row['med_depth_all']:4.1f}m "
                  f"GNC rot={g['rot_deg']:.3f} trans={g['trans_m']:.3f} "
                  f"ok={g['success']}/{g['success_loose']} fe={dt_fe:.2f}s", flush=True)
    agg = summarize(pair_rows, records)
    if write:
        cm.json_dump(OUT / f"kitti_stereo_{tag}.json",
                     dict(settings=dict(c_robust_default=C_ROBUST, z_max=Z_MAX,
                                        configs=[list(c) for c in configs],
                                        pairs=[list(p) for p in pairs],
                                        kitti=str(DRIVE),
                                        success=dict(rot_deg=SUCCESS_ROT,
                                                     trans_m=SUCCESS_TRANS,
                                                     trans_loose_m=SUCCESS_TRANS_LOOSE)),
                          pairs=pair_rows, records=records, summary=agg))
        write_md(OUT / f"kitti_stereo_{tag}.md", agg, pair_rows)
    return pair_rows, records, agg


def summarize(pair_rows, records):
    agg = {}
    keys = sorted({(p["frontend"], p["detector"], p["c_label"]) for p in pair_rows})
    for fe, kind, c in keys:
        rows = [p for p in pair_rows if (p["frontend"], p["detector"], p["c_label"]) == (fe, kind, c)]
        ok = [p for p in rows if not p.get("skipped")]
        recs = [r for r in records if (r["frontend"], r["detector"], r["c_label"]) == (fe, kind, c)]
        s = dict(n_pairs=len(rows), n_ok=len(ok),
                 n_corr_p50=float(np.median([p["n"] for p in ok])) if ok else 0,
                 n_corr_min=int(min([p["n"] for p in ok])) if ok else 0,
                 inlier_rate_p50=float(np.median([p["inlier_rate"] for p in ok])) if ok else 0,
                 inlier_rate_min=float(min([p["inlier_rate"] for p in ok])) if ok else 0,
                 med_depth_inlier_p50=float(np.median([p["med_depth_inlier"] for p in ok
                                                       if p["med_depth_inlier"]])) if ok else 0,
                 gt_rot_p50=float(np.median([p["gt_rot_deg"] for p in ok])) if ok else 0,
                 gt_trans_p50=float(np.median([p["gt_trans"] for p in ok])) if ok else 0,
                 invert_ratio_p50=float(np.median([p["med_resid_inverse"] / max(p["med_resid_all"], 1e-9)
                                                   for p in ok])) if ok else 0,
                 fe_time_p50_s=float(np.median([p["frontend_time_s"] for p in ok])) if ok else 0,
                 fe_time_p95_s=float(np.percentile([p["frontend_time_s"] for p in ok], 95)) if ok else 0)
        for m in METHODS:
            re = np.array([r["rot_deg"] for r in recs if r["method"] == m])
            te = np.array([r["trans_m"] for r in recs if r["method"] == m])
            su = np.array([r["success"] for r in recs if r["method"] == m])
            sl = np.array([r["success_loose"] for r in recs if r["method"] == m])
            tm = np.array([r["time_ms"] for r in recs if r["method"] == m])
            s[m] = dict(success=float(su.mean()) if len(su) else float("nan"),
                        success_loose=float(sl.mean()) if len(sl) else float("nan"),
                        med_rot=float(np.median(re)) if len(re) else float("nan"),
                        med_trans=float(np.median(te)) if len(te) else float("nan"),
                        p90_rot=float(np.percentile(re, 90)) if len(re) else float("nan"),
                        p50_ms=float(np.median(tm)) if len(tm) else float("nan"),
                        p95_ms=float(np.percentile(tm, 95)) if len(tm) else float("nan"),
                        max_ms=float(tm.max()) if len(tm) else float("nan"))
        fl = [p["floor_inlier_kabsch"] for p in ok if p["floor_inlier_kabsch"]["rot_deg"] == p["floor_inlier_kabsch"]["rot_deg"]]
        s["floor"] = dict(success=float(np.mean([f["success"] for f in fl])) if fl else float("nan"),
                          success_loose=float(np.mean([f["success_loose"] for f in fl])) if fl else float("nan"),
                          med_rot=float(np.median([f["rot_deg"] for f in fl])) if fl else float("nan"),
                          med_trans=float(np.median([f["trans_m"] for f in fl])) if fl else float("nan"))
        agg[f"{fe}/{kind}/c{c}"] = s
    return agg


def write_md(path, agg, pair_rows):
    L = ["# KITTI raw stereo front-end -> 3D-3D robust layer (fallback, not In-The-Dark)", "",
         "Real hand-crafted stereo front-end (SIFT/ORB + sub-pixel + SGBM or sparse "
         "triangulation), natural mismatch outliers, no injected outliers and no trials "
         "(one deterministic instance per frame pair). `floor` = Kabsch on GT-labeled "
         "inliers only: the front-end noise floor. Success = rot < 5 deg and trans < 0.1 m "
         "(loose: trans < 0.25 m).", ""]
    for key, s in agg.items():
        L += [f"## {key}  ({s['n_ok']}/{s['n_pairs']} pairs)", "",
              f"- correspondences median {s['n_corr_p50']:.0f} (min {s['n_corr_min']}), "
              f"natural inlier rate median {s['inlier_rate_p50']:.1%} (min {s['inlier_rate_min']:.1%}), "
              f"inlier median depth {s['med_depth_inlier_p50']:.1f} m",
              f"- GT motion: rot {s['gt_rot_p50']:.3f} deg, trans {s['gt_trans_p50']:.2f} m; "
              f"inverse-pose residual ratio {s['invert_ratio_p50']:.1f}x",
              f"- front-end time p50 {s['fe_time_p50_s']:.2f} s / p95 {s['fe_time_p95_s']:.2f} s", "",
              "| method | success@0.1 m | success@0.25 m | med rot (deg) | med trans (m) | p50 ms | p95 ms | max ms |",
              "|-|-|-|-|-|-|-|-|"]
        for m in METHODS + ["floor"]:
            d = s[m]
            L.append(f"| {m} | {d['success']:.0%} | {d['success_loose']:.0%} | {d['med_rot']:.3f} "
                     f"| {d['med_trans']:.3f} | {d.get('p50_ms', float('nan')):.2f} "
                     f"| {d.get('p95_ms', float('nan')):.2f} | {d.get('max_ms', float('nan')):.2f} |")
        L.append("")
    L += ["## Per-pair front-end stats (dense/SIFT/c=0.5)", "",
          "| pair | n | inlier rate | med depth (m) | GT trans (m) |", "|-|-|-|-|-|"]
    for p in pair_rows:
        if (p["frontend"], p["detector"], p["c_label"]) != ("dense", "sift", 0.5):
            continue
        if p.get("skipped"):
            L.append(f"| {p['pair']} | {p['n']} | skipped | - | - |")
        else:
            L.append(f"| {p['pair']} | {p['n']} | {p['inlier_rate']:.1%} "
                     f"| {p['med_depth_inlier']:.1f} | {p['gt_trans']:.2f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
