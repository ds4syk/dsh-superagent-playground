#!/usr/bin/env python3
"""KITTI odometry: extend the robust 3D-3D table to sequences 01/02/03/04/06/08/09/10.

Same protocol as ``../kitti_odom/run_kitti_odom.py`` (imported read-only):

* GT-mapped nearest-neighbour correspondences on real LiDAR
  (voxel 0.15 m, range 1-30 m, tau 0.45 m), up to 1000 base pairs, 300 sampled
  per trial;
* random outliers at p in {0, 0.3, 0.5, 0.7};
* five methods (Kabsch / RANSAC-1000 / RANSAC-adaptive / IRLS-GM / GNC-GM);
* c = 0.25 m, success = rot < 5 deg and |t| < 0.1 m, 10 consecutive-frame
  pairs per sequence.

Two deliberate differences from the parent run, both documented:

1. **10 trials/cell instead of 20**, to fit 8 new sequences in the time
   budget. Success is a Monte-Carlo mean; the parent 00/05/07 rows are
   *not* re-run - they are read from ``../kitti_odom/results/kitti_odom.json``
   at the matching p values, so the reader must compare means with different
   trial counts (not paired tests).
2. Random outliers only (the parent's coherent/boundary cells and the motion
   /frame-gap phases are not repeated here).

Data: only the ~33 frames per sequence touched by the protocol were fetched
with per-frame HTTP Range requests from the HuggingFace mirror of the official
``data_odometry_velodyne.zip`` (84.8 GB); see ``fetch_fast.py`` and
``/mnt/storage1/syk/data/kitti_odom/manifest_fast.json``.  This means the
claim is **"all 11 odometry sequences 00-10 are represented, each by the 33
selected frames the protocol uses"** - NOT "the full 00-10 LiDAR streams".

Run:  taskset -c 4-7 /home/syk/miniconda3/envs/WM/bin/python run_kitti_more.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "kitti_odom"
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))

import load_kitti_odom as K  # noqa: E402
import run_kitti_odom as R  # noqa: E402  (protocol functions, main is guarded)
from run_kitti_odom import METHODS, N, C, VOXEL, _seed  # noqa: E402

DATA = K.DATA
OUT = HERE / "results"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

SEQS = ["01", "02", "03", "04", "06", "08", "09", "10"]
BASELINE_SEQS = ["00", "05", "07"]
P_LIST = [0.0, 0.3, 0.5, 0.7]
TRIALS = 10
N_PAIRS = 10


def load_manifest():
    p = DATA / "manifest_fast.json"
    if p.exists():
        return json.loads(p.read_text())
    return json.loads((DATA / "manifest.json").read_text())


def run_sequence(seq, methods):
    t_seq = time.time()
    poses = K.load_poses(seq)
    T_cam_velo = K.read_calib_tr(seq)
    n_frames = len(poses)
    step = max(2, (n_frames - 2) // N_PAIRS)
    pairs = [(k * step, k * step + 1) for k in range(N_PAIRS)]
    idxs = sorted({z for pr in pairs for z in pr})
    clouds = K.load_clouds(seq, idxs, voxel=VOXEL, max_range=30.0)

    bases = []
    for k, pair in enumerate(pairs):
        rng = np.random.default_rng(_seed(70, int(seq), k))
        b = R.prepare_base(clouds, poses, T_cam_velo, pair, rng)
        if len(b["x"]) < N:
            print(f"  [seq {seq}] skip pair {pair}: {len(b['x'])} base pairs", flush=True)
            continue
        b["pair_k"] = k
        bases.append(b)

    rows, pair_rows = [], []
    for b in bases:
        pair_rows.append(dict(pair=f"{b['i']}-{b['j']}", t_m=b["t_m"],
                              rot_deg=b["rot_deg"], n_base=len(b["x"]),
                              natural_inlier=b["natural_inlier"]))
    for p in P_LIST:
        per = {m: dict(re=[], te=[], time=[], base_in=[]) for m in METHODS}
        for trial in range(TRIALS):
            rng = np.random.default_rng(_seed(71, int(seq), 1, int(round(100 * p)), trial))
            b = bases[int(rng.integers(len(bases)))]
            take = rng.choice(len(b["x"]), N, replace=False)
            base = dict(b)
            base["x"], base["y"], base["d"] = b["x"][take], b["y"][take], b["d"][take]
            y, _meta = R.inject(base, rng, p, "random", clouds[b["j"]])
            base_in = float(np.mean(base["d"] < C))
            for name, Rm, t, info, dt in R.fit_all(methods, base["x"], y, p, trial,
                                                   tag=(seq, "random")):
                re, te = R.cm.pose_errors(Rm, t, base["Rgt"], base["tgt"])
                per[name]["re"].append(re)
                per[name]["te"].append(te)
                per[name]["time"].append(dt)
                per[name]["base_in"].append(base_in)
        for m in METHODS:
            a = per[m]
            s = R.cm.summarize_errors(a["re"], a["te"], a["time"])
            rows.append(dict(dataset=f"odom{seq}", p=p, outlier_type="random",
                             method=m, n_trials=len(a["re"]),
                             base_inlier=float(np.mean(a["base_in"])), **s))
        print(f"  [seq {seq}] p={p:.1f} done ({time.time()-t_seq:.0f}s)", flush=True)

    out = dict(n_frames=n_frames, step=step, pairs=pairs, n_clouds=len(clouds),
               pairs_used=pair_rows, rows=rows,
               elapsed_s=round(time.time() - t_seq, 1))
    print(f"[seq {seq}] table done in {out['elapsed_s']}s "
          f"({len(bases)} pairs, {len(clouds)} clouds)", flush=True)
    return out


def baseline():
    path = PARENT / "results" / "kitti_odom.json"
    d = json.loads(path.read_text())
    rows = []
    for seq in BASELINE_SEQS:
        for r in d["per_sequence"][seq]["rows"]:
            if r["outlier_type"] == "random" and any(
                    abs(r["p"] - p) < 1e-9 for p in P_LIST):
                rows.append(dict(r, source="kitti_odom/results/kitti_odom.json"))
    return dict(config=d["config"], rows=rows,
                per_sequence={s: dict(n_frames=d["per_sequence"][s]["n_frames"],
                                      step=d["per_sequence"][s]["step"],
                                      pairs_used=d["per_sequence"][s]["base"])
                              for s in BASELINE_SEQS})


def pooled(rows_by_dataset, datasets, label):
    out = []
    for p in P_LIST:
        for m in METHODS:
            sel = [r for ds in datasets for r in rows_by_dataset[ds]
                   if abs(r["p"] - p) < 1e-9 and r["method"] == m]
            if not sel:
                continue
            s = R.cm.summarize_errors([r["rot_median"] for r in sel],
                                      [r["trans_median"] for r in sel],
                                      [r["time_ms_p50"] / 1000.0 for r in sel])
            # pooled success: trial-count-weighted mean of per-sequence success
            n = np.array([r["n_trials"] for r in sel], float)
            succ = float(np.average([r["success"] for r in sel], weights=n))
            out.append(dict(dataset=label, p=p, outlier_type="random", method=m,
                            n_trials=int(n.sum()),
                            success=succ,
                            rot_median=float(np.median([r["rot_median"] for r in sel])),
                            trans_median=float(np.median(
                                [r["trans_median"] for r in sel])),
                            time_ms_p50=s["time_ms_p50"],
                            n_sequences=len(sel)))
    return out


def main():
    t0 = time.time()
    manifest = load_manifest()
    methods = R.make_methods()
    per_sequence = {}
    for seq in SEQS:
        per_sequence[seq] = run_sequence(seq, methods)

    rows_by_dataset = {f"odom{s}": per_sequence[s]["rows"] for s in SEQS}
    base = baseline()
    for r in base["rows"]:
        rows_by_dataset.setdefault(r["dataset"], []).append(r)

    out = dict(
        config=dict(sequences=SEQS, baseline_sequences=BASELINE_SEQS,
                    n=N, c=C, tau=R.TAU, voxel=VOXEL, p=P_LIST, trials=TRIALS,
                    pairs_per_seq=N_PAIRS, methods=METHODS,
                    success="rot < 5 deg and |t| < 0.1 m",
                    data_root=str(DATA),
                    note="exact odometry GT; 33 selected frames/sequence, "
                         "not the full 00-10 LiDAR streams"),
        protocol_source="../kitti_odom/run_kitti_odom.py (reused via import)",
        per_sequence=per_sequence,
        baseline=base,
        pooled_new=pooled(rows_by_dataset, [f"odom{s}" for s in SEQS], "pooled_new8"),
        pooled_all11=pooled(rows_by_dataset, [f"odom{s}" for s in SEQS + BASELINE_SEQS],
                            "pooled_all11"),
        download=manifest,
        elapsed_s=None,
    )
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT / "kitti_more.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    write_markdown(out)
    print(f"[done] {len(per_sequence)} sequences in {out['elapsed_s']}s", flush=True)


def write_markdown(o):
    cfg = o["config"]
    L = ["# KITTI odometry 01/02/03/04/06/08/09/10: robust 3D-3D extension", ""]
    L.append(f"Protocol imported unchanged from `../kitti_odom/run_kitti_odom.py`: "
             f"N={cfg['n']}, c={cfg['c']} m, tau={cfg['tau']} m, voxel={cfg['voxel']} m, "
             f"random outliers at p in {cfg['p']}, **{cfg['trials']} trials/cell**, "
             f"{cfg['pairs_per_seq']} consecutive-frame pairs per sequence. "
             f"Success = rot < 5 deg and |t| < 0.1 m. The parent 00/05/07 rows use "
             f"20 trials/cell (read from `../kitti_odom/results/kitti_odom.json`), "
             f"so compare means, not paired tests.")
    L.append("")
    L.append("## Data actually used")
    L.append("")
    L.append("| seq | GT frames | frames fetched by protocol | pairs used | MB |")
    L.append("|---|---|---|---|---|")
    for s in cfg["sequences"]:
        d = o["download"].get(s, {})
        L.append(f"| {s} | {d.get('n_frames','?')} | {len(d.get('frames',[]))} | "
                 f"{len(o['per_sequence'][s]['pairs_used'])} | "
                 f"{d.get('fetched_bytes',0)/1e6:.1f} |")
    L.append("")
    L.append("The official odometry Velodyne archive is a single 84.8 GB zip; only the "
             "~33 frames per sequence touched by the protocol were range-fetched from "
             "the HuggingFace mirror. **Coverage claim: all 11 sequences 00-10 are "
             "represented by the frames this protocol uses; the full 00-10 LiDAR "
             "streams are NOT downloaded.**")
    L.append("")
    L.append("## Success per sequence (GNC / RANSAC-1000), p = 0.0/0.3/0.5/0.7")
    L.append("")
    L.append("| seq | base inlier | GNC p0 | GNC p.3 | GNC p.5 | GNC p.7 | RANSAC p.7 |")
    L.append("|---|---|---|---|---|---|---|")
    rows = {s: o["per_sequence"][s]["rows"] for s in cfg["sequences"]}
    for s in cfg["sequences"]:
        def get(m, p):
            c = [r for r in rows[s] if r["method"] == m and abs(r["p"] - p) < 1e-9]
            return c[0] if c else None
        base_in = np.mean([r["base_inlier"] for r in rows[s] if abs(r["p"]) < 1e-9])
        vals = [get("GNC-GM", p)["success"] for p in P_LIST]
        rp7 = get("RANSAC-1000", 0.7)["success"]
        L.append(f"| {s} | {base_in:.3f} | " + " | ".join(f"{v:.2f}" for v in vals)
                 + f" | {rp7:.2f} |")
    L.append("")
    L.append("## Pooled")
    L.append("")
    L.append("| set | p | GNC | RANSAC-1000 | IRLS | RANSAC/GNC time |")
    L.append("|---|---|---|---|---|---|")
    for r in o["pooled_new"]:
        g = r
        rr = [x for x in o["pooled_new"] if x["p"] == r["p"]
              and x["method"] == "RANSAC-1000"]
        il = [x for x in o["pooled_new"] if x["p"] == r["p"]
              and x["method"] == "IRLS-GM"]
        if r["method"] != "GNC-GM" or not rr:
            continue
        L.append(f"| new 8 | {r['p']:.1f} | {g['success']:.2f} | "
                 f"{rr[0]['success']:.2f} | {il[0]['success'] if il else float('nan'):.2f} | "
                 f"{rr[0]['time_ms_p50']/g['time_ms_p50']:.1f}x |")
    for r in o["pooled_all11"]:
        if r["method"] != "GNC-GM":
            continue
        rr = [x for x in o["pooled_all11"] if x["p"] == r["p"]
              and x["method"] == "RANSAC-1000"]
        L.append(f"| all 11 | {r['p']:.1f} | {r['success']:.2f} | "
                 f"{rr[0]['success']:.2f} | "
                 f"{[x for x in o['pooled_all11'] if x['p']==r['p'] and x['method']=='IRLS-GM'][0]['success']:.2f} | "
                 f"{rr[0]['time_ms_p50']/r['time_ms_p50']:.1f}x |")
    L.append("")
    L.append("## Baseline 00/05/07 at the same p (20 trials/cell, parent run)")
    L.append("")
    L.append("| seq | GNC p0 | GNC p.3 | GNC p.5 | GNC p.7 |")
    L.append("|---|---|---|---|---|")
    for s in cfg["baseline_sequences"]:
        br = [r for r in o["baseline"]["rows"] if r["dataset"] == f"odom{s}"]
        vals = []
        for p in P_LIST:
            c = [r for r in br if r["method"] == "GNC-GM" and abs(r["p"] - p) < 1e-9]
            vals.append(c[0]["success"] if c else float("nan"))
        L.append(f"| {s} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    L.append("")
    (OUT / "kitti_more.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
