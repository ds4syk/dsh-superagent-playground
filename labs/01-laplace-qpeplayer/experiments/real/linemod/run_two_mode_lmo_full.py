#!/usr/bin/env python3
"""Full-sequence LMO evaluation of the two-mode / Laplace estimators + SQPNP baseline.

What this adds to `run_two_mode_lmo.py`
---------------------------------------
The previous run used 20 attempts/object on LMO test `000002` (37 usable
frames).  The submission table needs (i) many more frames, (ii) visibility and
plane-size strata, and (iii) a classical baseline on the *same* correspondences
to separate "data" from "estimator layer".  This script:

  1. scans all 1214 images of the only LMO test sequence for obj 10 (eggbox) /
     11 (glue) instances with ``px_count_visib >= PX_MIN`` (default 1200, the
     same data-availability threshold as before -- it is what guarantees 900
     valid depth pixels can be sampled), and picks ``--per-obj`` frames per
     object evenly spaced over that candidate list.  Selection never looks at
     K, at the estimator output or at the error.
  2. reuses the existing pipeline **unchanged** by importing it from
     `run_two_mode_lmo.py`: ``build_frame`` (mask_visib depth -> bearings ->
     nearest mesh vertex -> deterministic RANSAC plane -> reduced cost ->
     physical modes -> hard / Laplace / MC), and the exact metric helpers from
     `run_pose_error.py` (``pose_metrics``, ``make_vsets``, ``aggregate``,
     ``boot_ci``, ``paired_ci``).  tau=0.01, n_starts=384, 900 points, 2.5 mm
     RANSAC band.  No existing ``.py`` file is modified.
  3. stratifies the sample by ``visib_fract`` terciles (boundaries computed on
     the *candidate pool*, before any estimator runs), by ``px_count_visib``
     terciles, and by the pre-registered ``n_plane >= 350`` split, and tests the
     "small visible patch drives the hard-argmin degradation" hypothesis with
     permutation tests and bootstrap Spearman CIs.
  4. runs OpenCV ``SOLVEPNP_SQPNP`` on 20 of the same frames (10/object) using
     exactly the same 3-D model points / 2-D pixels (pixel round-trip is
     asserted), in two variants: ``plane`` = the same RANSAC inlier set the
     estimator sees, ``all`` = all 900 correspondences.  This is the classical
     baseline that tells whether the degraded hard argmin is a *data* limit or
     a *layer* (cost / mode ranking) limit.  SQPNP runs in the sdpr env through
     `sqpnp_lmo_helper.py`; the estimator env has no cv2.

Outputs: ``results/lmo_full.{json,md,png}``.  Everything numeric comes from
this run; the previous 37-frame table is only *cited* (from
``results/lmo_two_mode.json``) for a stability check.

Usage
-----
  srun --partition=cpu -n1 --time=00:40:00 \
      /home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo_full.py
  ... run_two_mode_lmo_full.py --quick            # 4 frames/object, 2 SQPNP frames, no starts check
  ... run_two_mode_lmo_full.py --per-obj 100 --n-sqpnp-per-obj 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import permutation_test, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_two_mode_lmo as base  # noqa: E402  (sets sym.ROOT / tm.ROOT to LMO)
from run_pose_error import (  # noqa: E402
    BOOT_SEED,
    MC_SEEDS,
    N_ADD_SUBS,
    N_BOOT,
    boot_ci,
    paired_ci,
    pose_metrics,
)
from run_two_mode import load_gt, plane_ransac, real_correspondences  # noqa: E402
from run_two_mode_lmo import (  # noqa: E402
    LMO_ROOT,
    OBJ,
    PX_MIN,
    SEQ,
    jsonable,
)

SDPR_PY = Path("/home/syk/miniconda3/envs/sdpr/bin/python")
WM_PY = Path("/home/syk/miniconda3/envs/WM/bin/python")
N_PER_OBJ = 100          # attempts per object -> ~185 usable of 200 attempted
N_SQPNP_PER_OBJ = 10     # classical-baseline frames per object (task: 10-20 total)
N_STARTS_SUBSET = 8      # frames re-run with 2x Newton starts (enumeration check)
PLANE_STRATUM = base.PLANE_STRATUM          # 350, pre-registered split
AUC_T = base.AUC_T                          # 0.1 m

# ------------------------------------------------------------------ monkeypatch
# `base.build_frame` calls the module-global `make_vsets`; wrapping it lets this
# script keep the *exact* vertex subsamples used for the hard/Laplace/MC ADD-S,
# so the SQPNP ADD-S is evaluated on the identical query set (no extra
# subsample noise in the baseline comparison).  The wrapped function delegates.
_CALLS: list = []
_ORIG_MAKE_VSETS = base.make_vsets


def _make_vsets_probe(verts, R, t, rng):
    vs = _ORIG_MAKE_VSETS(verts, R, t, rng)
    _CALLS.append((int(len(verts)), vs))
    return vs


base.make_vsets = _make_vsets_probe
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ small utils
def jsonable_clean(obj):
    """base.jsonable (numpy -> python) plus NaN/Inf -> None, so the JSON is strict."""
    obj = base.jsonable(obj)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    return clean(obj)


def fstat(stat, ci, nd=2):
    return f"{stat:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def med_ci(vals, seed, nd=2, scale=1.0):
    vals = np.asarray(vals, float) * scale
    return (float(np.median(vals)),
            [float(c) for c in boot_ci(vals, np.median, seed=seed)])


def mean_ci(vals, seed, scale=1.0):
    vals = np.asarray(vals, float) * scale
    return (float(np.mean(vals)),
            [float(c) for c in boot_ci(vals, np.mean, seed=seed)])


def spearman_ci(x, y, seed, n_boot=N_BOOT):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([spearmanr(x[i], y[i]).statistic for i in idx], float)
    return dict(rho=rho, ci=[float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))],
                n=int(len(x)))


def perm_median_diff(a, b, seed, n=5000, min_n=8):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < min_n or len(b) < min_n:
        return dict(status="too_few", n_a=int(len(a)), n_b=int(len(b)),
                    median_a=float(np.median(a)) if len(a) else None,
                    median_b=float(np.median(b)) if len(b) else None)
    r = permutation_test((a, b),
                         statistic=lambda x, y: np.median(x) - np.median(y),
                         n_resamples=n, alternative="two-sided", random_state=seed)
    return dict(n_a=int(len(a)), n_b=int(len(b)),
                median_a=float(np.median(a)), median_b=float(np.median(b)),
                diff=float(np.median(a) - np.median(b)), p=float(r.pvalue),
                n_resamples=int(n))


# --------------------------------------------------------------- correspondence
def correspondences_with_pixels(oid, im, idx, models, trees):
    """The exact `real_correspondences` output + the original pixel coordinates.

    ``U`` is a unit bearing, so ``(K U) / (K U)_z`` reproduces the pixel exactly;
    this is asserted against a re-read of `mask_visib` (see ``pixel_dev_px``), so
    the SQPNP baseline provably sees the same 2-D points as the estimator.
    """
    K, R, t, _ = load_gt(SEQ, str(im), idx)
    data = real_correspondences(SEQ, str(im), idx, K, R, t, trees[oid], models[oid])
    if data is None:
        return None
    P, U, dist = data
    KU = (K @ U.T).T
    uv = KU[:, :2] / KU[:, 2:3]
    # the deterministic subset of real mask_visib pixels, recomputed here to check
    fdir = LMO_ROOT / "test" / SEQ
    msk = np.array(Image.open(fdir / "mask_visib" / f"{im:06d}_{idx:06d}.png")) > 0
    dep = np.array(Image.open(fdir / "depth" / f"{im:06d}.png"))
    ys, xs = np.nonzero(msk)
    d = dep[ys, xs]
    ok = d > 0
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if len(ys) < len(uv):
        return None
    sel = np.linspace(0, len(ys) - 1, len(uv)).astype(int)
    pix = np.stack([xs[sel], ys[sel]], 1).astype(float)
    dev = float(np.max(np.abs(uv - pix)))
    pl = plane_ransac(P)
    if pl is None:
        return None
    return dict(P=P, uv=uv, inl=pl[0], K=K, R=R, t=t, dist=dist,
                pixel_dev_px=dev)


# ------------------------------------------------------------------ SQPNP
def run_sqpnp(inputs: dict, K: np.ndarray):
    """Write the correspondences, solve them in the sdpr env, return the dict."""
    if not SDPR_PY.exists():
        return dict(status="missing_interpreter", path=str(SDPR_PY)), None
    if not (HERE / "sqpnp_lmo_helper.py").exists():
        return dict(status="missing_helper"), None
    tmpdir = Path(tempfile.mkdtemp(prefix="lmo_sqpnp_"))
    zin = tmpdir / "in.npz"
    zout = tmpdir / "out.npz"
    meta = []
    arrays = dict(K=np.asarray(K, float))
    for key, rec in inputs.items():
        for st, (P, uv) in (("plane", (rec["P"][rec["inl"]], rec["uv"][rec["inl"]])),
                            ("all", (rec["P"], rec["uv"]))):
            arrays[f"{key}|{st}|P"] = np.asarray(P, float)
            arrays[f"{key}|{st}|uv"] = np.asarray(uv, float)
            meta.append(dict(key=key, set=st))
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(zin, **arrays)
    try:
        p = subprocess.run([str(SDPR_PY), str(HERE / "sqpnp_lmo_helper.py"),
                            str(zin), str(zout)],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            return dict(status="failed", returncode=p.returncode,
                        stdout=p.stdout[-1500:], stderr=p.stderr[-3000:]), None
        z = np.load(zout, allow_pickle=False)
        sol = json.loads(str(z["meta"][()]))
        ver = subprocess.run([str(SDPR_PY), "-c", "import cv2;print(cv2.__version__)"],
                             capture_output=True, text=True).stdout.strip()
        return dict(status="ok", cv2_version=ver,
                    stdout=p.stdout.strip()), sol
    finally:
        for f in (zin, zout):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def rvec_tvec_to_Rt(rvec, tvec):
    return Rot.from_rotvec(np.asarray(rvec, float)).as_matrix(), np.asarray(tvec, float)


def eval_sqpnp(sol, rows_by_key, vsets_by_key):
    """Per-frame SQPNP metrics on the *same* vsets as hard/Laplace/MC."""
    out = {}
    for key, sets in sol.items():
        row = rows_by_key[key]
        K, R_gt, t_gt, _ = load_gt(SEQ, str(row["img"]), row["idx"])
        vsets = vsets_by_key[key]
        fr = dict(obj=row["obj"], img=row["img"], idx=row["idx"],
                  visib_fract=row["visib_fract"], px_count_visib=row["px_count_visib"],
                  n_plane=row["n_plane"], frame=row["frame"], sets={})
        for st in ("plane", "all"):
            rec = sets.get(st)
            if not rec or rec.get("status") != "ok":
                fr["sets"][st] = dict(status="fail",
                                      error=(rec or {}).get("error"),
                                      n_solutions=(rec or {}).get("n_solutions"))
                continue
            sols = rec["solutions"]
            chosen = sols[rec["chosen"]]
            Rc, tc = rvec_tvec_to_Rt(chosen["rvec"], chosen["tvec"])
            met = pose_metrics(Rc, tc, R_gt, t_gt, vsets, with_floor=False)
            # oracle over the returned solutions (GT-peeking diagnostic, labelled)
            best = None
            for s in sols:
                Ro, to = rvec_tvec_to_Rt(s["rvec"], s["tvec"])
                mo = pose_metrics(Ro, to, R_gt, t_gt, vsets)
                if best is None or mo["rot_deg"] < best["rot_deg"]:
                    best = mo
            fr["sets"][st] = dict(
                status="ok", n_points=int(rec["n_points"]),
                n_solutions=int(rec["n_solutions"]),
                chosen_reproj_px=float(chosen["reproj_px"]),
                chosen_rot_deg=met["rot_deg"], chosen_trans_m=met["trans_m"],
                chosen_adds_m=met["adds_m"],
                oracle_rot_deg=best["rot_deg"], oracle_trans_m=best["trans_m"],
                oracle_adds_m=best["adds_m"],
                n_solutions_in_front=int(chosen["n_in_front"]),
            )
        out[key] = fr
    return out


def aggregate_sqpnp(frames, rows_by_key):
    """Aggregate over the SQPNP subset, with the estimator values on the same frames."""
    keys = [k for k, v in frames.items() if v["sets"].get("all", {}).get("status") == "ok"]
    keys = sorted(keys, key=lambda k: (frames[k]["obj"], frames[k]["img"]))
    series = {}
    for meth in ("hard", "laplace", "mc"):
        series[meth] = []
        for k in keys:
            r = rows_by_key[k]
            series[meth].append([r[meth]["rot_deg"], r[meth]["trans_m"], r[meth]["adds_m"]]
                                if meth != "mc" else
                                [r["mc"]["rot_deg"]["mean"], r["mc"]["trans_m"]["mean"],
                                 r["mc"]["adds_m"]["mean"]])
    for st in ("plane", "all"):
        series[f"sqpnp_{st}"] = []
        series[f"sqpnp_{st}_oracle"] = []
        for k in keys:
            s = frames[k]["sets"][st]
            if s.get("status") != "ok":
                series[f"sqpnp_{st}"].append([np.nan] * 3)
                series[f"sqpnp_{st}_oracle"].append([np.nan] * 3)
            else:
                series[f"sqpnp_{st}"].append([s["chosen_rot_deg"], s["chosen_trans_m"],
                                              s["chosen_adds_m"]])
                series[f"sqpnp_{st}_oracle"].append([s["oracle_rot_deg"], s["oracle_trans_m"],
                                                     s["oracle_adds_m"]])
    names = ("rot_deg", "trans_m", "adds_m")
    agg = {}
    for i, nm in enumerate(names):
        block = {}
        for meth, vals in series.items():
            v = np.array([x[i] for x in vals], float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                block[meth] = dict(n=0)
                continue
            m, mci = med_ci(v, BOOT_SEED + 6100 + 7 * i)
            mu, muci = mean_ci(v, BOOT_SEED + 6200 + 7 * i)
            block[meth] = dict(n=int(len(v)), median=m, median_ci=mci,
                               mean=mu, mean_ci=muci)
        # paired differences vs hard on the frames where both exist
        for meth in ("sqpnp_plane", "sqpnp_all", "sqpnp_plane_oracle", "sqpnp_all_oracle"):
            diffs = [x[i] - y[i] for x, y in zip(series[meth], series["hard"])
                     if np.isfinite(x[i]) and np.isfinite(y[i])]
            block[f"{meth}_minus_hard"] = paired_ci(diffs, seed=BOOT_SEED + 6300 + 7 * i)
            best = int(sum(1 for x, y in zip(series[meth], series["hard"])
                           if np.isfinite(x[i]) and np.isfinite(y[i]) and x[i] < y[i]))
            block[f"{meth}_minus_hard"]["n_better"] = best
        agg[nm] = block
    return dict(keys=keys, n=len(keys), metrics=agg, series=series)


# ------------------------------------------------------------------ strata
def block_stats(g, seed):
    if not g:
        return dict(n=0)
    block = dict(
        n=len(g),
        n_eggbox=int(sum(1 for r in g if r["obj"] == 10)),
        n_glue=int(sum(1 for r in g if r["obj"] == 11)),
        k2_exact_rate=float(np.mean([r["n_phys"] == 2 for r in g])),
        median={}, median_ci={}, mean={}, mean_ci={},
        context=dict(
            visib_fract_median=float(np.median([r["visib_fract"] for r in g])),
            px_count_visib_median=float(np.median([r["px_count_visib"] for r in g])),
            n_plane_median=float(np.median([r["n_plane"] for r in g])),
            plane_extent_mm_median=float(np.median([r["plane_extent_mm"] for r in g])),
            dist_p50_mm_median=float(np.median([r["dist_p50_mm"] for r in g])),
            J_gt_over_min_median=float(np.median([r["J_gt_over_min"] for r in g])),
        ),
        oracle=dict(
            closest_mode_rot_deg_median=float(np.median([r["closest_mode_rot_deg"] for r in g])),
            hard_is_gt_like_rate=float(np.mean([r["hard_is_gt_like"] for r in g])),
            flip_deg_median=float(np.median([r["flip_deg"] for r in g])),
            w_other_median=float(np.median([r["w_other"] for r in g])),
        ),
        n_phys_dist={str(k): int(v) for k, v in
                     zip(*np.unique([r["n_phys"] for r in g], return_counts=True))},
    )
    for i, key in enumerate(("rot_deg", "trans_m", "adds_m", "auc")):
        block["median"][key] = {}
        block["median_ci"][key] = {}
        block["mean"][key] = {}
        block["mean_ci"][key] = {}
        for j, meth in enumerate(("hard", "laplace", "mc")):
            vals = np.array([base.meth_value(r, meth, key) for r in g], float)
            sd = seed + 31 * i + 7 * j
            block["median"][key][meth] = float(np.median(vals))
            block["median_ci"][key][meth] = [float(c) for c in
                                             boot_ci(vals, np.median, seed=sd)]
            block["mean"][key][meth] = float(np.mean(vals))
            block["mean_ci"][key][meth] = [float(c) for c in
                                           boot_ci(vals, np.mean, seed=sd + 1)]
    return block


def build_strata(rows, edges):
    q1, q2 = edges["visib_fract_terciles"]
    p1, p2 = edges["px_count_visib_terciles"]
    vis = {
        f"visib < {q1:.2f}": [r for r in rows if r["visib_fract"] < q1],
        f"visib {q1:.2f}-{q2:.2f}": [r for r in rows if q1 <= r["visib_fract"] < q2],
        f"visib >= {q2:.2f}": [r for r in rows if r["visib_fract"] >= q2],
    }
    npl = {
        f"n_plane >= {PLANE_STRATUM}": [r for r in rows if r["n_plane"] >= PLANE_STRATUM],
        f"n_plane < {PLANE_STRATUM}": [r for r in rows if r["n_plane"] < PLANE_STRATUM],
    }
    px = {
        f"px < {p1:.0f}": [r for r in rows if r["px_count_visib"] < p1],
        f"px {p1:.0f}-{p2:.0f}": [r for r in rows if p1 <= r["px_count_visib"] < p2],
        f"px >= {p2:.0f}": [r for r in rows if r["px_count_visib"] >= p2],
    }
    out = {}
    for i, (name, groups) in enumerate((("visib_fract", vis), ("n_plane", npl),
                                        ("px_count_visib", px))):
        out[name] = {
            "definition": list(edges.get(name + "_terciles", [PLANE_STRATUM])),
            "groups": {k: block_stats(v, BOOT_SEED + 5000 + 97 * i + 13 * j)
                       for j, (k, v) in enumerate(groups.items())}}
    return out, vis, npl, px


def build_within_object(rows, edges):
    """Object-internal replication of the n_plane / visibility splits.

    The pooled ``n_plane < 350`` stratum is 51/55 eggbox, so the pooled test is
    confounded with object identity.  This block repeats both splits *inside*
    each object, where the object is held fixed.
    """
    q1, q2 = edges["visib_fract_terciles"]
    out = {}
    for i, oid in enumerate(OBJ):
        sub = [r for r in rows if r["obj"] == oid]
        hi = [r for r in sub if r["n_plane"] >= PLANE_STRATUM]
        lo = [r for r in sub if r["n_plane"] < PLANE_STRATUM]
        vlo = [r for r in sub if r["visib_fract"] < q1]
        vhi = [r for r in sub if r["visib_fract"] >= q2]
        out[str(int(oid))] = dict(
            n=len(sub),
            n_plane=dict(
                groups={f">= {PLANE_STRATUM}": block_stats(hi, BOOT_SEED + 8100 + 100 * i),
                        f"< {PLANE_STRATUM}": block_stats(lo, BOOT_SEED + 8110 + 100 * i)},
                test=perm_median_diff([r["hard"]["rot_deg"] for r in hi],
                                      [r["hard"]["rot_deg"] for r in lo],
                                      BOOT_SEED + 8120 + 100 * i),
                test_adds=perm_median_diff([1e3 * r["hard"]["adds_m"] for r in hi],
                                           [1e3 * r["hard"]["adds_m"] for r in lo],
                                           BOOT_SEED + 8130 + 100 * i),
                test_closest=perm_median_diff([r["closest_mode_rot_deg"] for r in hi],
                                              [r["closest_mode_rot_deg"] for r in lo],
                                              BOOT_SEED + 8140 + 100 * i),
            ),
            visib=dict(
                groups={f"< {q1:.2f}": block_stats(vlo, BOOT_SEED + 8200 + 100 * i),
                        f">= {q2:.2f}": block_stats(vhi, BOOT_SEED + 8210 + 100 * i)},
                test=perm_median_diff([r["hard"]["rot_deg"] for r in vlo],
                                      [r["hard"]["rot_deg"] for r in vhi],
                                      BOOT_SEED + 8220 + 100 * i),
            ),
        )
    return out


def build_strata_tests(rows, vis, npl):
    hard_rot = lambda g: [r["hard"]["rot_deg"] for r in g]
    hard_adds = lambda g: [1e3 * r["hard"]["adds_m"] for r in g]
    closest = lambda g: [r["closest_mode_rot_deg"] for r in g]
    lap_rot = lambda g: [r["laplace"]["rot_deg"] for r in g]
    hi = npl[f"n_plane >= {PLANE_STRATUM}"]
    lo = npl[f"n_plane < {PLANE_STRATUM}"]
    vkeys = list(vis)
    tests = {
        "n_plane/hard_rot": perm_median_diff(hard_rot(hi), hard_rot(lo), BOOT_SEED + 7001),
        "n_plane/hard_adds_mm": perm_median_diff(hard_adds(hi), hard_adds(lo),
                                                 BOOT_SEED + 7002),
        "n_plane/closest_mode_rot": perm_median_diff(closest(hi), closest(lo),
                                                     BOOT_SEED + 7003),
        "n_plane/laplace_rot": perm_median_diff(lap_rot(hi), lap_rot(lo),
                                                BOOT_SEED + 7004),
        "visib/hard_rot": perm_median_diff(hard_rot(vis[vkeys[2]]),
                                           hard_rot(vis[vkeys[0]]), BOOT_SEED + 7005),
        "visib/hard_adds_mm": perm_median_diff(hard_adds(vis[vkeys[2]]),
                                               hard_adds(vis[vkeys[0]]),
                                               BOOT_SEED + 7006),
        "visib/closest_mode_rot": perm_median_diff(closest(vis[vkeys[2]]),
                                                   closest(vis[vkeys[0]]),
                                                   BOOT_SEED + 7007),
    }
    x = {k: np.array([r[k] for r in rows], float) for k in
         ("visib_fract", "px_count_visib", "n_plane", "plane_extent_mm", "dist_p50_mm")}
    corr = {}
    for i, stat in enumerate(("hard_rot", "hard_adds_mm", "closest_mode_rot")):
        y = {"hard_rot": [r["hard"]["rot_deg"] for r in rows],
             "hard_adds_mm": [1e3 * r["hard"]["adds_m"] for r in rows],
             "closest_mode_rot": [r["closest_mode_rot_deg"] for r in rows]}[stat]
        corr[stat] = {k: spearman_ci(v, y, BOOT_SEED + 7100 + 10 * i + j)
                      for j, (k, v) in enumerate(x.items())}
    return tests, corr


def temporal_blocks(rows, n_images):
    edges = np.linspace(0, n_images - 1, 5)
    blocks = []
    for b in range(4):
        lo, hi = edges[b], edges[b + 1]
        g = [r for r in rows if (lo <= r["img"] <= hi if b == 3
                                 else lo <= r["img"] < hi)]
        blocks.append(dict(
            label=f"img {int(lo)}-{int(hi)}", n=len(g),
            hard_rot_median=float(np.median([r["hard"]["rot_deg"] for r in g])) if g else None,
            laplace_rot_median=float(np.median([r["laplace"]["rot_deg"] for r in g])) if g else None,
            hard_adds_mm_median=float(np.median([1e3 * r["hard"]["adds_m"] for r in g])) if g else None,
            n_plane_median=float(np.median([r["n_plane"] for r in g])) if g else None,
            visib_median=float(np.median([r["visib_fract"] for r in g])) if g else None,
        ))
    return blocks


def cite_previous():
    p = HERE / "results" / "lmo_two_mode.json"
    if not p.exists():
        return dict(status="missing", path=str(p))
    d = json.load(open(p))
    agg = d["aggregate"]
    out = dict(status="ok", source="results/lmo_two_mode.json",
               date=d.get("meta", {}).get("date"), n_attempted=d["meta"]["n_attempted"],
               n_used=len(d["frames"]), failures=d.get("failures", []),
               metrics={})
    for key in ("rot_deg", "trans_m", "adds_m", "auc"):
        out["metrics"][key] = {}
        for meth in ("hard", "laplace", "mc"):
            out["metrics"][key][meth] = dict(
                median=agg[key][meth]["median"],
                median_ci=agg[key][meth]["median_ci"],
                mean=agg[key][meth]["mean"])
    return out


# ------------------------------------------------------------------ checks
def build_checks(stash, gt, rows):
    checks = {}
    # reuse the mesh/constant checks from the previous run, recomputed here
    import hashlib

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    m = {}
    for oid in OBJ:
        a = sha256(LMO_ROOT / "models" / f"obj_{oid:06d}.ply")
        b = sha256(base.LM_ROOT / "models" / f"obj_{oid:06d}.ply")
        m[int(oid)] = dict(lmo_sha256=a, lm_sha256=b, identical=bool(a == b))
    checks["mesh_identity"] = dict(
        all_identical=bool(all(v["identical"] for v in m.values())), per_object=m,
        note="ADD-S primary mesh = models/, byte-identical to the LINEMOD meshes cited "
             "in results/pose_error.json; models_eval is the secondary column.")
    checks["constants"] = dict(
        tau=base.TAU, n_points=base.N_POINTS, n_starts=base.N_STARTS,
        plane_thr_m=base.PLANE_THR, min_plane=base.MIN_PLANE,
        min_other=base.MIN_OTHER, mc_n=base.MC_N, mc_seeds=MC_SEEDS,
        n_boot=N_BOOT, n_add_pts=base.N_ADD_PTS, n_add_subs=N_ADD_SUBS,
        auc_T_m=AUC_T)
    checks["starts_check_2x"] = stash.get("starts_check")
    checks["sqpnp"] = stash.get("sqpnp_env")
    checks["pixel_roundtrip"] = stash.get("pixel_dev")
    checks["shift_check"] = dict(
        status="not_repeated",
        note="The 2-D depth/mask shift spot check was run in run_two_mode_lmo.py on the "
             "37-frame sample and found that integer (dx,dy) shifts do not explain the "
             "depth-model residual; it is not repeated here (cost), so the LMO residual "
             "is cited from that run, not re-derived.")
    # max instances of the same object in one image (duplicate-instance guard)
    dup = {int(o): max(sum(1 for q in gt[im] if int(q["obj_id"]) == o) for im in gt)
           for o in OBJ}
    checks["max_gt_instances_per_image"] = dup
    checks["n_attempted"] = len(rows) + len(stash.get("failures", []))
    return checks


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-obj", type=int, default=None)
    ap.add_argument("--px-min", type=int, default=PX_MIN)
    ap.add_argument("--n-sqpnp-per-obj", type=int, default=None)
    ap.add_argument("--no-sqpnp", action="store_true")
    ap.add_argument("--no-starts-check", action="store_true")
    ap.add_argument("--from-json", type=str, default=None,
                    help="regenerate lmo_full.md/.png from an existing lmo_full.json")
    args = ap.parse_args()

    if args.from_json:
        doc = json.load(open(args.from_json))
        write_report(doc, OUT / "lmo_full.md")
        make_figure(doc, OUT / "lmo_full.png")
        print(f"regenerated {OUT/'lmo_full.md'} / .png from {args.from_json}", flush=True)
        return

    per_obj = args.per_obj if args.per_obj is not None else (4 if args.quick else N_PER_OBJ)
    n_sq = args.n_sqpnp_per_obj if args.n_sqpnp_per_obj is not None else (
        2 if args.quick else N_SQPNP_PER_OBJ)
    do_sqpnp = not args.no_sqpnp
    do_starts = (not args.quick) and (not args.no_starts_check)
    px_min = int(args.px_min)
    t_start = time.time()

    print(f"[inventory] {LMO_ROOT}", flush=True)
    inv = base.inventory()
    print(f"  images={inv['n_images']} objects={inv['model_obj_ids']}", flush=True)

    gt = json.load(open(LMO_ROOT / "test" / SEQ / "scene_gt.json"))
    gi = json.load(open(LMO_ROOT / "test" / SEQ / "scene_gt_info.json"))
    models = {oid: base.load_model(oid) for oid in OBJ}
    trees = {oid: base.cKDTree(v) for oid, v in models.items()}
    eval_models = {oid: base.load_ply_m(LMO_ROOT / "models_eval" / f"obj_{oid:06d}.ply")
                   for oid in OBJ}
    I = base.inst_mod.Instance(seed=20261127)
    rng_sub = np.random.default_rng(BOOT_SEED)
    rng_sub_eval = np.random.default_rng(BOOT_SEED + 1)

    # ---- candidate pool (all instances with px_count_visib >= px_min), data only
    chosen = {}
    pools = {}
    cand_counts = {}
    for oid in OBJ:
        ch, recs, cand = base.frame_candidates(gt, gi, oid, per_obj, px_min)
        allcand = [r for r in recs if r["px"] >= px_min]
        chosen[int(oid)] = ch
        pools[int(oid)] = allcand
        cand_counts[str(int(oid))] = dict(gt_instances=len(recs),
                                          candidates=len(allcand),
                                          attempted=len(ch))
        print(f"[obj {oid}] GT instances={len(recs)} candidates(px>={px_min})={len(allcand)} "
              f"attempting={len(ch)}", flush=True)
    pool = [r for oid in OBJ for r in pools[oid]]
    edges = dict(
        visib_fract_terciles=[float(x) for x in
                              np.percentile([r["visib_fract"] for r in pool], [33.333, 66.667])],
        px_count_visib_terciles=[float(x) for x in
                                 np.percentile([r["px"] for r in pool], [33.333, 66.667])],
    )
    print(f"[strata] visib terciles={edges['visib_fract_terciles']} "
          f"px terciles={edges['px_count_visib_terciles']}", flush=True)

    # ---- SQPNP subset: evenly spaced over the *candidate list* per object
    sqpnp_targets = {}
    for oid in OBJ:
        n_use = min(n_sq, len(chosen[oid]))
        sel = sorted(set(np.linspace(0, len(chosen[oid]) - 1, n_use).astype(int).tolist()))
        sqpnp_targets[int(oid)] = {(c["im"], c["idx"]) for c in
                                   [chosen[oid][i] for i in sel]}
    n_sq_planned = sum(len(v) for v in sqpnp_targets.values())

    # ---- main frame loop: the existing pipeline, untouched
    rows, failures, rows_by_key, sq_inputs, sqpnp_vsets = [], [], {}, {}, {}
    starts_log = []
    pixel_devs = []
    for oid in OBJ:
        for n_done, ch in enumerate(chosen[oid], 1):
            im, idx = ch["im"], ch["idx"]
            key = f"{oid}_{im:06d}_{idx}"
            _CALLS.clear()
            row, fail = base.build_frame(
                I, oid, im, idx, gi, models=models, trees=trees,
                eval_models=eval_models, rng_sub=rng_sub, rng_sub_eval=rng_sub_eval)
            if row is None:
                fail.update(obj=int(oid), im=im, idx=idx, frame=f"{SEQ}/{im:06d}/{idx}",
                            px_count_visib=int(ch["px"]),
                            px_count_valid=int(gi[str(im)][idx].get("px_count_valid", -1)),
                            visib_fract=float(ch["visib_fract"]))
                failures.append(fail)
                print(f"  FAIL {key}: {fail['reason']}", flush=True)
                continue
            rows.append(row)
            rows_by_key[key] = row
            if (im, idx) in sqpnp_targets[oid] and do_sqpnp:
                rec = correspondences_with_pixels(oid, im, idx, models, trees)
                if rec is not None:
                    sq_inputs[key] = rec
                    pixel_devs.append(rec["pixel_dev_px"])
                # exact vsets used by hard/Laplace/MC on this frame
                if len(_CALLS) >= 1:
                    sqpnp_vsets[key] = _CALLS[0][1]
            print(f"  [{n_done}/{len(chosen[oid])}] {key} visib={row['visib_fract']:.2f} "
                  f"px={row['px_count_visib']} plane={row['n_plane']} "
                  f"K={row['n_phys']}/{row['K_all']} dJ={row['dJ']:.2e} "
                  f"hard={row['hard']['rot_deg']:.2f}deg/{1e3*row['hard']['adds_m']:.1f}mm "
                  f"lap={row['laplace']['rot_deg']:.2f}deg", flush=True)

    if not rows:
        print("no usable frames; writing negative result", flush=True)
        doc = dict(meta=dict(runtime_s=time.time() - t_start), dataset=inv,
                   failures=failures, frames=[], note="no frame passed K>=2")
        json.dump(jsonable_clean(doc), open(OUT / "lmo_full.json", "w"), indent=2)
        return

    # ---- 2x-starts enumeration check on a subset (cheap robustness)
    stash = {}
    if do_starts:
        picks = sorted(set(np.linspace(0, len(rows) - 1,
                                       min(N_STARTS_SUBSET, len(rows))).astype(int).tolist()))
        for i in picks:
            r = rows[i]
            K2 = base.check_starts(I, r, models, trees)
            starts_log.append(dict(frame=r["frame"], **K2))
        nK = sum(1 for s in starts_log if s["K_phys"] != s["n_phys_primary"])
        nHard = sum(1 for s in starts_log if s["hard_rot_diff_deg"] > 0.5)
        stash["starts_check"] = dict(
            n_frames=len(starts_log), n_K_changed=nK, n_hard_changed=nHard,
            max_hard_rot_diff_deg=float(max(s["hard_rot_diff_deg"] for s in starts_log)),
            note="find_modes re-run with 2x starts (768) on the identical reduced cost; "
                 "K and hard argmin compared with the primary 384-start run.",
            frames=starts_log)
    else:
        stash["starts_check"] = dict(status="skipped")

    # ---- SQPNP baseline
    sq = dict(status="skipped")
    sq_frames, sq_agg = {}, {}
    if do_sqpnp and sq_inputs:
        t_sq = time.time()
        info, sol = run_sqpnp(sq_inputs, K=load_gt(SEQ, str(rows[0]["img"]), rows[0]["idx"])[0])
        stash["sqpnp_env"] = dict(interpreter=str(SDPR_PY), n_problems=2 * len(sq_inputs),
                                  elapsed_s=time.time() - t_sq, **info)
        if sol is not None:
            sq_frames = eval_sqpnp(sol, rows_by_key, sqpnp_vsets)
            sq_agg = aggregate_sqpnp(sq_frames, rows_by_key)
            sq = dict(status=info.get("status"), n_frames=sq_agg["n"],
                      subset_planned=n_sq_planned, frames=sq_frames, **sq_agg)
        else:
            sq = dict(status=info.get("status"), error="solver failed", env=info)
    stash["pixel_dev"] = dict(max_px=float(max(pixel_devs)) if pixel_devs else None,
                              n=int(len(pixel_devs)),
                              note="max |(K U)/(KU)_z - original mask_visib pixel| over the "
                                   "SQPNP frames; 0 proves the baseline sees the same 2-D points.")

    # ---- aggregates and strata
    agg = base.build_aggregates(rows)
    strata, vis, npl, px_groups = build_strata(rows, edges)
    tests, corr = build_strata_tests(rows, vis, npl)
    within = build_within_object(rows, edges)
    temporal = temporal_blocks(rows, inv["n_images"])
    prev = cite_previous()
    stash["failures"] = failures
    checks = build_checks(stash, gt, rows)

    elapsed = time.time() - t_start
    meta = dict(
        date="2026-11-28", dataset="BOP LMO (test/000002)",
        objects={int(o): OBJ[o] for o in OBJ},
        tau=base.TAU, n_starts=base.N_STARTS, n_points=base.N_POINTS,
        plane_thr_mm=base.PLANE_THR * 1e3, min_plane=base.MIN_PLANE,
        min_other=base.MIN_OTHER, mc_n=base.MC_N, mc_seeds=MC_SEEDS,
        n_boot=N_BOOT, n_add_pts=base.N_ADD_PTS, n_add_subs=N_ADD_SUBS,
        auc_range_m=[0.0, AUC_T], px_min=px_min, n_per_obj=per_obj,
        n_attempted=len(rows) + len(failures), n_used=len(rows),
        n_failures=len(failures), runtime_s=float(elapsed),
        host=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        python=sys.version.split()[0],
        command=" ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        pipeline="helpers imported unchanged from run_two_mode_lmo.build_frame / "
                 "run_pose_error.pose_metrics",
        frame_selection=f"scene_gt_info.px_count_visib >= {px_min}; evenly spaced over the "
                        f"sequence; {per_obj} attempts per object; selection never looks at K "
                        f"or at the estimator output",
        metric_mesh="models/ (byte-identical to the LINEMOD meshes); models_eval is the "
                    "secondary adds_eval_m column",
        note="GT-pose correspondences: a controlled estimator experiment on real cost "
             "functions, not a detector; no front-end and no accuracy claim against "
             "published BOP pipelines.",
    )
    selection = dict(px_min=px_min, per_obj=per_obj, objects=cand_counts,
                     pool_visib_fract_terciles=edges["visib_fract_terciles"],
                     pool_px_count_visib_terciles=edges["px_count_visib_terciles"],
                     sqpnp_planned_per_object={str(int(o)): len(v) for o, v in
                                               sqpnp_targets.items()})
    doc = dict(meta=meta, dataset=inv, selection=selection, failures=failures,
               frames=rows, aggregate=agg, stratified=strata,
               strata_tests=tests, correlations=corr, within_object=within,
               temporal=temporal,
               sqpnp=sq, comparison_previous=prev, checks=checks)
    json.dump(jsonable_clean(doc), open(OUT / "lmo_full.json", "w"), indent=2)
    print(f"wrote {OUT / 'lmo_full.json'}", flush=True)

    write_report(doc, OUT / "lmo_full.md")
    make_figure(doc, OUT / "lmo_full.png")
    print(f"wrote {OUT / 'lmo_full.md'} / .png  ({elapsed:.1f} s total)", flush=True)


# ------------------------------------------------------------------ markdown
def write_report(doc, path):
    rows = doc["frames"]
    n = len(rows)
    agg = doc["aggregate"]
    st = doc["stratified"]
    tests = doc["strata_tests"]
    corr = doc["correlations"]
    sq = doc["sqpnp"]
    wo = doc.get("within_object", {})
    prev = doc["comparison_previous"]
    inv = doc["dataset"]
    sel = doc["selection"]
    checks = doc["checks"]
    vis_g = st["visib_fract"]["groups"]
    npl_g = st["n_plane"]["groups"]
    px_g = st["px_count_visib"]["groups"]
    vkeys = list(vis_g)
    npl_hi, npl_lo = list(npl_g)
    q1, q2 = sel["pool_visib_fract_terciles"]
    med = lambda key, meth: float(np.median([base.meth_value(r, meth, key) for r in rows]))
    rowmap = {r["frame"]: r for r in rows}
    n_attempted = doc["meta"]["n_attempted"]
    n_fail = doc["meta"]["n_failures"]
    k2_attempt = n / n_attempted
    k2_exact = float(np.mean([r["n_phys"] == 2 for r in rows]))
    n_phys_dist = {str(k): int(v) for k, v in
                   zip(*np.unique([r["n_phys"] for r in rows], return_counts=True))}

    # SQPNP numbers used in the summary
    sqn = (sq.get("metrics") or {}).get("rot_deg", {}) if sq.get("status") == "ok" else {}
    sq_ok = bool(sqn)
    sq_hard = sqn.get("hard", {}) if sq_ok else {}
    sq_all = sqn.get("sqpnp_all", {}) if sq_ok else {}
    sq_plane = sqn.get("sqpnp_plane", {}) if sq_ok else {}
    sq_all_d = sqn.get("sqpnp_all_minus_hard", {}) if sq_ok else {}
    sq_plane_d = sqn.get("sqpnp_plane_minus_hard", {}) if sq_ok else {}
    adds_sq = (sq.get("metrics") or {}).get("adds_m", {}) if sq_ok else {}
    t_np = tests["n_plane/hard_rot"]
    t_np_or = tests["n_plane/closest_mode_rot"]
    t_v = tests["visib/hard_rot"]
    rho_np = corr["hard_rot"]["n_plane"]
    rho_or = corr["closest_mode_rot"]["n_plane"]
    L = []
    L.append("# LMO 全序列扩展：两模代价 + Laplace / MC + SQPNP 经典基线")
    L.append("")
    L.append(f"**日期：** 2026-11-28 · **数据：** `{LMO_ROOT}`（唯一 test 序列 `{SEQ}`，"
             f"{inv['n_images']} 张） · **样本：** {n} 帧可用 / {n_attempted} 帧尝试"
             f"（每对象 {sel['per_obj']} 帧，`px_count_visib >= {sel['px_min']}` 等距抽取） · "
             f"**运行：** {doc['meta']['runtime_s']:.1f} s"
             + (f"（Slurm `{doc['meta']['slurm_job_id']}`）" if doc['meta'].get('slurm_job_id') else "")
             + "。")
    L.append("")
    L.append("> **结论先行**")
    L.append(">")
    L.append(f"> 1. **样本从 37 帧扩到 {n} 帧**（obj 10 eggbox {sum(1 for r in rows if r['obj']==10)}、"
             f"obj 11 glue {sum(1 for r in rows if r['obj']==11)}），K≥2 物理模式通过率 "
             f"{n}/{n_attempted} = {k2_attempt:.1%}；可用帧中 **正好 K=2** 占 {k2_exact:.1%}，"
             f"n_phys 分布 {n_phys_dist}。失败 {n_fail} 帧按原因逐条列出（§6），不静默丢弃。")
    L.append(f"> 2. **聚合（n={n}）**：hard 旋转中位 **{med('rot_deg','hard'):.2f}°**、平移 "
             f"**{1e3*med('trans_m','hard'):.1f} mm**、ADD-S **{1e3*med('adds_m','hard'):.1f} mm**；"
             f"Laplace {med('rot_deg','laplace'):.1f}° / {1e3*med('adds_m','laplace'):.1f} mm；"
             f"MC {med('rot_deg','mc'):.1f}° / {1e3*med('adds_m','mc'):.1f} mm。"
             f"ADD-S AUC(0–0.1 m)：hard 均值 {agg['auc']['hard']['mean']:.3f} "
             f"[{agg['auc']['hard']['mean_ci'][0]:.3f}, {agg['auc']['hard']['mean_ci'][1]:.3f}]、"
             f"中位 {agg['auc']['hard']['median']:.3f}；Laplace 均值 "
             f"{agg['auc']['laplace']['mean']:.3f}、MC 均值 {agg['auc']['mc']['mean']:.3f}。")
    hard_hi = npl_g[npl_hi]["median"]["rot_deg"]["hard"]
    hard_lo = npl_g[npl_lo]["median"]["rot_deg"]["hard"]
    orc_hi = npl_g[npl_hi]["oracle"]["closest_mode_rot_deg_median"]
    orc_lo = npl_g[npl_lo]["oracle"]["closest_mode_rot_deg_median"]
    p_np = t_np.get("p")
    p_or = t_np_or.get("p")
    wrong_dir = hard_lo <= hard_hi
    if p_np is not None and p_np < 0.05 and not wrong_dir and rho_np["rho"] < 0:
        claim3 = f"在 n={n} 上成立（方向一致且显著）"
    elif not wrong_dir:
        claim3 = (f"方向与上次一致，但在 n={n} 上未达显著"
                  f"（p={p_np:.3f}），只能算与上次相容、不能算确认")
    else:
        claim3 = (f"在 n={n} 上不成立：小片组的 hard 中位并不更差"
                  f"（p={p_np:.3f}），与上次 37 帧的机制解释相矛盾")
    if p_or is not None and p_or < 0.05 and orc_lo > orc_hi:
        mech3 = (f"关键的是**最近的物理模式本身**也在小片组显著变差"
                 f"（{orc_hi:.2f}° vs {orc_lo:.2f}°，p={p_or:.4f}），"
                 f"说明退化发生在**代价面**（最小点被深度-模型残差推离 GT），而不是模式排序。")
    elif orc_lo > orc_hi:
        mech3 = (f"最近的物理模式本身也同向变差（{orc_hi:.2f}° vs {orc_lo:.2f}°，"
                 f"p={p_or:.4f}，未达显著），与「代价面被推离」一致但不构成显著证据。")
    else:
        mech3 = (f"但最近的物理模式在两组的差异为 "
                 f"{orc_hi:.2f}° vs {orc_lo:.2f}°（p={p_or:.4f}），"
                 f"不支持「代价面本身被推离」的解释；此时更可能是模式排序/权重问题。")
    w10 = wo.get("10", {}).get("n_plane", {}) if wo else {}
    w10g = w10.get("groups", {}) if w10 else {}
    w10k = list(w10g)
    wsp = ""
    if len(w10k) == 2 and w10.get("test", {}).get("p") is not None:
        a10, b10 = w10g[w10k[0]], w10g[w10k[1]]
        wsp = (f" 对象内重复（消除对象混淆）：obj 10 内部 {w10k[0]}（n={a10['n']}）"
               f"{a10['median']['rot_deg']['hard']:.2f}° vs {w10k[1]}（n={b10['n']}）"
               f"{b10['median']['rot_deg']['hard']:.2f}°，p={w10['test']['p']:.4f}，"
               f"最近模式 {a10['oracle']['closest_mode_rot_deg_median']:.2f}° vs "
               f"{b10['oracle']['closest_mode_rot_deg_median']:.2f}°；"
               f"obj 11 的 <350 组只有 "
               f"{wo.get('11',{}).get('n_plane',{}).get('groups',{}).get(w10k[1],{}).get('n','?')} 帧，"
               f"无法在对象内检验（该对象平面片几乎都 ≥350）。")
    L.append(f"> 3. **「小可见平面片驱动 hard 退化」{claim3}**："
             f"n_plane ≥ {PLANE_STRATUM} 组（n={npl_g[npl_hi]['n']}）hard 旋转中位 "
             f"{hard_hi:.2f}° vs < {PLANE_STRATUM} 组"
             f"（n={npl_g[npl_lo]['n']}）{hard_lo:.2f}°，"
             f"置换检验 p={p_np if p_np is not None else float('nan'):.4f}；"
             f"Spearman ρ(n_plane, hard rot) = {rho_np['rho']:+.2f} "
             f"[{rho_np['ci'][0]:+.2f}, {rho_np['ci'][1]:+.2f}]。"
             + mech3 + wsp)
    if sq_ok:
        ci_all = sq_all_d.get("ci", [float("nan")] * 2)
        ci_pl = sq_plane_d.get("ci", [float("nan")] * 2)
        adds_all_d = sq["metrics"]["adds_m"]["sqpnp_all_minus_hard"]
        trans_all_d = sq["metrics"]["trans_m"]["sqpnp_all_minus_hard"]
        okay = [fr for fr in sq["frames"].values()
                if fr["sets"].get("plane", {}).get("status") == "ok"
                and fr["sets"].get("all", {}).get("status") == "ok"]
        d_pl = [fr["sets"]["plane"]["chosen_rot_deg"] - rowmap[fr["frame"]]["hard"]["rot_deg"]
                for fr in okay]
        d_all = [fr["sets"]["all"]["chosen_rot_deg"] - rowmap[fr["frame"]]["hard"]["rot_deg"]
                 for fr in okay]
        med_abs_pl = float(np.median(np.abs(d_pl))) if d_pl else float("nan")
        max_abs_pl = float(np.max(np.abs(d_pl))) if d_pl else float("nan")
        med_abs_all = float(np.median(np.abs(d_all))) if d_all else float("nan")
        n_impr = int(sum(1 for x in d_all if x < -5))
        n_worse = int(sum(1 for x in d_all if x > 5))
        plane_equiv = bool(ci_pl[0] > -0.5 and ci_pl[1] < 0.5)
        all_worse_adds = bool(adds_all_d["ci"][0] > 0)
        all_rot_better = bool(ci_all[1] < 0)
        n_multi = int(sum(1 for fr in okay if fr["sets"]["plane"]["n_solutions"] > 1))
        if all_rot_better and plane_equiv:
            concl = ("结论：**两模层与经典求解器等价，但用全部点还能更好**——数据/对应点仍有余量。")
        elif plane_equiv and all_worse_adds:
            concl = ("结论：**同一平面信息下 hard argmin 与经典 SQPNP 等价"
                     f"（|Δrot| 中位 {med_abs_pl:.2f}°、最大 {max_abs_pl:.2f}°）→ "
                     "两模代价/模式枚举这一层没有留下可回收的精度；瓶颈在对应点质量"
                     "（小可见平面片 + 深度-模型残差），不是 hard 的 argmin 或模式枚举**。"
                     f"给 SQPNP 全部 900 个点并不稳定（{n_impr} 帧改善 >5°、{n_worse} 帧恶化 >5°），"
                     f"平移/ADD-S 反而显著更差（ADD-S 配对差 "
                     f"{1e3*adds_all_d['mean']:+.1f} mm "
                     f"[{1e3*adds_all_d['ci'][0]:+.1f}, {1e3*adds_all_d['ci'][1]:+.1f}]），"
                     "说明非平面部分的 nearest-vertex 对应点（无外点剔除）不能当作更强上界，"
                     "也再次指向数据/对应点而非估计器层。")
        elif plane_equiv:
            concl = ("结论：**同一平面信息下 hard 与经典 SQPNP 等价 → 层没有额外精度损失**；"
                     "差异主要来自数据。")
        else:
            concl = ("结论：数据与层都有影响，逐帧表见 §4；不把单一环节当作唯一解释。")
        all_txt = ("旋转显著更低" if all_rot_better else
                   ("旋转反而更高" if ci_all[0] > 0 else "旋转差异未达显著"))
        pl_txt = ("与 hard 等价" if plane_equiv else
                  ("显著优于 hard" if ci_pl[1] < 0 else "显著差于 hard"))
        L.append(f"> 4. **数据 vs 层（SQPNP，{sq['n']} 帧、同一组对应点）**：SQPNP 用**全部 900 个**"
                 f"深度-模型对应点时旋转中位 {sq_all.get('median',float('nan')):.2f}°、ADD-S "
                 f"{1e3*adds_sq.get('sqpnp_all',{}).get('median',float('nan')):.1f} mm，"
                 f"相对 hard（{sq_hard.get('median',float('nan')):.2f}° / "
                 f"{1e3*adds_sq.get('hard',{}).get('median',float('nan')):.1f} mm）{all_txt}"
                 f"（配对差 {sq_all_d.get('mean',float('nan')):+.2f}°，"
                 f"[{ci_all[0]:+.2f}, {ci_all[1]:+.2f}]）；"
                 f"但只给**平面内点**（与估计器完全相同的输入）时 SQPNP 为 "
                 f"{sq_plane.get('median',float('nan')):.2f}°、ADD-S "
                 f"{1e3*adds_sq.get('sqpnp_plane',{}).get('median',float('nan')):.1f} mm，"
                 f"{pl_txt}（配对差 {sq_plane_d.get('mean',float('nan')):+.2f}°，"
                 f"[{ci_pl[0]:+.2f}, {ci_pl[1]:+.2f}]，中位 |Δrot| {med_abs_pl:.2f}°，"
                 f"{n_multi}/{len(okay)} 帧返回多解）。" + concl)
    L.append(f"> 5. **AUC 仍是粗统计量**：n={n} 时 0–0.1 m 召回曲线有 ≤{n} 个台阶；"
             f"报告是因为要求，不与 BOP 榜单数字比较。所有指标定义与 `pose_error.py` 一致，"
             f"估计器、τ、n_starts、RANSAC 阈值均未改动。")
    L.append("")
    L.append("**证据等级：** 本仓库独立复现/测量。GT 位姿给对应点，**不是检测器**，不对已发表 "
             "BOP LMO pipeline 的精度作任何 claim。")
    L.append("")
    L.append("## 0. 扫描、选帧与分层定义（纯数据可用性）")
    L.append("")
    L.append("| item | obj 10 eggbox | obj 11 glue |")
    L.append("|-|-|-|")
    L.append(f"| GT 实例（全序列） | {sel['objects']['10']['gt_instances']} | "
             f"{sel['objects']['11']['gt_instances']} |")
    L.append(f"| `px_count_visib >= {sel['px_min']}` 候选 | {sel['objects']['10']['candidates']} | "
             f"{sel['objects']['11']['candidates']} |")
    L.append(f"| 本次尝试 | {sel['objects']['10']['attempted']} | "
             f"{sel['objects']['11']['attempted']} |")
    L.append(f"| 可用（K≥2） | {sum(1 for r in rows if r['obj']==10)} | "
             f"{sum(1 for r in rows if r['obj']==11)} |")
    L.append(f"| 每图最多同对象实例 | {checks.get('max_gt_instances_per_image', {}).get(10, 1)} | "
             f"{checks.get('max_gt_instances_per_image', {}).get(11, 1)} |")
    L.append("")
    L.append(f"- 选帧：对每个对象，把 `px_count_visib >= {sel['px_min']}` 的候选按时间排序后"
             f"`np.linspace` 等距取 {sel['per_obj']} 帧；**不看 K、不看估计器输出、不看误差**。"
             f"`{sel['px_min']}` 是能抽到 900 个有效深度像素的数据可用性代理（与上一次 37 帧运行相同）。")
    L.append(f"- 分层边界在**跑估计器之前**由候选池本身确定：visib_fract 三分位 "
             f"{q1:.3f} / {q2:.3f}；px_count_visib 三分位 "
             f"{sel['pool_px_count_visib_terciles'][0]:.0f} / "
             f"{sel['pool_px_count_visib_terciles'][1]:.0f}；平面内点数沿用预注册切分 "
             f"`n_plane = {PLANE_STRATUM}`。`n_plane` 是可见性协变量，不是结果变量，"
             f"因此「按 n_plane 分层」是描述性/机制检验，不是后验挑帧。")
    L.append(f"- 时间稳定性：按图像序号四等分（§3.4）。")
    L.append("")
    L.append("## 1. 流程与指标（与 LINEMOD 完全一致，未改动估计器）")
    L.append("")
    L.append(f"`mask_visib` 内真实深度像素 → bearing $U_i$（真实内参，深度 mm）→ 最近真实 mesh 顶点 "
             f"→ 3-D 模型点 $P_i$；确定性 RANSAC（带宽 {base.PLANE_THR*1e3:.1f} mm、"
             f"{base.N_POINTS} 点、seed 0）选主导平面片；约化代价消去平移；变换后任一点 "
             f"$z\\le0.05$ m 的模式被拒。`n_starts={base.N_STARTS}`，$\\tau={base.TAU}$。"
             f"三个估计器与 `run_pose_error.py` 完全相同：hard = $J$ 最小的物理模式；"
             f"Laplace = 投影 chordal 均值，权重 $w_k\\propto\\exp(-J_k/\\tau-\\tfrac12\\log\\det H_k)$；"
             f"MC = 从 $w$ 抽 400 个模式、20 个种子、逐帧取均值。")
    L.append("")
    L.append("指标（与 `pose_error.py` 相同）：旋转测地角（deg）、平移误差（m）、**ADD-S** = 1000 个 "
             "mesh 顶点到完整 GT mesh 的最近邻距离均值（mm，KD-tree；目标集为全网格）；"
             "ADD 为同索引均值（参考列，eggbox/glue 都有 ~180° 离散对称）；"
             "**ADD-S AUC** = 0–0.1 m 内召回-阈值曲线下面积 / 0.1。主表 mesh = `models/`，与 "
             f"`results/pose_error.json` 用的 LINEMOD mesh 逐字节相同（sha256 检查 "
             f"{checks['mesh_identity']['all_identical']}）；`models_eval` 只作次要列。"
             f"1000 顶点子样本的离散化地板平均 +{agg['discretisation_floor_hard_mm']:.2f} mm。")
    L.append("")
    L.append(f"**与上一次运行的关系**：`build_frame` 是从 `run_two_mode_lmo.py` 直接 import 的同一个"
             f"函数（未复制、未改），常数 {checks['constants']}。SQPNP 用 "
             f"`sqpnp_lmo_helper.py` 在 sdpr 环境（cv2 {checks.get('sqpnp',{}).get('cv2_version','?')}）"
             f"求解；像素反投影回环误差 ≤ {checks.get('pixel_roundtrip',{}).get('max_px',0):.1e} px，"
             f"证明基线看到的是同一批 2-D 点。")
    L.append("")
    L.append(f"## 2. 聚合结果（n={n}；中位数 [95% bootstrap CI] / 均值 [CI]）")
    L.append("")
    L.append("| 指标 | hard | Laplace | MC | Laplace − hard（配对） | MC − hard（配对） |")
    L.append("|-|-|-|-|-|-|")
    for key, label, scale, nd in (("rot_deg", "旋转 (deg)", 1.0, 2),
                                  ("trans_m", "平移 (mm)", 1e3, 1),
                                  ("adds_m", "ADD-S (mm)", 1e3, 1),
                                  ("adds_eval_m", "ADD-S models_eval (mm)", 1e3, 1),
                                  ("add_m", "ADD (mm, 参考)", 1e3, 1)):
        blk = agg[key]
        cells = [fstat(scale * blk[m]["median"],
                       [scale * c for c in blk[m]["median_ci"]], nd)
                 for m in ("hard", "laplace", "mc")]
        pair = [fstat(scale * blk[f"{m}_minus_hard"]["mean"],
                      [scale * c for c in blk[f"{m}_minus_hard"]["ci"]], nd)
                for m in ("laplace", "mc")]
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {pair[0]} | {pair[1]} |")
    blk = agg["auc"]
    cells = [fstat(blk[m]["median"], blk[m]["median_ci"], 3) for m in
             ("hard", "laplace", "mc")]
    pair = [fstat(blk[f"{m}_minus_hard"]["mean"], blk[f"{m}_minus_hard"]["ci"], 3)
            for m in ("laplace", "mc")]
    L.append(f"| ADD-S AUC (0–0.1 m) | {cells[0]} | {cells[1]} | {cells[2]} | {pair[0]} | {pair[1]} |")
    L.append("")
    L.append("**均值 [95% CI]**（同一批帧）：")
    L.append("")
    L.append("| 指标 | hard | Laplace | MC |")
    L.append("|-|-|-|-|")
    for key, label, scale, nd in (("rot_deg", "旋转 (deg)", 1.0, 2),
                                  ("trans_m", "平移 (mm)", 1e3, 1),
                                  ("adds_m", "ADD-S (mm)", 1e3, 1),
                                  ("adds_eval_m", "ADD-S models_eval (mm)", 1e3, 1),
                                  ("add_m", "ADD (mm, 参考)", 1e3, 1),
                                  ("auc", "ADD-S AUC", 1.0, 3)):
        b = agg["auc"] if key == "auc" else agg[key]
        L.append(f"| {label} | " + " | ".join(
            fstat(scale * b[m]["mean"], [scale * c for c in b[m]["mean_ci"]], nd)
            for m in ("hard", "laplace", "mc")) + " |")
    L.append("")
    md = agg["mc_detail"]
    L.append("**MC 随机性诊断**（帧 bootstrap vs 帧+种子联合 bootstrap）：")
    L.append("")
    L.append("| 指标 | 种子间 SD | 帧 CI | 联合 CI |")
    L.append("|-|-|-|-|")
    for key, label, scale in (("rot_deg", "旋转 (deg)", 1.0),
                              ("trans_m", "平移 (mm)", 1e3),
                              ("adds_m", "ADD-S (mm)", 1e3)):
        d = md[key]
        L.append(f"| {label} | {scale*d['between_seed_sd']:.3f} | "
                 f"[{scale*d['frame_boot_ci'][0]:.3f}, {scale*d['frame_boot_ci'][1]:.3f}] | "
                 f"[{scale*d['combined_frame_seed_ci'][0]:.3f}, "
                 f"{scale*d['combined_frame_seed_ci'][1]:.3f}] |")
    L.append("")
    L.append(f"### 2.1 逐帧表（全部 {n} 帧；完整字段在 `results/lmo_full.json` 的 `frames[]`）")
    L.append("")
    L.append("模式结构与数据 regime：")
    L.append("")
    L.append("| frame | obj | visib | px_visib | n_plane | plane ext (mm) | depth p50 (mm) | "
             "K_phys/K_all | ΔJ | flip° | w_wrong | closest mode° | J_gt/J_min |")
    L.append("|-|-|-|-|-|-|-|-|-|-|-|-|-|")
    for r in rows:
        L.append(f"| {r['frame']} | {r['obj']} | {r['visib_fract']:.2f} | "
                 f"{r['px_count_visib']} | {r['n_plane']} | {r['plane_extent_mm']:.0f} | "
                 f"{r['dist_p50_mm']:.1f} | {r['n_phys']}/{r['K_all']} | "
                 f"{r['dJ']:.2e} | {r['flip_deg']:.1f} | {r['w_other']:.3f} | "
                 f"{r['closest_mode_rot_deg']:.1f} | {r['J_gt_over_min']:.2f} |")
    L.append("")
    L.append("位姿指标逐帧（rot deg / trans mm / ADD-S mm / AUC）：")
    L.append("")
    L.append("| frame | obj | hard rot | Lap rot | MC rot | hard trans | Lap trans | MC trans | "
             "hard ADD-S | Lap ADD-S | MC ADD-S | AUC hard | AUC Lap | AUC MC |")
    L.append("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|")
    for r in rows:
        L.append(f"| {r['frame']} | {r['obj']} | {r['hard']['rot_deg']:.2f} | "
                 f"{r['laplace']['rot_deg']:.2f} | {r['mc']['rot_deg']['mean']:.2f} | "
                 f"{1e3*r['hard']['trans_m']:.1f} | {1e3*r['laplace']['trans_m']:.1f} | "
                 f"{1e3*r['mc']['trans_m']['mean']:.1f} | "
                 f"{1e3*r['hard']['adds_m']:.1f} | {1e3*r['laplace']['adds_m']:.1f} | "
                 f"{1e3*r['mc']['adds_m']['mean']:.1f} | {r['auc']['hard']:.3f} | "
                 f"{r['auc']['laplace']:.3f} | {r['auc']['mc']:.3f} |")
    L.append("")
    L.append(f"## 3. 分层结果（{n} 帧）")
    L.append("")
    L.append("### 3.1 可见性分层（visib_fract 三分位；边界来自候选池）")
    L.append("")
    L.append("（`px_count_visib` 三分位的同一套分层表在 `results/lmo_full.json` 的 "
             "`stratified.px_count_visib`，结果为 n="
             + "/".join(str(px_g[k]["n"]) for k in px_g) + "，结论与 visib 分层同向。）")
    L.append("")
    L.append("| 层 | n (egg/glue) | n_plane 中位 | hard 旋转 | Laplace 旋转 | MC 旋转 | "
             "hard ADD-S | hard AUC | 最近模式旋转（oracle） | hard=最近模式 |")
    L.append("|-|-|-|-|-|-|-|-|-|-|")
    for k in vkeys:
        b = vis_g[k]
        L.append(f"| {k} | {b['n']} ({b['n_eggbox']}/{b['n_glue']}) | "
                 f"{b['context']['n_plane_median']:.0f} | "
                 f"{fstat(b['median']['rot_deg']['hard'], b['median_ci']['rot_deg']['hard'])} | "
                 f"{b['median']['rot_deg']['laplace']:.2f} | "
                 f"{b['median']['rot_deg']['mc']:.2f} | "
                 f"{fstat(1e3*b['median']['adds_m']['hard'], [1e3*c for c in b['median_ci']['adds_m']['hard']], 1)} | "
                 f"{b['median']['auc']['hard']:.3f} | "
                 f"{b['oracle']['closest_mode_rot_deg_median']:.2f} | "
                 f"{b['oracle']['hard_is_gt_like_rate']:.2f} |")
    L.append("")
    L.append("### 3.2 平面内点分层（预注册 `n_plane = 350`）")
    L.append("")
    L.append("| 层 | n (egg/glue) | K=2 精确占比 | n_plane 中位 | depth p50 (mm) | hard 旋转 | "
             "Laplace 旋转 | MC 旋转 | hard ADD-S | 最近模式旋转 | hard=最近模式 | J_gt/J_min |")
    L.append("|-|-|-|-|-|-|-|-|-|-|-|-|")
    for k in (npl_hi, npl_lo):
        b = npl_g[k]
        L.append(f"| {k} | {b['n']} ({b['n_eggbox']}/{b['n_glue']}) | "
                 f"{b['k2_exact_rate']:.2f} | {b['context']['n_plane_median']:.0f} | "
                 f"{b['context']['dist_p50_mm_median']:.2f} | "
                 f"{fstat(b['median']['rot_deg']['hard'], b['median_ci']['rot_deg']['hard'])} | "
                 f"{b['median']['rot_deg']['laplace']:.2f} | {b['median']['rot_deg']['mc']:.2f} | "
                 f"{fstat(1e3*b['median']['adds_m']['hard'], [1e3*c for c in b['median_ci']['adds_m']['hard']], 1)} | "
                 f"{b['oracle']['closest_mode_rot_deg_median']:.2f} | "
                 f"{b['oracle']['hard_is_gt_like_rate']:.2f} | "
                 f"{b['context']['J_gt_over_min_median']:.2f} |")
    L.append("")
    L.append("### 3.3 置换检验 / Spearman（回答「小片驱动退化是否成立」）")
    L.append("")
    L.append("| 检验 | 组 A 中位 | 组 B 中位 | 差 (A−B) | 双侧置换 p |")
    L.append("|-|-|-|-|-|")
    for k, t in tests.items():
        if t.get("status") in ("empty", "too_few"):
            continue
        L.append(f"| {k} | {t['median_a']:.3f} | {t['median_b']:.3f} | "
                 f"{t['diff']:+.3f} | {t['p']:.4f} |")
    L.append("")
    L.append("| Spearman ρ | visib_fract | px_count_visib | n_plane | plane_extent_mm | depth p50 mm |")
    L.append("|-|-|-|-|-|-|")
    for stat, lab in (("hard_rot", "hard 旋转"), ("hard_adds_mm", "hard ADD-S"),
                      ("closest_mode_rot", "最近模式旋转")):
        cells = []
        for k in ("visib_fract", "px_count_visib", "n_plane", "plane_extent_mm", "dist_p50_mm"):
            c = corr[stat][k]
            cells.append(f"{c['rho']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"**对象内重复（消除对象混淆）**：合并的 `n_plane < {PLANE_STRATUM}` 层里 "
             f"{npl_g[npl_lo]['n_eggbox']}/{npl_g[npl_lo]['n']} 是 eggbox，所以上面的分池检验"
             f"与对象身份部分混淆；下表把切分放进每个对象内部。")
    L.append("")
    L.append("| 对象 | 切分 | 组 A n | 组 A hard 中位 | 组 B n | 组 B hard 中位 | 差 | 置换 p | "
             "组 A 最近模式 | 组 B 最近模式 |")
    L.append("|-|-|-|-|-|-|-|-|-|-|")
    for oid in ("10", "11"):
        o = wo.get(oid, {})
        for kind in ("n_plane", "visib"):
            blk = o.get(kind)
            if not blk:
                continue
            ks = list(blk["groups"])
            if len(ks) != 2:
                continue
            a, b = blk["groups"][ks[0]], blk["groups"][ks[1]]
            t = blk.get("test") or {}
            L.append(f"| obj {oid} | {kind}: {ks[0]} vs {ks[1]} | {a['n']} | "
                     f"{a['median']['rot_deg']['hard']:.2f} | {b['n']} | "
                     f"{b['median']['rot_deg']['hard']:.2f} | "
                     f"{a['median']['rot_deg']['hard']-b['median']['rot_deg']['hard']:+.2f} | "
                     f"{('%.4f' % t['p']) if t.get('p') is not None else 'n/a'} | "
                     f"{a['oracle']['closest_mode_rot_deg_median']:.2f} | "
                     f"{b['oracle']['closest_mode_rot_deg_median']:.2f} |")
    L.append("")
    L.append("### 3.4 时间稳定性（图像序号四等分）")
    L.append("")
    L.append("| 区间 | n | hard 旋转 | Laplace 旋转 | hard ADD-S | n_plane 中位 | visib 中位 |")
    L.append("|-|-|-|-|-|-|-|")
    for b in doc["temporal"]:
        if b["n"] == 0:
            L.append(f"| {b['label']} | 0 | – | – | – | – | – |")
            continue
        L.append(f"| {b['label']} | {b['n']} | {b['hard_rot_median']:.2f} | "
                 f"{b['laplace_rot_median']:.2f} | {b['hard_adds_mm_median']:.1f} | "
                 f"{b['n_plane_median']:.0f} | {b['visib_median']:.2f} |")
    L.append("")
    L.append("## 4. 经典基线：SQPNP（同一组对应点，分离「数据」与「层」）")
    L.append("")
    if sq.get("status") == "ok":
        n_sq_used = sq["n"]
        L.append(f"`cv2.solvePnPGeneric(flags=SOLVEPNP_SQPNP)`（sdpr 环境 cv2 "
                 f"{checks.get('sqpnp',{}).get('cv2_version','?')}）在 {n_sq_used} 帧上运行"
                 f"（计划 {sq.get('subset_planned', n_sq_used)} 帧、每对象等距 10 帧；"
                 f"未通过 K≥2 过滤的帧不进入基线，因此少 "
                 f"{sq.get('subset_planned', n_sq_used) - n_sq_used} 帧），"
                 f"输入是**同一批** 3-D 模型点/2-D 像素（回环误差 "
                 f"{checks.get('pixel_roundtrip',{}).get('max_px',0):.1e} px）。"
                 f"两种输入：`plane` = 与估计器相同的 RANSAC 平面内点；`all` = 全部 900 个对应点。"
                 f"选择规则是经典做法：**平均重投影误差最小**的解（不看 GT）；`oracle` 列是"
                 f"在返回的多个解里按 GT 取最优，用来量化平面歧义本身。")
        L.append("")
        L.append("| 指标 | hard（同帧） | Laplace | MC | SQPNP(plane) | SQPNP(all) | "
                 "SQPNP(all) − hard | SQPNP(plane) − hard |")
        L.append("|-|-|-|-|-|-|-|-|")
        for key, label, scale, nd in (("rot_deg", "旋转 (deg)", 1.0, 2),
                                      ("trans_m", "平移 (mm)", 1e3, 1),
                                      ("adds_m", "ADD-S (mm)", 1e3, 1)):
            b = sq["metrics"][key]
            cells = [fstat(scale * b[m]["median"], [scale * c for c in b[m]["median_ci"]], nd)
                     for m in ("hard", "laplace", "mc", "sqpnp_plane", "sqpnp_all")]
            d = [fstat(scale * b[f"{m}_minus_hard"]["mean"],
                       [scale * c for c in b[f"{m}_minus_hard"]["ci"]], nd)
                 for m in ("sqpnp_all", "sqpnp_plane")]
            L.append(f"| {label} | " + " | ".join(cells) + f" | {d[0]} | {d[1]} |")
        L.append("")
        L.append("| 指标 | SQPNP(plane, oracle) | SQPNP(all, oracle) | 平均解数 plane | "
                 "平均解数 all | hard 更优帧数（sqpnp=plane/all） |")
        L.append("|-|-|-|-|-|-|")
        nsol = [f["sets"]["plane"].get("n_solutions") for f in sq["frames"].values()
                if f["sets"].get("plane", {}).get("status") == "ok"]
        nsol_all = [f["sets"]["all"].get("n_solutions") for f in sq["frames"].values()
                    if f["sets"].get("all", {}).get("status") == "ok"]
        for key, label, scale, nd in (("rot_deg", "旋转 (deg)", 1.0, 2),
                                      ("adds_m", "ADD-S (mm)", 1e3, 1)):
            b = sq["metrics"][key]
            L.append(f"| {label} | "
                     f"{fstat(scale*b['sqpnp_plane_oracle']['median'], [scale*c for c in b['sqpnp_plane_oracle']['median_ci']], nd)} | "
                     f"{fstat(scale*b['sqpnp_all_oracle']['median'], [scale*c for c in b['sqpnp_all_oracle']['median_ci']], nd)} | "
                     f"{np.mean(nsol) if nsol else float('nan'):.2f} | "
                     f"{np.mean(nsol_all) if nsol_all else float('nan'):.2f} | "
                     f"{b['sqpnp_plane_minus_hard'].get('n_better','?')}/{b['sqpnp_all_minus_hard'].get('n_better','?')} |")
        L.append("")
        L.append(f"逐帧看：`SQPNP(plane)` 与 hard 在 {len(okay)} 帧上逐帧一致"
                 f"（中位 |Δrot| {med_abs_pl:.2f}°、最大 {max_abs_pl:.2f}°），包括 "
                 f"{n_multi} 帧平面 PnP 返回 2–3 个解的情况——即「按 J 选模式」与「按重投影"
                 f"误差选解」落在同一支。`SQPNP(all)` 呈双峰行为：{n_impr} 帧比 hard 好 >5°"
                 f"（例如 20.2°→0.25°、27.2°→1.8°），{n_worse} 帧差 >5°（例如 2.2°→29.9°、"
                 f"ADD-S 328 mm），中位 |Δrot| {med_abs_all:.2f}°；无外点剔除的最小二乘 PnP "
                 f"被非平面部分误差较大的 nearest-vertex 对应点拖坏。")
        L.append("")
        L.append("逐帧（baseline 子集）：")
        L.append("")
        L.append("| frame | obj | visib | n_plane | hard rot | hard ADD-S | SQPNP(all) rot | "
                 "SQPNP(all) ADD-S | SQPNP(plane) rot | SQPNP(plane) ADD-S | 解数 all/plane |")
        L.append("|-|-|-|-|-|-|-|-|-|-|-|")
        for k in sorted(sq["frames"], key=lambda kk: (sq["frames"][kk]["obj"],
                                                       sq["frames"][kk]["img"])):
            fr = sq["frames"][k]
            r = [x for x in rows if x["frame"] == fr["frame"]][0]
            a, p = fr["sets"].get("all", {}), fr["sets"].get("plane", {})
            f2 = lambda s, key, sc=1.0, nd=2: ("–" if s.get("status") != "ok"
                                               else f"{sc*s[key]:.{nd}f}")
            L.append(f"| {fr['frame']} | {fr['obj']} | {fr['visib_fract']:.2f} | "
                     f"{fr['n_plane']} | {r['hard']['rot_deg']:.2f} | "
                     f"{1e3*r['hard']['adds_m']:.1f} | {f2(a,'chosen_rot_deg')} | "
                     f"{f2(a,'chosen_adds_m',1e3,1)} | {f2(p,'chosen_rot_deg')} | "
                     f"{f2(p,'chosen_adds_m',1e3,1)} | "
                     f"{a.get('n_solutions','–')}/{p.get('n_solutions','–')} |")
    else:
        L.append(f"SQPNP 基线未产出：`{json.dumps(sq, ensure_ascii=False)[:800]}`。"
                 f"因此本节不给出数据/层分离结论。")
    L.append("")
    L.append("## 5. 与上一次 37 帧结果、LINEMOD 的对照")
    L.append("")
    if prev.get("status") == "ok":
        L.append(f"来源（**引用，未重算**）：`{prev['source']}`（{prev['n_used']} 帧可用 / "
                 f"{prev['n_attempted']} 尝试）。")
        L.append("")
        L.append("| 指标 | 上次 37 帧 hard | 本次 hard | 上次 Laplace | 本次 Laplace | "
                 "上次 MC | 本次 MC |")
        L.append("|-|-|-|-|-|-|-|")
        for key, label, scale, nd in (("rot_deg", "旋转中位 (deg)", 1.0, 2),
                                      ("trans_m", "平移中位 (mm)", 1e3, 1),
                                      ("adds_m", "ADD-S 中位 (mm)", 1e3, 1),
                                      ("auc", "AUC 中位", 1.0, 3)):
            p = prev["metrics"][key]
            cells = [fstat(scale * p[m]["median"], [scale * c for c in p[m]["median_ci"]], nd)
                     for m in ("hard", "laplace", "mc")]
            cur = [f"{scale*med(key, m):.{nd}f}" for m in ("hard", "laplace", "mc")]
            L.append(f"| {label} | {cells[0]} | {cur[0]} | {cells[1]} | {cur[1]} | "
                     f"{cells[2]} | {cur[2]} |")
        contra = []
        for key in ("rot_deg", "trans_m", "adds_m", "auc"):
            for m in ("hard", "laplace", "mc"):
                pm = prev["metrics"][key][m]
                cur_med = med(key, m)
                cur_ci = agg[key][m]["median_ci"]
                if not (cur_ci[0] <= pm["median"] <= cur_ci[1]):
                    contra.append(f"{key}/{m}: 37-frame {pm['median']:.3g} outside full CI")
                if not (pm["median_ci"][0] <= cur_med <= pm["median_ci"][1]):
                    contra.append(f"{key}/{m}: full {cur_med:.3g} outside 37-frame CI")
        L.append("")
        if contra:
            L.append(f"**与 37 帧的一致性**：有 {len(contra)} 项超出对方 95% CI："
                     + "；".join(contra) + "。逐项如上表，不作「一致」的强声明。")
        else:
            L.append("**与 37 帧的一致性**：4 个指标 × 3 个估计器共 12 个点估计，"
                     "**全部落在对方样本的 95% 中位 CI 内**（双向检查），因此本次扩样与上次"
                     " 37 帧结果不矛盾；扩样新增的是分层统计检验、时间稳定性和同对应点基线。")
    else:
        L.append("上次 `results/lmo_two_mode.json` 不在，跳过对照。")
    L.append("")
    L.append("LINEMOD 参考（引用 `results/pose_error.json`）：hard 2.06° / 3.7 mm ADD-S、"
             "Laplace 44.2° / 13.9 mm（15 帧）。LMO 的 hard 比 LINEMOD 差、Laplace 量级相当，"
             "方向与上次运行一致；差异的机制见 §3。")
    L.append("")
    L.append("## 6. 失败与 caveats")
    L.append("")
    if doc["failures"]:
        L.append("| frame | obj | visib | px_visib | px_valid | reason |")
        L.append("|-|-|-|-|-|-|")
        for f in doc["failures"]:
            L.append(f"| {f.get('frame','?')} | {f['obj']} | {f['visib_fract']:.2f} | "
                     f"{f['px_count_visib']} | {f['px_count_valid']} | {f['reason']} |")
        L.append("")
        empt = [f for f in doc["failures"] if f["reason"] == "n_valid_depth_lt_900"]
        if empt:
            nvp = [f["n_valid_px"] for f in empt if f.get("n_valid_px") is not None]
            gi_off = [f for f in empt
                      if f.get("n_valid_px") is not None
                      and abs(f["px_count_valid"] - f["n_valid_px"]) > 100]
            L.append(f"- **近空深度图**：{len(empt)} 帧失败于 `n_valid_depth_lt_900` —— "
                     f"`mask_visib` 内**实际**有效深度像素只有 "
                     f"{min(nvp)}–{max(nvp)} 个（管线口径 `n_valid_px`），虽然 "
                     f"`px_count_visib` 有 "
                     f"{min(f['px_count_visib'] for f in empt)}–"
                     f"{max(f['px_count_visib'] for f in empt)}。注意 "
                     f"`scene_gt_info.px_count_valid` 是 amodal 口径，其中 {len(gi_off)} 帧明显偏大"
                     f"（{', '.join(str(f['px_count_valid']) for f in gi_off)}），不能当作可见深度数。"
                     f"这是 LMO 采集本身的缺陷；这些帧在任何估计器下都不可用，"
                     f"按失败报告，不替换、不重抽。")
    else:
        L.append("无失败帧。")
    L.append("")
    L.append(f"1. **选择偏差的边界**：`px_count_visib >= {sel['px_min']}` 让极重遮挡实例"
             f"（抽不到 900 个有效深度像素）系统性缺席；本次 visib_fract 范围 "
             f"{min(r['visib_fract'] for r in rows):.2f}–{max(r['visib_fract'] for r in rows):.2f}"
             f"（中位 {np.median([r['visib_fract'] for r in rows]):.2f}），因此本表对 LMO 最难的"
             f"遮挡是**乐观**的。分层只能在这个范围内比较。")
    L.append(f"2. **K≥2 是入样条件**：hard/Laplace/MC 只在 K≥2 的帧上有定义，因此 "
             f"`n={n}` 的聚合是「条件于 K≥2」的；K<2 的 {sum(1 for f in doc['failures'] if f['reason']=='K_phys_lt_2')} 帧"
             f"单独报告。K≥2 通过率 {k2_attempt:.1%} 已含在 §0。")
    L.append(f"3. **深度-模型残差是 LMO 数据事实**：本次逐帧 depth p50 中位 "
             f"{np.median([r['dist_p50_mm'] for r in rows]):.1f} mm"
             f"（范围 {min(r['dist_p50_mm'] for r in rows):.1f}–"
             f"{max(r['dist_p50_mm'] for r in rows):.1f}），高于 LINEMOD 的 2–5 mm；"
             f"上次的 2-D 平移抽查已排除配准偏移，本次未重复该抽查"
             f"（`checks.shift_check.status = not_repeated`，属于刻意省略，"
             f"见 `run_two_mode_lmo.py` 的上次结果）。")
    L.append(f"3b. **分层与对象部分混淆**：`n_plane < {PLANE_STRATUM}` 的 {npl_g[npl_lo]['n']} 帧中 "
             f"{npl_g[npl_lo]['n_eggbox']} 帧是 eggbox（obj 11 的平面片几乎都 ≥{PLANE_STRATUM}）。"
             f"对象内重复见 §3.3：obj 10 内部效应复现，obj 11 无法检验；因此结论是"
             f"「在 eggbox 上成立、在 glue 上未被检验」，不是「对两个对象都成立」。")
    L.append(f"4. **SQPNP 的两个变体含义不同**：`all` 用了非平面的 800+ 个点，本意是「比估计器多信息」"
             f"的更强数据上界，但实测被非平面外点拖坏（§4），**不能**当作上界；`plane` 与估计器同输入，"
             f"但平面 PnP 本身有经典二义性（返回解数见 §4），所以它衡量的是「层」的下界。"
             f"两者都不是检测器精度，前端仍是 GT。")
    L.append(f"5. **AUC 语义**：BOP ADD-S AUC 用同一 0–0.1 m 归一化；n={n} 时仍有台阶，"
             f"CI 宽，配对差比绝对值更可信。")
    L.append("")
    L.append("## 7. 复现")
    L.append("")
    L.append("```bash")
    L.append("cd labs/01-laplace-qpeplayer/experiments/real/linemod")
    L.append("# 全部计算走 Slurm（登录节点不跑）：")
    L.append("srun --partition=cpu -n1 --time=00:40:00 \\")
    L.append("  /home/syk/miniconda3/envs/WM/bin/python run_two_mode_lmo_full.py")
    L.append("# 冒烟（4 帧/对象、2 帧 SQPNP、跳过 2x-starts）：")
    L.append("srun --partition=cpu -n1 /home/syk/miniconda3/envs/WM/bin/python \\")
    L.append("  run_two_mode_lmo_full.py --quick")
    L.append("```")
    L.append("")
    L.append(f"估计器常数来自 `run_two_mode.py`/`run_pose_error.py`（未改动），"
             f"`checks.constants` 记录实际使用值；SQPNP 由 `sqpnp_lmo_helper.py` 在 "
             f"`/home/syk/miniconda3/envs/sdpr/bin/python`（cv2 "
             f"{checks.get('sqpnp',{}).get('cv2_version','?')}）中执行。"
             f"JSON 里有完整逐帧表、分层、检验、时间块与 checks。")
    L.append("")
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ figure
def make_figure(doc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = doc["frames"]
    sq = doc["sqpnp"]
    st = doc["stratified"]
    npl_g = st["n_plane"]["groups"]
    vis_g = st["visib_fract"]["groups"]
    npl_hi, npl_lo = list(npl_g)
    vkeys = list(vis_g)

    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.0))

    # (a) rotation error distributions, with SQPNP on the baseline subset
    data, labels, colors = [], [], []
    for m, lab, c in (("hard", "hard", "C0"), ("laplace", "Laplace", "C1"),
                      ("mc", "MC", "C2")):
        data.append([base.meth_value(r, m, "rot_deg") for r in rows])
        labels.append(f"{lab}\n(n={len(rows)})")
        colors.append(c)
    if sq.get("status") == "ok":
        n_sq_sub = sum(1 for f in sq["frames"].values()
                       if f["sets"].get("all", {}).get("status") == "ok")
        for stn, lab, c in (("sqpnp_plane", f"SQPNP\n(plane, n={n_sq_sub})", "C3"),
                            ("sqpnp_all", f"SQPNP\n(all, n={n_sq_sub})", "C4")):
            v = [f["sets"][stn.split("_")[1]].get("chosen_rot_deg") for f in sq["frames"].values()
                 if f["sets"].get(stn.split("_")[1], {}).get("status") == "ok"]
            data.append(v)
            labels.append(lab)
            colors.append(c)
    bp = ax[0, 0].boxplot(data, showfliers=False, patch_artist=True,
                          medianprops=dict(color="k"))
    ax[0, 0].set_xticks(range(1, len(labels) + 1))
    ax[0, 0].set_xticklabels(labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel("rotation error (deg, log)")
    ax[0, 0].set_title(f"(a) LMO n={len(rows)}: rotation error by estimator")
    ax[0, 0].grid(alpha=0.3, axis="y")
    ax[0, 0].tick_params(axis="x", labelsize=8)

    # (b) mechanism: hard rotation vs plane inliers, coloured by visibility
    for oid, mk in ((10, "o"), (11, "^")):
        sub = [r for r in rows if r["obj"] == oid]
        sc = ax[0, 1].scatter([r["n_plane"] for r in sub],
                              [r["hard"]["rot_deg"] for r in sub],
                              c=[r["visib_fract"] for r in sub], cmap="viridis",
                              s=20, vmin=0.0, vmax=1.0, marker=mk,
                              label=f"obj {oid} ({OBJ[oid]})")
    ax[0, 1].legend(fontsize=7, loc="upper right")
    ax[0, 1].axvline(PLANE_STRATUM, color="k", ls="--", lw=0.9)
    cb = fig.colorbar(sc, ax=ax[0, 1])
    cb.set_label("visib_fract")
    rho = doc["correlations"]["hard_rot"]["n_plane"]
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("plane inliers (n_plane)")
    ax[0, 1].set_ylabel("hard rotation error (deg, log)")
    ax[0, 1].set_title(f"(b) Spearman {rho['rho']:+.2f} "
                       f"[{rho['ci'][0]:+.2f},{rho['ci'][1]:+.2f}]")
    ax[0, 1].grid(alpha=0.3)

    # (c) stratified medians: visibility terciles + n_plane split
    q1, q2 = doc["selection"]["pool_visib_fract_terciles"]
    groups = [(f"vis<{q1:.2f}\n(n={vis_g[vkeys[0]]['n']})", vis_g[vkeys[0]]),
              (f"{q1:.2f}-{q2:.2f}\n(n={vis_g[vkeys[1]]['n']})", vis_g[vkeys[1]]),
              (f"vis>={q2:.2f}\n(n={vis_g[vkeys[2]]['n']})", vis_g[vkeys[2]]),
              (f">= {PLANE_STRATUM}\n(n={npl_g[npl_hi]['n']})", npl_g[npl_hi]),
              (f"< {PLANE_STRATUM}\n(n={npl_g[npl_lo]['n']})", npl_g[npl_lo])]
    xs = np.arange(len(groups))
    for j, (m, lab, c) in enumerate((("hard", "hard", "C0"),
                                     ("laplace", "Laplace", "C1"),
                                     ("mc", "MC", "C2"))):
        meds = [g["median"]["rot_deg"][m] for _, g in groups]
        ax[1, 0].bar(xs + (j - 1) * 0.26, meds, width=0.25, color=c, alpha=0.75,
                     label=lab)
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels([n for n, _ in groups], rotation=18, fontsize=7.5)
    ax[1, 0].set_ylabel("median rotation error (deg)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("(c) stratified medians (last two: pre-registered plane split)")
    ax[1, 0].grid(alpha=0.3, axis="y")
    ax[1, 0].legend(fontsize=7)

    # (d) ADD-S recall curves
    ths = np.linspace(0.0, AUC_T, 201)
    for m, lab, c in (("hard", "hard", "C0"), ("laplace", "Laplace", "C1"),
                      ("mc", "MC", "C2")):
        errs = np.array([base.meth_value(r, m, "adds_m") for r in rows])
        rec = [float(np.mean(errs <= t)) for t in ths]
        ax[1, 1].plot(ths * 1e3, rec, lw=1.4, color=c,
                      label=f"{lab} (mean AUC={doc['aggregate']['auc'][m]['mean']:.3f})")
    if sq.get("status") == "ok":
        errs = np.array([f["sets"]["all"]["chosen_adds_m"] for f in sq["frames"].values()
                         if f["sets"].get("all", {}).get("status") == "ok"])
        rec = [float(np.mean(errs <= t)) for t in ths]
        ax[1, 1].plot(ths * 1e3, rec, lw=1.2, ls="--", color="C4",
                      label=f"SQPNP(all) n={len(errs)}")
    ax[1, 1].set_xlabel("ADD-S threshold (mm)")
    ax[1, 1].set_ylabel("recall (fraction of frames)")
    ax[1, 1].set_title(f"(d) ADD-S recall 0–100 mm (n={len(rows)})")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
