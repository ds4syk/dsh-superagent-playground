#!/usr/bin/env python3
"""Real-data rigorous certificates for P2 (rounds S11-S19).

Two jobs:

1. `collect <ds>` — replay `real/bop/run_bop.py` with `RIGOROUS_ONLY=1`
   (bit-identical instance/solver RNG stream to the published `raw.json`),
   computing for every ALM row three certificate configurations with
   `repair_dual=True, schur=True`:
       base : no cuts
       rlt  : zero-dimensional RLT  g_i g_j >= 0
       brlt : bound-RLT             g_i (y_k - lo_k) <= 0 (the useful one)
   Each config yields `value` (solver value, NOT a bound), `rigorous`
   (dual-repair lower bound), `rigorous_schur` (P2-Cert 13-dim Schur bound),
   so `repair_dual=False` is just `value`.
   `collect` then aggregates to `experiments/results/rigorous_<ds>.{json,md}`.

2. `depth-tilt` — for the LM-O rows that stay uncertified under brlt, add a
   positive-depth constraint (`b=-e3`) and an upright/roll-pitch cone around
   the table normal (`n^T R(q) u >= cos(alpha)`, i.e. the camera-frame form of
   `q1^2+q2^2 <= sin^2(alpha/2)`), re-solve ALM and re-certify.  This changes
   the feasible set: it certifies the *augmented physical* problem, not the
   original 16-plane-only one (reported explicitly).

3. `final` — merge per-dataset JSONs into `results/rigorous_real.{json,md}`
   and the 9-round log `results/round_rigorous.md` is written separately.

Usage
-----
    python code/run_rigorous_real.py collect lm
    python code/run_rigorous_real.py collect lmo
    python code/run_rigorous_real.py collect ycbv
    python code/run_rigorous_real.py depth-tilt
    python code/run_rigorous_real.py final

All computations are small (13x13 SDPs); run directly on the login node with
`nice -n 10` and single-thread BLAS.  No Slurm is used (cpu partition busy).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent          # experiments/code
EXP = HERE.parent                                # experiments
REAL = EXP / "real"
RESULTS = EXP / "results"
BOP = REAL / "bop"
CODE = HERE

sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402
from certificate import shor_bound  # noqa: E402

DATA_ROOT = Path(os.environ.get("BOP_DATA_ROOT", "/mnt/storage1/syk/data/bop"))
PY = sys.executable

DATASETS = {
    # env reproduces the published n=90/72/84 runs (see each results/meta.json)
    "lm": dict(out=REAL / "linemod" / "results",
               env=dict(BOP_MAX_FRAMES="15", BOP_MAX_IMG="2",
                        BOP_OBJ_IDS=",".join(str(i) for i in range(1, 16)))),
    "lmo": dict(out=BOP / "results" / "lmo",
                env=dict(BOP_MAX_FRAMES="1", BOP_MAX_IMG="8")),
    "ycbv": dict(out=BOP / "results" / "ycbv",
                 env=dict(BOP_MAX_FRAMES="12", BOP_MAX_IMG="2")),
}
CONFIGS = ["base", "rlt", "brlt"]
CONFIG_LABEL = {"base": "base 14x14", "rlt": "zero-dim RLT g_i g_j",
                "brlt": "bound-RLT g_i(y_k-lo_k)"}
GAP_TOL = 1e-6          # J - bound <= GAP_TOL  => certified
SOUND_TOL = 1e-8        # numerical slack for "bound <= J"


# ---------------------------------------------------------------------------
# small stats helpers
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(a, b):
    if a + b == 0:
        return 1.0, a, b
    return float(binomtest(a, a + b, 0.5).pvalue), a, b


def _finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _gap_stats(J, b):
    vals = np.asarray([J[i] - b[i] for i in range(len(J)) if _finite(b[i])],
                      dtype=float)
    if vals.size == 0:
        return dict(n=0)
    return dict(n=int(vals.size), min=float(vals.min()),
                median=float(np.median(vals)), max=float(vals.max()),
                mean=float(vals.mean()),
                n_negative=int((vals < -SOUND_TOL).sum()),
                n_le_tol=int((vals <= GAP_TOL).sum()))


def _time_stats(ts):
    v = np.asarray([t for t in ts if _finite(t)], dtype=float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()),
                median=float(np.median(v)), max=float(v.max()))


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def collect(ds, extra_env=None):
    cfg = DATASETS[ds]
    out = cfg["out"]
    env = dict(os.environ)
    env.update(cfg["env"])
    env.update(dict(RIGOROUS_ONLY="1", BOP_OUT_DIR=str(out),
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"))
    if extra_env:
        env.update(extra_env)
    cmd = ["nice", "-n", "10", PY, str(BOP / "run_bop.py"), ds]
    print("+", " ".join(f"{k}={v}" for k, v in sorted(cfg["env"].items())),
          "RIGOROUS_ONLY=1", flush=True)
    print("+", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, check=True)
    print("collect wall %.1f s" % (time.perf_counter() - t0), flush=True)
    return aggregate(ds)


def _load(ds):
    out = DATASETS[ds]["out"] / "rigorous"
    meta = json.load(open(out / "rigorous_rows.json"))["meta"]
    rows = json.load(open(out / "rigorous_rows.json"))["rows"]
    arrays = dict(np.load(out / "rigorous_arrays.npz"))
    return meta, rows, arrays


def _reproduction_check(ds, rows):
    raw_path = DATASETS[ds]["out"] / "raw.json"
    if not raw_path.exists():
        return dict(ok=False, reason="raw.json missing")
    raw = json.load(open(raw_path))
    ref = {(r["frame"], r["img"], r["obj"], float(r["bias_px"])): r["J"]
           for r in raw if r["method"] == "ALM-QPEP"}
    diffs, missing = [], 0
    for r in rows:
        key = (r["frame"], r["img"], r["obj"], float(r["bias_px"]))
        if key not in ref:
            missing += 1
            continue
        diffs.append(abs(float(r["J"]) - float(ref[key])))
    return dict(ok=bool(diffs and missing == 0 and max(diffs) <= 1e-12),
                n_compared=len(diffs), n_missing=missing,
                max_abs_dJ=float(max(diffs)) if diffs else None,
                note="RIGOROUS_ONLY replay vs published raw.json ALM J")


def _config_stats(rows, name):
    n = len(rows)
    cert, cert_schur, naive, naive_schur = [], [], [], []
    gaps, gaps_s, times, status = [], [], [], []
    shifts, lammins, cases, tts, nineqs = [], [], [], [], []
    for r in rows:
        c = r["certs"].get(name, {})
        J = float(r["J"])
        feas = float(r["viol"]) <= 1e-6
        rg = c.get("rigorous")
        sc = c.get("rigorous_schur")
        va = c.get("value")
        cert.append(bool(feas and _finite(rg) and (J - float(rg) <= GAP_TOL)))
        cert_schur.append(bool(feas and _finite(sc) and (J - float(sc) <= GAP_TOL)))
        naive.append(bool(feas and _finite(va) and (J - float(va) <= GAP_TOL)))
        naive_schur.append(bool(feas and _finite(sc)
                                and (J - float(sc) <= GAP_TOL)))
        if _finite(rg):
            gaps.append(J - float(rg))
        if _finite(sc):
            gaps_s.append(J - float(sc))
        times.append(c.get("time_s"))
        status.append(c.get("status"))
        shifts.append(c.get("dual_shift"))
        lammins.append(c.get("dual_lam_min"))
        cases.append(c.get("schur_case"))
        tts.append(c.get("schur_tt_cond"))
        nineqs.append(c.get("n_ineq"))
    k = int(sum(cert))
    ks = int(sum(cert_schur))
    p, lo, hi = wilson(k, n)
    ps, los, his = wilson(ks, n)
    delta = np.asarray([float(r["certs"][name]["rigorous_schur"])
                        - float(r["certs"][name]["rigorous"])
                        for r in rows
                        if _finite(r["certs"].get(name, {}).get("rigorous"))
                        and _finite(r["certs"].get(name, {}).get("rigorous_schur"))])
    from collections import Counter
    return dict(
        n=n,
        cert_dual_repair=k, rate_dual_repair=p, ci_lo=lo, ci_hi=hi,
        cert_schur=ks, rate_schur=ps, ci_lo_schur=los, ci_hi_schur=his,
        naive_value_cert=int(sum(naive)),
        schur_minus_dual=dict(n=int(delta.size),
                              median=float(np.median(delta)) if delta.size else None,
                              max=float(delta.max()) if delta.size else None,
                              min=float(delta.min()) if delta.size else None,
                              n_strictly_tighter=int((delta > 1e-12).sum()),
                              n_strictly_looser=int((delta < -1e-12).sum())),
        gap_dual_repair=_gap_stats([float(r["J"]) for r in rows],
                                   [r["certs"].get(name, {}).get("rigorous") for r in rows]),
        gap_schur=_gap_stats([float(r["J"]) for r in rows],
                             [r["certs"].get(name, {}).get("rigorous_schur") for r in rows]),
        time_s=_time_stats(times),
        schur_time_s=_time_stats([r["certs"].get(name, {}).get("schur_time_s") for r in rows]),
        status=dict(Counter(status)),
        schur_case=dict(Counter(cases)),
        n_ineq_median=float(np.median([x for x in nineqs if x is not None]))
        if any(x is not None for x in nineqs) else None,
        dual_shift_max=max([float(x) for x in shifts if _finite(x)], default=None),
        dual_lam_min_min=min([float(x) for x in lammins if _finite(x)], default=None),
        schur_tt_cond_max=max([float(x) for x in tts if _finite(x)], default=None),
        n_cert_both=int(sum(1 for a, b in zip(cert, cert_schur) if a and b)),
        n_schur_only=int(sum(1 for a, b in zip(cert, cert_schur) if (not a) and b)),
        n_dual_only=int(sum(1 for a, b in zip(cert, cert_schur) if a and (not b))),
    )


def _soundness(rows):
    out = dict(
        alm_rows=len(rows), alm_feasible=int(sum(float(r["viol"]) <= 1e-6
                                                 for r in rows)),
        alm_viol_max=max(float(r["viol"]) for r in rows),
        dual_repair_le_J=[], schur_le_J=[], schur_le_value=[],
        dual_invalid_shift=[], nan_schur=0, cases_b_noRt=0,
    )
    worst = dict(dual_repair=-np.inf, schur=-np.inf, schur_minus_value=-np.inf)
    for r in rows:
        J = float(r["J"])
        for name in CONFIGS:
            c = r["certs"].get(name, {})
            rg, sc, va = c.get("rigorous"), c.get("rigorous_schur"), c.get("value")
            if _finite(rg) and float(rg) > J + SOUND_TOL:
                out["dual_repair_le_J"].append(
                    dict(frame=r["frame"], img=r["img"], obj=r["obj"],
                         bias=r["bias_px"], cfg=name, violation=float(rg) - J))
                worst["dual_repair"] = max(worst["dual_repair"], float(rg) - J)
            if _finite(sc) and float(sc) > J + SOUND_TOL:
                out["schur_le_J"].append(
                    dict(frame=r["frame"], img=r["img"], obj=r["obj"],
                         bias=r["bias_px"], cfg=name, violation=float(sc) - J))
                worst["schur"] = max(worst["schur"], float(sc) - J)
            if _finite(sc) and _finite(va) and float(sc) > float(va) + SOUND_TOL:
                out["schur_le_value"].append(
                    dict(frame=r["frame"], img=r["img"], obj=r["obj"],
                         bias=r["bias_px"], cfg=name,
                         violation=float(sc) - float(va)))
                worst["schur_minus_value"] = max(worst["schur_minus_value"],
                                                 float(sc) - float(va))
            if _finite(c.get("dual_lam_min")) and float(c["dual_lam_min"]) < -1e-10:
                out["dual_invalid_shift"].append(
                    dict(frame=r["frame"], img=r["img"], obj=r["obj"],
                         bias=r["bias_px"], cfg=name,
                         dual_lam_min=float(c["dual_lam_min"])))
            if not _finite(sc):
                out["nan_schur"] += 1
            if str(c.get("schur_case", "")).startswith("B_noRt"):
                out["cases_b_noRt"] += 1
    out["worst"] = {k: (None if not np.isfinite(v) else float(v))
                    for k, v in worst.items()}
    out["ok"] = bool(not out["dual_repair_le_J"] and not out["schur_le_J"]
                     and not out["dual_invalid_shift"])
    return out


def _paired(rows, a, b):
    ca = [bool(_finite(r["certs"].get(a, {}).get("rigorous"))
               and float(r["J"]) - float(r["certs"][a]["rigorous"]) <= GAP_TOL
               and float(r["viol"]) <= 1e-6) for r in rows]
    cb = [bool(_finite(r["certs"].get(b, {}).get("rigorous"))
               and float(r["J"]) - float(r["certs"][b]["rigorous"]) <= GAP_TOL
               and float(r["viol"]) <= 1e-6) for r in rows]
    n01 = int(sum(1 for x, y in zip(ca, cb) if (not x) and y))
    n10 = int(sum(1 for x, y in zip(ca, cb) if x and (not y)))
    pval, _, _ = mcnemar_exact(n01, n10)
    return dict(a=a, b=b, n=len(rows), cert_a=int(sum(ca)), cert_b=int(sum(cb)),
                flips_0to1=n01, flips_1to0=n10,
                delta_pp=100.0 * (sum(cb) - sum(ca)) / max(len(rows), 1),
                mcnemar_p=pval)


def aggregate(ds):
    meta, rows, arrays = _load(ds)
    rep = _reproduction_check(ds, rows)
    out = dict(dataset=ds, n=len(rows), meta=meta, reproduction=rep,
               configs={c: _config_stats(rows, c) for c in CONFIGS},
               soundness=_soundness(rows),
               rlt_paired=[_paired(rows, "base", "brlt"),
                           _paired(rows, "base", "rlt")],
               generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / f"rigorous_{ds}.json", "w"), indent=2,
              default=float)
    write_dataset_md(ds, out)
    print(json.dumps({k: out[k] for k in ("dataset", "n")}, default=float))
    for c in CONFIGS:
        s = out["configs"][c]
        print(f"  {c:5s} dual-repair {s['cert_dual_repair']}/{s['n']} "
              f"({100*s['rate_dual_repair']:.1f}%), schur {s['cert_schur']}/{s['n']} "
              f"({100*s['rate_schur']:.1f}%), naive {s['naive_value_cert']}, "
              f"gap_schur med {s['gap_schur'].get('median', float('nan')):.2e}, "
              f"max {s['gap_schur'].get('max', float('nan')):.2e}")
    print("  soundness ok", out["soundness"]["ok"],
          "worst", out["soundness"]["worst"])
    return out


def write_dataset_md(ds, out):
    L = [f"# Rigorous real-data certificates — {ds.upper()} (n={out['n']})", ""]
    r = out["reproduction"]
    L += ["## Rebuild fidelity", "",
          f"- replay vs published `raw.json`: max |ΔJ| = {r.get('max_abs_dJ')}, "
          f"compared {r.get('n_compared')}, missing {r.get('n_missing')} → "
          f"{'identical instances' if r.get('ok') else 'MISMATCH'}", ""]
    L += ["## Certificate rates (95% Wilson CI)", "",
          "`cert` = ALM feasible and `J - bound <= 1e-6`, bound = rigorous lower bound.",
          "`naive` = same rule with the solver value (NOT a bound; diagnostic only).", "",
          "| cuts | n | dual-repair cert | CI | Schur cert | CI | naive(value) cert |",
          "|-|-|-|-|-|-|-|"]
    for c in CONFIGS:
        s = out["configs"][c]
        L.append(f"| {CONFIG_LABEL[c]} | {s['n']} | {s['cert_dual_repair']} "
                 f"({100*s['rate_dual_repair']:.1f}%) | "
                 f"[{100*s['ci_lo']:.1f}, {100*s['ci_hi']:.1f}] | "
                 f"{s['cert_schur']} ({100*s['rate_schur']:.1f}%) | "
                 f"[{100*s['ci_lo_schur']:.1f}, {100*s['ci_hi_schur']:.1f}] | "
                 f"{s['naive_value_cert']} |")
    L += ["", "## Rigorous gap `J - bound` (<=1e-6 = certified)", "",
          "| cuts | bound | n | min | median | max | #negative | time/solve |",
          "|-|-|-|-|-|-|-|-|"]
    for c in CONFIGS:
        s = out["configs"][c]
        for key, lab in (("gap_dual_repair", "dual-repair"),
                         ("gap_schur", "Schur")):
            g = s[key]
            L.append(f"| {CONFIG_LABEL[c]} | {lab} | {g.get('n')} | "
                     f"{g.get('min', float('nan')):.3e} | "
                     f"{g.get('median', float('nan')):.3e} | "
                     f"{g.get('max', float('nan')):.3e} | "
                     f"{g.get('n_negative')} | "
                     f"{s['time_s'].get('median', float('nan')):.2f} |")
    L += ["", "## Schur case / solver status", ""]
    for c in CONFIGS:
        s = out["configs"][c]
        L.append(f"- {c}: cases {s['schur_case']}, status {s['status']}, "
                 f"max dual_shift {s['dual_shift_max']}, "
                 f"min λmin(S_rep) {s['dual_lam_min_min']}, "
                 f"max cond(H_tt) {s['schur_tt_cond_max']}")
    L += ["", "## Soundness audit", "",
          f"- ALM feasible: {out['soundness']['alm_feasible']}/{out['soundness']['alm_rows']}"
          f" (max viol {out['soundness']['alm_viol_max']:.2e})",
          f"- dual-repair bound > J + {SOUND_TOL}: {len(out['soundness']['dual_repair_le_J'])} rows",
          f"- Schur bound > J + {SOUND_TOL}: {len(out['soundness']['schur_le_J'])} rows",
          f"- Schur bound > solver value + {SOUND_TOL}: {len(out['soundness']['schur_le_value'])} rows",
          f"- repaired slack λmin < -1e-10: {len(out['soundness']['dual_invalid_shift'])} rows",
          f"- worst violations: {out['soundness']['worst']}", "",
          "## RLT paired (rigorous cert flags)", ""]
    for p in out["rlt_paired"]:
        L.append(f"- {p['a']} → {p['b']}: {p['cert_a']}/{p['n']} → {p['cert_b']}/{p['n']} "
                 f"({p['delta_pp']:+.1f} pp), 0→1 {p['flips_0to1']}, "
                 f"1→0 {p['flips_1to0']}, McNemar p={p['mcnemar_p']:.3g}")
    (RESULTS / f"rigorous_{ds}.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# depth + tilt augmentation (LM-O uncertified rows)
# ---------------------------------------------------------------------------
def _load_model(ds, obj_id):
    from plyfile import PlyData
    p = DATA_ROOT / ds / "models" / f"obj_{obj_id:06d}.ply"
    v = PlyData.read(str(p))["vertex"]
    return np.stack([np.asarray(v[a], dtype=float) for a in "xyz"], axis=1) * 0.001


def _augment(A, B, V, R_gt, n, alpha_deg, n_depth, zmin):
    """Append positive-depth and upright-cone rows to (A,B)."""
    A2 = [A]
    B2 = [B]
    # positive depth for the n_depth vertices closest to the camera (smallest GT z)
    x_gt = (R_gt @ V.T).T
    order = np.argsort(x_gt[:, 2])
    for j in order[:n_depth]:
        a, b = core.c_depth(V[j], zmin)
        A2.append(a[None, :])
        B2.append(b[None, :])
    # upright cone around the table normal: n^T R(q) u >= cos(alpha), u = R_gt^T n
    u = R_gt.T @ n
    c_alpha = float(np.cos(np.deg2rad(alpha_deg)))
    m = core.m_rows(u)
    a_tilt = -(n @ m) + c_alpha * core.E_TR
    A2.append(a_tilt[None, :])
    B2.append(np.zeros((1, 3)))
    Aa = np.vstack(A2)
    Ba = np.vstack(B2)
    return Aa, Ba, a_tilt, u


def depth_tilt(ds="lmo", alpha_deg=12.0, n_depth=4, zmin=0.02,
               cfg="base", max_rows=None):
    """Augment the rows that are uncertified under `cfg` (default: honest base
    dual-repair/Schur; the published note's '6 LM-O plane-only cases')."""
    meta, rows, arrays = _load(ds)
    out = RESULTS / f"rigorous_{ds}_depth_tilt_{cfg}.json" if cfg != "base" \
        else RESULTS / f"rigorous_{ds}_depth_tilt.json"
    sel = []
    for i, r in enumerate(rows):
        c = r["certs"].get(cfg, {})
        J = float(r["J"])
        rg = c.get("rigorous_schur")
        feas = float(r["viol"]) <= 1e-6
        cert = bool(feas and _finite(rg) and (J - float(rg) <= GAP_TOL))
        if not cert:
            sel.append(i)
    if max_rows:
        sel = sel[:max_rows]
    print(f"{ds}: {len(sel)} uncertified rows under {cfg} "
          f"(of {len(rows)}); augment with {n_depth} depth + 1 tilt rows, "
          f"alpha={alpha_deg} deg, zmin={zmin} m", flush=True)
    out_rows = []
    t0 = time.perf_counter()
    for k, i in enumerate(sel):
        r = rows[i]
        M0, D0, G0 = arrays["M0"][i], arrays["D0"][i], arrays["G0"][i]
        A, B = arrays["A"][i], arrays["B"][i]
        q_alm, t_alm = arrays["q_alm"][i], arrays["t_alm"][i]
        q_unc, t_unc = arrays["q_unc"][i], arrays["t_unc"][i]
        q_gt = arrays["q_gt"][i]
        R_gt = core.R_of(q_gt)
        n = arrays["plane_n"][i]
        V = _load_model(ds, int(r["obj"]))
        # GT depth / upright checks for the augmentation's validity
        xg = (R_gt @ V.T).T
        gmin_gt = float(xg[:, 2].min())
        u = R_gt.T @ n
        tilt_gt = float(n @ R_gt @ u)
        xa = (core.R_of(q_alm) @ V.T).T + t_alm
        gmin_alm = float(xa[:, 2].min())
        tilt_alm = float(n @ core.R_of(q_alm) @ u)
        Aa, Ba, _, _ = _augment(A, B, V, R_gt, n, alpha_deg, n_depth, zmin)
        # ALM on the augmented problem, same init scheme as run_bop
        alm = core.alm_qpep(M0, D0, G0, Aa, Ba, q_unc, t_unc, rho0=10.0,
                            n_starts=6, rng=np.random.default_rng(1000 + k))
        q_new, t_new = alm["q"], alm["t"]
        J_new = core.obj_value(q_new, t_new, M0, D0, G0)
        viol_new = core.max_viol(Aa, Ba, q_new, t_new)
        res = shor_bound(M0, D0, G0, Aa, Ba, add_bound_rlt=True,
                         repair_dual=True, schur=True,
                         solver="SCS", eps=1e-9, max_iters=50000)
        rg_new = res.get("rigorous_schur")
        rg_old = r["certs"][cfg].get("rigorous_schur")
        row = dict(
            index=i, frame=r["frame"], img=r["img"], obj=r["obj"],
            bias_px=r["bias_px"],
            J_old=float(r["J"]), bound_old=float(rg_old) if _finite(rg_old) else None,
            gap_old=float(r["J"]) - float(rg_old) if _finite(rg_old) else None,
            cert_old=False,
            J_new=float(J_new), bound_new=float(rg_new) if _finite(rg_new) else None,
            gap_new=float(J_new) - float(rg_new) if _finite(rg_new) else None,
            cert_new=bool(_finite(rg_new) and viol_new <= 1e-6
                          and J_new - float(rg_new) <= GAP_TOL),
            viol_new=float(viol_new), schur_case=res.get("schur_case"),
            status=res.get("status"), n_ineq=res.get("n_ineq"),
            n_constraints=int(Aa.shape[0]),
            gt_depth_min=gmin_gt, alm_depth_min=gmin_alm,
            gt_upright=tilt_gt, alm_upright=tilt_alm,
            cos_alpha=float(np.cos(np.deg2rad(alpha_deg))),
            time_s=float(res.get("time_s", float("nan"))),
            J_shift=float(J_new - r["J"]),
            bound_shift=float(rg_new - rg_old) if (_finite(rg_new) and _finite(rg_old)) else None,
        )
        out_rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    res = dict(dataset=ds, config=cfg, alpha_deg=alpha_deg, n_depth=n_depth,
               zmin=zmin, n_uncertified=len(sel), rows=out_rows,
               wall_s=time.perf_counter() - t0,
               note=("depth+upright constraints RESTRICT the feasible set; a "
                     "certificate here is for the augmented physical problem, "
                     "not for the original 16-plane-only problem.  J_new=J_old "
                     "means the ALM pose already satisfies the added rows."))
    json.dump(res, open(out, "w"), indent=2, default=float)
    kc = int(sum(r["cert_new"] for r in out_rows))
    print(f"certified after augmentation: {kc}/{len(out_rows)}; wall "
          f"{res['wall_s']:.1f} s; wrote {out}", flush=True)
    return res


# ---------------------------------------------------------------------------
# final merge
# ---------------------------------------------------------------------------
def final():
    parts = {}
    for ds in ("lm", "lmo", "ycbv"):
        raw = DATASETS[ds]["out"] / "rigorous" / "rigorous_rows.json"
        if raw.exists():
            aggregate(ds)          # re-aggregate from stored rows (no SDP)
            update_tables(ds)
            parts[ds] = json.load(open(RESULTS / f"rigorous_{ds}.json"))
    dt = RESULTS / "rigorous_lmo_depth_tilt.json"
    depth_tilt_res = json.load(open(dt)) if dt.exists() else None
    merged = dict(datasets=parts, depth_tilt=depth_tilt_res,
                  gap_tol=GAP_TOL, sound_tol=SOUND_TOL,
                  generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(merged, open(RESULTS / "rigorous_real.json", "w"), indent=2,
              default=float)
    write_final_md(merged)
    print("wrote", RESULTS / "rigorous_real.json",
          RESULTS / "rigorous_real.md")


def write_final_md(m):
    L = ["# P2 rigorous real-data certificates (S11-S19)", "",
         "Honest lower bounds from `certificate.py`: `repair_dual` (dual slack + "
         "PSD shift) and the P2-Cert 13-dim Schur bound, on the *published* "
         "BOP instances (replayed with identical RNG).", "",
         "`cert` = ALM feasible and `J - bound <= 1e-6`. Naive one-sided "
         "(solver value) is listed only to show how much of the old rate was "
         "not honest.", "",
         "## Per-dataset rigorous certificate rates", "",
         "| dataset | n | cuts | dual-repair cert (95% CI) | Schur cert (95% CI) | naive(value) | median/max Schur gap |",
         "|-|-|-|-|-|-|-|"]
    for ds, p in m["datasets"].items():
        for c in CONFIGS:
            s = p["configs"][c]
            L.append(f"| {ds} | {s['n']} | {c} | "
                     f"{s['cert_dual_repair']} ({100*s['rate_dual_repair']:.1f}%; "
                     f"[{100*s['ci_lo']:.1f},{100*s['ci_hi']:.1f}]) | "
                     f"{s['cert_schur']}/"
                     f"{100*s['rate_schur']:.1f}% "
                     f"[{100*s['ci_lo_schur']:.1f},{100*s['ci_hi_schur']:.1f}] | "
                     f"{s['naive_value_cert']} | "
                     f"{s['gap_schur'].get('median', float('nan')):.2e}/"
                     f"{s['gap_schur'].get('max', float('nan')):.2e} |")
    L += ["", "## Soundness audit", "",
          "| dataset | ALM feasible | dual>J | Schur>J | Schur>value | invalid shift | worst (Schur-J) |",
          "|-|-|-|-|-|-|-|"]
    for ds, p in m["datasets"].items():
        s = p["soundness"]
        L.append(f"| {ds} | {s['alm_feasible']}/{s['alm_rows']} | "
                 f"{len(s['dual_repair_le_J'])} | {len(s['schur_le_J'])} | "
                 f"{len(s['schur_le_value'])} | {len(s['dual_invalid_shift'])} | "
                 f"{s['worst'].get('schur')} |")
    L += ["", "## RLT paired (rigorous flags)", ""]
    for ds, p in m["datasets"].items():
        for q in p["rlt_paired"]:
            L.append(f"- {ds} {q['a']}→{q['b']}: {q['cert_a']}/{q['n']} → "
                     f"{q['cert_b']}/{q['n']} ({q['delta_pp']:+.1f} pp), "
                     f"0→1 {q['flips_0to1']}, 1→0 {q['flips_1to0']}, "
                     f"p={q['mcnemar_p']:.3g}")
    d = m.get("depth_tilt")
    if d:
        kc = sum(1 for r in d["rows"] if r["cert_new"])
        L += ["", "## LM-O depth+upright augmentation", "",
              f"- uncertified under {d['config']}: {d['n_uncertified']}; "
              f"certified after augmentation: {kc}/{len(d['rows'])}", "",
              "| frame | img | obj | bias | J_old | gap_old | J_new | gap_new | cert |",
              "|-|-|-|-|-|-|-|-|-|"]
        for r in d["rows"]:
            L.append(f"| {r['frame']} | {r['img']} | {r['obj']} | {r['bias_px']} | "
                     f"{r['J_old']:.4e} | {r['gap_old']:.2e} | {r['J_new']:.4e} | "
                     f"{r['gap_new']:.2e} | {r['cert_new']} |")
        L += ["", "Caveat: depth/upright rows restrict the feasible set; this "
              "certifies the augmented physical problem only.", ""]
    L += ["", "## Reproduction", ""]
    for ds, p in m["datasets"].items():
        r = p["reproduction"]
        L.append(f"- {ds}: n={p['n']}, max|ΔJ| vs raw.json = {r.get('max_abs_dJ')}, "
                 f"missing {r.get('n_missing')} → {'OK' if r.get('ok') else 'CHECK'}")
    (RESULTS / "rigorous_real.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# S18: append the rigorous columns to the generated real table + meta.json
# ---------------------------------------------------------------------------
RIG_BEGIN = "<!-- RIGOROUS_CERT:BEGIN (generated by code/run_rigorous_real.py) -->"
RIG_END = "<!-- RIGOROUS_CERT:END -->"


def update_tables(ds):
    """Append (idempotently) a rigorous-certificate block to main_table.md and
    store the same numbers under `meta.json["rigorous"]`.  Never hand-edited."""
    out = DATASETS[ds]["out"]
    rj = RESULTS / f"rigorous_{ds}.json"
    if not rj.exists() or not (out / "main_table.md").exists():
        return
    d = json.load(open(rj))
    L = [RIG_BEGIN, "",
         "## Rigorous certificates (supersede the one-sided numbers above)", "",
         "Generated by `code/run_rigorous_real.py`; `cert` = ALM feasible and "
         "`J - bound <= 1e-6` with `bound` a *proven* lower bound "
         "(`repair_dual` / P2-Cert Schur). The one-sided `value` columns above "
         "are upper bounds on the sound rate and are kept only for comparison.", "",
         "| cuts | n | dual-repair cert (95% CI) | Schur cert (95% CI) | naive(value) | Schur gap min/med/max |",
         "|-|-|-|-|-|-|"]
    for c in CONFIGS:
        s = d["configs"][c]
        g = s["gap_schur"]
        L.append(f"| {CONFIG_LABEL[c]} | {s['n']} | "
                 f"{s['cert_dual_repair']} ({100*s['rate_dual_repair']:.1f}%; "
                 f"[{100*s['ci_lo']:.1f},{100*s['ci_hi']:.1f}]) | "
                 f"{s['cert_schur']} ({100*s['rate_schur']:.1f}%; "
                 f"[{100*s['ci_lo_schur']:.1f},{100*s['ci_hi_schur']:.1f}]) | "
                 f"{s['naive_value_cert']} | "
                 f"{g.get('min', float('nan')):.2e} / "
                 f"{g.get('median', float('nan')):.2e} / "
                 f"{g.get('max', float('nan')):.2e} |")
    L += ["", "Soundness: " + json.dumps(
        {k: d["soundness"][k] for k in
         ("alm_rows", "alm_feasible", "alm_viol_max", "nan_schur")}
        | {"dual_gt_J": len(d["soundness"]["dual_repair_le_J"]),
           "schur_gt_J": len(d["soundness"]["schur_le_J"]),
           "schur_gt_value": len(d["soundness"]["schur_le_value"])}), "",
        RIG_END]
    txt = (out / "main_table.md").read_text()
    if RIG_BEGIN in txt:
        txt = txt.split(RIG_BEGIN)[0].rstrip("\n") + "\n\n" + "\n".join(L) + "\n"
    else:
        txt = txt.rstrip("\n") + "\n\n" + "\n".join(L) + "\n"
    (out / "main_table.md").write_text(txt)
    meta = json.load(open(out / "meta.json"))
    meta["rigorous"] = dict(
        generated_by="code/run_rigorous_real.py",
        gap_tol=GAP_TOL, sound_tol=SOUND_TOL,
        configs={c: {k: v for k, v in d["configs"][c].items()
                     if k in ("n", "cert_dual_repair", "rate_dual_repair",
                              "ci_lo", "ci_hi", "cert_schur", "rate_schur",
                              "ci_lo_schur", "ci_hi_schur", "naive_value_cert",
                              "gap_schur", "schur_minus_dual")}
                 for c in CONFIGS},
        soundness={k: d["soundness"][k] for k in
                   ("ok", "alm_rows", "alm_feasible", "alm_viol_max",
                    "worst", "nan_schur")}
        | {"dual_gt_J": len(d["soundness"]["dual_repair_le_J"]),
           "schur_gt_J": len(d["soundness"]["schur_le_J"]),
           "schur_gt_value": len(d["soundness"]["schur_le_value"])},
        reproduction=d["reproduction"])
    json.dump(meta, open(out / "meta.json", "w"), indent=2, default=float)
    print("updated", out / "main_table.md", out / "meta.json", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["collect", "depth-tilt", "final"])
    ap.add_argument("dataset", nargs="?", default=None,
                    choices=["lm", "lmo", "ycbv"])
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--n-depth", type=int, default=4)
    ap.add_argument("--zmin", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--select-config", default="base", choices=CONFIGS)
    a = ap.parse_args()
    if a.command == "collect":
        if not a.dataset:
            ap.error("collect needs a dataset")
        collect(a.dataset)
    elif a.command == "depth-tilt":
        depth_tilt("lmo", alpha_deg=a.alpha, n_depth=a.n_depth, zmin=a.zmin,
                   cfg=a.select_config, max_rows=a.max_rows)
    else:
        final()


if __name__ == "__main__":
    main()
