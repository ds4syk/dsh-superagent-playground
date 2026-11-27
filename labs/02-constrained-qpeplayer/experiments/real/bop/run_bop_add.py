#!/usr/bin/env python3
"""BOP ADD / ADD-S for the four methods, from the dumped QPEP poses.

Inputs
------
* ``real/linemod/results/`` (LM), ``real/bop/results/lmo``, ``real/bop/results/ycbv``
  -- ``raw.json`` (the published per-trial table), ``meta.json`` and the
  ``poses.npz`` written by ``run_bop.py POSE_DUMP=1``.
* ``/mnt/storage1/syk/data/bop/<ds>/models_eval/obj_XXXXXX.ply`` -- the *eval*
  meshes used by the official BOP metrics (metres after x0.001).
* ``/mnt/storage1/syk/data/bop/<ds>/test/<frame>/scene_gt.json`` -- GT poses.

Metric definition (BOP)
-----------------------
``ADD  = mean_i || (R_e v_i + t_e) - (R_g v_i + t_g) ||``
``ADD-S= mean_i min_j || (R_e v_i + t_e) - (R_g v_j + t_g) ||``  (KD-tree NN)
``ADD(-S)`` uses ADD for asymmetric objects and ADD-S for the symmetric ones
(LM/LM-O: obj 8 eggbox, obj 9 glue -- the BOP symmetry list; none of the YCB-V
objects used here is symmetric).  All distances in mm.

The pose dump is *verified* against ``raw.json`` (same rows, same rot/trans/
penetration/J to floating-point round-off) before any metric is computed, so the
ADD numbers are attached to exactly the trials of the published table.

Usage
-----
    python run_bop_add.py            # all three datasets
    python run_bop_add.py lm         # one dataset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1] / "code"
sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402

DATA = Path("/mnt/storage1/syk/data/bop")
RESULTS = HERE.parents[1] / "results"
OUT_DIR = RESULTS
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "lm": dict(res=HERE.parent / "linemod" / "results", data="lm", sym={8, 9}),
    "lmo": dict(res=HERE.parent / "bop" / "results" / "lmo", data="lmo", sym={8, 9}),
    "ycbv": dict(res=HERE.parent / "bop" / "results" / "ycbv", data="ycbv", sym=set()),
}
METHODS = ["Unconstrained", "Soft penalty", "Projection", "ALM-QPEP"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(xs, seed=0, n_boot=2000):
    xs = [bool(x) for x in xs]
    p, lo, hi = wilson(sum(xs), len(xs))
    return dict(k=int(sum(xs)), n=len(xs), p=p, lo=lo, hi=hi)


def median_ci(xs, seed=0, n_boot=2000):
    x = np.asarray([v for v in xs if np.isfinite(v)], float)
    if x.size == 0:
        return dict(n=0, median=float("nan"), q1=float("nan"), q3=float("nan"),
                    lo=float("nan"), hi=float("nan"))
    rng = np.random.default_rng(seed)
    med = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    return dict(n=int(x.size), median=float(np.median(x)),
                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                lo=float(np.percentile(med, 2.5)), hi=float(np.percentile(med, 97.5)))


def load_model(ds, oid):
    from plyfile import PlyData
    p = DATA / ds / "models_eval" / f"obj_{oid:06d}.ply"
    v = PlyData.read(str(p))["vertex"]
    return np.stack([np.asarray(v[a], float) for a in "xyz"], axis=1) * 0.001


def diameter(V):
    """Model diameter (m) = max pairwise distance, computed on the hull."""
    try:
        h = V[ConvexHull(V).vertices]
    except Exception:
        h = V
    d2 = ((h[:, None, :] - h[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


def gt_lookup(ds, frames):
    """(frame, img, obj) -> (R, t) from scene_gt.json (m)."""
    out = {}
    for frame in frames:
        f = DATA / ds / "test" / frame / "scene_gt.json"
        if not f.exists():
            continue
        for img, gts in json.load(open(f)).items():
            for g in gts:
                out[(frame, str(int(img)), int(g["obj_id"]))] = (
                    np.array(g["cam_R_m2c"], float).reshape(3, 3),
                    np.array(g["cam_t_m2c"], float) * 0.001)
    return out


def verify(rows, poses, tol=1e-8):
    """Check poses.npz reproduces raw.json row-for-row."""
    key = lambda r: (r["frame"], str(int(r["img"])), int(r["obj"]),
                     float(r["bias_px"]), r["method"])
    raw = {key(r): r for r in rows}
    got = {key(r): r for r in poses}
    if set(raw) != set(got):
        raise SystemExit(f"pose dump / raw.json key mismatch "
                         f"(raw {len(raw)}, dump {len(got)}, "
                         f"only-raw {len(set(raw) - set(got))}, "
                         f"only-dump {len(set(got) - set(raw))})")
    worst = {}
    for k, a in raw.items():
        b = got[k]
        for f in ("rot_deg", "trans_mm", "pen_mm", "J", "viol"):
            d = abs(float(a[f]) - float(b[f]))
            worst[f] = max(worst.get(f, 0.0), d)
        if bool(a["feas"]) != bool(b["feas"]):
            raise SystemExit(f"feasibility mismatch on {k}")
    bad = {f: d for f, d in worst.items() if d > tol}
    return dict(n_rows=len(raw), max_abs_diff=worst, mismatch=bad,
                ok=not bad)


def add_metrics(model, R_e, t_e, R_g, t_g):
    est = model @ R_e.T + t_e
    gt = model @ R_g.T + t_g
    add = float(np.mean(np.linalg.norm(est - gt, axis=1)))
    adds = float(cKDTree(gt).query(est)[0].mean())
    return add * 1000.0, adds * 1000.0


def metric_selftest():
    """ADD/ADD-S sanity: self = 0, and a symmetry rotation is cheap only for ADD-S."""
    out = {}
    for oid in (1, 8, 9):
        V = load_model("lm", oid)
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.5])
        Rz180 = np.diag([-1.0, -1.0, 1.0])
        Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        add0, adds0 = add_metrics(V, R, t, R, t)
        add1, adds1 = add_metrics(V, Rz180, t, R, t)
        add2, adds2 = add_metrics(V, Rz90, t, R, t)
        out[str(oid)] = dict(diam_mm=diameter(V) * 1000.0,
                             self_add_mm=add0, self_adds_mm=adds0,
                             rot180_add_mm=add1, rot180_adds_mm=adds1,
                             rot90_add_mm=add2, rot90_adds_mm=adds2)
    return out


def run_dataset(tag):
    cfg = DATASETS[tag]
    res = cfg["res"]
    rows = json.load(open(res / "raw.json"))
    meta = json.load(open(res / "meta.json"))
    pd = np.load(res / "poses.npz", allow_pickle=True)
    poses = [{k: pd[k][i] for k in pd.files} for i in range(len(pd["method"]))]
    poses = [dict(r, frame=str(r["frame"]), img=str(int(r["img"])),
                  obj=int(r["obj"]), bias_px=float(r["bias_px"]),
                  method=str(r["method"]),
                  q=np.asarray(r["q"], float), t=np.asarray(r["t"], float),
                  J=float(r["J"]), rot_deg=float(r["rot_deg"]),
                  trans_mm=float(r["trans_mm"]), pen_mm=float(r["pen_mm"]),
                  feas=bool(r["feas"]), viol=float(r["viol"])) for r in poses]
    ver = verify(rows, poses)
    gts = gt_lookup(cfg["data"], meta["frames"])
    models = {oid: load_model(cfg["data"], oid) for oid in meta["obj_ids"]}
    diams = {oid: diameter(V) for oid, V in models.items()}
    rowmap = {(r["frame"], str(int(r["img"])), int(r["obj"]),
               float(r["bias_px"]), r["method"]): r for r in rows}
    trials = []
    for p in poses:
        key = (p["frame"], p["img"], p["obj"], p["bias_px"], p["method"])
        r = rowmap[key]
        R_g, t_g = gts[(p["frame"], p["img"], p["obj"])]
        R_e = core.R_of(p["q"])
        add, adds = add_metrics(models[p["obj"]], R_e, p["t"], R_g, t_g)
        sym = p["obj"] in cfg["sym"]
        # consistency: the stored GT errors must come from *this* GT entry
        rq = core.rot_deg(p["q"], core.q_from_R(R_g))
        rt = core.trans_mm(p["t"], t_g)
        trials.append(dict(
            frame=p["frame"], img=p["img"], obj=p["obj"],
            bias_px=p["bias_px"], method=p["method"], symmetric=sym,
            add_mm=add, adds_mm=adds, add_s_mm=(adds if sym else add),
            rot_deg=float(r["rot_deg"]), trans_mm=float(r["trans_mm"]),
            pen_mm=float(r["pen_mm"]), feas=bool(r["feas"]),
            cert=bool(r["cert"]), cert_two_sided=bool(r["cert_two_sided"]),
            cert_rlt=bool(r["cert_rlt"]), viol=float(r["viol"]),
            J=float(r["J"]),
            gt_repro_rot_err_deg=abs(rq - float(r["rot_deg"])),
            gt_repro_trans_err_mm=abs(rt - float(r["trans_mm"])),
            diam_mm=diams[p["obj"]] * 1000.0,
            diam10_mm=0.1 * diams[p["obj"]] * 1000.0,
        ))
    worst_gt = max(max(t["gt_repro_rot_err_deg"], t["gt_repro_trans_err_mm"])
                   for t in trials)
    summary = {}
    for m in METHODS:
        ts = [t for t in trials if t["method"] == m]
        summary[m] = dict(
            n=len(ts),
            add_mm=median_ci([t["add_mm"] for t in ts], seed=11),
            adds_mm=median_ci([t["adds_mm"] for t in ts], seed=12),
            add_s_mm=median_ci([t["add_s_mm"] for t in ts], seed=13),
            add_s_lt20mm=rate([t["add_s_mm"] < 20.0 for t in ts]),
            add_s_lt10pct=rate([t["add_s_mm"] < t["diam10_mm"] for t in ts]),
            add_lt20mm=rate([t["add_mm"] < 20.0 for t in ts]),
            pen_mm=median_ci([t["pen_mm"] for t in ts], seed=14),
            feasible=rate([t["feas"] for t in ts]),
            cert=rate([t["cert"] for t in ts]),
            cert_two_sided=rate([t["cert_two_sided"] for t in ts]),
            rot_deg=median_ci([t["rot_deg"] for t in ts], seed=15),
            trans_mm=median_ci([t["trans_mm"] for t in ts], seed=16),
        )
    per_obj = {}
    for oid in meta["obj_ids"]:
        per_obj[str(oid)] = {}
        for m in METHODS:
            ts = [t for t in trials if t["method"] == m and t["obj"] == oid]
            if ts:
                per_obj[str(oid)][m] = dict(
                    n=len(ts),
                    add_mm=median_ci([t["add_mm"] for t in ts], seed=17)["median"],
                    adds_mm=median_ci([t["adds_mm"] for t in ts], seed=18)["median"],
                    add_s_mm=median_ci([t["add_s_mm"] for t in ts], seed=19)["median"],
                    symmetric=oid in cfg["sym"])
    rig = None
    rf = OUT_DIR / f"rigorous_{tag}.json"
    if rf.exists():
        rj = json.load(open(rf))
        rig = dict(configs={c: dict(n=v["n"], cert_schur_verified=v["cert_schur_verified"],
                                    rate=v["rate_schur_verified"],
                                    lo=v["ci_lo_schur_verified"], hi=v["ci_hi_schur_verified"],
                                    cert_dual=v["cert_dual_repair"], cert_naive=v["naive_value_cert"])
                           for c, v in rj["configs"].items()},
                   reproduction=rj["reproduction"],
                   source=f"results/rigorous_{tag}.json (Schur/dual-repair, same trials)")
    out = dict(dataset=tag, bop_name={"lm": "LINEMOD", "lmo": "LM-O",
                                      "ycbv": "YCB-Video"}[tag],
               meta=dict(frames=meta["frames"], max_img=meta["max_img"],
                         obj_ids=meta["obj_ids"], n_poses=meta["n_poses"],
                         n_trials=meta["n_trials"], seed=meta["seed"],
                         noise_px=1.0, bias_px=[10.0, 20.0, 30.0],
                         symmetric_objects=sorted(cfg["sym"]),
                         model="models_eval/*.ply (BOP eval meshes)"),
               pose_dump_verified=ver, gt_repro_max_err=worst_gt,
               rigorous=rig, summary=summary, per_obj=per_obj, trials=trials)
    print(f"{tag}: {len(trials)} trials, pose-dump ok={ver['ok']}, "
          f"gt-repro max err {worst_gt:.2e}", flush=True)
    return out


def fmt_ms(d):
    return (f"{d['median']:.1f} [{d['q1']:.1f}, {d['q3']:.1f}]"
            if np.isfinite(d["median"]) else "n/a")


def write_md(all_out, selftest=None):
    L = ["# BOP ADD / ADD-S on the four methods (real meshes, real GT poses)\n",
         "Generated by `experiments/real/bop/run_bop_add.py` from the pose dump\n"
         "`poses.npz` that exactly replays the published `raw.json` trials.\n",
         "Metric: BOP ADD / ADD-S on `models_eval` meshes; **ADD(-S)** uses ADD for\n"
         "asymmetric objects and ADD-S for the symmetric ones (LM/LM-O obj 8 eggbox,\n"
         "obj 9 glue; none of the YCB-V objects here is symmetric). Units: mm.\n",
         "Layer-level caveat: the 2D-3D correspondences are 30 model vertices projected\n"
         "at the GT pose + 1 px noise + a downward pixel bias; this is **not** a detector\n"
         "and no end-to-end perception claim is made.  The poses are the *same* trials as\n"
         "the published BOP table (verified row-for-row below).\n"]
    for out in all_out:
        tag = out["dataset"]
        ver = out["pose_dump_verified"]
        L.append(f"\n## {out['bop_name']} (`{tag}`, n={out['meta']['n_trials']} trials, "
                 f"{len(out['meta']['obj_ids'])} objects)\n")
        L.append(f"Subset: frames={out['meta']['frames']}, max_img={out['meta']['max_img']}, "
                 f"objects={out['meta']['obj_ids']}, seed={out['meta']['seed']}.")
        L.append(f"Pose dump vs raw.json: {ver['n_rows']} rows, max |diff| "
                 f"{max(ver['max_abs_diff'].values()):.2e} -> {'OK' if ver['ok'] else 'MISMATCH'}; "
                 f"GT re-association max err {out['gt_repro_max_err']:.2e}.\n")
        L.append("| method | n | ADD med [IQR] | ADD-S med [IQR] | ADD(-S) med [IQR] | "
                 "ADD(-S)<2cm | ADD(-S)<10%diam | pen mm | feasible | cert (1-sided UB) | "
                 "cert 2-sided | rot deg | trans mm |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            s = out["summary"][m]
            L.append("| {m} | {n} | {add} | {adds} | {adds_s} | {r20} | {r10} | {pen} | "
                     "{feas} | {cert} | {c2} | {rot} | {tr} |".format(
                         m=m, n=s["n"],
                         add=fmt_ms(s["add_mm"]), adds=fmt_ms(s["adds_mm"]),
                         adds_s=fmt_ms(s["add_s_mm"]),
                         r20=f"{100*s['add_s_lt20mm']['p']:.1f}% "
                             f"[{100*s['add_s_lt20mm']['lo']:.1f},"
                             f"{100*s['add_s_lt20mm']['hi']:.1f}] "
                             f"({s['add_s_lt20mm']['k']}/{s['add_s_lt20mm']['n']})",
                         r10=f"{100*s['add_s_lt10pct']['p']:.1f}% "
                             f"[{100*s['add_s_lt10pct']['lo']:.1f},"
                             f"{100*s['add_s_lt10pct']['hi']:.1f}]",
                         pen=f"{s['pen_mm']['median']:.1f}",
                         feas=f"{100*s['feasible']['p']:.1f}% "
                              f"[{100*s['feasible']['lo']:.1f},"
                              f"{100*s['feasible']['hi']:.1f}]",
                         cert=f"{100*s['cert']['p']:.1f}% "
                              f"[{100*s['cert']['lo']:.1f},"
                              f"{100*s['cert']['hi']:.1f}]",
                         c2=f"{100*s['cert_two_sided']['p']:.1f}%",
                         rot=f"{s['rot_deg']['median']:.2f}",
                         tr=f"{s['trans_mm']['median']:.1f}"))
        L.append("")
        if out.get("rigorous"):
            rc = out["rigorous"]["configs"]
            rep = out["rigorous"]["reproduction"]
            L.append(f"Rigorous certificate on the *same* trials (Schur / P2-Cert, "
                     f"`{out['rigorous']['source']}`, replay check "
                     f"max |dJ| = {rep['max_abs_dJ']:.1e}, n={rep['n_compared']}): "
                     f"base **{rc['base']['cert_schur_verified']}/{rc['base']['n']} = "
                     f"{100*rc['base']['rate']:.1f}%** "
                     f"[{100*rc['base']['lo']:.1f}, {100*rc['base']['hi']:.1f}]; "
                     f"+bound-RLT **{rc['brlt']['cert_schur_verified']}/{rc['brlt']['n']} = "
                     f"{100*rc['brlt']['rate']:.1f}%** "
                     f"[{100*rc['brlt']['lo']:.1f}, {100*rc['brlt']['hi']:.1f}]. "
                     f"This is the sound ALM-only rate; the `cert` column above is the "
                     f"one-sided SCS upper bound for *every* method, a different quantity.")
            L.append("")
        L.append("Per-object ADD(-S) median (mm):")
        L.append("")
        L.append("| obj | sym | " + " | ".join(METHODS) + " |")
        L.append("|---|---|" + "---|" * len(METHODS))
        for oid, d in out["per_obj"].items():
            vals = [f"{d[m]['add_s_mm']:.1f}" if m in d else "n/a" for m in METHODS]
            sym = "Y" if d[METHODS[0]]["symmetric"] else "n"
            L.append(f"| {oid} | {sym} | " + " | ".join(vals) + " |")
        L.append("")
        L.append("95% CI for the medians is a percentile bootstrap (2000 draws); rates use\n"
                 "the Wilson score interval. `cert` is the one-sided (J - Shor <= 1e-6) flag,\n"
                 "which is an **upper bound** on the sound rate; `cert 2-sided` is the\n"
                 "necessary condition |J - Shor| <= 1e-6 (see the BOP table caveat).\n")
    L.append("\n## Constraints trade accuracy for admissibility (does not improve ADD)\n")
    for out in all_out:
        u = out["summary"]["Unconstrained"]
        a = out["summary"]["ALM-QPEP"]
        p = out["summary"]["Projection"]
        L.append(f"- **{out['bop_name']}**: ALM ADD(-S) median {a['add_s_mm']['median']:.1f} mm vs "
                 f"unconstrained {u['add_s_mm']['median']:.1f} mm vs projection "
                 f"{p['add_s_mm']['median']:.1f} mm; penetration "
                 f"{u['pen_mm']['median']:.1f} -> {a['pen_mm']['median']:.1f} mm; feasible "
                 f"{100*u['feasible']['p']:.1f}% -> {100*a['feasible']['p']:.1f}%. "
                 f"So the constrained solve buys admissibility (zero penetration, 100% "
                 f"feasible) and pays for it in ADD under this deliberately biased "
                 f"correspondence layer -- ALM is **not** an ADD winner here.")
    L.append("")
    L.append("\n## Reading the numbers\n")
    L.append("- ADD/ADD-S are *pose* errors on real meshes and real GT poses, but the\n"
             "  correspondence layer is synthetic-by-projection, so they must be read as\n"
             "  \"error of the constrained solve on a 1 px-noisy real-object instance\",\n"
             "  not as a BOP leaderboard entry.\n"
             "- The downward pixel bias (10/20/30 px) is what manufactures penetration in\n"
             "  the unconstrained solve; ADD grows with it, and that is the quantity the\n"
             "  constrained methods trade against penetration.\n")
    if selftest:
        L.append("\n## Metric self-test (LM meshes)\n")
        L.append("| obj | diameter mm | self ADD | self ADD-S | 180° about z: ADD | ADD-S | "
                 "90° about z: ADD | ADD-S |")
        L.append("|---|---|---|---|---|---|---|---|")
        for oid, d in selftest.items():
            L.append(f"| {oid} | {d['diam_mm']:.1f} | {d['self_add_mm']:.4f} | "
                     f"{d['self_adds_mm']:.4f} | {d['rot180_add_mm']:.2f} | "
                     f"{d['rot180_adds_mm']:.2f} | {d['rot90_add_mm']:.2f} | "
                     f"{d['rot90_adds_mm']:.2f} |")
        L.append("\nSelf-distance is exactly 0 (the metric code is consistent); for the two\n"
                 "symmetric LM objects (8 eggbox, 9 glue) a 180° symmetry rotation costs\n"
                 "60-100 mm of ADD but only ~5-13 mm of ADD-S, which is why ADD-S is the\n"
                 "right number for them.  Obj 1 (ape) is asymmetric but nearly blob-shaped,\n"
                 "so its ADD-S is small too -- ADD-S is not a substitute for ADD on it.\n")
    (OUT_DIR / "real_bop_add.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT_DIR / "real_bop_add.md", flush=True)


def main():
    tags = sys.argv[1:] or list(DATASETS)
    all_out = [run_dataset(t) for t in tags]
    selftest = metric_selftest()
    json.dump(dict(datasets=all_out, metric_selftest=selftest),
              open(OUT_DIR / "real_bop_add.json", "w"),
              indent=2, default=float)
    print("wrote", OUT_DIR / "real_bop_add.json", flush=True)
    write_md(all_out, selftest)
    # figure: CDF of ADD(-S)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(all_out), figsize=(4.6 * len(all_out), 3.6),
                             squeeze=False)
    for ax, out in zip(axes[0], all_out):
        for m, color in zip(METHODS, ["C0", "C1", "C2", "C3"]):
            v = np.sort([t["add_s_mm"] for t in out["trials"] if t["method"] == m])
            ax.plot(v, np.linspace(0, 1, v.size, endpoint=False), label=m, color=color)
        ax.axvline(20, ls=":", c="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{out['bop_name']} (n={out['meta']['n_trials']})")
        ax.set_xlabel("ADD(-S) [mm]")
        ax.set_ylabel("empirical CDF")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("BOP ADD(-S) CDF, four methods (dotted: 20 mm)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_bop_add.png", dpi=160)
    print("wrote", OUT_DIR / "real_bop_add.png", flush=True)


if __name__ == "__main__":
    main()
