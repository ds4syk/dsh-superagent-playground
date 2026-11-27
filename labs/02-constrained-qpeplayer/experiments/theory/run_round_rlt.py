#!/usr/bin/env python3
"""P2 round-rlt: sound cut families + rigorous dual repair on the 14x14 Shor SDP.

Question (round hypothesis): are the zero-dimensional RLT cuts g_i g_j >= 0 too
weak at high noise, and is "certified" solver-tolerance-dependent?  This driver
measures, on the 4 (noise,bias) cells x 4 seeds of the P2 synthetic benchmark:

  base      : 14x14 Veronese Shor SDP
  rlt       : + g_i g_j >= 0                        (existing)
  bounds    : + proven box cuts (y_k-lo)(y_k-hi)<=0 (linear in X)
  bound_rlt : + g_i (y_k - lo_k) <= 0 / g_i (hi_k-y_k) <= 0 (sound, linear in X)
  all       : rlt + bounds + bound_rlt + box-RLT

For every solve we also run the **rigorous dual-feasible repair** (certificate.py
`repair_dual=True`) and record the rigorous lower bound.  Soundness audits:
  * every cut family is evaluated on sampled truly-feasible points (GT / ALM /
    random q + LP-feasible t)  -> max violation must be <= 1e-8;
  * bound_with_cuts >= base_bound - 1e-6 (naive and rigorous);
  * no NaN;  if rank-1 then bound <= J + 1e-6.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import qpep_core as core  # noqa: E402
import certificate as C  # noqa: E402
from certificate import shor_bound  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = [(0.02, 0.10), (0.05, 0.20), (0.10, 0.30), (0.20, 0.40)]
SEEDS = [0, 1, 2, 3]
import os
if os.environ.get("ROUND_SMOKE"):        # 冒烟测试：只跑 1 个 cell / 1 个 seed
    CELLS = CELLS[:1]
    SEEDS = [0]
VARIANTS = {
    "base": {},
    "rlt": dict(add_rlt=True),
    "bounds": dict(add_bounds=True),
    "bound_rlt": dict(add_bound_rlt=True),
    "all": dict(add_rlt=True, add_bounds=True, add_bound_rlt=True, add_box_rlt=True),
}
SOLVERS = [("SCS", dict(eps=1e-9, max_iters=50000)), ("CLARABEL", dict(eps=1e-9))]
EPS_CERT = 1e-6


def cut_families(A, B):
    """返回 {family: [M matrices]}，每行表示 <M,X> <= b（b=0 或单侧界）。"""
    n = 14
    e14 = C._e14(n)
    cl = [C._v2_row(A, B, j, n) for j in range(A.shape[0])]
    fams = {}
    fams["g"] = [(C._g_matrix(c, e14), 0.0) for c in cl]
    fams["rlt"] = [(-C._sym(np.outer(cl[i], cl[j])), 0.0)
                   for i in range(len(cl)) for j in range(i, len(cl))]
    fams["bounds"] = C._bound_specs(A, B, n)
    fams["bound_rlt"] = C._bound_rlt_specs(A, B, n)
    fams["box_rlt"] = C._box_rlt_specs(A, B, n)
    return fams


def sample_feasible(A, B, rng, n_try=120, n_max=25):
    """随机可行点：随机 q（tilt 锥内）+ 解 t 的可行性 LP。返回 [(q,t)]。"""
    out = []
    for _ in range(n_try):
        q = rng.normal(size=4)
        q[1] *= 0.04
        q[2] *= 0.04
        q = q / np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        x = core.v2(q)
        res = linprog(np.zeros(3), A_ub=np.asarray(B, float), b_ub=-np.asarray(A, float) @ x,
                      bounds=[(-10.0, 10.0)] * 3, method="highs")
        if not res.success:
            continue
        t = np.asarray(res.x, float)
        if core.max_viol(A, B, q, t) > 1e-9:
            continue
        out.append((q, t))
        if len(out) >= n_max:
            break
    return out


def audit_cuts(inst, A, B, rng):
    """在所有采样可行点上检查每个切族：返回 max_family_violation (<=0 表示满足)。"""
    pts = [("gt", inst["q_gt"], inst["t_gt"])]
    for s in range(3):
        alm = core.alm_qpep(inst["M0"], inst["D0"], inst["G0"], A, B,
                            inst["q_gt"], inst["t_gt"], rho0=10.0, n_starts=6,
                            rng=np.random.default_rng(200 + s))
        if core.max_viol(A, B, alm["q"], alm["t"]) <= 1e-9:
            pts.append((f"alm{s}", alm["q"], alm["t"]))
    for k, (q, t) in enumerate(sample_feasible(A, B, rng)):
        pts.append((f"lp{k}", q, t))
    fams = cut_families(A, B)
    worst = {}
    for fam, specs in fams.items():
        mx = -np.inf
        for _nm, q, t in pts:
            y = np.concatenate([core.v2(q), t, [1.0]])
            X = np.outer(y, y)
            for (M, b) in specs:
                mx = max(mx, float(np.sum(M * X) - b))
        worst[fam] = float(mx)
    return worst, [p[0] for p in pts]


def _fmt(x, spec, dash="—"):
    """None/NaN/失败行安全格式化（CLARABEL 在少数实例上会直接 raise）。"""
    try:
        if x is None:
            return dash
        v = float(x)
        if not np.isfinite(v):
            return dash
        return format(v, spec)
    except Exception:
        return dash


def main():
    t_start = time.time()
    rng = np.random.default_rng(20261127)
    rows = []
    audits = []
    for noise, bias in CELLS:
        for seed in SEEDS:
            inst = core.make_instance(seed, 16, noise=noise, bias_z=bias)
            M0, D0, G0, A, B = inst["M0"], inst["D0"], inst["G0"], inst["A"], inst["B"]
            alm = core.alm_qpep(M0, D0, G0, A, B, inst["q_gt"], inst["t_gt"],
                                rho0=10.0, n_starts=6, rng=np.random.default_rng(seed))
            J = core.obj_value(alm["q"], alm["t"], M0, D0, G0)
            feas = core.max_viol(A, B, alm["q"], alm["t"]) <= 1e-6
            worst, pt_names = audit_cuts(inst, A, B, rng)
            audits.append(dict(noise=noise, bias=bias, seed=seed, n_points=len(pt_names),
                               worst_violation=worst))
            print(f"[cell {noise:.2f}/{bias:.2f} seed {seed}] J={J:.6e} feas={feas} "
                  f"audit={ {k: round(v, 10) for k, v in worst.items()} }", flush=True)
            for vname, vkw in VARIANTS.items():
                for sname, skw in SOLVERS:
                    r = shor_bound(M0, D0, G0, A, B, solver=sname,
                                   repair_dual=True, **skw, **vkw)
                    rec = dict(noise=noise, bias=bias, seed=seed, J=float(J),
                               alm_feasible=bool(feas), variant=vname, solver=sname)
                    for k in ("value", "status", "time_s", "rigorous", "ratio", "lam1",
                              "feas_ineq", "eq_resid", "n_eq", "n_ineq", "dual_obj",
                              "dual_shift", "dual_lam_min", "dual_n_negative_lam",
                              "dual_method", "x_lam_min", "rank1"):
                        v = r.get(k)
                        rec[k] = (bool(v) if isinstance(v, (bool, np.bool_)) else
                                  (float(v) if isinstance(v, (int, float, np.floating)) else v))
                    rec["gap"] = float(J - r["value"]) if np.isfinite(r["value"]) else float("nan")
                    rec["rigorous_gap"] = float(J - r["rigorous"]) if np.isfinite(r["rigorous"]) else float("nan")
                    rec["cert_naive"] = bool(feas and np.isfinite(r["value"]) and J - r["value"] <= EPS_CERT)
                    rec["cert_rigorous"] = bool(feas and np.isfinite(r["rigorous"]) and J - r["rigorous"] <= EPS_CERT)
                    rec["rigorous_valid"] = bool(np.isfinite(r["rigorous"]) and
                                                 np.isfinite(r.get("dual_lam_min", np.nan)) and
                                                 r.get("dual_lam_min", -1) >= -1e-10)
                    rows.append(rec)
            json.dump(dict(rows=rows, audits=audits), open(OUT / "round_rlt.json", "w"), indent=2)
    print("elapsed", time.time() - t_start, "rows", len(rows))

    # ---------------- soundness checks ----------------
    checks = []
    audit_max = {f: max(a["worst_violation"][f] for a in audits)
                 for f in ("g", "rlt", "bounds", "bound_rlt", "box_rlt")}
    checks.append(dict(name="cuts_satisfied_on_sampled_feasible_points",
                       max_violation={k: float(v) for k, v in audit_max.items()},
                       tol=1e-8,
                       ok=bool(all(v <= 1e-8 for v in audit_max.values()))))
    # bound_with_cuts >= base - 1e-6, both naive and rigorous
    for mode in ("value", "rigorous"):
        bad = []
        for noise, bias in CELLS:
            for seed in SEEDS:
                for sname, _ in SOLVERS:
                    grp = {(r["variant"], r["solver"]): r for r in rows
                           if r["noise"] == noise and r["seed"] == seed}
                    b = grp.get(("base", sname))
                    if b is None or not np.isfinite(b[mode]):
                        continue
                    for vname in ("rlt", "bounds", "bound_rlt", "all"):
                        v = grp.get((vname, sname))
                        if v is None or not np.isfinite(v[mode]):
                            continue
                        if v[mode] < b[mode] - 1e-6:
                            bad.append(dict(noise=noise, seed=seed, solver=sname,
                                            variant=vname, base=b[mode], cut=v[mode]))
        checks.append(dict(name=f"bound_with_cuts_ge_base[{mode}]", n_violations=len(bad),
                           violations=bad[:8], ok=len(bad) == 0))
    nan_rows = [dict(noise=r["noise"], seed=r["seed"], variant=r["variant"], solver=r["solver"])
                for r in rows if not np.isfinite(r["value"])]
    checks.append(dict(name="no_nan_value", n=len(nan_rows), ok=len(nan_rows) == 0))
    rank1_bad = [dict(noise=r["noise"], seed=r["seed"], variant=r["variant"], solver=r["solver"],
                      gap=r["gap"]) for r in rows
                 if r["rank1"] and (not np.isfinite(r["gap"]) or r["gap"] < -1e-6)]
    checks.append(dict(name="rank1_implies_bound_le_J", n_violations=len(rank1_bad),
                       violations=rank1_bad[:8], ok=len(rank1_bad) == 0))
    inval = [dict(noise=r["noise"], seed=r["seed"], variant=r["variant"], solver=r["solver"],
                  lam_min=r["dual_lam_min"]) for r in rows
             if r["variant"] != "base" and not r["rigorous_valid"]]
    checks.append(dict(name="rigorous_dual_psd_after_repair", n_invalid=len(inval),
                       invalid=inval[:8], ok=len(inval) == 0))

    # ---------------- aggregates ----------------
    def agg(filt):
        rs = [r for r in rows if filt(r)]
        if not rs:
            return {}
        return dict(
            n=len(rs),
            cert_naive=float(np.mean([r["cert_naive"] for r in rs])),
            cert_rigorous=float(np.mean([r["cert_rigorous"] for r in rs])),
            med_gap=float(np.median([r["gap"] for r in rs])),
            med_rgap=float(np.median([r["rigorous_gap"] for r in rs])),
            med_ratio=float(np.median([r["ratio"] for r in rs])),
            med_time=float(np.median([r["time_s"] for r in rs])),
            med_nineq=float(np.median([r["n_ineq"] for r in rs])),
            n_inaccurate=int(np.sum([not str(r["status"]).startswith("optimal") for r in rs])),
        )

    aggs = {}
    for sname, _ in SOLVERS:
        for vname in VARIANTS:
            for noise, bias in CELLS:
                aggs[f"{sname}|{vname}|{noise:.2f}"] = agg(
                    lambda r, s=sname, v=vname, nz=noise: r["solver"] == s and r["variant"] == v and r["noise"] == nz)
    json.dump(dict(rows=rows, audits=audits, checks=checks, aggregates=aggs,
                   meta=dict(cells=CELLS, seeds=SEEDS, variants=list(VARIANTS),
                             solvers=[s for s, _ in SOLVERS], eps_cert=EPS_CERT,
                             start_time=time.strftime("%Y-%m-%d %H:%M:%S"))),
              open(OUT / "round_rlt.json", "w"), indent=2)

    # ---------------- markdown tables ----------------
    L = ["# round-rlt raw tables", "",
         "Generated by `experiments/theory/run_round_rlt.py`. `value` = solver-reported",
         "`<Q,X>`; `rig` = rigorous dual bound (`certificate.py` repair_dual).", ""]
    L += ["## Soundness checks", "", "| check | ok | detail |", "|-|-|-|"]
    for c in checks:
        det = {k: v for k, v in c.items() if k not in ("name", "ok")}
        L.append(f"| {c['name']} | {c['ok']} | `{json.dumps(det, default=float)}` |")
    for sname, _ in SOLVERS:
        L += ["", f"## per-instance ({sname})", "",
              "| noise | seed | J | variant | value | gap | rigorous | rig gap | ratio | λmin(S_rep) | status | n_ineq | t(s) | cert naive | cert rig |",
              "|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|"]
        for noise, bias in CELLS:
            for seed in SEEDS:
                for vname in VARIANTS:
                    r = next(x for x in rows if x["noise"] == noise and x["seed"] == seed
                             and x["variant"] == vname and x["solver"] == sname)
                    L.append("| {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                                 r["noise"], r["seed"], vname, _fmt(r["J"], ".4e"),
                                 _fmt(r["value"], ".4e"), _fmt(r["gap"], "+.2e"),
                                 _fmt(r["rigorous"], ".4e"), _fmt(r["rigorous_gap"], "+.2e"),
                                 _fmt(r["ratio"], ".1e"), _fmt(r["dual_lam_min"], "+.1e"),
                                 r["status"], r["n_ineq"], _fmt(r["time_s"], ".2f"),
                                 r["cert_naive"], r["cert_rigorous"]))
    L += ["", "## per-cell aggregates", "",
          "| solver | variant | noise | n | cert naive | cert rigorous | med gap | med rig gap | med ratio | med n_ineq | med t(s) |",
          "|-|-|-|-|-|-|-|-|-|-|-|"]
    for key, a in aggs.items():
        s, v, nz = key.split("|")
        L.append("| {} | {} | {} | {} | {:.2f} | {:.2f} | {} | {} | {} | {:.0f} | {:.2f} |".format(
                 s, v, nz, a['n'], a['cert_naive'], a['cert_rigorous'],
                 _fmt(a['med_gap'], ".2e"), _fmt(a['med_rgap'], ".2e"),
                 _fmt(a['med_ratio'], ".1e"), a['med_nineq'], a['med_time']))
    (OUT / "round_rlt_tables.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "round_rlt.json", OUT / "round_rlt_tables.md")
    for c in checks:
        print("CHECK", c["name"], "OK" if c["ok"] else "FAIL", {k: v for k, v in c.items() if k not in ("name", "ok", "violations", "invalid")})


if __name__ == "__main__":
    main()
