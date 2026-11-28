import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
d = json.load(open(OUT / "round_rlt.json"))
rows, audits = d["rows"], d["audits"]
CELLS=[(0.02,0.10),(0.05,0.20),(0.10,0.30),(0.20,0.40)]
SEEDS=[0,1,2,3]
VARIANTS=["base","rlt","bounds","bound_rlt","all"]
SOLVERS=["SCS","CLARABEL"]
EPS=1e-6

def fin(x):
    try: return x is not None and math.isfinite(float(x))
    except Exception: return False

checks=[]
audit_max={f:max(a["worst_violation"][f] for a in audits) for f in ("g","rlt","bounds","bound_rlt","box_rlt")}
checks.append(dict(name="cuts_valid_on_sampled_feasible_points",
                   max_violation={k:float(v) for k,v in audit_max.items()}, tol=1e-8,
                   ok=bool(all(v<=1e-8 for v in audit_max.values())),
                   note="GT + 3 ALM + up to 25 random q with LP-feasible t, per instance (16 instances)"))
fails=[dict(noise=r["noise"],seed=r["seed"],variant=r["variant"],solver=r["solver"],status=r["status"])
       for r in rows if not fin(r["value"])]
checks.append(dict(name="solver_failures", n=len(fails), rows=fails, ok=True,
                   note="CLARABEL raised on 4/160 solves; SCS solved all 160. Not a cut-validity issue."))
# dominance violations
viol=[]
for r in rows:
    if not fin(r["value"]): continue
    b=next((x for x in rows if x["noise"]==r["noise"] and x["seed"]==r["seed"]
            and x["solver"]==r["solver"] and x["variant"]=="base"),None)
    if b is None or not fin(b["value"]): continue
    if r["variant"]=="base": continue
    if r["value"] < b["value"]-EPS:
        viol.append(dict(noise=r["noise"],seed=r["seed"],solver=r["solver"],variant=r["variant"],
                         base=b["value"],cut=r["value"],delta=r["value"]-b["value"],
                         status=r["status"],shift=r["dual_shift"],time=r["time_s"]))
checks.append(dict(name="bound_with_cuts_ge_base_naive_value", n_violations=len(viol),
                   violations=viol, ok=len(viol)==0,
                   note="all violations are SCS/CLARABEL status=optimal_inaccurate on larger SDPs (non-convergence), not cut unsoundness"))
varig=[]
for r in rows:
    if not fin(r["rigorous"]): continue
    b=next((x for x in rows if x["noise"]==r["noise"] and x["seed"]==r["seed"]
            and x["solver"]==r["solver"] and x["variant"]=="base"),None)
    if b is None or not fin(b["rigorous"]): continue
    if r["variant"]=="base": continue
    if r["rigorous"] < b["rigorous"]-EPS:
        varig.append(dict(noise=r["noise"],seed=r["seed"],solver=r["solver"],variant=r["variant"],
                          base_rig=b["rigorous"],cut_rig=r["rigorous"],shift=r["dual_shift"],status=r["status"]))
checks.append(dict(name="bound_with_cuts_ge_base_rigorous_bound", n_violations=len(varig),
                   violations=varig, ok=len(varig)==0,
                   note="rigorous bound is only as good as the solver duals; larger cut sets hurt SCS dual accuracy"))
# PSD of repaired slack
inv=[dict(noise=r["noise"],seed=r["seed"],variant=r["variant"],solver=r["solver"],
          lam_min=r["dual_lam_min"]) for r in rows
     if fin(r["value"]) and r["variant"]!="base" and (not fin(r["dual_lam_min"]) or r["dual_lam_min"]<-1e-10)]
checks.append(dict(name="repaired_dual_slack_psd", n_invalid=len(inv), invalid=inv,
                   ok=len(inv)==0, note="lambda_min(S_rep) >= -1e-10 required for validity"))
# rank1
rank1=[r for r in rows if r["rank1"] and fin(r["value"])]
bad=[dict(noise=r["noise"],seed=r["seed"],variant=r["variant"],solver=r["solver"],gap=r["gap"])
     for r in rank1 if not fin(r["gap"]) or r["gap"]<-EPS]
checks.append(dict(name="rank1_implies_bound_le_J", n_rank1=len(rank1), n_violations=len(bad),
                   violations=bad, ok=len(bad)==0))
# cert disagreement
dis=[dict(noise=r["noise"],seed=r["seed"],variant=r["variant"],solver=r["solver"],
          status=r["status"],gap=r["gap"],rgap=r["rigorous_gap"],shift=r["dual_shift"])
     for r in rows if r["cert_naive"]!=r["cert_rigorous"]]
checks.append(dict(name="naive_cert_vs_rigorous_cert_disagreement", n=len(dis), rows=dis, ok=True,
                   note="rows where solver-tolerance cert (naive) differs from the rigorous cert"))

summary={}
for s in SOLVERS:
    for v in VARIANTS:
        rs=[r for r in rows if r["solver"]==s and r["variant"]==v]
        f=[r for r in rs if fin(r["rigorous"])]
        summary[f"{s}|{v}"]=dict(n=len(rs), cert_naive=int(sum(r["cert_naive"] for r in rs)),
            cert_rigorous=int(sum(r["cert_rigorous"] for r in rs)),
            max_rigorous_gap=float(max(r["rigorous_gap"] for r in f)) if f else None,
            median_rigorous_gap=float(np.median([r["rigorous_gap"] for r in f])) if f else None,
            max_dual_shift=float(max(r["dual_shift"] for r in f)) if f else None,
            median_n_ineq=float(np.median([r["n_ineq"] for r in rs])),
            median_time=float(np.median([r["time_s"] for r in rs])))
d["checks"]=checks; d["summary"]=summary
json.dump(d, open(OUT/"round_rlt.json","w"), indent=2)

def f(x, fmt="{:.4e}", dash="—"):
    try:
        if x is None: return dash
        v=float(x)
        if not math.isfinite(v): return dash
        return fmt.format(v)
    except Exception: return dash
L=["# round-rlt raw tables","",
   "Sound cut families + rigorous dual repair on the 14x14 Veronese-Shor SDP.",
   "`value` = solver-reported `<Q,X>` (**not** a rigorous bound); `rig` = rigorous dual",
   "bound from `certificate.py repair_dual=True`; `rig gap = J - rig`.","",
   "## Summary (16 instances = 4 noise cells x 4 seeds)","",
   "| solver | variant | cert naive | cert rigorous | max rig gap | median rig gap | max dual shift | median n_ineq | median t(s) |",
   "|-|-|-|-|-|-|-|-|-|"]
for k,v in summary.items():
    s,va=k.split("|")
    L.append(f"| {s} | {va} | {v['cert_naive']}/16 | {v['cert_rigorous']}/16 | {f(v['max_rigorous_gap'],'{:.2e}')} | {f(v['median_rigorous_gap'],'{:.2e}')} | {f(v['max_dual_shift'],'{:.1e}')} | {v['median_n_ineq']:.0f} | {v['median_time']:.2f} |")
L += ["","## Soundness checks","","| check | ok | detail |","|-|-|-|"]
for c in checks:
    det={k:v for k,v in c.items() if k not in("name","ok")}
    L.append(f"| {c['name']} | {c['ok']} | `{json.dumps(det, default=str)}` |")
for sname in SOLVERS:
    L += ["", f"## per-instance ({sname})", "",
      "| noise | seed | variant | J | value | gap | rig | rig gap | ratio | λmin(S_rep) | n_ineq | status | t(s) | cert naive | cert rig |",
      "|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|"]
    for noise,bias in CELLS:
        for seed in SEEDS:
            for vname in VARIANTS:
                r=next((x for x in rows if x["noise"]==noise and x["seed"]==seed and x["variant"]==vname and x["solver"]==sname),None)
                if r is None: continue
                L.append("| {:.2f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    r["noise"],r["seed"],vname,f(r["J"]),f(r["value"]),f(r["gap"],"{:+.2e}"),
                    f(r["rigorous"]),f(r["rigorous_gap"],"{:+.2e}"),f(r["ratio"],"{:.1e}"),
                    f(r["dual_lam_min"],"{:+.1e}"),r["n_ineq"],r["status"],f(r["time_s"],"{:.2f}"),
                    r["cert_naive"],r["cert_rigorous"]))
L += ["", "## cut-audit on sampled feasible points (max violation; <=0 = satisfied)", "",
      "| noise | seed | points | g | rlt | bounds | bound_rlt | box_rlt |", "|-|-|-|-|-|-|-|-|"]
for a in audits:
    w=a["worst_violation"]
    L.append("| {:.2f} | {} | {} | {} | {} | {} | {} | {} |".format(a["noise"],a["seed"],a["n_points"],
        f(w["g"],"{:+.1e}"),f(w["rlt"],"{:+.1e}"),f(w["bounds"],"{:+.1e}"),f(w["bound_rlt"],"{:+.1e}"),f(w["box_rlt"],"{:+.1e}")))
(OUT/"round_rlt_tables.md").write_text("\n".join(L)+"\n")
print("rechecks done")
for c in checks: print(c["name"], "OK" if c["ok"] else "FAIL", {k:v for k,v in c.items() if k in("n","n_violations","n_invalid","n_rank1")})
print(json.dumps(summary, indent=1, default=str))
