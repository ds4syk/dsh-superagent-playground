"""Experiment 1 (Q1): certificate / tightness / corank / SOSC for the
Section VI-A sixth-order polynomial example.

Outputs: results/exp1_certificate.json , figures/fig1_certificate.png
Run: python exp1_certificate.py
"""
import json, os, warnings
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sdpr_poly as sp

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SCS_KW = dict(eps=1e-11, max_iters=300000)
SOLVER = "SCS"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def local_minima(theta):
    """all local minima (value, x) of the polynomial, exact."""
    theta = np.asarray(theta, float)
    d = np.array([i * theta[i] for i in range(1, 7)])
    dc = d.copy()
    while len(dc) > 1 and abs(dc[-1]) == 0:
        dc = dc[:-1]
    roots = np.roots(dc[::-1])
    out = [(sp.poly_y(r.real, theta), r.real) for r in roots
           if abs(r.imag) < 1e-12 and sp.poly_d2y(r.real, theta) > 0]
    return sorted(out)


def tie_theta3():
    def gap(t3):
        th = sp.THETA0.copy(); th[3] = t3
        m = local_minima(th)
        left = [v for v, x in m if x < 1.0]; right = [v for v, x in m if x > 1.0]
        return left[0] - right[0]
    return float(brentq(gap, -0.67, -0.65, xtol=1e-15, rtol=1e-15))


def study_case(theta, label, solver=SOLVER, kw=SCS_KW, exact_x=None):
    """solve + certificate report for one theta."""
    sol = sp.solve_sdp(theta, solver=solver, solver_kw=kw)
    rep = sp.certificate_report(sol)
    x1_ex = sp.exact_minimizer(theta)[0]
    y_ex = sp.poly_y(x1_ex, theta)
    out = dict(label=label, theta=np.asarray(theta, float).tolist(),
               status=sol["status"], pstar=float(sol["pstar"]),
               pstar_vs_exact=float(sol["pstar"] - y_ex), y_exact=float(y_ex),
               x1_exact=float(x1_ex),
               x1_sdp=float(sol["x"][1]), x_sdp=sol["x"],
               eigX=sol["eigX"].tolist(), rank1_ratio=float(sol["rank1_ratio"]),
               lam=sol["lam"], H=sol["H"],
               eigH=rep["eigH"], H_psd=rep["H_psd"], corankH=rep["corankH"],
               Hx_res=rep["Hx_res"], rankG=rep["rankG"],
               dim_kerG=rep["dim_kerG"], sosc_min=rep["sosc_min"],
               sosc=rep["sosc"], tight_by_ratio=bool(rep["tightness_ratio"] > 1e5),
               # the paper's practical tightness indicator, App. C, L1315-L1326
               check_tightness_ratio=float(sol["eigX"][0] / sol["eigX"][1])
               if sol["eigX"][1] > 0 else float("inf"))
    # SOSC with the SDP certificate evaluated at the *exact* minimizer
    m, kd, G = sp.sosc_margin(sol["H"], sp.exact_x(theta))
    out["sosc_at_exact_min"] = dict(margin=float(m), dim_kerG=kd,
                                    Hx_exact=float(np.linalg.norm(sol["H"] @ sp.exact_x(theta))))
    return out, sol


def main():
    res = {}

    # ---------------------------------------------------------------- base case
    base, sol = study_case(sp.THETA0, "base (Table I / theta0)")
    res["base"] = base
    print("[base] p*=%.12f  x1=%.8f  rank1=%.2e  eigH=%s  corank=%d  sosc=%.3e"
          % (base["pstar"], base["x1_sdp"], base["rank1_ratio"],
             np.array2string(np.array(base["eigH"]), precision=3),
             base["corankH"], base["sosc_min"]))
    print("       H_psd=%s  Hx_res=%.2e  rankG=%d (m+1=4)  tight=%s"
          % (base["H_psd"], base["Hx_res"], base["rankG"], base["tight_by_ratio"]))

    # at base theta the exact minimizer and the SDP factor agree to solver precision
    base["x_factor_vs_exact_inf"] = float(np.linalg.norm(sol["x"] - sp.exact_x(sp.THETA0), np.inf))

    # ------------------------------------------------- theta3 scan (switch/tie)
    tstar = tie_theta3()
    res["tie_theta3"] = tstar
    grid = np.linspace(-0.80, -0.50, 151)
    scan = dict(theta3=grid.tolist(), x1_sdp=[], x1_exact=[], rank1_ratio=[],
                corankH=[], sosc_min=[], eigX2=[], pstar=[], gap=[])
    for t3 in grid:
        th = sp.THETA0.copy(); th[3] = t3
        s = sp.solve_sdp(th, solver=SOLVER, solver_kw=SCS_KW)
        if s["X"] is None:
            for k in scan: scan[k].append(float("nan"))
            continue
        rep = sp.certificate_report(s)
        scan["x1_sdp"].append(float(s["x"][1]))
        scan["x1_exact"].append(float(sp.exact_minimizer(th)[0]))
        scan["rank1_ratio"].append(float(s["rank1_ratio"]))
        scan["corankH"].append(int(rep["corankH"]))
        scan["sosc_min"].append(float(rep["sosc_min"]) if np.isfinite(rep["sosc_min"]) else None)
        scan["eigX2"].append(float(s["eigX"][1] / s["eigX"][0]))
        scan["pstar"].append(float(s["pstar"]))
        scan["gap"].append(float(s["pstar"] - sp.poly_y(sp.exact_minimizer(th)[0], th)))
    res["theta3_scan"] = scan
    print("[scan] %d theta3 values, tie at theta3*=%.12f" % (len(grid), tstar))

    # ------------------------------------------------------------- exact tie case
    th_tie = sp.THETA0.copy(); th_tie[3] = tstar
    tie, sol_tie = study_case(th_tie, "tie (two global minima)")
    mins = local_minima(th_tie)
    tie["local_minima"] = [dict(y=float(v), x=float(x)) for v, x in mins]
    # the two rank-1 factors spanning range(X*) at the tie
    w, V = np.linalg.eigh(sol_tie["X"])
    Xr = V[:, np.argsort(w)[::-1][:2]]
    tie["eigX_nonzero"] = [float(v) for v in np.sort(w)[::-1][:2]]
    # H kills both global minimizers -> they are both certified
    xl = np.array([1.0, mins[0][1], mins[0][1] ** 2, mins[0][1] ** 3])
    xr = np.array([1.0, mins[-1][1], mins[-1][1] ** 2, mins[-1][1] ** 3])
    Htie = sol_tie["H"]
    tie["Hx_left"] = float(np.linalg.norm(Htie @ xl))
    tie["Hx_right"] = float(np.linalg.norm(Htie @ xr))
    v = xl - xr
    tie["sosc_at_global_min"] = []
    for xg in (xl, xr):
        m, kd, _ = sp.sosc_margin(Htie, xg)
        tie["sosc_at_global_min"].append(dict(margin=float(m), dim_kerG=kd,
                                              Hx=float(np.linalg.norm(Htie @ xg))))
    tie["sosc_failure_direction"] = dict(
        v=v, v_norm=float(np.linalg.norm(v)),
        vHv=float(v @ Htie @ v),
        Gv=float(np.linalg.norm(np.vstack([C @ v for C in sp.CONS] + [sp.A0 @ v]))))
    res["tie"] = tie
    print("[tie] minima y=%s x=%s" % ([round(v, 10) for v, _ in mins], [round(x, 8) for _, x in mins]))
    print("      eigX=%s  rank1=%.3e (NOT tight)  corankH=%d  Hx_left=%.1e Hx_right=%.1e  vHv=%.2e"
          % (np.array2string(np.array(tie["eigX"]), precision=3), tie["rank1_ratio"],
             tie["corankH"], tie["Hx_left"], tie["Hx_right"], tie["sosc_failure_direction"]["vHv"]))

    # ------------------------------------------------------- random theta sample
    rng = np.random.default_rng(0)
    rand = {}
    for sigma in (0.05, 0.2, 0.5):
        recs = []
        n_ok = 0
        while n_ok < 40:
            th = sp.THETA0 + rng.normal(0.0, sigma, 7)
            if th[-1] < 0.02:      # keep the polynomial bounded below (theta6>0)
                continue
            s = sp.solve_sdp(th, solver=SOLVER, solver_kw=SCS_KW)
            if s["status"] not in ("optimal", "optimal_inaccurate") or s["X"] is None:
                continue
            n_ok += 1
            rep = sp.certificate_report(s)
            recs.append(dict(rank1_ratio=float(s["rank1_ratio"]), corankH=int(rep["corankH"]),
                             corank_confident=rep["corank_confident"],
                             eigH_rel=rep["eigH_rel"],
                             H_psd=bool(rep["H_psd"]), sosc=bool(rep["sosc"]),
                             sosc_min=float(rep["sosc_min"]),
                             tight=bool(s["rank1_ratio"] > 1e5),
                             gap=float(s["pstar"] - sp.poly_y(sp.exact_minimizer(th)[0], th))))
        rand["sigma_%.2f" % sigma] = dict(
            n=len(recs), records=recs,
            frac_tight=float(np.mean([r["tight"] for r in recs])),
            frac_corank1=float(np.mean([r["corankH"] == 1 for r in recs])),
            frac_corank1_confident=float(np.mean([r["corank_confident"] == "1" for r in recs])),
            frac_Hpsd=float(np.mean([r["H_psd"] for r in recs])),
            frac_sosc=float(np.mean([r["sosc"] for r in recs])),
            min_rank1_ratio=float(np.min([r["rank1_ratio"] for r in recs])),
            max_abs_gap=float(np.max(np.abs([r["gap"] for r in recs]))))
        print("[random sigma=%.2f] tight=%.2f corank1=%.2f Hpsd=%.2f sosc=%.2f min_rank1=%.1e"
              % (sigma, rand["sigma_%.2f" % sigma]["frac_tight"],
                 rand["sigma_%.2f" % sigma]["frac_corank1"],
                 rand["sigma_%.2f" % sigma]["frac_Hpsd"],
                 rand["sigma_%.2f" % sigma]["frac_sosc"],
                 rand["sigma_%.2f" % sigma]["min_rank1_ratio"]))
    res["random"] = rand

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    a = ax[0]
    a.plot(scan["theta3"], scan["x1_exact"], "k-", lw=1.2, label="exact global min (roots)")
    a.plot(scan["theta3"], scan["x1_sdp"], "o", ms=2.2, color="tab:red", label="SDP rank-1 factor")
    a.axvline(tstar, color="tab:gray", ls="--", lw=1, label="tie $\\theta_3^*$")
    a.set_xlabel(r"$\theta_3$"); a.set_ylabel(r"$x_1^\star$"); a.set_title("global minimizer switches")
    a.legend(fontsize=7, loc="best")

    a = ax[1]
    a.semilogy(scan["theta3"], np.maximum(scan["rank1_ratio"], 1.0), color="tab:blue", lw=1.0)
    a.axhline(1e5, color="tab:orange", ls="--", lw=1, label="check_tightness threshold $10^5$")
    a.axvline(tstar, color="tab:gray", ls="--", lw=1)
    a.set_xlabel(r"$\theta_3$"); a.set_ylabel(r"$\lambda_1(X^\star)/\lambda_2(X^\star)$")
    a.set_title("tightness indicator"); a.legend(fontsize=7)

    a = ax[2]
    a.plot(scan["theta3"], scan["corankH"], color="tab:green", lw=1.2, label="corank $H$")
    sm = np.array([v if v is not None else np.nan for v in scan["sosc_min"]])
    a2 = a.twinx()
    a2.plot(scan["theta3"], sm, color="tab:purple", lw=1.0, label="SOSC margin")
    a.set_xlabel(r"$\theta_3$"); a.set_ylabel("corank $H$")
    a2.set_ylabel(r"$\min_{v\in\ker G} v^T H v$")
    a.set_title("corank / SOSC")
    lines = a.get_lines() + a2.get_lines()
    a.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="best")
    for a_ in (a2,):
        pass
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_certificate.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(RES, "exp1_certificate.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print("[done] results/exp1_certificate.json figures/fig1_certificate.png")


if __name__ == "__main__":
    main()
