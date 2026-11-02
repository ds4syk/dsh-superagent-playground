"""Experiment 2 (Q3): Theorem-2 Jacobian  J = -P Mr^+ N_r  vs finite differences,
SDPR-CIFT, dual non-uniqueness, alternative row selections, and the
parameterized-constraint block of N.

Outputs: results/exp2_jacobian.json , figures/fig2_jacobian.png
Run: python exp2_jacobian.py
"""
import itertools, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sdpr_poly as sp

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SCS = dict(eps=1e-12, max_iters=500000)
CLAR = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)


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


def main():
    th = sp.THETA0
    Jex = sp.exact_jacobian_theta(th)          # ground truth (analytic)
    res = {"theta0": th.tolist(), "J_exact": Jex.tolist()}
    methods = {}

    # --- 1. Thm2-IS with two solvers (different dual certificates)
    for name, solver, kw in [("SDPR-IS (SCS)", "SCS", SCS),
                             ("SDPR-IS (Clarabel)", "CLARABEL", CLAR)]:
        sol = sp.solve_sdp(th, solver=solver, solver_kw=kw)
        jac = sp.jacobian_thm2(sol)
        Jt = jac["J"][:, :16] @ sp.Q_BASIS
        methods[name] = dict(Jtheta=Jt.tolist(), rel_err=sp.rel_jac_error(Jt, Jex),
                             lam=sol["lam"].tolist(), eigH=np.linalg.eigvalsh(sol["H"]).tolist(),
                             corankH=sp.certificate_report(sol)["corankH"],
                             condMr=jac["cond"] if "cond" in jac else float(np.linalg.cond(jac["Mr"])),
                             sel=jac["sel"], r=jac["r"],
                             Hx_res=float(np.linalg.norm(sol["H"] @ sol["x"])))
    sol = sp.solve_sdp(th, solver="SCS", solver_kw=SCS)

    # --- 2. SDPR-CIFT (eqs. 22-24)
    cift = sp.cift_jacobian(sol)
    methods["SDPR-CIFT"] = dict(Jtheta=cift["Jtheta"].tolist(),
                                rel_err=sp.rel_jac_error(cift["Jtheta"], Jex),
                                eigHr=cift["eigHr"].tolist(), condMr=cift["condMr"],
                                sel=cift["sel"], lam_r=cift["lam_r"].tolist())

    # --- 3. finite differences
    fd = {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        Jfd = sp.fd_jacobian_theta(th, h=h, use_sdp=True, solver="SCS", solver_kw=SCS)
        Jfx = sp.fd_jacobian_theta(th, h=h, use_sdp=False)
        fd["h=%.0e" % h] = dict(
            sdp_fd_rel_err=sp.rel_jac_error(Jfd, Jex),
            sdp_fd_vs_thm2=sp.rel_jac_error(Jfd, np.array(methods["SDPR-IS (SCS)"]["Jtheta"])),
            exact_fd_rel_err=sp.rel_jac_error(Jfx, Jex))
        print("[fd h=%.0e] SDP-FD vs exact %.2e | exact-FD vs exact %.2e"
              % (h, fd["h=%.0e" % h]["sdp_fd_rel_err"], fd["h=%.0e" % h]["exact_fd_rel_err"]))
    res["fd"] = fd
    res["methods"] = methods

    # --- 4. selection invariance (which r rows of G are kept)
    G = np.vstack([C @ sol["x"] for C in sp.CONS] + [sp.A0 @ sol["x"]])
    sel_stats = []
    for sel in itertools.combinations(range(G.shape[0]), 3):
        if np.linalg.matrix_rank(G[list(sel)]) < 3:
            continue
        jac = sp.jacobian_thm2(sol, sel=list(sel))
        Jt = jac["J"][:, :16] @ sp.Q_BASIS
        sel_stats.append(dict(sel=list(sel), rel_err=sp.rel_jac_error(Jt, Jex),
                              condMr=float(np.linalg.cond(jac["Mr"])), Jtheta=Jt))
    spread = max(sp.rel_jac_error(s["Jtheta"], sel_stats[0]["Jtheta"]) for s in sel_stats)
    res["selection_invariance"] = dict(
        n=len(sel_stats),
        max_rel_err=float(max(s["rel_err"] for s in sel_stats)),
        max_condMr=float(max(s["condMr"] for s in sel_stats)),
        max_spread=float(spread),
        selections=[s["sel"] for s in sel_stats])
    print("[selection] %d independent selections, max rel err %.2e, spread %.2e"
          % (len(sel_stats), res["selection_invariance"]["max_rel_err"], spread))

    # --- 5. dual non-uniqueness: walk the dual optimal face (H >= 0 kept)
    certs = sp.alternative_certificates(sol, n_grid=401, tmax=1.0)
    Js = []
    for t, lam_t, H_t, cr, ev in certs:
        J, Jt, _ = sp.jacobian_from_H(sol, H_t, lam_t)
        Js.append(dict(t=t, rel_err=sp.rel_jac_error(Jt, Jex),
                       corankH=sp.corank(H_t), eigH=ev.tolist(),
                       lam=lam_t.tolist(), Jtheta=Jt))
    J0 = Js[0]["Jtheta"]
    spread_cert = max(sp.rel_jac_error(c["Jtheta"], J0) for c in Js)
    res["dual_nonuniqueness"] = dict(
        n_valid_certificates=len(Js),
        t_min=float(min(c["t"] for c in Js)), t_max=float(max(c["t"] for c in Js)),
        max_rel_err_vs_exact=float(max(c["rel_err"] for c in Js)),
        max_spread_across_certificates=float(spread_cert),
        coranks=sorted(set(int(c["corankH"]) for c in Js)),
        records=[dict(t=c["t"], rel_err=c["rel_err"], corankH=c["corankH"],
                      eigH=c["eigH"]) for c in Js[::20]])
    print("[dual face] %d valid certificates, t in [%.2f, %.2f], coranks %s, J spread %.2e"
          % (len(Js), res["dual_nonuniqueness"]["t_min"], res["dual_nonuniqueness"]["t_max"],
             res["dual_nonuniqueness"]["coranks"], spread_cert))

    # --- 6. dX consistency:  dX = dx x^T + x dx^T  (SDPR-SDP vs SDPR-IS)
    dxs = []
    for j in range(7):
        e = np.zeros(7); e[j] = 1e-5
        Xp = sp.solve_sdp(th + e, solver="SCS", solver_kw=SCS)["X"]
        Xm = sp.solve_sdp(th - e, solver="SCS", solver_kw=SCS)["X"]
        dX_fd = (Xp - Xm) / 2e-5
        dx_thm = np.array(methods["SDPR-IS (SCS)"]["Jtheta"])[:, j]
        x = sol["x"]
        dX_thm = np.outer(dx_thm, x) + np.outer(x, dx_thm)
        den = max(float(np.linalg.norm(dX_fd)), 1e-8)
        dxs.append(dict(j=j, rel_err=float(np.linalg.norm(dX_thm - dX_fd) / den)))
    res["dX_vs_dx"] = dict(per_column=dxs,
                           max_rel_err=float(max(d["rel_err"] for d in dxs[1:])),
                           note="column j=0 is structurally zero (dx*/dtheta0=0)")
    print("[dX] max rel err of  dx x^T + x dx^T  vs  FD(dX): %.2e" % res["dX_vs_dx"]["max_rel_err"])

    # --- 7. parameterized-constraint blocks of N (paper never exercises them):
    #     replace A1 -> A1 + theta*A3 (A3 is already a constraint, so the SDP
    #     feasible set is *identical* for every theta  =>  dx/dtheta = 0).
    As = [sp.A1 + 0.3 * sp.A3, sp.A2, sp.A3]
    solp = sp.solve_sdp(th, As=As, solver="SCS", solver_kw=SCS)
    jacp = sp.jacobian_thm2(solp, As=As)
    cons_blocks = {}
    for k, D in enumerate([sp.A3, sp.A2, sp.A1]):        # dA1/dtheta candidates
        dnu = np.zeros((len(As) + 1) * 16)
        dnu[16:32] = D.reshape(-1)                       # A1 block
        dx = jacp["J"] @ dnu
        cons_blocks["dA1=%s" % ["A3", "A2", "A1"][k]] = dict(
            dx=dx.tolist(), norm=float(np.linalg.norm(dx)))
    res["parameterized_constraint_block"] = dict(
        note="As = [A1+0.3 A3, A2, A3]: same SDP feasible set for all theta, "
             "so the exact Jacobian vanishes; the N-block for A1 must return 0.",
        blocks=cons_blocks,
        max_norm=float(max(b["norm"] for b in cons_blocks.values())),
        rel_err_at_reference=sp.rel_jac_error(np.array(methods["SDPR-IS (SCS)"]["Jtheta"]), Jex))
    print("[A-block] max |dx| for feasible-set-invariant constraint perturbation: %.2e"
          % res["parameterized_constraint_block"]["max_norm"])

    # --- 8. edge case: exact tie between two global minima (corank 2, X rank 2)
    from scipy.optimize import brentq

    def local_minima(theta):
        d = np.array([i * theta[i] for i in range(1, 7)])
        dc = d.copy()
        while len(dc) > 1 and abs(dc[-1]) == 0:
            dc = dc[:-1]
        roots = np.roots(dc[::-1])
        return sorted([(sp.poly_y(r.real, theta), r.real) for r in roots
                       if abs(r.imag) < 1e-12 and sp.poly_d2y(r.real, theta) > 0])

    def tie_gap(t3):
        t2 = sp.THETA0.copy(); t2[3] = t3
        m = local_minima(t2)
        return ([v for v, x in m if x < 1][0] - [v for v, x in m if x > 1][0])

    tstar = float(brentq(tie_gap, -0.67, -0.65, xtol=1e-15, rtol=1e-15))
    tt = sp.THETA0.copy(); tt[3] = tstar
    sol_tie = sp.solve_sdp(tt, solver="SCS", solver_kw=SCS)
    mins = local_minima(tt)
    tie = dict(theta3_star=tstar, eigX=sol_tie["eigX"].tolist(),
               rank1_ratio=float(sol_tie["rank1_ratio"]), corankH=int(sp.corank(sol_tie["H"])),
               minima=[dict(y=float(v), x=float(x)) for v, x in mins], branches=[])
    for tag, (v, x1b) in zip(["left", "right"], mins):
        x = np.array([1.0, x1b, x1b ** 2, x1b ** 3])
        sol_b = dict(sol_tie); sol_b["x"] = x
        G = np.vstack([C @ x for C in sp.CONS] + [sp.A0 @ x])
        sel = sp.independent_rows(G)
        d2 = sp.poly_d2y(x1b, tt)
        Jex_b = np.zeros((4, 7))
        for j in range(1, 7):
            d = -j * x1b ** (j - 1) / d2
            Jex_b[1, j] = d; Jex_b[2, j] = 2 * x1b * d; Jex_b[3, j] = 3 * x1b ** 2 * d
        J, Jt, info = sp.jacobian_from_H(sol_b, sol_tie["H"], sol_tie["lam"], sel=sel)
        m, kd, _ = sp.sosc_margin(sol_tie["H"], x)
        tie["branches"].append(dict(tag=tag, sel=list(sel), rel_err_branch_jac=float(
            sp.rel_jac_error(Jt, Jex_b)), condMr=float(info["condMr"]),
            sosc_margin=float(m), Hx_res=float(np.linalg.norm(sol_tie["H"] @ x))))
    res["tie_edge_case"] = tie
    print("[tie] theta3*=%.12f  eigX=%.2f,%.2f (rank 2, NOT tight) corankH=%d" %
          (tstar, tie["eigX"][0], tie["eigX"][1], tie["corankH"]))
    for b in tie["branches"]:
        print("      %s branch: Thm.2 (corank-2 cert H) vs branch Jacobian rel err %.2e, "
              "SOSC margin %.3f, cond Mr %.0f"
              % (b["tag"], b["rel_err_branch_jac"], b["sosc_margin"], b["condMr"]))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    a = ax[0]
    labels = ["IS(SCS)", "IS(Clar.)", "CIFT", "FD-SDP", "FD-exact"]
    vals = [methods["SDPR-IS (SCS)"]["rel_err"], methods["SDPR-IS (Clarabel)"]["rel_err"],
            methods["SDPR-CIFT"]["rel_err"], fd["h=1e-05"]["sdp_fd_rel_err"],
            fd["h=1e-05"]["exact_fd_rel_err"]]
    a.bar(labels, vals, color=["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:gray"])
    a.set_yscale("log"); a.set_ylabel(r"$\|J_{est}-J_{true}\|_\infty/\|J_{true}\|_\infty$")
    a.set_title("Jacobian accuracy vs analytic truth (eq. 29)"); a.tick_params(axis="x", rotation=15)

    a = ax[1]
    hs = [float(k.split("=")[1]) for k in fd]
    a.loglog(hs, [fd[k]["sdp_fd_rel_err"] for k in fd], "o-",
             label="FD by re-solving SDP")
    a.loglog(hs, [fd[k]["exact_fd_rel_err"] for k in fd], "s--",
             label="FD on exact minimizer")
    a.axhline(1e-14, color="k", ls=":", lw=0.8)
    a.set_xlabel("finite-difference step $h$"); a.set_ylabel("relative error")
    a.set_title("finite differences converge to Thm.2"); a.legend(fontsize=7)

    a = ax[2]
    ts = np.array([c["t"] for c in Js])
    dev = np.array([sp.rel_jac_error(c["Jtheta"], J0) for c in Js])
    a.semilogy(ts, np.maximum(dev, 1e-17), ".", ms=3, color="tab:purple")
    a.set_xlabel("dual-face parameter $t$  ($H_t=H+t\\sum v_iA_i\\succeq0$)")
    a.set_ylabel("Jacobian deviation from $t=0$")
    a.set_title("%d valid certificates, same $x$-Jacobian" % len(Js))

    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_jacobian.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(RES, "exp2_jacobian.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print("[done] results/exp2_jacobian.json figures/fig2_jacobian.png")


if __name__ == "__main__":
    main()
