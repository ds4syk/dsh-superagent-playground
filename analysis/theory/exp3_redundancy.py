"""Experiment 3 (Q3): redundant constraints, singular KKT matrix M, and the
pseudo-inverse formula  J = -P Mr^+ N_r  (Thm.2 / Implicit Selections).

Outputs: results/exp3_redundancy.json , figures/fig3_redundancy.png
Run: /home/syk/miniconda3/envs/WM/bin/python exp3_redundancy.py
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


def sv_ratio(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s, float(s[-1] / s[0]) if s[0] > 0 else float("nan")


def analyse_case(name, As, extra=(), th=None):
    """As: QCQP constraint matrices (m of them); A0 added automatically."""
    th = sp.THETA0 if th is None else th
    Jex = sp.exact_jacobian_theta(th)
    sol = sp.solve_sdp(th, extra_cons=extra, solver="SCS", solver_kw=SCS, As=As)
    x = sol["x"]; n = x.size; m = len(As)
    G = np.vstack([C @ x for C in As] + [sp.A0 @ x])       # (m+1) x n
    svG, Grank = np.linalg.svd(G, compute_uv=False), np.linalg.matrix_rank(G)
    sel = sp.independent_rows(G)                            # r independent rows
    r = len(sel); Gr = G[sel]
    # --- full (singular) KKT matrix M
    M = 2 * np.block([[sol["H"], G.T], [G, np.zeros((m + 1, m + 1))]])
    # --- reduced (full row rank) Mr
    Mr = 2 * np.block([[sol["H"], Gr.T], [Gr, np.zeros((r, r))]])
    Nfull = sp.jacobian_N(x, sol["lam"], m)
    # reduce rows: keep n stationarity rows + selected constraint rows
    Pfull = np.hstack([np.eye(n), np.zeros((n, m + 1))])
    # --- Thm.2 J via Mr^+ (pseudo-inverse formula)
    jac = sp.jacobian_thm2(sol, sel=sel, As=As)
    Jt = jac["J"][:, :16] @ sp.Q_BASIS
    # --- certificate-reconstruction variants (the solver dual can be poor when
    #     the problem is extremely degenerate): (i) least-squares multipliers
    #     enforcing G^T lam = -Qx  (stationarity), (ii) SDPR-CIFT eq. (22)
    lam_ls = np.linalg.lstsq(G.T, -sol["Q"] @ x, rcond=None)[0]
    H_ls = sol["Q"] + lam_ls[-1] * sp.A0 + sum(lam_ls[i] * As[i] for i in range(m))
    sol_ls = dict(sol); sol_ls["lam"] = lam_ls; sol_ls["H"] = H_ls
    jac_ls = sp.jacobian_thm2(sol_ls, sel=sel, As=As)
    J_ls = jac_ls["J"][:, :16] @ sp.Q_BASIS
    cift = sp.cift_jacobian(sol, As=As)
    # --- naive classical IFT: J = -P M^{-1} N  with the singular M
    try:
        Jnaive = -Pfull @ np.linalg.solve(M, Nfull)
        naive_err = sp.rel_jac_error(Jnaive[:, :16] @ sp.Q_BASIS, Jex)
        naive_note = "solved (numerically singular)"
    except np.linalg.LinAlgError as e:
        naive_err = float("nan")
        naive_note = "LinAlgError: singular matrix"
    # --- how much does the row-selection affect J?
    sels = [s for s in itertools.combinations(range(m + 1), r)
            if np.linalg.matrix_rank(G[list(s)]) == r]
    Jlist = []
    conds = []
    for s in sels:
        j = sp.jacobian_thm2(sol, sel=list(s), As=As)
        Jlist.append(j["J"][:, :16] @ sp.Q_BASIS)
        conds.append(float(np.linalg.cond(j["Mr"])))
    spread = max(sp.rel_jac_error(J, Jlist[0]) for J in Jlist) if Jlist else float("nan")
    out = dict(
        name=name, m=m, n_constraint_rows=m + 1, G_rank=int(Grank),
        svG=svG.tolist(),
        n_redundant=int(m + 1 - Grank),
        size_M=int(M.shape[0]),
        cond_M=float(np.linalg.cond(M)),
        sv_ratio_M=float(sv_ratio(M)[1]),
        eig_M=np.linalg.eigvalsh(M).tolist(),
        size_Mr=int(Mr.shape[0]),
        cond_Mr=float(np.linalg.cond(Mr)),
        rows_kept=sel,
        rel_err_thm2=float(sp.rel_jac_error(Jt, Jex)),
        Hx_res_solver=float(np.linalg.norm(sol["H"] @ x)),
        eigH_solver=np.linalg.eigvalsh(sol["H"]).tolist(),
        eigH_ls=np.linalg.eigvalsh(H_ls).tolist(),
        Hx_res_ls=float(np.linalg.norm(H_ls @ x)),
        rel_err_thm2_LS=float(sp.rel_jac_error(J_ls, Jex)),
        condMr_LS=float(np.linalg.cond(jac_ls["Mr"])),
        rel_err_cift=float(sp.rel_jac_error(cift["Jtheta"], Jex)),
        condMr_cift=float(cift["condMr"]),
        naive_ift=dict(note=naive_note, rel_err=naive_err),
        n_independent_selections=len(sels),
        selection_spread=float(spread),
        selection_cond_Mr=(float(min(conds)), float(max(conds))),
        rank1_ratio=float(sol["rank1_ratio"]),
        corankH=sp.certificate_report(sol, As=As)["corankH"],
        sosc=sp.certificate_report(sol, As=As)["sosc_min"],
        status=sol["status"])
    return out, sol, Jt, Jex, Mr, M


def main():
    res = {}
    th = sp.THETA0

    # ---- case A: the paper's own setup (A3 is a redundant QCQP constraint)
    outA, solA, JtA, Jex, MrA, MA = analyse_case("paper setup (1 redundant: A3)", list(sp.CONS))
    res["case_A"] = outA
    print("[A] G rank=%d/%d  cond(M)=%.2e  cond(Mr)=%.2f  Thm2 err=%.2e  naive-IFT: %s"
          % (outA["G_rank"], outA["n_constraint_rows"], outA["cond_M"],
             outA["cond_Mr"], outA["rel_err_thm2"], outA["naive_ift"]["note"]))
    print("    eig M =", np.array2string(np.array(outA["eig_M"]), precision=2))
    print("    %d independent row selections, spread=%.2e, cond(Mr) in [%.1f, %.1f]"
          % (outA["n_independent_selections"], outA["selection_spread"],
             *outA["selection_cond_Mr"]))
    for key in ("case_A", "case_B", "case_C"):
        pass
    print("    Hx res (solver dual)=%.2e  eigH0=%.1e | LS: Hx=%.1e, Thm2 err=%.1e | CIFT err=%.1e"
          % (outA["Hx_res_solver"], outA["eigH_solver"][0], outA["Hx_res_ls"],
             outA["rel_err_thm2_LS"], outA["rel_err_cift"]))

    # ---- case B: add 3 further redundant linear combinations of A1,A2,A3
    extra = [2 * sp.A3, sp.A1 + sp.A2, sp.A2 - sp.A1]
    AsB = list(sp.CONS) + extra
    outB, solB, JtB, JexB, MrB, MB = analyse_case(
        "3 extra redundant constraints", AsB, extra=extra)
    res["case_B"] = outB
    print("[B] G rank=%d/%d  cond(M)=%.2e  cond(Mr)=%.2f  Thm2 err=%.2e  naive-IFT: %s"
          % (outB["G_rank"], outB["n_constraint_rows"], outB["cond_M"],
             outB["cond_Mr"], outB["rel_err_thm2"], outB["naive_ift"]["note"]))
    print("    %d independent row selections, spread=%.2e, cond(Mr) in [%.1f, %.1f]"
          % (outB["n_independent_selections"], outB["selection_spread"],
             *outB["selection_cond_Mr"]))
    print("    Hx res (solver dual)=%.2e  eigH0=%.1e | LS: Hx=%.1e, Thm2 err=%.1e | CIFT err=%.1e"
          % (outB["Hx_res_solver"], outB["eigH_solver"][0], outB["Hx_res_ls"],
             outB["rel_err_thm2_LS"], outB["rel_err_cift"]))

    # ---- case C: duplicate every redundant constraint twice (extreme degeneracy)
    extra2 = [sp.A3, sp.A3, 3 * sp.A3, sp.A1 + sp.A2]
    AsC = list(sp.CONS) + extra2
    outC, solC, JtC, _, MrC, MC = analyse_case("duplicated redundant constraints", AsC,
                                               extra=extra2)
    res["case_C"] = outC
    print("[C] G rank=%d/%d  cond(M)=%.2e  cond(Mr)=%.2f  Thm2 err=%.2e  naive-IFT: %s"
          % (outC["G_rank"], outC["n_constraint_rows"], outC["cond_M"],
             outC["cond_Mr"], outC["rel_err_thm2"], outC["naive_ift"]["note"]))
    print("    Hx res (solver dual)=%.2e | LS: Thm2 err=%.1e | CIFT err=%.1e"
          % (outC["Hx_res_solver"], outC["rel_err_thm2_LS"], outC["rel_err_cift"]))

    # ---- how the number of redundant constraints scales the conditioning
    scaling = []
    for k in range(6):
        extra_k = [sp.A3] + [sp.A3] * k
        out, sol, Jt, Jexk, Mr, M = analyse_case("k=%d" % k, list(sp.CONS) + extra_k,
                                                 extra=extra_k)
        scaling.append(dict(k=k, size_M=out["size_M"], cond_M=out["cond_M"],
                            cond_Mr=out["cond_Mr"], rel_err=out["rel_err_thm2"],
                            naive=out["naive_ift"].get("rel_err")))
    res["scaling"] = scaling
    print("[scaling] k extras: cond(M) =", [("%.1e" % s["cond_M"]) for s in scaling])

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    a = ax[0]
    for M, lab, st in [(MA, "M (full KKT, %dx%d)" % MA.shape, "o-"),
                       (MrA, "Mr (reduced, %dx%d)" % MrA.shape, "s--")]:
        s = np.linalg.svd(M, compute_uv=False)
        a.semilogy(np.arange(1, len(s) + 1), s / s[0], st, label=lab)
    a.set_xlabel("singular value index"); a.set_ylabel(r"$\sigma_i/\sigma_1$")
    a.set_title("case A (paper setup: 1 redundant constraint)"); a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[1]
    ks = [s["k"] for s in scaling]
    a.semilogy(ks, [s["cond_M"] for s in scaling], "o-", label=r"cond $M$ (singular)")
    a.semilogy(ks, [s["cond_Mr"] for s in scaling], "s--", label=r"cond $M_r$")
    a.axhline(1e16, color="k", ls=":", lw=0.8)
    a.set_xlabel("number of duplicate redundant constraints added")
    a.set_ylabel("condition number")
    a.set_title("redundancy degrades $M$ but not $M_r$"); a.legend(fontsize=7)
    a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_redundancy.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(RES, "exp3_redundancy.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print("[done] results/exp3_redundancy.json figures/fig3_redundancy.png")


if __name__ == "__main__":
    main()
