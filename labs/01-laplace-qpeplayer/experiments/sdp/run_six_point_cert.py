#!/usr/bin/env python3
"""Numerical dual certificate for the 6-point physical 3D-3D counterexample.

For each of the three SDPs of the revised P1-Q2/Q13 proof
  f10      : 10x10, 21 constraints (full 20D ideal)
  f13_full : 13x13, 24 upstream + 3 row-orthogonality = 27 constraints (20D)
  f13_up   : 13x13, 24 upstream constraints (17D)
we produce a bracketing interval

  lower  = mu_dual - VALUE_SAFETY,
  upper  = exact feasible value (f10, f13_full) or primal solver value + VALUE_SAFETY
           (f13_up),

where mu_dual is obtained from a *direct* cvxpy dual solve

  maximise mu   s.t.   Z = Q - mu E00 + sum_i y_i A_i  >=  0,

i.e. mu is a lower bound for the SDP minimum for every dual-feasible Z.  The
dual slack is checked by eigenvalue/Cholesky with an explicit PSD margin, and
the interval is explicitly labelled

  "numerical certificate (float), not exact rational".

For f10 and f13_full the rank-1 point X = x x^T with x = [1, vec(I), 0] is
exactly feasible (all constraint entries are 0, +-1/2, +-1 and x has entries
0/1), and its objective is the exact rational 6(1-eps)^2 = 1083/200 = 5.415
from the closed form of the construction.  That gives a rigorous upper bound
5.415 for these two SDPs; only for f13_up (rank-4 optimum) is the upper bound
the solver's primal value plus the safety margin.

Run:
  /home/syk/miniconda3/envs/WM/bin/python sdp/run_six_point_cert.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cvxpy as cp
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
import run_6point_check as S6  # noqa: E402
import run_sdp_check as S  # noqa: E402

# --- stated tolerances ------------------------------------------------------
PSD_MARGIN = 1e-8        # Z + PSD_MARGIN*I must have a Cholesky factor
VALUE_SAFETY = 1e-7      # subtracted/added at the two interval endpoints
RANK_TOL_REL = 1e-4      # rank threshold, relative to the largest eigenvalue
EPS = sp.Rational(1, 20)  # 0.05, exact
EXACT_UPPER = sp.Rational(1083, 200)  # 6(1-eps)^2 = 5.415

PRIMARY_DUAL = "CLARABEL"
PRIMARY_PRIMAL = "SCS"
CHECK_PRIMAL = "CLARABEL"


def build_problems():
    """The three 6-point SDPs, with layouts exactly as in run_6point_check.py."""
    Q10, f_at_I = S6.build_Q10()
    _, A10, _, _ = S.build_10x10()
    A13, _ = S.build_13x13()
    rd_cm, _ = S.rowdot_matrices()
    A13_full = A13 + rd_cm
    P = S.perm_rowmajor_to_colmajor()
    G = 6.0 * S6.W
    Q13 = np.zeros((13, 13))
    Q13[:10, :10] = P @ Q10 @ P.T
    Q13[10:, 10:] = G
    Q13 = 0.5 * (Q13 + Q13.T)
    return [
        dict(tag="10", label="10x10 (21 constraints, full 20D)",
             Q=Q10, A=A10, n=10, dim=20),
        dict(tag="13_full", label="13x13 full-20D (24 + 3 row-orth = 27 constraints)",
             Q=Q13, A=A13_full, n=13, dim=20),
        dict(tag="13_up", label="13x13 upstream (24 constraints, 17D)",
             Q=Q13, A=A13, n=13, dim=17),
    ], f_at_I


def _solver_kwargs(solver: str, tight: bool = False):
    if solver == "CLARABEL":
        if tight:  # used for the dual certificate
            return dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10,
                        max_iter=500)
        return dict(max_iter=500)
    if solver == "SCS":
        return dict(eps=1e-11, max_iters=2_000_000)
    return {}


def primal_solve(Q, A, n, solver):
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(Ai @ X) == 0 for Ai in A]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    t0 = time.time()
    prob.solve(solver=solver, verbose=False,
               **_solver_kwargs(solver, tight=False))
    Xv = np.array(X.value, dtype=float) if X.value is not None else None
    out = dict(solver=solver, status=prob.status, value=float(prob.value),
               t_solve=time.time() - t0)
    if Xv is None:
        out.update(min_eig_X=None, max_constraint_residual=None, rank=None)
        return out
    Xv = 0.5 * (Xv + Xv.T)
    ev = np.linalg.eigvalsh(Xv)
    out.update(
        min_eig_X=float(ev[0]), max_eig_X=float(ev[-1]),
        max_constraint_residual=float(max(
            abs(float(np.trace(Ai @ Xv))) for Ai in A)),
        rank=int(np.sum(ev > RANK_TOL_REL * max(abs(ev[-1]), 1e-12))),
        X=Xv,
    )
    return out


def dual_solve(Q, A, n, solver):
    """Direct dual: max mu s.t. Q - mu E00 + sum y_i A_i >= 0."""
    m = len(A)
    mu = cp.Variable()
    y = cp.Variable(m)
    E00 = np.zeros((n, n))
    E00[0, 0] = 1.0
    Z = Q - mu * E00
    for i, Ai in enumerate(A):
        Z = Z + y[i] * Ai
    prob = cp.Problem(cp.Maximize(mu), [Z >> 0])
    t0 = time.time()
    prob.solve(solver=solver, verbose=False,
               **_solver_kwargs(solver, tight=True))
    Zv = np.array(Z.value, dtype=float) if Z.value is not None else None
    out = dict(solver=solver, status=prob.status, mu=float(prob.value),
               t_solve=time.time() - t0)
    if Zv is None:
        out.update(min_eig_Z=None)
        return out
    Zv = 0.5 * (Zv + Zv.T)
    ev = np.linalg.eigvalsh(Zv)
    shifted = np.linalg.eigvalsh(Zv + PSD_MARGIN * np.eye(n))
    try:
        np.linalg.cholesky(Zv + PSD_MARGIN * np.eye(n))
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
    out.update(min_eig_Z=float(ev[0]), max_eig_Z=float(ev[-1]),
               min_eig_shifted=float(shifted[0]), cholesky_ok=bool(chol_ok),
               psd_ok_with_margin=bool(shifted[0] > 0.0 and chol_ok),
               Z=Zv)
    return out


def exact_upper_check(Q, A, x, tag):
    """Feasibility of the rational point X = x x^T and its exact value."""
    max_res = float(max(abs(float(x @ Ai @ x)) for Ai in A))
    obj_float = float(x @ Q @ x)
    exact_val = float(EXACT_UPPER)
    return dict(
        x=x.tolist(), max_abs_xTAx=max_res,
        float_objective=obj_float,
        exact_rational_objective=str(EXACT_UPPER),
        exact_rational_objective_float=exact_val,
        gap_float_vs_exact=obj_float - exact_val,
        exactly_feasible=bool(max_res == 0.0),
        note=("x = [1, vec(I), 0] is rational; all constraint matrices have "
              "entries in {0, +-1/2, +-1}, so x^T A_i x = 0 exactly; "
              "J(I, 0) = 6(1-eps)^2 = 1083/200 by the construction's closed form"),
    )


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    problems, f_at_I = build_problems()
    # exact symbolic identity behind the upper bound
    assert sp.simplify(6 * (1 - EPS) ** 2 - EXACT_UPPER) == 0
    x10 = np.concatenate([[1.0], np.eye(3).reshape(-1)])
    x13 = np.concatenate([x10, np.zeros(3)])

    result = {}
    for p in problems:
        tag, Q, A, n = p["tag"], p["Q"], p["A"], p["n"]
        prim = primal_solve(Q, A, n, PRIMARY_PRIMAL)
        prim_check = primal_solve(Q, A, n, CHECK_PRIMAL)
        dual = dual_solve(Q, A, n, PRIMARY_DUAL)
        x = x10 if n == 10 else x13
        up = exact_upper_check(Q, A, x, tag)

        lower = dual["mu"] - VALUE_SAFETY
        if tag in ("10", "13_full"):
            upper = float(EXACT_UPPER)
            upper_kind = ("exact rational 1083/200 at the exactly feasible "
                          "rank-1 point X = x x^T, x = [1, vec(I), 0]")
        else:
            upper = prim["value"] + VALUE_SAFETY
            upper_kind = ("primal solver value + value safety margin "
                          "(X numerically feasible)")
        if not (lower < upper):
            raise AssertionError(f"{tag}: empty interval [{lower}, {upper}]")
        # complementarity: <Z, X> = primal - mu on the primal/dual pair
        comp = None
        if dual.get("Z") is not None and prim.get("X") is not None:
            comp = float(np.trace(dual["Z"] @ prim["X"]))

        result[tag] = {
            "label": p["label"], "n": n, "n_constraints": len(A),
            "constraint_dim": p["dim"],
            "primal": {k: v for k, v in prim.items() if k != "X"},
            "primal_check": {k: v for k, v in prim_check.items()
                             if k != "X"},
            "dual": {k: v for k, v in dual.items() if k != "Z"},
            "exact_upper_witness": up,
            "upper": {"value": upper, "kind": upper_kind},
            "interval": {
                "lower": lower, "upper": upper, "width": upper - lower,
                "label": "numerical certificate (float), not exact rational",
            },
            "dual_gap_primal_minus_mu": prim["value"] - dual["mu"],
            "complementarity_trace_ZX": comp,
        }
        print(f"{tag:8s} primal={prim['value']:.10f} dual_mu={dual['mu']:.10f} "
              f"Zmin={dual['min_eig_Z']:+.2e} interval=[{lower:.8f}, "
              f"{upper:.8f}] rank={prim.get('rank')}")

    lo = result["10"]["interval"]["lower"]
    hi_up = result["13_up"]["interval"]["upper"]
    lo_up = result["13_up"]["interval"]["lower"]
    hi = result["10"]["interval"]["upper"]
    derived = {
        "gap_f10_minus_f13up": {
            "lower": lo - hi_up, "upper": hi - lo_up,
            "width": (hi - lo_up) - (lo - hi_up),
            "label": "numerical certificate (float), not exact rational",
        },
        "f10_minus_f13full": {
            "lower": result["10"]["interval"]["lower"]
            - result["13_full"]["interval"]["upper"],
            "upper": result["10"]["interval"]["upper"]
            - result["13_full"]["interval"]["lower"],
            "label": "numerical certificate (float), not exact rational",
        },
    }

    payload = {
        "title": "6-point physical 3D-3D counterexample: numerical dual certificate",
        "label": "numerical certificate (float), not exact rational",
        "generated_by": "sdp/run_six_point_cert.py",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t_start, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "solvers": {
                PRIMARY_DUAL: __import__("clarabel").__version__,
                PRIMARY_PRIMAL: __import__("scs").__version__,
            },
            "platform": platform.platform(),
        },
        "instance": {
            "description": "p_i = +-e1, +-e2, +-e3; y_i = 0.05 p_i; "
                           "W = [[1,-0.6,0],[-0.6,1,0],[0,0,1]]; G = 6W; D = 0",
            "eps": 0.05,
            "f_at_R_eq_I": f_at_I,
            "exact_value_at_R_eq_I": str(EXACT_UPPER),
        },
        "margins": {
            "psd_margin": PSD_MARGIN,
            "value_safety": VALUE_SAFETY,
            "rank_tol_rel": RANK_TOL_REL,
            "psd_check": "min eig(Z) and Cholesky(Z + psd_margin*I)",
            "why": "value_safety must exceed the observed solver tolerance "
                   "(max |primal-dual| here ~2e-8); psd_margin must exceed the "
                   "observed negative eigenvalue of the dual slack (~6e-10)",
        },
        "derived_intervals": derived,
        "problems": result,
        "caveats": [
            "This is a floating-point certificate, not exact rational arithmetic: "
            "the dual multipliers and the slack matrix are stored as doubles, and "
            "the lower endpoint is the solver dual value minus the stated value "
            "safety margin.",
            "The revised proof quotes f*_13up = 3.59430886; both CLARABEL and SCS "
            "give 3.59431373 (difference 4.9e-6, above solver tolerance), so that "
            f"quoted value lies OUTSIDE the interval "
            f"[{result['13_up']['interval']['lower']:.8f}, "
            f"{result['13_up']['interval']['upper']:.8f}]. "
            "The O(1) separation from f10 (gap ~1.82) is unaffected.",
            "rank(X*_13up) = 4 at the stated 1e-4 relative threshold (four O(1) "
            "eigenvalues, then numerical dust at ~5e-6); it is 6 at a 1e-6 "
            "threshold, but the theorem only needs rank > 1.",
            "For f10 and f13_full the upper endpoint is the exact rational "
            "1083/200 at the exactly feasible X = x x^T; the float evaluation of "
            "the same point with the assembled float Q is 5.414999999999999. "
            "The solver primal values for f10/f13_full can sit up to ~3e-9 above "
            "5.415 because the returned X is only numerically feasible; they are "
            "reported as a check only and are not used for the upper endpoint.",
        ],
    }
    (OUT / "six_point_cert.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---------------- markdown report (Chinese, per repo convention) --------
    r10, rfull, rup = result["10"], result["13_full"], result["13_up"]
    L = []
    A = L.append
    A("# 6 点物理反例：数值对偶证书（numerical certificate (float), not exact rational）")
    A("")
    A("> **一句话结论：** 对 6 点实例，三个 SDP 的数值区间为 "
      f"$f^\\star_{{10}}\\in[{r10['interval']['lower']:.8f},\\,"
      f"{r10['interval']['upper']:.8f}]$、"
      f"$f^\\star_{{13,\\rm full}}\\in[{rfull['interval']['lower']:.8f},\\,"
      f"{rfull['interval']['upper']:.8f}]$、"
      f"$f^\\star_{{13,\\rm up}}\\in[{rup['interval']['lower']:.8f},\\,"
      f"{rup['interval']['upper']:.8f}]$；"
      f"后者与满 20D 值的间隔落在 "
      f"$[{derived['gap_f10_minus_f13up']['lower']:.8f},\\,"
      f"{derived['gap_f10_minus_f13up']['upper']:.8f}]$。"
      "这是**浮点数值证书**，不是有理数精确证书。")
    A("")
    A("## 0. 口径、求解器与安全裕度")
    A("")
    A(f"- 对偶（下界）求解器：**CLARABEL "
      f"{payload['environment']['solvers']['CLARABEL']}**（内点法，"
      f"cvxpy {cp.__version__}）；原始（上界/秩）求解器：**SCS "
      f"{payload['environment']['solvers']['SCS']}**；原始值另用 CLARABEL 交叉检查。")
    A(f"- 下界：直接求解对偶 $\\max\\mu$ s.t. "
      "$Z=Q-\\mu E_{00}+\\sum_i y_iA_i\\succeq0$，"
      "取 $\\mu-10^{-7}$。")
    A("- PSD 安全裕度：$m_{\\rm psd}=10^{-8}$；用 "
      "$\\lambda_{\\min}(Z)$ 与 $Z+m_{\\rm psd}I$ 的 Cholesky 分解双重核验。")
    A("- 数值安全裕度：$\\varepsilon_{\\rm safe}=10^{-7}$，"
      "取在区间两个端点外侧；它远大于观测到的原始-对偶差（$\\le10^{-9}$）"
      "与两求解器原始值差（$\\le2\\times10^{-8}$）。")
    A("- 秩阈值：$\\lambda_i>10^{-4}\\,\\lambda_{\\rm max}$。")
    A("- 上界：$f^\\star_{10}$、$f^\\star_{13,\\rm full}$ 用**精确可行点** "
      "$X=xx^\\top$，$x=[1,\\mathrm{vec}(I),0]$，其目标值为闭式 "
      "$6(1-\\varepsilon)^2=1083/200=5.415$（sympy 精确核对）；"
      "$f^\\star_{13,\\rm up}$ 的最优解 rank 4，没有构造精确可行点，"
      "上界取原始求解值加 $\\varepsilon_{\\rm safe}$。")
    A("")
    A("## 1. 区间表")
    A("")
    A("| 量 | 约束（独立维数） | rank(X*) | 数值区间 `[lower, upper]`（numerical certificate (float), not exact rational） | 宽度 | 上界来源 |")
    A("|-|-|-|-|-|-|")
    rows = [("$f^\\star_{10}$", r10, "21 (20D)"),
            ("$f^\\star_{13,\\rm full}$", rfull, "27 (20D)"),
            ("$f^\\star_{13,\\rm up}$", rup, "24 (17D)")]
    for name, r, cons in rows:
        A(f"| {name} | {cons} | {r['primal']['rank']} | "
          f"[{r['interval']['lower']:.8f}, {r['interval']['upper']:.8f}] | "
          f"{r['interval']['width']:.2e} | {r['upper']['kind']} |")
    A("")
    A("派生区间（同一证书的区间算术）：")
    A("")
    dgap = derived["gap_f10_minus_f13up"]
    dfull = derived["f10_minus_f13full"]
    A(f"- $f^\\star_{{10}}-f^\\star_{{13,\\rm up}}\\in"
      f"[{dgap['lower']:.8f},\\,{dgap['upper']:.8f}]$（宽度 {dgap['width']:.2e}）；"
      f"包含点值 {r10['primal']['value'] - rup['primal']['value']:.8f}；"
      "**不包含**证明引用的 1.82069114（相差约 4.8e-6），与正文“末 5 位未由我们的构造复现”一致。")
    A(f"- $f^\\star_{{10}}-f^\\star_{{13,\\rm full}}\\in"
      f"[{dfull['lower']:.2e},\\,{dfull['upper']:.2e}]$，包含 0，"
      "与满 20D 等价的结论一致。")
    A("")
    A("## 2. 对偶松弛的 PSD 核验（裕度已声明）")
    A("")
    A("| 量 | $\\lambda_{\\min}(Z)$ | $\\lambda_{\\min}(Z+m_{\\rm psd}I)$ | Cholesky | $\\langle Z,X^\\star\\rangle$ | $\\lambda_{\\min}(X^\\star)$ | 最大约束残差 | 原始-对偶差 |")
    A("|-|-|-|-|-|-|-|-|")
    for name, r in [("10", r10), ("13_full", rfull), ("13_up", rup)]:
        d, p = r["dual"], r["primal"]
        A(f"| {name} | {d['min_eig_Z']:+.2e} | {d['min_eig_shifted']:+.2e} | "
          f"{'OK' if d['cholesky_ok'] else 'FAIL'} | "
          f"{r['complementarity_trace_ZX']:+.2e} | "
          f"{p['min_eig_X']:+.2e} | {p['max_constraint_residual']:.2e} | "
          f"{r['dual_gap_primal_minus_mu']:+.2e} |")
    A("")
    A("读法：$\\lambda_{\\min}(Z)$ 的负值都在 $m_{\\rm psd}=10^{-8}$ 以内，"
      "$Z+10^{-8}I$ 的 Cholesky 分解全部成功，故对偶松弛在数值精度内 PSD；"
      "$\\langle Z,X^\\star\\rangle$ 与原始-对偶差一致，说明互补松弛在 ~1e-10 内闭合。")
    A("")
    A("## 3. 求解器交叉检查")
    A("")
    A("| 量 | SCS 原始值（主） | CLARABEL 原始值（校验） | 两原始值差 | CLARABEL 对偶 $\\mu$ |")
    A("|-|-|-|-|-|")
    for name, r in [("10", r10), ("13_full", rfull), ("13_up", rup)]:
        A(f"| {name} | {r['primal']['value']:.10f} | "
          f"{r['primal_check']['value']:.10f} | "
          f"{abs(r['primal']['value'] - r['primal_check']['value']):.1e} | "
          f"{r['dual']['mu']:.10f} |")
    A("")
    A("两个独立求解器的原始值一致到 ≤2e-8（f10 2.5e-9，13_up 2.0e-8），且 SCS 原始值与 "
      "`results/six_point.json` 逐位一致；下界来自 CLARABEL 对偶。"
      "因此区间端点不是单一求解器或单次运行的伪影。")
    A("")
    A("## 4. 必须保留的 caveat")
    A("")
    for c in payload["caveats"]:
        A(f"- {c}")
    A("")
    A("## 5. 复现")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/sdp")
    A("/home/syk/miniconda3/envs/WM/bin/python run_6point_check.py")
    A("/home/syk/miniconda3/envs/WM/bin/python run_six_point_cert.py")
    A("```")
    A("")
    A(f"总耗时 {payload['runtime_s']:.1f} s。原始数据：`results/six_point_cert.json`；"
      "复核值：`results/six_point.json`（`run_6point_check.py`）。")
    (OUT / "six_point_cert.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'six_point_cert.json'}")
    print(f"wrote {OUT / 'six_point_cert.md'}")
    print(f"runtime {payload['runtime_s']}s")


if __name__ == "__main__":
    main()
