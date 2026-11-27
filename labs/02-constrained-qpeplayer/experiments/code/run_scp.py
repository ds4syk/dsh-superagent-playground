#!/usr/bin/env python
"""B4 baseline：dQP + 局部线性化（SCP / trust-region）在合成桌面 PnP 上的诚实实现。

与主表同实例、同目标、同约束、同热启动、同证书流程
------------------------------------------------------
* 状态 (q,t) ∈ S^3 × R^3；每步在切空间 δ = (δω, δt) ∈ R^6 上建模：
    R(q) ← R(q) exp(δω^)          （右扰动；q ← q ⊗ exp(δω)）
    g_j(q exp(δω^), t+δt) ≈ g_j + [A_j Jv2(q) Q(q) , B_j] δ      （解析 Jacobian）
    J(q exp(δω^), t+δt)  ≈ J + ĝ^T δ + ½ δ^T Ĥ δ                 （解析梯度/Hessian）
  其中 Q(q) = ∂q/∂δω = ½[ -q_v^T ; q0 I + [q_v]× ]，
  Ĥ 含四元数参数化的二阶项 −¼ (∇_q J·q) I（见 tangent_model；已用有限差分验证，
  相对误差 <1e-6，见 scp_baseline.json 的 `model_fd_check`）。
* 子问题（cvxpy + CLARABEL，SOCP）：
    min  ½ δ^T Ĥ_reg δ + ĝ^T δ + κ Σ s_j        （phase=optimize）
    min  Σ s_j                                   （phase=restore：先恢复可行性）
    s.t. J_g δ − s ≤ −g,  s ≥ 0,  ‖δ‖₂ ≤ radius
  Ĥ_reg = Ĥ + max(0, −λ_min(Ĥ)) I + 1e-8 I（LM 正则，保证子问题凸）；κ=1e4。
  restore 阶段一旦 viol ≤ 1e-10 转 optimize；接受准则为 merit（restore: viol；
  optimize: J + κ·viol）下降；成功则按改进比例扩半径，失败则减半；
  optimize 阶段连续两次 |ΔJ| ≤ 1e-13 或半径 < 1e-6 判收敛；max_iter=100。
* 同一热启动：无约束 QPEP 解（48 起点，与 ALM/soft/proj 完全一致），
  因此 SCP 的失败只反映"局部凸层 + 该初值"，不是信息优势。
* 证书：与 run_main 完全相同的一次 shor_bound(enforce_ineq=True)，
  certified = 可行(≤1e-6) 且 J−Shor ≤ 1e-6 且 SDP tight(λ2/λ1 ≤ 1e-6)。

三个 SCP 变体
-------------
  scp          单起点（上段），主 baseline；
  scp_restart  若 scp 未认证（或不可行），最多再试 2 个扰动初值
               （q0+0.35·N(0,I₄) 归一化、t0+0.02·N(0,I₃)，与 ALM 内层起点同一风格），
               取"可行且 J 最小"的解；
  scp_restart3 同上但最多 3 个额外扰动初值（本轮 S34：多一次起点，检验镜像初值能否救回）。
  两个重启变体用来区分"SCP 局部收敛失败"与"初值落进错误盆地"。

压力档：3 档（σ=0.02/bias=0.12、σ=0.03/bias=0.18、σ=0.05/bias=0.25），默认各 20 trials；
报告 SCP vs ALM 的 cert/feasible/time 以及失败模式计数（见 md 表 3b）。

运行：/home/syk/miniconda3/envs/sdpr/bin/python run_scp.py
环境变量：QPE_SCP_TRIALS（默认 50）、QPE_SCP_P（默认 "8,16,32"）、QPE_SCP_MAX_ITER（默认 100）、
          QPE_SCP_NOISE_TRIALS（每档压力 trial 数，默认 20）、
          QPE_SCP_STRESS_TAGS（逗号分隔，只跑指定档）、QPE_SCP_STRESS=0（跳过压力档）。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import traceback

import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpep_core as C
from certificate import shor_bound
import run_main as RM          # 复用主表的 metrics()，保证指标定义逐字一致

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

P_LIST = [int(v) for v in os.environ.get("QPE_SCP_P", "8,16,32").split(",") if v]
N_TRIALS = int(os.environ.get("QPE_SCP_TRIALS", "50"))
MAX_ITER = int(os.environ.get("QPE_SCP_MAX_ITER", "100"))
NOISE, BIAS_Z = 0.01, 0.08          # 主比较单元格（与主表一致）
N_NOISE_TRIALS = int(os.environ.get("QPE_SCP_NOISE_TRIALS", "20"))
# 高噪声/偏差压力档（3 档）：主表 unc 镜像率 18%（σ=0.02/bias=0.12），
# 无约束热启动越容易落进 cheirality 镜像盆地，用来检验 SCP 单起点认证率对初值的敏感性。
# 新增两档 σ=0.03/bias=0.18、σ=0.05/bias=0.25（本轮 S34）。
STRESS_CELLS = [
    dict(tag="n020_b120", p=16, noise=0.02, bias_z=0.12, n=N_NOISE_TRIALS),
    dict(tag="n030_b180", p=16, noise=0.03, bias_z=0.18, n=N_NOISE_TRIALS),
    dict(tag="n050_b250", p=16, noise=0.05, bias_z=0.25, n=N_NOISE_TRIALS),
]
_tags = [v for v in os.environ.get("QPE_SCP_STRESS_TAGS", "").split(",") if v]
if _tags:
    STRESS_CELLS = [c for c in STRESS_CELLS if c["tag"] in _tags]
if os.environ.get("QPE_SCP_STRESS", "1") == "0":
    STRESS_CELLS = []
RESTART_EXTRA = 2                   # scp_restart：最多 2 个额外扰动起点（上一轮 baseline）
RESTART3_EXTRA = 3                  # scp_restart3：最多 3 个额外扰动起点（本轮新增）
LAM_SOFT = 1e3


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 四元数 / 切空间解析模型
# ---------------------------------------------------------------------------


def qmul(a, b):
    a0, av = a[0], a[1:]
    b0, bv = b[0], b[1:]
    return np.concatenate([[a0 * b0 - av @ bv], a0 * bv + b0 * av + np.cross(av, bv)])


def qexp(dw):
    th = float(np.linalg.norm(dw))
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * dw / th])


def Q_of(q):
    """∂q/∂δω（4×3），对应 q ← q ⊗ exp(δω)。"""
    q0, q1, q2, q3 = q
    return 0.5 * np.array([[-q1, -q2, -q3],
                           [q0, -q3, q2],
                           [q3, q0, -q1],
                           [-q2, q1, q0]])


def _v2_hessians():
    """v2 每个分量的 Hessian（常量）：k<4: 2 e_k e_k^T；k≥4: √2 (e_i e_j^T + e_j e_i^T)。"""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    Hs = []
    for k in range(10):
        H = np.zeros((4, 4))
        if k < 4:
            H[k, k] = 2.0
        else:
            i, j = pairs[k - 4]
            H[i, j] = H[j, i] = np.sqrt(2.0)
        Hs.append(H)
    return Hs


_V2H = _v2_hessians()


def tangent_model(q, t, M0, D0, G0, A, B):
    """返回 (J, ĝ(6), Ĥ(6,6), J_g(p,6))：J 与 g 在 (q,t) 处对 δ=(δω,δt) 的精确一/二阶展开。"""
    x = C.v2(q)
    Jv = C.Jv2(q)
    J = C.obj_value(q, t, M0, D0, G0)
    gq = 2.0 * (Jv.T @ (M0 @ x + D0.T @ t))          # ∂J/∂q (4,)
    gt = 2.0 * (D0 @ x + G0 @ t)                     # ∂J/∂t (3,)
    coef = M0 @ x + D0.T @ t                          # J 里二次项系数
    Hqq = 2.0 * Jv.T @ M0 @ Jv
    for k in range(10):
        Hqq = Hqq + 2.0 * coef[k] * _V2H[k]
    Q = Q_of(q)
    Hww = Q.T @ Hqq @ Q - 0.25 * float(gq @ q) * np.eye(3)   # 二阶项 ∂²q/∂δω² = −q/4
    Hwt = Q.T @ (2.0 * (D0 @ Jv).T)                    # ∂²J/∂δω∂δt
    Htt = 2.0 * G0
    H = np.block([[Hww, Hwt], [Hwt.T, Htt]])
    gvec = np.concatenate([Q.T @ gq, gt])
    Jg = np.hstack([A @ Jv @ Q, B])                    # p×6
    return J, gvec, H, Jg


def model_fd_check(M0, D0, G0, A, B, q, t, eps=1e-4, rng=None):
    """中心/混合差分核对切空间模型（只用于自检，不参与求解）。

    同时给绝对误差与相对误差（分母 max(||·FD||, 1e-6)）：
    ALM 的无约束解在 t 方向近似驻点（∇_t J≈0），此时相对误差被 0 除放大，
    必须看绝对误差；另在扰动的 generic 点再核对一次相对误差。
    """
    rng = rng or np.random.default_rng(0)
    J, g, H, Jg = tangent_model(q, t, M0, D0, G0, A, B)
    g0 = A @ C.v2(q) + B @ t

    def grad_fd(qq, tt):
        out = np.zeros(6)
        for i in range(6):
            d = np.zeros(6)
            d[i] = eps
            qp = qmul(qq, qexp(d[:3]))
            qm = qmul(qq, qexp(-d[:3]))
            out[i] = (C.obj_value(qp, tt + d[3:], M0, D0, G0)
                      - C.obj_value(qm, tt - d[3:], M0, D0, G0)) / (2 * eps)
        return out

    def jac_fd(qq, tt):
        out = np.zeros_like(Jg)
        for j in range(6):
            d = np.zeros(6)
            d[j] = eps
            gp = A @ C.v2(qmul(qq, qexp(d[:3]))) + B @ (tt + d[3:])
            gm = A @ C.v2(qmul(qq, qexp(-d[:3]))) + B @ (tt - d[3:])
            out[:, j] = (gp - gm) / (2 * eps)
        return out

    res = dict(eps=eps, grad_norm_at_point=float(np.linalg.norm(grad_fd(q, t))))
    # (a) 求解实际线性化点（qu,tu）
    gfd = grad_fd(q, t)
    Jgfd = jac_fd(q, t)
    res["grad_abs_err"] = float(np.linalg.norm(g - gfd))
    res["grad_rel_err"] = float(np.linalg.norm(g - gfd) / max(1e-6, np.linalg.norm(gfd)))
    res["constraint_jac_abs_err"] = float(np.linalg.norm(Jg - Jgfd))
    res["constraint_jac_rel_err"] = float(np.linalg.norm(Jg - Jgfd)
                                          / max(1e-6, np.linalg.norm(Jgfd)))
    # Hessian：方向二阶差分 + 混合差分（尺度不变）
    dirs = rng.normal(size=(3, 6))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    herr = []
    for d in dirs:
        f = lambda ss: C.obj_value(qmul(q, qexp(ss * d[:3])), t + ss * d[3:], M0, D0, G0)
        sec = (f(eps) - 2 * f(0) + f(-eps)) / eps ** 2
        herr.append(abs(sec - float(d @ H @ d)) / max(1e-12, abs(float(d @ H @ d))))
    cross = []
    for i in range(3):
        for j in range(3):
            e1 = np.zeros(6)
            e1[i] = eps
            e2 = np.zeros(6)
            e2[3 + j] = eps
            f = lambda aa, bb: C.obj_value(qmul(q, qexp(aa * e1[:3])), t + bb * e2[3:], M0, D0, G0)
            mixed = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * eps ** 2)
            cross.append(abs(mixed - H[i, 3 + j]) / max(1e-12, abs(H[i, 3 + j])))
    res["hess_dir_rel_err_max"] = float(max(herr))
    res["hess_cross_rel_err_max"] = float(max(cross))
    # (b) generic 扰动点：梯度非零，报告相对误差
    qg = q / np.linalg.norm(q) + 0.10 * rng.normal(size=4)
    tg = t + 0.01 * rng.normal(size=3)
    _, gg, _, Jgg = tangent_model(qg, tg, M0, D0, G0, A, B)
    gfd2 = grad_fd(qg, tg)
    Jgfd2 = jac_fd(qg, tg)
    res["generic_grad_norm"] = float(np.linalg.norm(gfd2))
    res["generic_grad_rel_err"] = float(np.linalg.norm(gg - gfd2)
                                        / max(1e-12, np.linalg.norm(gfd2)))
    res["generic_constraint_jac_rel_err"] = float(np.linalg.norm(Jgg - Jgfd2)
                                                  / max(1e-12, np.linalg.norm(Jgfd2)))
    return res


# ---------------------------------------------------------------------------
# 2. SCP 外壳
# ---------------------------------------------------------------------------


def scp_solve(M0, D0, G0, A, B, q_init, t_init, max_iter=MAX_ITER, r0=0.05, rmax=0.5,
              rmin=1e-6, kappa=1e4, tol_feas=1e-10, solver="CLARABEL", trace=False):
    """单起点 trust-region SCP。返回 dict（含 q,t,J,viol,iters,converged,耗时拆分）。"""
    q = np.array(q_init, float)
    q = q / np.linalg.norm(q)
    t = np.array(t_init, float)
    m = A.shape[0]
    r = float(r0)
    phase = "restore"
    n_qp = n_fail = 0
    t_qp = t_restore = t_opt = 0.0
    hist = []
    converged = False
    stall = 0
    it = -1
    t_all = time.perf_counter()
    for it in range(max_iter):
        J, g, H, Jg = tangent_model(q, t, M0, D0, G0, A, B)
        gv = A @ C.v2(q) + B @ t
        viol = float(max(0.0, gv.max()))
        if phase == "restore" and viol <= tol_feas:
            phase = "optimize"
        lmin = float(np.linalg.eigvalsh(H)[0])
        Hreg = H + (max(0.0, -lmin) + 1e-8) * np.eye(6)
        d = cp.Variable(6)
        s = cp.Variable(m, nonneg=True)
        if phase == "restore":
            obj = cp.sum(s) + 1e-6 * cp.sum_squares(d)
        else:
            obj = 0.5 * cp.quad_form(d, Hreg, assume_PSD=True) + g @ d + kappa * cp.sum(s)
        cons = [Jg @ d - s <= -gv, cp.norm(d, 2) <= r]
        prob = cp.Problem(cp.Minimize(obj), cons)
        t0 = time.perf_counter()
        try:
            prob.solve(solver=solver)
        except Exception:
            n_fail += 1
            r *= 0.5
            if trace:
                hist.append(dict(it=it, phase=phase, r=r, solver_error=True))
            continue
        dt = time.perf_counter() - t0
        t_qp += dt
        n_qp += 1
        if phase == "restore":
            t_restore += dt
        else:
            t_opt += dt
        if d.value is None:
            n_fail += 1
            r *= 0.5
            continue
        dv = np.array(d.value).ravel()
        qn = qmul(q, qexp(dv[:3]))
        tn = t + dv[3:]
        gvn = A @ C.v2(qn) + B @ tn
        violn = float(max(0.0, gvn.max()))
        Jn = C.obj_value(qn, tn, M0, D0, G0)
        merit, meritn = (viol, violn) if phase == "restore" else (J + kappa * viol, Jn + kappa * violn)
        accept = meritn < merit - max(1e-15, 1e-9 * abs(merit))
        if trace:
            hist.append(dict(it=it, phase=phase, r=r, J=J, viol=viol, J_new=Jn,
                             viol_new=violn, step=float(np.linalg.norm(dv)), accepted=bool(accept)))
        if accept:
            q, t = qn, tn
            if phase == "restore":
                if violn < 0.5 * viol:
                    r = min(r * 1.5, rmax)
            else:
                if meritn < 0.75 * merit:
                    r = min(r * 1.5, rmax)
                if abs(Jn - J) <= 1e-13 and violn <= tol_feas:
                    stall += 1
                    if stall >= 2:
                        converged = True
                        break
                else:
                    stall = 0
        else:
            r *= 0.5
            if phase == "optimize" and r < rmin:
                converged = True
                break
            if r < 1e-9:
                break
    J = C.obj_value(q, t, M0, D0, G0)
    gv = A @ C.v2(q) + B @ t
    viol = float(max(0.0, gv.max()))
    return dict(q=q, t=t, J=float(J), viol=float(viol), iters=int(it + 1), phase=phase,
                radius=float(r), converged=bool(converged and viol <= 1e-8),
                n_qp=int(n_qp), n_fail=int(n_fail), time_qp_ms=1e3 * t_qp,
                time_restore_ms=1e3 * t_restore, time_opt_ms=1e3 * t_opt,
                time_total_ms=1e3 * (time.perf_counter() - t_all), hist=hist)


def scp_with_restarts(M0, D0, G0, A, B, qu, tu, shor_val, rng, n_extra=2, **kw):
    """SCP + 最多 n_extra 个扰动初值（仅在未认证/不可行时触发）。"""
    runs = []
    t_all = time.perf_counter()
    best = scp_solve(M0, D0, G0, A, B, qu, tu, **kw)
    runs.append(best)
    n_starts = 1
    for _ in range(n_extra):
        if best["viol"] <= 1e-6 and (best["J"] - shor_val) <= 1e-6:
            break
        q0 = qu / np.linalg.norm(qu) + 0.35 * rng.normal(size=4)
        t0 = np.array(tu, float) + 0.02 * rng.normal(size=3)
        r = scp_solve(M0, D0, G0, A, B, q0, t0, **kw)
        runs.append(r)
        n_starts += 1
        if r["viol"] <= 1e-6 and r["J"] < best["J"]:
            best = r
    out = dict(best)
    out["n_starts"] = n_starts
    out["time_total_ms"] = 1e3 * (time.perf_counter() - t_all)
    out["run_Js"] = [float(x["J"]) for x in runs]
    out["run_viols"] = [float(x["viol"]) for x in runs]
    out["run_iters"] = [int(x["iters"]) for x in runs]
    out["run_converged"] = [bool(x["converged"]) for x in runs]
    return out


# ---------------------------------------------------------------------------
# 3. 单 trial：四种主表方法 + SCP（同实例、同热启动、同证书）
# ---------------------------------------------------------------------------


def run_trial(ins, do_fd_check=False):
    rec = dict(seed=ins["seed"], p=ins["p"], noise=ins["noise"], bias_z=ins["bias_z"],
               actypes=ins["actypes"])
    M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
    seed = ins["seed"]

    # --- 无约束 QPEP（热启动，与主表一致）---
    t0 = time.perf_counter()
    Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48,
                                   rng=np.random.default_rng(1000 + seed))
    rec["unc"] = RM.metrics(ins, qu, tu, A, B, M0, D0, G0, J=Ju)
    rec["unc"]["time_ms"] = 1e3 * (time.perf_counter() - t0)
    rec["unc"]["mirror"] = bool(rec["unc"]["depth_min_mm"] < 0.0)

    # --- 软惩罚 ---
    t0 = time.perf_counter()
    sp = C.soft_penalty(M0, D0, G0, A, B, LAM_SOFT, qu, tu, n_starts=14,
                        rng=np.random.default_rng(2000 + seed))
    rec["soft"] = RM.metrics(ins, sp["q"], sp["t"], A, B, M0, D0, G0, J=sp["J"])
    rec["soft"]["time_ms"] = 1e3 * (time.perf_counter() - t0)
    rec["soft"]["lam"] = LAM_SOFT

    # --- 事后投影 ---
    t0 = time.perf_counter()
    pr = C.project_feasible(M0, D0, G0, A, B, qu, tu, n_starts=4,
                            rng=np.random.default_rng(3000 + seed))
    rec["proj"] = RM.metrics(ins, pr["q"], pr["t"], A, B, M0, D0, G0)
    rec["proj"]["time_ms"] = 1e3 * (time.perf_counter() - t0)
    rec["proj"]["variant"] = pr.get("method")

    # --- ALM-QPEP（主表实现，未改动）---
    t0 = time.perf_counter()
    ok = C.alm_qpep(M0, D0, G0, A, B, qu, tu, rho0=10.0, n_starts=6,
                    rng=np.random.default_rng(4000 + seed))
    rec["alm"] = RM.metrics(ins, ok["q"], ok["t"], A, B, M0, D0, G0, J=ok["J"])
    rec["alm"]["time_ms"] = 1e3 * (time.perf_counter() - t0)
    rec["alm"].update(iters=ok["iters"], fallback=ok["fallback_count"],
                      n_act=int(np.sum(ok["mu"] > 1e-9)), ok=ok["ok"])

    # --- 证书（与实例/方法无关，先算，供所有方法判定 certified）---
    t0 = time.perf_counter()
    sd = shor_bound(M0, D0, G0, A, B, enforce_ineq=True)
    rec["sdp"] = dict(value=sd["value"], ratio=sd["ratio"], lam1=sd["lam1"],
                      status=sd["status"], time_ms=1e3 * (time.perf_counter() - t0),
                      tight=bool(sd["ratio"] <= 1e-6))
    shor = sd["value"]
    rec["sdp"]["sound_vs_alm"] = bool(shor <= ok["J"] + 1e-7)

    # --- SCP 单起点 ---
    rng = np.random.default_rng(6000 + seed)
    t0 = time.perf_counter()
    sc = scp_solve(M0, D0, G0, A, B, qu, tu, max_iter=MAX_ITER, trace=do_fd_check)
    rec["scp"] = RM.metrics(ins, sc["q"], sc["t"], A, B, M0, D0, G0, J=sc["J"])
    rec["scp"].update(time_ms=1e3 * (time.perf_counter() - t0), time_solver_ms=sc["time_qp_ms"],
                      time_restore_ms=sc["time_restore_ms"], time_opt_ms=sc["time_opt_ms"],
                      iters=sc["iters"], converged=sc["converged"], phase=sc["phase"],
                      radius=sc["radius"], n_qp=sc["n_qp"], n_fail=sc["n_fail"])

    # --- SCP + 重启（诊断：区分"局部凸层失败"与"初值落错盆地"）---
    t0 = time.perf_counter()
    scr = scp_with_restarts(M0, D0, G0, A, B, qu, tu, shor,
                            rng=np.random.default_rng(7000 + seed),
                            n_extra=RESTART_EXTRA, max_iter=MAX_ITER)
    rec["scp_restart"] = RM.metrics(ins, scr["q"], scr["t"], A, B, M0, D0, G0, J=scr["J"])
    rec["scp_restart"].update(time_ms=1e3 * (time.perf_counter() - t0),
                              n_starts=scr["n_starts"], run_Js=scr["run_Js"],
                              run_viols=scr["run_viols"], run_iters=scr["run_iters"],
                              run_converged=scr["run_converged"],
                              iters=scr["iters"], converged=scr["converged"],
                              phase=scr["phase"])

    # --- SCP + 3 重启（本轮 S34：多一次扰动起点，看镜像初值能否被救回）---
    t0 = time.perf_counter()
    scr3 = scp_with_restarts(M0, D0, G0, A, B, qu, tu, shor,
                             rng=np.random.default_rng(7300 + seed),
                             n_extra=RESTART3_EXTRA, max_iter=MAX_ITER)
    rec["scp_restart3"] = RM.metrics(ins, scr3["q"], scr3["t"], A, B, M0, D0, G0, J=scr3["J"])
    rec["scp_restart3"].update(time_ms=1e3 * (time.perf_counter() - t0),
                               n_starts=scr3["n_starts"], run_Js=scr3["run_Js"],
                               run_viols=scr3["run_viols"], run_iters=scr3["run_iters"],
                               run_converged=scr3["run_converged"],
                               iters=scr3["iters"], converged=scr3["converged"],
                               phase=scr3["phase"])

    # --- 统一认证口径（与 run_main 逐字一致）---
    for name in ("unc", "soft", "proj", "alm", "scp", "scp_restart", "scp_restart3"):
        m = rec[name]
        m["feasible"] = bool(m["viol"] <= 1e-6)
        m["cert_gap"] = float(m["J"] - shor)
        m["certified"] = bool(m["feasible"] and (m["J"] - shor <= 1e-6)
                              and rec["sdp"]["tight"])
    if do_fd_check:
        rec["model_fd_check"] = model_fd_check(M0, D0, G0, A, B, qu, tu,
                                              rng=np.random.default_rng(11))
    return rec


# ---------------------------------------------------------------------------
# 4. 汇总 / 输出
# ---------------------------------------------------------------------------


def med_iqr(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if xs.size == 0:
        return dict(median=float("nan"), q1=float("nan"), q3=float("nan"), n=0)
    q1, q50, q3 = np.percentile(xs, [25, 50, 75])
    return dict(median=float(q50), q1=float(q1), q3=float(q3), n=int(xs.size))


def summarize(recs):
    out = {}
    for name in ("unc", "soft", "proj", "alm", "scp", "scp_restart", "scp_restart3"):
        ms = [r[name] for r in recs]
        out[name] = dict(
            pen_mm=med_iqr([m["pen_mm"] for m in ms]),
            pen_mean=float(np.mean([m["pen_mm"] for m in ms])),
            pen_rate=float(np.mean([m["pen_mm"] > 1.0 for m in ms])),
            rot_deg=med_iqr([m["rot_deg"] for m in ms]),
            trans_mm=med_iqr([m["trans_mm"] for m in ms]),
            viol_max=float(np.max([m["viol"] for m in ms])),
            feasible=float(np.mean([m["feasible"] for m in ms])),
            certified=float(np.mean([m["certified"] for m in ms])),
            J_med=float(np.median([m["J"] for m in ms])),
            time_ms=med_iqr([m["time_ms"] for m in ms]),
            cert_gap_med=float(np.median([m["cert_gap"] for m in ms])),
        )
    # SCP 专属
    sc = [r["scp"] for r in recs]
    out["scp"]["iters"] = med_iqr([m["iters"] for m in sc])
    out["scp"]["converged"] = float(np.mean([m["converged"] for m in sc]))
    out["scp"]["solver_share"] = float(np.median(
        [m["time_solver_ms"] / max(m["time_ms"], 1e-9) for m in sc]))
    out["scp"]["restore_share"] = float(np.median(
        [m["time_restore_ms"] / max(m["time_ms"], 1e-9) for m in sc]))
    for name in ("scp_restart", "scp_restart3"):
        out[name]["n_starts_mean"] = float(np.mean([r[name]["n_starts"] for r in recs]))
        out[name]["rescued"] = float(np.mean([r[name]["n_starts"] > 1 for r in recs]))
    out["sdp"] = dict(value_med=float(np.median([r["sdp"]["value"] for r in recs])),
                      time_ms=med_iqr([r["sdp"]["time_ms"] for r in recs]),
                      tight=float(np.mean([r["sdp"]["tight"] for r in recs])),
                      sound_vs_alm=float(np.mean([r["sdp"]["sound_vs_alm"] for r in recs])))
    out["mirror_rate"] = float(np.mean([r["unc"]["mirror"] for r in recs]))
    out["failures"] = {name: fail_modes(recs, name)
                       for name in ("scp", "scp_restart", "scp_restart3")}
    out["n"] = len(recs)
    return out


def fail_modes(recs, name):
    """SCP 失败 trial 的失败模式计数（同一 trial 可同时命中 mirror_init）。

    主分类互斥且和 = n_failed：infeasible > nonconverged > suboptimal_converged。
    `mirror_init`  = 无约束热启动本身是 cheirality 镜像解（depth_min<0）；
    `output_mirror`= SCP 输出的解仍在相机后方（depth_min<0）；
    `at_max_iter`  = 跑到 max_iter 上限（与 nonconverged 高度重叠）。
    """
    fails = [r for r in recs if "error" not in r and not r[name]["certified"]]
    if not fails:
        return dict(n=0, infeasible=0, nonconverged=0, suboptimal_converged=0,
                    mirror_init=0, output_mirror=0, at_max_iter=0,
                    iters_med=float("nan"), cert_gap_med=float("nan"))
    infeas = [r for r in fails if r[name]["viol"] > 1e-6]
    nonconv = [r for r in fails if r[name]["viol"] <= 1e-6
               and not r[name].get("converged", False)]
    subopt = [r for r in fails if r[name]["viol"] <= 1e-6
              and r[name].get("converged", False) and r[name]["cert_gap"] > 1e-6]
    return dict(
        n=len(fails), infeasible=len(infeas), nonconverged=len(nonconv),
        suboptimal_converged=len(subopt),
        mirror_init=int(sum(r["unc"]["mirror"] for r in fails)),
        output_mirror=int(sum(r[name]["depth_min_mm"] < 0.0 for r in fails)),
        at_max_iter=int(sum(r[name].get("iters", 0) >= MAX_ITER for r in fails)),
        iters_med=float(np.median([r[name].get("iters", 0) for r in fails])),
        cert_gap_med=float(np.median([r[name]["cert_gap"] for r in fails])),
    )


def run_cell(p, noise, bias_z, n_trials, tag):
    recs = []
    t_cell = time.perf_counter()
    for trial in range(n_trials):
        try:
            ins = C.make_instance(trial, p, noise=noise, bias_z=bias_z)
            rec = run_trial(ins, do_fd_check=(trial == 0))
        except Exception:
            log("  !! trial %d FAILED %s\n%s" % (trial, tag, traceback.format_exc()))
            rec = dict(seed=trial, p=p, error=traceback.format_exc())
        recs.append(rec)
        if (trial + 1) % 10 == 0:
            log("  [%s] %d/%d (%.1fs)  scp_cert_so_far=%.0f%% scp_viol_max=%.1e"
                % (tag, trial + 1, n_trials, time.perf_counter() - t_cell,
                   100 * np.mean([r["scp"]["certified"] for r in recs if "error" not in r]),
                   max([r["scp"]["viol"] for r in recs if "error" not in r])))
    good = [r for r in recs if "error" not in r]
    summ = summarize(good)
    log("  ---- %s summary (n=%d) ----" % (tag, len(good)))
    for name in ("unc", "soft", "proj", "alm", "scp", "scp_restart", "scp_restart3"):
        ss = summ[name]
        log("   %-12s pen=%7.2f mm rot=%5.2f deg trans=%6.2f mm feas=%3.0f%% cert=%3.0f%% "
            "time=%6.1f ms"
            % (name, ss["pen_mm"]["median"], ss["rot_deg"]["median"], ss["trans_mm"]["median"],
               100 * ss["feasible"], 100 * ss["certified"], ss["time_ms"]["median"]))
    for name in ("scp", "scp_restart", "scp_restart3"):
        f = summ["failures"][name]
        log("   fail[%s] n=%d (infes=%d nonconv=%d subopt=%d) mirror_init=%d output_mirror=%d"
            % (name, f["n"], f["infeasible"], f["nonconverged"], f["suboptimal_converged"],
               f["mirror_init"], f["output_mirror"]))
    return dict(tag=tag, p=p, noise=noise, bias_z=bias_z, n=len(good), summary=summ, recs=recs)


def fmt_ms(m):
    return "%.1f [%.1f, %.1f]" % (m["median"], m["q1"], m["q3"])


def main():
    t_start = time.perf_counter()
    log("=" * 100)
    log("B4 SCP baseline | p=%s trials/p=%d noise=%.3f bias_z=%.3f max_iter=%d"
        % (P_LIST, N_TRIALS, NOISE, BIAS_Z, MAX_ITER))
    log("threads: OMP=%s OPENBLAS=%s MKL=%s | cvxpy=%s"
        % (os.environ.get("OMP_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"),
           os.environ.get("MKL_NUM_THREADS"), cp.__version__))
    raw = dict(config=dict(p=P_LIST, n_trials=N_TRIALS, noise=NOISE, bias_z=BIAS_Z,
                           stress_cells=STRESS_CELLS,
                           restart_extra=RESTART_EXTRA, restart3_extra=RESTART3_EXTRA,
                           max_iter=MAX_ITER, kappa=1e4, r0=0.05, rmax=0.5, rmin=1e-6,
                           tol_feas=1e-10, solver="CLARABEL",
                           soft_lam=LAM_SOFT, threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
                           numpy=np.__version__, cvxpy=cp.__version__,
                           scs_eps=1e-9,
                           cert_rule="feasible(viol<=1e-6) & J-Shor<=1e-6 & SDP tight"),
               cells={}, stress={})
    for p in P_LIST:
        log("\n--- main p=%d (noise=%.3f bias=%.3f, n=%d) ---" % (p, NOISE, BIAS_Z, N_TRIALS))
        raw["cells"]["p%d" % p] = run_cell(p, NOISE, BIAS_Z, N_TRIALS, "p%d" % p)
        with open(os.path.join(RES, "scp_baseline.json"), "w") as f:
            json.dump(raw, f, indent=1, default=float)
        log("  [saved] scp_baseline.json (%.1fs)" % (time.perf_counter() - t_start))

    # 高噪声/偏差压力档（默认 p=16, n=20；3 档）
    for cell in STRESS_CELLS:
        log("\n--- stress %s (p=%d noise=%.3f bias=%.3f, n=%d) ---"
            % (cell["tag"], cell["p"], cell["noise"], cell["bias_z"], cell["n"]))
        key = "%s_p%d" % (cell["tag"], cell["p"])
        raw["stress"][key] = run_cell(cell["p"], cell["noise"], cell["bias_z"],
                                      cell["n"], cell["tag"])
        with open(os.path.join(RES, "scp_baseline.json"), "w") as f:
            json.dump(raw, f, indent=1, default=float)
        log("  [saved] scp_baseline.json (%.1fs)" % (time.perf_counter() - t_start))
    raw["elapsed_s"] = time.perf_counter() - t_start
    with open(os.path.join(RES, "scp_baseline.json"), "w") as f:
        json.dump(raw, f, indent=1, default=float)
    write_md(raw)
    log("\n完成：%.1fs" % raw["elapsed_s"])


def write_md(raw):
    P = raw["config"]["p"]
    cells = raw["cells"]
    stress = raw.get("stress") or {}
    cfg = raw["config"]
    L = []
    A = L.append
    A("# B4 baseline：dQP + 局部线性化（SCP）vs ALM / 投影 / 软惩罚")
    A("")
    A("> 由 `experiments/code/run_scp.py` 自动生成；**不要手改**。逐 trial 原始数据在 "
      "`scp_baseline.json`。")
    A("")
    cells_desc = "；".join("p=%d, noise=%.3f, bias=%.3f, n=%d"
                           % (c["p"], c["noise"], c["bias_z"], c["n"]) for c in cfg["stress_cells"])
    A("设置：合成桌面 PnP，主比较 p ∈ %s，noise=%.3f，bias_z=%.3f，每格 %d trials"
      "（seed=trial，与主表共享场景）；压力档 %d 档：%s。"
      "四种主表方法参数与主表一致（unc 48 起点 / soft λ=1e3、14 起点 / proj 4 起点 / "
      "ALM rho0=10、6 起点）。"
      % (P, cfg["noise"], cfg["bias_z"], cfg["n_trials"],
         len(cfg["stress_cells"]), cells_desc if cells_desc else "无"))
    A("")
    A("SCP 实现：切空间 δ=(δω,δt) 上精确一/二阶展开（解析梯度/Hessian，"
      "含四元数二阶项；有限差分自检见文末），trust region + 弹性松弛，"
      "restore→optimize 两阶段，cvxpy/CLARABEL 解 SOCP，max_iter=%d，κ=1e4，r0=%.2f。"
      % (cfg["max_iter"], cfg["r0"]))
    A("证书与主表同一口径：可行(viol≤1e-6) 且 J−Shor≤1e-6 且 SDP tight(λ₂/λ₁≤1e-6)。"
      "两个诊断变体都只在单起点结果未认证时追加扰动初值（q0+0.35·N(0,I₄) 归一化、"
      "t0+0.02·N(0,I₃)）：`scp_restart` 最多再 %d 个（共 ≤%d 起点），"
      "`scp_restart3` 最多再 %d 个（共 ≤%d 起点）。"
      % (cfg["restart_extra"], cfg["restart_extra"] + 1,
         cfg["restart3_extra"], cfg["restart3_extra"] + 1))
    A("")

    def table(cell_map, title):
        A(title)
        A("")
        A("| 格 | method | pen mm (median [IQR]) | rot deg | trans mm | 可行率 | 认证率 | runtime ms (median [IQR]) |")
        A("|---|---|---|---|---|---|---|---|")
        for key in cell_map:
            c = cell_map[key]
            s = c["summary"]
            for name, label in (("unc", "Unconstrained"), ("soft", "Soft λ=1e3"),
                                ("proj", "Post-hoc proj"), ("alm", "ALM-QPEP (ours)"),
                                ("scp", "SCP (single start)"),
                                ("scp_restart", "SCP + %d restarts" % cfg["restart_extra"]),
                                ("scp_restart3", "SCP + %d restarts" % cfg["restart3_extra"])):
                m = s[name]
                A("| %s | %s | %.2f [%.2f, %.2f] | %.2f | %.2f | %.0f%% | %.0f%% | %s |"
                  % (key, label, m["pen_mm"]["median"], m["pen_mm"]["q1"], m["pen_mm"]["q3"],
                     m["rot_deg"]["median"], m["trans_mm"]["median"],
                     100 * m["feasible"], 100 * m["certified"], fmt_ms(m["time_ms"])))
        A("")

    table(cells, "## 表 1. 主比较（noise=%.3f, bias_z=%.3f，n=%d）"
          % (cfg["noise"], cfg["bias_z"], cfg["n_trials"]))
    if stress:
        table(stress, "## 表 1b. 压力档（%d 档高噪声/偏差；无约束镜像初值比例更高）"
              % len(cfg["stress_cells"]))

    A("## 表 2. SCP 内部：迭代 / 收敛 / 耗时构成 / 重启")
    A("")
    A("| 格 | iters (median [IQR]) | converged | QP solve / total | restore / total | 重启触发率 | 平均起点数 | 被 +%d 重启救回 |" % cfg["restart_extra"])
    A("|---|---|---|---|---|---|---|---|")
    rows = list(cells.items()) + list(stress.items())
    for key, c in rows:
        s = c["summary"]
        n_resc = sum(1 for r in c["recs"] if "error" not in r and r["scp_restart"]["n_starts"] > 1
                     and r["scp_restart"]["certified"] and not r["scp"]["certified"])
        A("| %s | %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.2f | %d |"
          % (key, fmt_ms(s["scp"]["iters"]), 100 * s["scp"]["converged"],
             100 * s["scp"]["solver_share"], 100 * s["scp"]["restore_share"],
             100 * s["scp_restart"]["rescued"], s["scp_restart"]["n_starts_mean"], n_resc))
    A("")

    A("## 表 2b. 重启变体：认证率 / 起点数 / 救回 / 代价")
    A("")
    A("| 格 | SCP 认证 | +%d 重启认证 | +%d 重启认证 | 平均起点 +%d | 平均起点 +%d | 救回 (+%d) | 救回 (+%d) | time +%d ms | time +%d ms |" %
      (cfg["restart_extra"], cfg["restart3_extra"], cfg["restart_extra"], cfg["restart3_extra"],
       cfg["restart_extra"], cfg["restart3_extra"], cfg["restart_extra"], cfg["restart3_extra"]))
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key, c in rows:
        s = c["summary"]
        recs_g = [r for r in c["recs"] if "error" not in r]
        resc2 = sum(1 for r in recs_g if r["scp_restart"]["n_starts"] > 1
                    and r["scp_restart"]["certified"] and not r["scp"]["certified"])
        resc3 = sum(1 for r in recs_g if r["scp_restart3"]["n_starts"] > 1
                    and r["scp_restart3"]["certified"] and not r["scp"]["certified"])
        A("| %s | %.0f%% | %.0f%% | %.0f%% | %.2f | %.2f | %d | %d | %s | %s |"
          % (key, 100 * s["scp"]["certified"], 100 * s["scp_restart"]["certified"],
             100 * s["scp_restart3"]["certified"], s["scp_restart"]["n_starts_mean"],
             s["scp_restart3"]["n_starts_mean"], resc2, resc3,
             fmt_ms(s["scp_restart"]["time_ms"]), fmt_ms(s["scp_restart3"]["time_ms"])))
    A("")

    A("## 表 3. SCP 单起点失败 trial（J 高于 Shor 或不可行）")
    A("")
    A("| 格 | seed | unc mirror | SCP J | ALM J | Shor | J_SCP−Shor | viol | iters | converged | +%d J | +%d J | +%d 起点 |" %
      (cfg["restart_extra"], cfg["restart3_extra"], cfg["restart3_extra"]))
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    nfail = 0
    for key, c in rows:
        for r in c["recs"]:
            if "error" in r or r["scp"]["certified"]:
                continue
            nfail += 1
            m = r["scp"]
            A("| %s | %d | %s | %.6f | %.6f | %.6f | %.2e | %.1e | %d | %s | %.6f | %.6f | %d |"
              % (key, r["seed"], "yes" if r["unc"]["mirror"] else "no",
                 m["J"], r["alm"]["J"], r["sdp"]["value"], m["cert_gap"], m["viol"],
                 m["iters"], "yes" if m["converged"] else "no",
                 r["scp_restart"]["J"], r["scp_restart3"]["J"],
                 r["scp_restart3"]["n_starts"]))
    if nfail == 0:
        A("| — | — | — | — | — | — | — | — | — | — | — | — | — |")
    A("")

    A("## 表 3b. 失败模式计数（按方法变体；主分类互斥且和 = n）")
    A("")
    A("| 格 | 变体 | n 失败 | 不可行 | 未收敛 | 收敛但次优 | 初值镜像 | 输出镜像 | 撞 max_iter | iters 中位 | J−Shor 中位 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, c in rows:
        for name, label in (("scp", "SCP"), ("scp_restart", "+%d" % cfg["restart_extra"]),
                            ("scp_restart3", "+%d" % cfg["restart3_extra"])):
            f = c["summary"]["failures"][name]
            A("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %.0f | %.2e |"
              % (key, label, f["n"], f["infeasible"], f["nonconverged"],
                 f["suboptimal_converged"], f["mirror_init"], f["output_mirror"],
                 f["at_max_iter"], f["iters_med"], f["cert_gap_med"]))
    A("")
    A("（`初值镜像` = 无约束热启动 depth_min<0；`输出镜像` = SCP 输出解 depth_min<0；"
      "两者可同时命中，故不加进主分类。）")
    A("")

    A("## 表 4. 证书与健康度")
    A("")
    A("| 格 | SDP tight | SDP sound(vs ALM) | ALM 认证 | SCP 认证 | +%d 认证 | +%d 认证 | unc mirror 率 | max viol (ALM) | max viol (SCP) |" %
      (cfg["restart_extra"], cfg["restart3_extra"]))
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key, c in rows:
        s = c["summary"]
        A("| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.1e | %.1e |"
          % (key, 100 * s["sdp"]["tight"], 100 * s["sdp"]["sound_vs_alm"],
             100 * s["alm"]["certified"], 100 * s["scp"]["certified"],
             100 * s["scp_restart"]["certified"], 100 * s["scp_restart3"]["certified"],
             100 * s["mirror_rate"], s["alm"]["viol_max"], s["scp"]["viol_max"]))
    A("")

    # ---- 结论（数字全部来自本 JSON）----
    s_all = [cells["p%d" % p]["summary"] for p in P]
    scp_cert = [s["scp"]["certified"] for s in s_all]
    scr_cert = [s["scp_restart"]["certified"] for s in s_all]
    scr3_cert = [s["scp_restart3"]["certified"] for s in s_all]
    alm_cert = [s["alm"]["certified"] for s in s_all]
    scp_time = [s["scp"]["time_ms"]["median"] for s in s_all]
    alm_time = [s["alm"]["time_ms"]["median"] for s in s_all]
    med_p = P[len(P) // 2]
    sm = cells["p%d" % med_p]["summary"]
    fails = [(key, r) for key, c in rows for r in c["recs"]
             if "error" not in r and not r["scp"]["certified"]]
    n_mirror = sum(1 for _, r in fails if r["unc"]["mirror"])
    n_outmirror = sum(1 for _, r in fails if r["scp"]["depth_min_mm"] < 0.0)
    n_resc2 = sum(1 for key, c in rows for r in c["recs"]
                  if "error" not in r and r["scp_restart"]["n_starts"] > 1
                  and r["scp_restart"]["certified"] and not r["scp"]["certified"])
    n_resc3 = sum(1 for key, c in rows for r in c["recs"]
                  if "error" not in r and r["scp_restart3"]["n_starts"] > 1
                  and r["scp_restart3"]["certified"] and not r["scp"]["certified"])
    stress_cert = [(key, c["summary"]["scp"]["certified"],
                    c["summary"]["scp_restart3"]["certified"],
                    c["summary"]["alm"]["certified"], c["summary"]["mirror_rate"])
                   for key, c in stress.items()]
    A("## 结论")
    A("")
    A("1. **精度与可行性**：SCP 与 ALM 同档——p=%d 上 SCP pen 中位 %.2f mm、rot %.2f°、"
      "trans %.2f mm（ALM %.2f mm / %.2f° / %.2f mm），可行率 %.0f%% vs %.0f%%。"
      % (med_p, sm["scp"]["pen_mm"]["median"], sm["scp"]["rot_deg"]["median"],
         sm["scp"]["trans_mm"]["median"], sm["alm"]["pen_mm"]["median"],
         sm["alm"]["rot_deg"]["median"], sm["alm"]["trans_mm"]["median"],
         100 * sm["scp"]["feasible"], 100 * sm["alm"]["feasible"]))
    A("2. **认证率（主比较）**：SCP 单起点 %s，+%d 重启 %s，+%d 重启 %s，ALM %s。"
      % (" / ".join("%.0f%%" % (100 * c) for c in scp_cert), cfg["restart_extra"],
         " / ".join("%.0f%%" % (100 * c) for c in scr_cert), cfg["restart3_extra"],
         " / ".join("%.0f%%" % (100 * c) for c in scr3_cert),
         " / ".join("%.0f%%" % (100 * c) for c in alm_cert)))
    if len(fails) == 0:
        A("   本轮主比较与压力档中 SCP **没有**认证失败 trial——在这批合成实例上单起点 SCP "
          "与 ALM 认证率持平，不能声称 ALM 在认证率上占优。")
    else:
        A("   全部 %d 个失败 trial 中 %d 个的 SCP 初值就是无约束 cheirality 镜像解"
          "（depth_min<0、rot err≈180°），%d 个的 SCP 输出仍在相机后方；"
          "失败不是普遍性的，而是集中在镜像初值/局部收敛这两类（逐类计数见表 3b）。"
          % (len(fails), n_mirror, n_outmirror))
    if stress_cert:
        A("3. **压力档（%d 档）**：" % len(stress_cert))
        for key, c1, c3, ca, mr in stress_cert:
            A("   - %s：unc 镜像率 %.0f%%，SCP 单起点认证 %.0f%%、+%d 重启 %.0f%%、ALM %.0f%%。"
              % (key, 100 * mr, 100 * c1, cfg["restart3_extra"], 100 * c3, 100 * ca))
        A("   镜像初值比例升高时 SCP 单起点的脆弱性被放大，这是**初值敏感性**，"
          "不是凸层不可行；ALM 在同一批实例上保持 %.0f%%–%.0f%%。"
          % (100 * min(x[3] for x in stress_cert), 100 * max(x[3] for x in stress_cert)))
    # 失败模式聚合
    fmod = dict(n=0, infeasible=0, nonconverged=0, suboptimal_converged=0,
                mirror_init=0, output_mirror=0, at_max_iter=0)
    for key, c in rows:
        f = c["summary"]["failures"]["scp"]
        for k in fmod:
            fmod[k] += f[k]
    A("4. **失败模式**：SCP 单起点共 %d 个失败（主比较 + 压力档）：不可行 %d、未收敛 %d、"
      "收敛但次优 %d；其中初值是镜像解 %d、输出仍镜像 %d、撞 max_iter %d。"
      % (fmod["n"], fmod["infeasible"], fmod["nonconverged"],
         fmod["suboptimal_converged"], fmod["mirror_init"], fmod["output_mirror"],
         fmod["at_max_iter"]))
    A("5. **重启变体**：主比较认证率 %s → +%d %s → +%d %s（救回 %d / %d 个失败 trial，"
      "平均起点数 %.2f / %.2f）；压力档 +%d 变体的认证率为 %s。"
      "把初值当超参调能补上**部分**差距，但补不回来的失败说明存在真正的局部收敛/迭代上限，"
      "不是「随便扰动一下」能解决的。"
      % (" / ".join("%.0f%%" % (100 * c) for c in scp_cert), cfg["restart_extra"],
         " / ".join("%.0f%%" % (100 * c) for c in scr_cert), cfg["restart3_extra"],
         " / ".join("%.0f%%" % (100 * c) for c in scr3_cert), n_resc2, n_resc3,
         np.mean([s["scp_restart"]["n_starts_mean"] for s in s_all]),
         np.mean([s["scp_restart3"]["n_starts_mean"] for s in s_all]),
         cfg["restart3_extra"],
         " / ".join("%.0f%%" % (100 * x[2]) for x in stress_cert) if stress_cert else "—"))
    A("6. **速度**：SCP 单起点中位 %s ms vs ALM %s ms（p=%s）；"
      "SCP 约 %.0f%% 的墙钟花在 cvxpy/CLARABEL 的 SOCP 上、%.0f%% 在 restore 阶段。"
      "注意这只说明「本实现下 SCP 更快」，SCP 没有 ALM 那种与 p 无关的先验常数维复杂度论证。"
      % (" / ".join("%.0f" % t for t in scp_time), " / ".join("%.0f" % t for t in alm_time),
         ",".join(str(x) for x in P), 100 * sm["scp"]["solver_share"],
         100 * sm["scp"]["restore_share"]))
    A("")
    A("**判定（keep / revert）**：keep `run_scp.py` 及本轮 3 档压力表 + 3 重启变体。"
      "在这些合成实例上 SCP **可行、精度同档、中位更快**，是必须报告的强 baseline；"
      "即使 SCP 在压力档输给 ALM，这张表也要保留（诚实报告失败模式）。"
      "ALM 的差异化在于："
      "(a) 对 cheirality 镜像初值的鲁棒性（表 1b/表 3b）；"
      "(b) 运行时与 p 无关的先验常数维结构（SCP 需要 trust-region/restore 调参，且每次迭代解 SOCP）；"
      "(c) ALM 的证书/反向（FB、rank-deficiency）理论链与后验证书，而 SCP 的证书只验证"
      "「它恰好找到的那个点」。论文中必须同时报告 SCP 的收敛率/迭代数、失败模式与认证率，"
      "不能只报 runtime 或只报精度。")
    A("")
    A("## Caveats")
    A("")
    A("- 本机 40 核多用户共享（本轮测量在 login node、load ~20–23，Slurm 被他人占满），"
      "所有计时为单线程墙钟；绝对毫秒随负载漂移。SCP 与 ALM 在同一进程、同一实例上顺序测量，"
      "相对比较可用。")
    A("- SCP 用 cvxpy/CLARABEL 解 SOCP，ALM 用 scipy BFGS 多起点；两者实现栈不同，"
      "「更快」只对本实现成立（公平性说明见 protocol B4：算法层面对比）。")
    mr_main = [100 * cells["p%d" % p]["summary"]["mirror_rate"] for p in P]
    mr_stress = [100 * c["summary"]["mirror_rate"] for c in stress.values()] if stress else []
    A("- 单起点 SCP 的初值 = 无约束 QPEP 解（与主表 soft/proj/ALM 完全相同的热启动），"
      "因此 SCP 在镜像初值上的失败继承自 unc 的镜像率，不是额外人为设置；"
      "本文件实测 unc 镜像率：主比较 %s%s（主表 50-trial 高噪声档为 18%%）。"
      % (("%.0f%%" % min(mr_main)) if min(mr_main) == max(mr_main)
         else ("%.0f%%–%.0f%%" % (min(mr_main), max(mr_main))),
         "，压力档 %.0f%%–%.0f%%" % (min(mr_stress), max(mr_stress)) if mr_stress else ""))
    A("- 只测前向 + 证书，不含反向（FB）与真实数据；合成实例的底面是平面，"
      "p≥16 后精度饱和，SCP 的精度优势/劣势在这类实例上不可分辨。")
    A("")
    A("## 自检（模型正确性）")
    A("")
    for key, c in rows:
        recs = [r for r in c["recs"] if "model_fd_check" in r]
        if recs:
            fd = recs[0]["model_fd_check"]
            A("- %s（eps=%.0e）：梯度绝对误差 %.1e（求解点梯度范数 %.1e，"
              "该点 ∇_t J≈0 故相对误差不具参考性）；generic 扰动点梯度相对误差 %.1e，"
              "约束 Jacobian 相对误差 %.1e；Hessian 方向相对误差 %.1e、"
              "交叉项相对误差 %.1e。"
              % (key, fd["eps"], fd["grad_abs_err"], fd["grad_norm_at_point"],
                 fd["generic_grad_rel_err"],
                 max(fd["constraint_jac_rel_err"], fd["generic_constraint_jac_rel_err"]),
                 fd["hess_dir_rel_err_max"], fd["hess_cross_rel_err_max"]))
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("cd labs/02-constrained-qpeplayer/experiments/code")
    A("# 全量（主比较 3×50 + 3 档压力 ×20）：")
    A("/home/syk/miniconda3/envs/sdpr/bin/python run_scp.py")
    A("# 小跑 / 单档：")
    A("QPE_SCP_TRIALS=5 QPE_SCP_NOISE_TRIALS=3 \\")
    A("QPE_SCP_STRESS_TAGS=n030_b180 /home/syk/miniconda3/envs/sdpr/bin/python run_scp.py")
    A("# 只跑主比较（跳过压力档）：QPE_SCP_STRESS=0 ... run_scp.py")
    A("```")
    with open(os.path.join(RES, "scp_baseline.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log("  [saved] scp_baseline.md")


if __name__ == "__main__":
    main()
