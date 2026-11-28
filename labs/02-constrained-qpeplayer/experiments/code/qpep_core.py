#!/usr/bin/env python
"""Constrained-QPEPLayer 合成桌面 PnP 实验 —— 核心数学与求解器。

约定（与 theory/proofs.md §0、verify/sanity_constraints_alm.py 一致）
------------------------------------------------------------------------
q ∈ S^3, v2(q) = [q0^2, q1^2, q2^2, q3^2, √2 q0q1, √2 q0q2, √2 q0q3,
                  √2 q1q2, √2 q1q3, √2 q2q3] ∈ R^10
J(q,t) = v2^T M0 v2 + 2 t^T D0 v2 + t^T G0 t          （bearing/重投影型 PnP）
g_j(q,t) = a_j^T v2(q) + b_j^T t <= 0, j=1..p        （物理不等式）
所有约束对 (v2,t) 仿射（C1 已验证）。

本文件刻意不依赖 cvxpy；SDP 证书在 make_cert.py / run_main.py 里调用。
解释器：/home/syk/miniconda3/envs/sdpr/bin/python
"""
import time

import numpy as np
from scipy.optimize import minimize

SQRT2 = np.sqrt(2.0)
E_TR = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# 1. 四元数 / 旋转 / v2 的精确代数
# ---------------------------------------------------------------------------


def v2(q):
    q0, q1, q2, q3 = q
    return np.array([
        q0 * q0, q1 * q1, q2 * q2, q3 * q3,
        SQRT2 * q0 * q1, SQRT2 * q0 * q2, SQRT2 * q0 * q3,
        SQRT2 * q1 * q2, SQRT2 * q1 * q3, SQRT2 * q2 * q3,
    ])


def Jv2(q):
    """dv2/dq ∈ R^{10×4}（解析）。"""
    q0, q1, q2, q3 = q
    J = np.zeros((10, 4))
    J[0, 0] = 2 * q0
    J[1, 1] = 2 * q1
    J[2, 2] = 2 * q2
    J[3, 3] = 2 * q3
    J[4] = [SQRT2 * q1, SQRT2 * q0, 0.0, 0.0]
    J[5] = [SQRT2 * q2, 0.0, SQRT2 * q0, 0.0]
    J[6] = [SQRT2 * q3, 0.0, 0.0, SQRT2 * q0]
    J[7] = [0.0, SQRT2 * q2, SQRT2 * q1, 0.0]
    J[8] = [0.0, SQRT2 * q3, 0.0, SQRT2 * q1]
    J[9] = [0.0, 0.0, SQRT2 * q3, SQRT2 * q2]
    return J


def R_of(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


def q_from_R(R):
    """旋转矩阵 -> 单位四元数（w>=0）。"""
    tr = np.trace(R)
    if tr > 0:
        w = np.sqrt(1.0 + tr) / 2.0
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                      (R[0, 2] - R[2, 0]) / (4 * w),
                      (R[1, 0] - R[0, 1]) / (4 * w)])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot_err_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def m_rows(p):
    """3×10 矩阵 m(p)，使 R(q) p = m(p) @ v2(q) 对一切 q∈S^3 精确成立。
    与 verify/sanity_constraints_alm.py 的 lstsq 版本等价（validate_code.py V1 交叉验证）。"""
    px, py, pz = p
    m = np.zeros((3, 10))
    m[0] = [px, px, -px, -px, 0.0, SQRT2 * pz, -SQRT2 * py, SQRT2 * py, SQRT2 * pz, 0.0]
    m[1] = [py, -py, py, -py, -SQRT2 * pz, 0.0, SQRT2 * px, SQRT2 * px, 0.0, SQRT2 * pz]
    m[2] = [pz, -pz, -pz, pz, SQRT2 * py, -SQRT2 * px, 0.0, 0.0, SQRT2 * px, SQRT2 * py]
    return m


def rand_unit(n, rng):
    g = rng.normal(size=(n, 4))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 2. 仿射约束（对 (v2,t)）
# ---------------------------------------------------------------------------


def c_plane(v, n, d):
    """平面 n^T(Rv+t)+d >= 0  <=>  -n^T(Rv+t)-d <= 0.  n=(0,0,1), d=-H_TAB."""
    Mm = m_rows(v)
    a = -(np.asarray(n, float) @ Mm) - d * E_TR
    return a, -np.asarray(n, float)


def c_depth(v, zmin):
    """(Rv+t)_z >= zmin."""
    Mm = m_rows(v)
    return -Mm[2] + zmin * E_TR, np.array([0.0, 0.0, -1.0])


def c_fov(v, tau, s):
    """s (Rv+t)_x - tau (Rv+t)_z <= 0（s=+1 左边界, s=-1 右边界）."""
    Mm = m_rows(v)
    return s * Mm[0] - tau * Mm[2], np.array([float(s), 0.0, -tau])


def c_tilt(alpha_half):
    """q1^2+q2^2 <= sin^2(alpha/2)（相机/物体横滚+俯仰锥）."""
    a = np.zeros(10)
    a[1] += 1.0
    a[2] += 1.0
    return a - np.sin(alpha_half) ** 2 * E_TR, np.zeros(3)


# ---------------------------------------------------------------------------
# 3. PnP 目标装配（bearing 型）
# ---------------------------------------------------------------------------


def build_objective(P, U):
    """J(q,t)=Σ_i (R p_i+t)^T (I-u_i u_i^T/(u_i^Tu_i)) (R p_i+t).
    返回 M0(10,10), D0(3,10), G0(3,3)。"""
    N = P.shape[0]
    M0 = np.zeros((10, 10))
    D0 = np.zeros((3, 10))
    G0 = np.zeros((3, 3))
    for i in range(N):
        u = U[i]
        Pm = np.eye(3) - np.outer(u, u) / (u @ u)
        Mm = m_rows(P[i])
        M0 += 0.5 * (Mm.T @ Pm @ Mm + Mm.T @ Pm.T @ Mm)  # = sym(Mm^T Pm Mm) since Pm sym
        D0 += Pm @ Mm
        G0 += Pm
    return M0, D0, G0


def obj_value(q, t, M, D, G):
    x = v2(q)
    return float(x @ M @ x + 2.0 * t @ (D @ x) + t @ G @ t)


def obj_min_t(M, D, G):
    """消去 t：t* = -G^{-1} D v2, 返回 (Mred, G, D)。要求 G≻0。"""
    Mred = M - D.T @ np.linalg.solve(G, D)
    return Mred


# ---------------------------------------------------------------------------
# 4. 内层：S^3 上四次型的多起点（数值全局替代）
# ---------------------------------------------------------------------------


def _sphere_obj_grad(u, Mred):
    """f(u) = v2(u)^T M v2(u) / ||u||^4（在方向 q=u/||u|| 上等价于 v2(q)^T M v2(q)）。
    解析梯度：df/du = 2 Jv2(u)^T M v2(u)/s^4 - 4 f u / s^2。"""
    s = np.linalg.norm(u)
    xu = v2(u)
    Mx = Mred @ xu
    f = float(xu @ Mx) / s ** 4
    g_u = 2.0 * (Jv2(u).T @ Mx) / s ** 4 - 4.0 * f * u / s ** 2
    return f, g_u


def solve_inner(Mred, n_starts=8, rng=None, q0=None, gtol=1e-10, maxiter=400):
    """min_{q∈S^3} v2(q)^T Mred v2(q) 的多起点 BFGS（返回 q 与值）。
    这是 A2（内层枚举完备性）的数值替代，不是证书。q0 给出时作为热启动之一。"""
    rng = rng or np.random.default_rng(0)
    best_f, best_q = np.inf, None
    starts = []
    if q0 is not None:
        starts.append(q0 / np.linalg.norm(q0))
        for _ in range(max(1, n_starts - 1)):
            v = q0 / np.linalg.norm(q0) + 0.35 * rng.normal(size=4)
            starts.append(v / np.linalg.norm(v))
    else:
        starts = list(rand_unit(n_starts, rng))
    for u0 in starts:
        res = minimize(lambda u: _sphere_obj_grad(u, Mred), u0, jac=True, method="BFGS",
                       options=dict(gtol=gtol, maxiter=maxiter))
        u = res.x
        nu = np.linalg.norm(u)
        if not np.isfinite(nu) or nu < 1e-12:
            continue
        q = u / nu
        f = float(v2(q) @ Mred @ v2(q))
        if f < best_f:
            best_f, best_q = f, q
    if best_q is None:
        raise RuntimeError("inner solve failed")
    return best_f, best_q


def solve_qp_global(M, D, G, n_starts=64, rng=None, q0=None):
    """无约束 (C-QPEP) 的数值全局替代：消 t 后对 q 多起点。返回 J, q, t。"""
    Mred = obj_min_t(M, D, G)
    f, q = solve_inner(Mred, n_starts=n_starts, rng=rng, q0=q0)
    t = -np.linalg.solve(G, D @ v2(q))
    return f, q, t


# ---------------------------------------------------------------------------
# 5. ALM（冻结活跃集 + 一致性检查；失败时回退到 full-L_rho 直接下降）
# ---------------------------------------------------------------------------


def _alm_quadratic(M0, D0, G0, A, B, mu, rho, act):
    MA, DA, GA = M0.copy(), D0.copy(), G0.copy()
    for j in act:
        aj, bj = A[j], B[j]
        MA += mu[j] / 2.0 * (np.outer(E_TR, aj) + np.outer(aj, E_TR)) + rho / 2.0 * np.outer(aj, aj)
        DA += mu[j] / 2.0 * np.outer(bj, E_TR) + rho / 2.0 * np.outer(bj, aj)
        GA += rho / 2.0 * np.outer(bj, bj)
    return MA, DA, GA


def _gfun(A, B, q, t):
    return A @ v2(q) + B @ t


def _rho_of(MA, DA, GA):
    return MA - DA.T @ np.linalg.solve(GA, DA)


def full_L(q, t, M0, D0, G0, A, B, mu, rho):
    g = _gfun(A, B, q, t)
    return obj_value(q, t, M0, D0, G0) + np.sum(np.maximum(mu + rho * g, 0.0) ** 2 - mu ** 2) / (2 * rho)


def full_L_grad(q, t, M0, D0, G0, A, B, mu, rho):
    """∇_{q,t} L_rho（piecewise smooth）。"""
    x = v2(q)
    g = A @ x + B @ t
    act = (mu + rho * g) > 0.0
    s = np.where(act, mu + rho * g, 0.0)          # = [mu+rho g]_+
    # ∇_q J = 2 Jv2^T (M0 x + D0^T t)
    dJ_q = 2.0 * (Jv2(q).T @ (M0 @ x + D0.T @ t))
    dJ_t = 2.0 * (D0 @ x + G0 @ t)
    dg_q = A @ Jv2(q)                              # p×4
    dg_t = B                                       # p×3
    dL_q = dJ_q + dg_q.T @ s
    dL_t = dJ_t + dg_t.T @ s
    return np.concatenate([dL_q, dL_t])


def alm_qpep(M0, D0, G0, A, B, q_init, t_init, mu0=None, rho0=10.0,
             max_outer=50, tol=1e-10, rho_max=1e7, n_starts=6, rng=None,
             verbose=False, inner="multistart", inner_kw=None):
    """ALM-QPEP 前向。

    返回 dict: q,t,mu,rho,iters,viol,J,hist, fallback_count, ok
    说明：
      * 每个外层 iteration 冻结活跃集 A = {j: mu_j + rho g_j > 0}，
        按定理 2.1 装配 M_A/D_A/G_A（10×10/3×10/3×3，与 p 无关），丢给内层；
      * 内层解出 (q,t) 后若活跃集不一致（Lemma 2.1(b)），重新识别并重解；
      * 连续 5 次不一致时，回退到对 full L_rho 做 7 维 BFGS（仍用同一 ALM 更新）。

    inner : "multistart"（默认，多起点 BFGS，与历史结果完全一致）或
            "exact"（quartic_roots.solve_inner_exact 的 KKT 全枚举）。
            exact 模式下若枚举掉了路径（fallback=True），solve_inner_exact 内部
            已回退到多起点，本函数只统计次数，不声称精确。
    """
    rng = rng or np.random.default_rng(0)
    inner_kw = dict(inner_kw or {})
    if inner not in ("multistart", "exact"):
        raise ValueError("inner must be 'multistart' or 'exact', got %r" % (inner,))
    _inner_stats = dict(exact_calls=0, exact_fallbacks=0, exact_time=0.0)
    p = A.shape[0]
    mu = np.zeros(p) if mu0 is None else np.array(mu0, float)
    q, t = q_init / np.linalg.norm(q_init), np.array(t_init, float)
    rho = float(rho0)
    hist = []
    fallback_count = 0
    prev_viol = np.inf
    for it in range(max_outer):
        g = _gfun(A, B, q, t)
        viol = max(0.0, float(g.max()))
        hist.append(dict(it=it, J=obj_value(q, t, M0, D0, G0), viol=viol,
                         mu=np.array(mu), rho=rho))
        if verbose:
            print("  ALM it=%2d J=%.10f viol=%.3e rho=%.1f ||mu||1=%.4f"
                  % (it, hist[-1]["J"], viol, rho, np.abs(mu).sum()))
        if viol <= tol and it > 0:
            break
        act = np.where(mu + rho * g > 0.0)[0]
        consistent = False
        for _ in range(5):
            MA, DA, GA = _alm_quadratic(M0, D0, G0, A, B, mu, rho, act)
            Mred = _rho_of(MA, DA, GA)
            if inner == "exact":
                import quartic_roots as _QR
                _t0 = time.perf_counter()
                _ex = _QR.solve_inner_exact(
                    Mred, seed=_inner_stats["exact_calls"], **inner_kw)
                _inner_stats["exact_time"] += time.perf_counter() - _t0
                _inner_stats["exact_calls"] += 1
                if _ex.get("fallback", False):
                    _inner_stats["exact_fallbacks"] += 1
                q_new = _ex["q"]
            else:
                _, q_new = solve_inner(Mred, n_starts=n_starts, rng=rng, q0=q)
            t_new = -np.linalg.solve(GA, DA @ v2(q_new))
            g_new = _gfun(A, B, q_new, t_new)
            act_new = np.where(mu + rho * g_new > 0.0)[0]
            q, t = q_new, t_new
            if np.array_equal(act_new, act):
                consistent = True
                break
            act = act_new
        if not consistent:
            fallback_count += 1
            # 回退：直接最小化 full L_rho（7 维），从当前点出发
            x0 = np.concatenate([q, t])

            def fg(z):
                nq = np.linalg.norm(z[:4])
                qq = z[:4] / nq
                tt = z[4:]
                f = full_L(qq, tt, M0, D0, G0, A, B, mu, rho)
                grad = full_L_grad(qq, tt, M0, D0, G0, A, B, mu, rho)
                gq, gt = grad[:4], grad[4:]
                # q = z_q/||z_q|| ⇒ ∇_{z_q} = (I - qq^T) ∇_q / ||z_q||
                gz = np.concatenate([(gq - qq * (qq @ gq)) / nq, gt])
                return f, gz
            res = minimize(fg, x0, jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=400))
            z = res.x
            q = z[:4] / np.linalg.norm(z[:4])
            t = z[4:]
        # ALM 乘子更新（保证 mu>=0）
        g = _gfun(A, B, q, t)
        mu = np.maximum(0.0, mu + rho * g)
        viol = max(0.0, float(g.max()))
        # rho 调度：violation 未按 2× 下降则放大
        if it >= 2 and viol > 0.5 * prev_viol and viol > tol:
            rho = min(rho * 5.0, rho_max)
        prev_viol = min(prev_viol, viol) if viol > 0 else viol
    g = _gfun(A, B, q, t)
    viol = max(0.0, float(g.max()))
    J = obj_value(q, t, M0, D0, G0)
    return dict(q=q, t=t, mu=mu, rho=rho, iters=it + 1, viol=viol, J=J,
                hist=hist, fallback_count=fallback_count,
                inner=inner, **_inner_stats,
                ok=bool(viol <= max(tol, 1e-8)))


# ---------------------------------------------------------------------------
# 6. 软惩罚与事后投影 baseline
# ---------------------------------------------------------------------------


def soft_penalty(M0, D0, G0, A, B, lam, q_init, t_init, n_starts=10, rng=None):
    """min J + lam * Σ [g_j]_+^2 （二次惩罚），7 维 BFGS 多起点。"""
    rng = rng or np.random.default_rng(0)

    def fg(z):
        q = z[:4] / np.linalg.norm(z[:4])
        t = z[4:]
        x = v2(q)
        g = A @ x + B @ t
        gp = np.maximum(g, 0.0)
        J = x @ (M0 @ x) + 2.0 * t @ (D0 @ x) + t @ (G0 @ t)
        f = J + lam * float(gp @ gp)
        dJ_q = 2.0 * (Jv2(q).T @ (M0 @ x + D0.T @ t))
        dJ_t = 2.0 * (D0 @ x + G0 @ t)
        dg_q = A @ Jv2(q)
        dg_t = B
        dL_q = dJ_q + 2.0 * lam * dg_q.T @ gp
        dL_t = dJ_t + 2.0 * lam * dg_t.T @ gp
        return f, np.concatenate([dL_q, dL_t])

    x0 = np.concatenate([q_init, t_init])
    res = minimize(fg, x0, jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=600))
    best = (res.fun, res.x)
    for k in range(n_starts):
        if k % 2 == 0:
            pert = np.concatenate([0.25 * rng.normal(size=4), 0.02 * rng.normal(size=3)])
        else:
            pert = np.concatenate([0.5 * rng.normal(size=4), 0.05 * rng.normal(size=3)])
        z0 = np.concatenate([q_init, t_init]) + pert
        res = minimize(fg, z0, jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=600))
        if res.fun < best[0]:
            best = (res.fun, res.x)
    z = best[1]
    q = z[:4] / np.linalg.norm(z[:4])
    t = z[4:]
    return dict(q=q, t=t, J=obj_value(q, t, M0, D0, G0), lam=lam)


def project_feasible(M0, D0, G0, A, B, q_u, t_u, n_starts=4, rng=None):
    """Post-hoc 投影：min ||q-q_u||^2 + ||t-t_u||^2 s.t. g<=0。
    SLSQP 多起点；失败则回退到『只投影 t』的凸 QP（cvxpy）。返回 q,t。"""
    rng = rng or np.random.default_rng(0)
    # 目标用四元数欧氏距离（q 与 -q 同姿态，取与 q_u 同号）
    def obj(z):
        q = z[:4] / np.linalg.norm(z[:4])
        if q @ q_u < 0:
            q = -q
        return float(np.sum((q - q_u) ** 2) + np.sum((z[4:] - t_u) ** 2))

    def cons(z):
        q = z[:4] / np.linalg.norm(z[:4])
        return -(A @ v2(q) + B @ z[4:])          # SLSQP: >=0

    best = None
    starts = [np.concatenate([q_u, t_u])]
    for _ in range(n_starts - 1):
        starts.append(np.concatenate([q_u + 0.15 * rng.normal(size=4), t_u + 0.02 * rng.normal(size=3)]))
    for z0 in starts:
        try:
            res = minimize(obj, z0, method="SLSQP",
                           constraints=[{"type": "ineq", "fun": cons}],
                           options=dict(maxiter=300, ftol=1e-12))
            z = res.x
            q = z[:4] / np.linalg.norm(z[:4])
            g = A @ v2(q) + B @ z[4:]
            viol = max(0.0, float(g.max()))
            if best is None or (viol < best[0] - 1e-12) or (abs(viol - best[0]) <= 1e-12 and res.fun < best[1]):
                best = (viol, res.fun, q, z[4:].copy())
        except Exception:
            continue
    if best is not None and best[0] <= 1e-6:
        return dict(q=best[2], t=best[3], viol=best[0], method="slsqp")
    # 回退：固定 q=q_u，只投影 t（3 变量凸 QP）
    import cvxpy as cp
    t = cp.Variable(3)
    c = A @ v2(q_u)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(t - t_u)), [c + B @ t <= 0])
    prob.solve(solver="OSQP", eps_abs=1e-10, eps_rel=1e-10)
    tval = np.array(t.value).ravel()
    g = A @ v2(q_u) + B @ tval
    return dict(q=q_u / np.linalg.norm(q_u), t=tval,
                viol=max(0.0, float(g.max())), method="t-only")


# ---------------------------------------------------------------------------
# 7. 合成桌面 PnP 实例
# ---------------------------------------------------------------------------

# 约束数分配（总 p；README 记录）：(pen, depth, fov_pairs, tilt)
P_SPLIT = {
    8:  dict(pen=5,  depth=2,  fov=0, tilt=1),
    16: dict(pen=9,  depth=4,  fov=1, tilt=1),
    32: dict(pen=17, depth=8,  fov=3, tilt=1),
    64: dict(pen=33, depth=12, fov=9, tilt=1),
}
# 说明：非穿透点从物体底面 6×6 网格取（含 4 角），p 越大覆盖越密。
# 由于底面是平面，≥3 个不共线接触点在刚性物体上已等价于整面非穿透，
# 所以 p 的增大主要检验"常数维/装配 O(p)/证书 p 无关成本"，而非精度提升
# （协议 A5 的预期：p>=16 后精度饱和）。README 有明确说明。

BOX_HX, BOX_HY, BOX_HZ = 0.09, 0.06, 0.05
H_TAB = 0.75
ZMIN = 0.05
TAU = 0.84          # tan(40°)
ALPHA = np.deg2rad(12.0)


def _object_points():
    hx, hy, hz = BOX_HX, BOX_HY, BOX_HZ
    corners = np.array([[sx * hx, sy * hy, sz * hz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    faces = np.array([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0], [0, 0, hz], [0, 0, -hz]], float)
    return np.vstack([corners, faces])


P_OBJ = _object_points()          # N=14
N_PTS = P_OBJ.shape[0]

_BOTTOM_GRID = None


def bottom_grid(n=6):
    """物体底面 z=-hz 上的 n×n 网格（含 4 角），用于非穿透约束点与穿透度量。"""
    global _BOTTOM_GRID
    if _BOTTOM_GRID is None or _BOTTOM_GRID[0].shape[0] != n * n:
        xs = np.linspace(-BOX_HX, BOX_HX, n)
        ys = np.linspace(-BOX_HY, BOX_HY, n)
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        _BOTTOM_GRID = (np.column_stack([Xg.ravel(), Yg.ravel(), -BOX_HZ * np.ones(Xg.size)]), n)
    return _BOTTOM_GRID[0]


def dense_bottom(n=21):
    G = bottom_grid(6)
    xs = np.linspace(-BOX_HX, BOX_HX, n)
    ys = np.linspace(-BOX_HY, BOX_HY, n)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([Xg.ravel(), Yg.ravel(), -BOX_HZ * np.ones(Xg.size)])


def make_instance(seed, p, noise=0.01, bias_z=0.08, bias_xy=0.005):
    """生成一个 6-DoF 桌面 PnP 实例（已知真值）。

    返回 dict，包含 GT、观测 bearing、J 的 (M0,D0,G0)、约束 (A,B) 及其分类。
    """
    if p not in P_SPLIT:
        raise ValueError("p must be in %s" % sorted(P_SPLIT))
    rng = np.random.default_rng(10000 + seed)
    roll = rng.uniform(-0.10, 0.10)
    pitch = rng.uniform(-0.10, 0.10)
    yaw = rng.uniform(-np.pi, np.pi)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_gt = Rz @ Ry @ Rx
    q_gt = q_from_R(R_gt)
    txy = rng.uniform(-0.04, 0.04, size=2)
    # GT 恰好接触桌面（底面网格最低点 = H_TAB）
    Vb = dense_bottom(41)
    zmin_gt = float((R_gt @ Vb.T).T[:, 2].min())
    t_gt = np.array([txy[0], txy[1], H_TAB - zmin_gt])
    Bw = P_OBJ @ R_gt.T + t_gt
    bias = np.array([bias_xy, -bias_xy, -bias_z])
    Bw_obs = Bw + bias + noise * rng.normal(size=Bw.shape)
    U = Bw_obs / np.linalg.norm(Bw_obs, axis=1, keepdims=True)
    M0, D0, G0 = build_objective(P_OBJ, U)

    # ---- 约束 ----
    sp = P_SPLIT[p]
    actypes = []
    A_l, B_l = [], []

    # (1) 非穿透：底面网格点（先 4 角，再固定顺序的其余点）
    grid = bottom_grid(6)
    corner_mask = (np.abs(np.abs(grid[:, 0]) - BOX_HX) < 1e-12) & (np.abs(np.abs(grid[:, 1]) - BOX_HY) < 1e-12)
    idx_corners = np.where(corner_mask)[0]
    idx_rest = np.array([i for i in range(grid.shape[0]) if i not in set(idx_corners.tolist())])
    rngp = np.random.default_rng(777 + seed)
    idx_rest = rngp.permutation(idx_rest)
    order = np.concatenate([idx_corners, idx_rest])[: sp["pen"]]
    for i in order:
        a, b = c_plane(grid[i], np.array([0.0, 0.0, 1.0]), -H_TAB)
        A_l.append(a); B_l.append(b); actypes.append("pen")
    # (2) 正深度：物体点（取前 depth 个，但不含底面点，避免与穿透重复语义）
    for i in range(sp["depth"]):
        a, b = c_depth(P_OBJ[i], ZMIN)
        A_l.append(a); B_l.append(b); actypes.append("depth")
    # (3) FoV 对：每个点两条边界
    for k in range(sp["fov"]):
        v = P_OBJ[(k * 3) % N_PTS]
        for s in (+1, -1):
            a, b = c_fov(v, TAU, s)
            A_l.append(a); B_l.append(b); actypes.append("fov")
    # (4) 倾角锥
    a, b = c_tilt(ALPHA / 2.0)
    A_l.append(a); B_l.append(b); actypes.append("tilt")

    A = np.array(A_l)
    B = np.array(B_l)
    assert A.shape[0] == p, (A.shape[0], p)

    Vd = dense_bottom(21)
    return dict(seed=seed, p=p, noise=noise, bias_z=bias_z, bias_xy=bias_xy,
                R_gt=R_gt, q_gt=q_gt, t_gt=t_gt, P=P_OBJ, U=U, Bw_obs=Bw_obs,
                M0=M0, D0=D0, G0=G0, A=A, B=B, actypes=actypes,
                H_TAB=H_TAB, V_dense=Vd,
                g_gt=A @ v2(q_gt) + B @ t_gt)


# ---------------------------------------------------------------------------
# 8. 指标
# ---------------------------------------------------------------------------


def penetration_mm(R, t, V_dense, H=H_TAB):
    """max_v [H - (R v + t)_z]_+ (mm)。V_dense 是物体坐标系下的密集底面点。"""
    z = (R @ V_dense.T).T[:, 2] + t[2]
    return float(max(0.0, H - z.min()) * 1000.0)


def mesh_penetration_mm(R, t, H=H_TAB):
    """只看 8 个角点/6 面心的 mesh 顶点（交叉检查用）。"""
    z = (R @ P_OBJ.T).T[:, 2] + t[2]
    return float(max(0.0, H - z.min()) * 1000.0)


def max_viol(A, B, q, t):
    return float(max(0.0, (A @ v2(q) + B @ t).max()))


def rot_deg(q, q_gt):
    return rot_err_deg(R_of(q), R_of(q_gt))


def trans_mm(t, t_gt):
    return float(1000.0 * np.linalg.norm(np.asarray(t) - np.asarray(t_gt)))


def active_jacobian_rank(q, t, A, B, act, eps=1e-7):
    """在 6 维切空间 (δω, δt) 上数值计算活跃约束 Jacobian 的秩。
    δω 为右扰动 R exp(δω^): g(q(δω), t+δt) 的一阶系数。
    这是『四脚桌 LICQ 失效』在真实实例上的对应量。"""
    R = R_of(q)
    rows = []
    for j in act:
        row = np.zeros(6)
        for k in range(3):
            w = np.zeros(3); w[k] = eps
            th = np.linalg.norm(w)
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            Rk = R @ (np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K))
            qk = q_from_R(Rk)
            gk = A[j] @ v2(qk) + B[j] @ t
            g0 = A[j] @ v2(q) + B[j] @ t
            row[k] = (gk - g0) / eps
        row[3:] = B[j]
        rows.append(row)
    rows = np.array(rows)
    return int(np.linalg.matrix_rank(rows, tol=1e-5)), rows
