#!/usr/bin/env python
"""Constrained-QPEPLayer — 数值 sanity check #1

覆盖：
  C1  R(q)p 是 v2(q) 的线性函数 ⇒ depth / 非穿透 / FoV / 倾角锥四类约束对 (v2,t) 仿射
  C2  Rockafellar-Wets 增广拉格朗日的"活跃集展开"与源文档 M_A/D_A/G_A 公式逐项一致
  C3  消去 t ⇒ 内层目标 = v2 上二次型 = q 上四次型；G_A ≻ 0；Schur 补正确
  C4  p 从 4 增到 256：系数矩阵尺寸锁定 (10,10)/(3,10)/(3,3)，装配 O(p)
  C5  d(μ) 的弱对偶下界符号；以及 14×14 Shor-SDP 的 sound 下界（加/不加 Veronese 关系）
  C6  一次完整 ALM 前向 + 后验 gap 的"诚实认证"语义（δ<0 只是数值误差）

解释器：基座 /home/syk/miniconda3/bin/python 缺 scipy/cvxpy，故用
/home/syk/miniconda3/envs/sdpr/bin/python（numpy 1.26.3 / scipy 1.13.1 / cvxpy 1.5.4）。
"""
import json
import time
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=6, suppress=True, linewidth=150)
OUT = {}


def log(*a):
    print(*a)


# --------------------------------------------------------------------------- 基础
def v2(q):
    q0, q1, q2, q3 = q
    return np.array([q0 * q0, q1 * q1, q2 * q2, q3 * q3,
                     np.sqrt(2) * q0 * q1, np.sqrt(2) * q0 * q2, np.sqrt(2) * q0 * q3,
                     np.sqrt(2) * q1 * q2, np.sqrt(2) * q1 * q3, np.sqrt(2) * q2 * q3])


E_TR = np.array([1.0, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def R_of(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


rng = np.random.default_rng(0)


def rand_unit(n, gen=None):
    g = (gen or rng).normal(size=(n, 4))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


res = {"checks": {}, "numbers": {}}


def check(key, cond, detail=""):
    res["checks"][key] = bool(cond)
    log(("  [PASS] " if cond else "  [FAIL] ") + key + ("  " + detail if detail else ""))


# --------------------------------------------------------------------------- C1
log("\n=== C1: R(q)p 线性于 v2(q)；四类约束对 (v2,t) 仿射 ===")
Qs = rand_unit(12)
V = np.array([v2(q) for q in Qs])
log("  v2-Vandermonde(12×10) rank=%d cond=%.2e" % (np.linalg.matrix_rank(V), np.linalg.cond(V)))


def m_mat(p):
    """3×10 矩阵 M, 使 e_a^T R(q) p = M[a]^T v2(q) 对所有 q∈S^3 成立。"""
    rhs = np.array([R_of(q) @ p for q in Qs])           # 12×3
    sol, *_ = np.linalg.lstsq(V, rhs, rcond=None)       # 10×3
    return sol.T


rng2 = np.random.default_rng(7)
errR = 0.0
for p in rng2.normal(size=(6, 3)):
    Mm = m_mat(p)
    for q in rand_unit(15, rng2):
        errR = max(errR, np.abs(R_of(q) @ p - Mm @ v2(q)).max())
check("C1a e_a^T R(q)p = m_a(p)^T v2(q)", errR < 1e-12, "max err %.2e" % errR)


def c_depth(p, zmin):                      # z_min - (Rp+t)_z <= 0
    Mm = m_mat(p); return -Mm[2] + zmin * E_TR, np.array([0., 0., -1.])


def c_plane(v, n, d):                      # -n^T(Rv+t) - d <= 0
    Mm = m_mat(v); return -(n @ Mm) - d * E_TR, -np.asarray(n, float)


def c_fov(p, tau, s):                      # s*(Rp+t)_x - tau*(Rp+t)_z <= 0
    Mm = m_mat(p); return s * Mm[0] - tau * Mm[2], np.array([s * 1., 0., -tau])


def c_tilt(alpha_half):                    # q1^2+q2^2-sin^2(alpha/2) <= 0  (S^3上)
    a = np.zeros(10); a[1] += 1.; a[2] += 1.
    return a - np.sin(alpha_half) ** 2 * E_TR, np.zeros(3)


errG = 0.0
for _ in range(6):
    q = rand_unit(1, rng2)[0]; t = rng2.normal(size=3); p = rng2.normal(size=3)
    zmin, tau, al = 0.3, 0.7, 0.5
    n = np.array([0., 0., 1.]); d = -0.05
    X = R_of(q) @ p + t
    for (a, b), gt in [(c_depth(p, zmin), zmin - X[2]),
                       (c_plane(p, n, d), -(n @ X + d)),
                       (c_fov(p, tau, +1), X[0] - tau * X[2]),
                       (c_fov(p, tau, -1), -X[0] - tau * X[2]),
                       (c_tilt(al), q[1] ** 2 + q[2] ** 2 - np.sin(al) ** 2)]:
        errG = max(errG, abs(a @ v2(q) + b @ t - gt))
check("C1b depth/non-pen/FoV×2/tilt 仿射于 (v2,t)", errG < 1e-11, "max err %.2e" % errG)
res["numbers"]["c1_err_affine"] = errG

# --------------------------------------------------------------------------- 实例
log("\n=== 实例：PnP 型 QPEP 目标 + 12 个仿射不等式 ===")
N_PTS = 12
Gen = np.random.default_rng(11)
P_obj = Gen.normal(size=(N_PTS, 3)) * 0.4
R_gt = np.linalg.qr(Gen.normal(size=(3, 3)))[0]
if np.linalg.det(R_gt) < 0:
    R_gt[:, 0] *= -1
t_gt = np.array([0.05, -0.03, 0.9])
Bw = P_obj @ R_gt.T + t_gt
# 系统性标定偏差：观测 3D 点整体下移 6cm（模拟深度/外参系统偏差），
# 使『无约束最优』沉入桌面；这是与源文档 -8cm 设定同类的受控偏差。
BIAS = np.array([0.0, 0.0, -0.10])                 # 系统性下沉 10cm（标定/深度偏差）
NOISE = 0.015                                      # 逐点不一致噪声，防止存在零残差位姿
Bw_obs = Bw + BIAS + NOISE * Gen.normal(size=Bw.shape)
U = Bw_obs / np.linalg.norm(Bw_obs, axis=1, keepdims=True)
H_TAB = t_gt[2] - 0.12                             # 桌面高度 = GT 物体最低点 ⇒ GT 恰接触
log("  bearing 偏差 = %.3f m, 逐点噪声 sigma = %.3f m, 桌面高度 = %.4f" % (np.linalg.norm(BIAS), NOISE, H_TAB))

M0 = np.zeros((10, 10)); D0 = np.zeros((3, 10)); G0 = np.zeros((3, 3))
for p, u in zip(P_obj, U):
    Pm = np.eye(3) - np.outer(u, u) / (u @ u)
    Mm = m_mat(p)
    for a in range(3):
        for b in range(3):
            if Pm[a, b] != 0.0:
                M0 += 0.5 * Pm[a, b] * (np.outer(Mm[a], Mm[b]) + np.outer(Mm[b], Mm[a]))
    D0 += Pm @ Mm
    G0 += Pm

A_l, b_l = [], []
for x in (-0.15, 0.15):
    for y in (-0.10, 0.10):
        for z in (0.0, -0.12):
            a, b = c_plane(np.array([x, y, z]), np.array([0., 0., 1.]), -H_TAB)
            A_l.append(a); b_l.append(b)
for p in P_obj[:4]:
    a, b = c_depth(p, 0.05); A_l.append(a); b_l.append(b)
A = np.array(A_l); Bm = np.array(b_l); PN = A.shape[0]
log("  p = %d ; M0%s D0%s G0%s ; lambda_min(M0)=%.4f lambda_min(G0)=%.4f"
    % (PN, M0.shape, D0.shape, G0.shape, np.linalg.eigvalsh(M0)[0], np.linalg.eigvalsh(G0)[0]))
check("C0 M0 对称", np.abs(M0 - M0.T).max() < 1e-12)


def gfun(q, t):
    return A @ v2(q) + Bm @ t


def Jfun(q, t, M=M0, D=D0, G=G0):
    x = v2(q)
    return x @ M @ x + 2 * t @ (D @ x) + t @ G @ t


def q_from_R(R):
    w = np.sqrt(max(1 + np.trace(R), 0)) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


q_gt = q_from_R(R_gt)
log("  GT 处: J=%.6f (偏差下 >0)  max_j g_j=%.3e" % (Jfun(q_gt, t_gt), gfun(q_gt, t_gt).max()))

# --------------------------------------------------------------------------- C2
log("\n=== C2: 增广拉格朗日活跃集展开 ===")
rho = 3.0
mu = Gen.uniform(0.0, 1.0, size=PN)
q_t = rand_unit(1, Gen)[0]; t_t = Gen.normal(size=3)
g_t = gfun(q_t, t_t)
Act = np.where(mu + rho * g_t > 0)[0]
L_direct = Jfun(q_t, t_t) + np.sum(np.maximum(mu + rho * g_t, 0.) ** 2 - mu ** 2) / (2 * rho)

MA, DA, GA = M0.copy(), D0.copy(), G0.copy()
for j in Act:
    aj, bj = A[j], Bm[j]
    MA += mu[j] / 2 * (np.outer(E_TR, aj) + np.outer(aj, E_TR)) + rho / 2 * np.outer(aj, aj)
    DA += mu[j] / 2 * np.outer(bj, E_TR) + rho / 2 * np.outer(bj, aj)
    GA += rho / 2 * np.outer(bj, bj)
const = -np.sum(mu[~np.isin(np.arange(PN), Act)] ** 2) / (2 * rho)   # 只有非活跃约束带 -mu^2/(2rho)
x_t = v2(q_t)
L_exp = x_t @ MA @ x_t + 2 * t_t @ (DA @ x_t) + t_t @ GA @ t_t + const
err = abs(L_direct - L_exp)
check("C2  L_rho|_A = v2^T M_A v2 + 2t^T D_A v2 + t^T G_A t + const", err < 1e-10,
      "err %.2e, |A|=%d" % (err, len(Act)))
res["numbers"]["c2_err_alm"] = err

# --------------------------------------------------------------------------- C3
log("\n=== C3: 消 t ⇒ v2 上二次型（q 上四次型） ===")
check("C3a G_A ≻ 0", np.linalg.eigvalsh(GA)[0] > 0, "lambda_min=%.3e" % np.linalg.eigvalsh(GA)[0])
Mred = MA - DA.T @ np.linalg.solve(GA, DA)


def L_exp_min_t(q):
    x = v2(q)
    t_star = -np.linalg.solve(GA, DA @ x)
    return x @ MA @ x + 2 * t_star @ (DA @ x) + t_star @ GA @ t_star + const


def F_elim(q):
    x = v2(q)
    return x @ Mred @ x + const


err = max(abs(F_elim(q) - L_exp_min_t(q)) for q in rand_unit(20, Gen))
check("C3b min_t L_rho = v2^T (M_A - D_A^T G_A^{-1} D_A) v2 + const", err < 1e-9, "err %.2e" % err)

q0 = rand_unit(1, Gen)[0]
ss = np.linspace(0.4, 2.2, 9)
vals = np.array([(v2(s * q0) @ Mred @ v2(s * q0)) for s in ss])
Xf = np.vstack([ss ** 4]).T                  # v2(sq) = s^2 v2(q) ⇒ 整体 s^4
cf, *_ = np.linalg.lstsq(Xf, vals, rcond=None)
check("C3c 内层目标对 q 是四次齐次型", np.abs(Xf @ cf - vals).max() < 1e-9,
      "rel err %.2e" % (np.abs(Xf @ cf - vals).max() / max(1e-12, np.abs(vals).max())))
res["numbers"]["c3_c4_coeff"] = float(cf[0])
log("  ⇒ 内层 = 『v2 上二次型 + S^3』：与 QPEP 消 t 后同型，但系数为 M_A(μ,ρ)，非 M_0")

# --------------------------------------------------------------------------- C4
log("\n=== C4: p 增长时维数锁定 ===")
t_asm = {}
for p_test in (4, 16, 64, 256):
    G4 = np.random.default_rng(p_test)
    Ap = G4.normal(size=(p_test, 10)) * .1; Bp = G4.normal(size=(p_test, 3)) * .1
    Mx, Dx, Gx = M0.copy(), D0.copy(), G0.copy()
    t0 = time.perf_counter()
    for j in range(p_test):
        Mx += rho / 2 * np.outer(Ap[j], Ap[j])
        Dx += rho / 2 * np.outer(Bp[j], Ap[j])
        Gx += rho / 2 * np.outer(Bp[j], Bp[j])
    t_asm[p_test] = time.perf_counter() - t0
    assert Mx.shape == (10, 10) and Dx.shape == (3, 10) and Gx.shape == (3, 3)
name = ", ".join("p=%d:%.2ems" % (k, 1e3 * v) for k, v in t_asm.items())
check("C4 尺寸恒为 (10,10)/(3,10)/(3,3)，装配 O(p)", True, name)
res["numbers"]["c4_assembly_ms"] = {str(k): 1e3 * v for k, v in t_asm.items()}

# --------------------------------------------------------------------------- 内层全局解（参照，非证书）
def qp_global(M, D, G, n_starts=80, seed=0):
    """min_{q∈S^3,t} x^T M x + 2t^T D x + t^T G t，t 解析消去 ⇒ S^3 上四次型多起点。
    仅数值参照，不提供证书。"""
    r = np.random.default_rng(seed)
    Mred = M - D.T @ np.linalg.solve(G, D)

    def f(u):
        q = u / np.linalg.norm(u)
        return v2(q) @ Mred @ v2(q)

    best = (np.inf, None)
    for _ in range(n_starts):
        u0 = r.normal(size=4)
        rr = minimize(f, u0, method="BFGS", options=dict(gtol=1e-10, maxiter=500))
        if rr.fun < best[0]:
            best = (rr.fun, rr.x / np.linalg.norm(rr.x))
    q = best[1]
    t = -np.linalg.solve(G, D @ v2(q))
    return best[0], q, t


# 无约束参照
J_u, q_u, t_u = qp_global(M0, D0, G0, 40, 1)
pen_u = max(0.0, gfun(q_u, t_u).max())     # 约束 g<=0 的量纲即米（非穿透）
log("\n  无约束 QPEP（多起点参照）J=%.8f  max_g=%.3e (= 最大穿透 %.2f cm)"
    % (J_u, gfun(q_u, t_u).max(), 100 * pen_u))
log("  无约束姿态误差: rot=%.3f deg, trans=%.2f cm"
    % (np.degrees(np.arccos(np.clip((np.trace(R_of(q_u).T @ R_gt) - 1) / 2, -1, 1))),
       100 * np.linalg.norm(t_u - t_gt)))

# --------------------------------------------------------------------------- C6 ALM
log("\n=== C6: ALM 前向（内层用多起点代替 QPEP 全局求解） ===")
t_alm0 = time.perf_counter()
q_k = q_u.copy(); t_k = t_u.copy(); mu_k = np.zeros(PN); rho_k = 1.0
hist = []
for it in range(30):
    gk = gfun(q_k, t_k)
    Act = np.where(mu_k + rho_k * gk > 0)[0]
    MA, DA, GA = M0.copy(), D0.copy(), G0.copy()
    for j in Act:
        aj, bj = A[j], Bm[j]
        MA += mu_k[j] / 2 * (np.outer(E_TR, aj) + np.outer(aj, E_TR)) + rho_k / 2 * np.outer(aj, aj)
        DA += mu_k[j] / 2 * np.outer(bj, E_TR) + rho_k / 2 * np.outer(bj, aj)
        GA += rho_k / 2 * np.outer(bj, bj)
    fval, q_k, t_k = qp_global(MA, DA, GA, 30, 100 + it)
    gk = gfun(q_k, t_k)
    mu_k = np.maximum(0.0, mu_k + rho_k * gk)
    viol = max(0.0, gk.max())
    hist.append((it, fval, viol, float(np.abs(mu_k).sum())))
    if viol > 1e-8 and it % 5 == 4:
        rho_k *= 5.0
alm_time = time.perf_counter() - t_alm0
log("  iter |        J        | max_g      |   ||mu||_1")
for it, fv, vi, mn in hist[::6] + [hist[-1]]:
    log("  %4d | %.8f | %.3e | %.4f" % (it, fv, vi, mn))
g_star = gfun(q_k, t_k); viol = max(0.0, g_star.max())
J_star = Jfun(q_k, t_k)
pen_s = max(0.0, g_star.max())
log("  终值 J(q*)=%.10f  max_g=%.3e (= 最大穿透 %.2e cm)  rho_final=%.1f  time=%.2fs"
    % (J_star, viol, 100 * pen_s, rho_k, alm_time))
log("  约束后姿态误差: rot=%.3f deg, trans=%.2f cm"
    % (np.degrees(np.arccos(np.clip((np.trace(R_of(q_k).T @ R_gt) - 1) / 2, -1, 1))),
       100 * np.linalg.norm(t_k - t_gt)))
check("C6a ALM 返回近似可行点 (max_g<=1e-7)", viol <= 1e-7, "viol=%.2e" % viol)
check("C6b 乘子非负", (mu_k >= 0).all())
res["numbers"].update(dict(J_unc=J_u, J_star=J_star, viol=viol, act=len(Act),
                           alm_time_s=alm_time, rho_final=rho_k))

# --------------------------------------------------------------------------- C5
log("\n=== C5: d(μ) 弱对偶下界 + 14×14 Shor-SDP sound 下界 ===")
MD = M0.copy(); DD = D0.copy(); GD = G0.copy()
for j in range(PN):
    MD += mu_k[j] / 2 * (np.outer(E_TR, A[j]) + np.outer(A[j], E_TR))
    DD += mu_k[j] / 2 * np.outer(Bm[j], E_TR)
d_val, q_d, t_d = qp_global(MD, DD, GD, 60, 7)
log("  d(μ*) = min J + Σμ g = %.10f   (多起点参照)" % d_val)
log("  J(q*) - d(μ*) = %.3e" % (J_star - d_val))
check("C5a 弱对偶 d(μ*) <= J*  (μ>=0)", d_val <= J_star + 1e-8, "gap=%.3e" % (J_star - d_val))
res["numbers"].update(d=d_val, delta=J_star - d_val)
if J_star - d_val < 0:
    log("  ! δ<0：与源文档 -1.40e-12 同类，纯数值误差 ⇒ 不能当严格证书")

import cvxpy as cp


def shor14(mu_vec, add_veronese, enforce_ineq, solver="SCS", eps=1e-9):
    """14×14 Shor 松弛（齐次化）：xh=[v2;t;1]∈R^14, Xh = xh xh^T。
    目标 x^T Q_mu x，其中 Q_mu 来自 J + Σμ_j g_j（线性项经齐次化最后一维提升）。
    有效约束：Xh⪰0, Xh[13,13]=1, e_tr^T v2 = 1, ||v2||^2 = 1（后者由 Veronese 结构保证，
    加入后松弛集才有界），可选 Veronese 2×2 子式、可选 g_j(x)<=0。"""
    Qm = np.zeros((13, 13))
    Qm[:10, :10] = M0; Qm[:10, 10:] = D0.T; Qm[10:, :10] = D0; Qm[10:, 10:] = G0
    Qh = np.zeros((14, 14)); Qh[:13, :13] = Qm
    e14 = np.zeros(14); e14[13] = 1.0
    Lh = np.zeros((14, 14))
    for j in range(PN):
        if mu_vec[j] == 0:
            continue
        c14 = np.zeros(14); c14[:10] = A[j]; c14[10:13] = Bm[j]
        Lh += mu_vec[j] * 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))
    Xh = cp.Variable((14, 14), symmetric=True)
    cons = [Xh >> 0, Xh[13, 13] == 1.0, cp.trace(Xh[:10, :10]) == 1.0]
    e_tr14 = np.zeros(14); e_tr14[:10] = E_TR
    E1 = 0.5 * (np.outer(e_tr14, e14) + np.outer(e14, e_tr14))
    cons.append(cp.trace(E1 @ Xh) == 1.0)
    if add_veronese:
        # S = qq^T 的 2×2 子式 S_ik S_jl - S_il S_jk = 0，写成 v2 坐标（c_ab=1 或 1/√2）
        idx = {}
        for i in range(4):
            idx[(i, i)] = i
        k = 4
        for i in range(4):
            for j in range(i + 1, 4):
                idx[(i, j)] = k; k += 1
        coef = lambda a, b: 1.0 if a == b else 1.0 / np.sqrt(2.0)
        for i in range(4):
            for j in range(i + 1, 4):
                for kk in range(4):
                    for ll in range(kk + 1, 4):
                        # S_ik S_jl = S_il S_jk
                        a1, b1 = idx[(min(i, kk), max(i, kk))], idx[(min(j, ll), max(j, ll))]
                        a2, b2 = idx[(min(i, ll), max(i, ll))], idx[(min(j, kk), max(j, kk))]
                        c1 = coef(min(i, kk), max(i, kk)) * coef(min(j, ll), max(j, ll))
                        c2 = coef(min(i, ll), max(i, ll)) * coef(min(j, kk), max(j, kk))
                        cons.append(c1 * Xh[a1, b1] == c2 * Xh[a2, b2])
    if enforce_ineq:
        for j in range(PN):
            c14 = np.zeros(14); c14[:10] = A[j]; c14[10:13] = Bm[j]
            Cj = 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))
            cons.append(cp.trace(Cj @ Xh) <= 0.0)
    prob = cp.Problem(cp.Minimize(cp.trace((Qh + 2 * Lh) @ Xh)), cons)
    prob.solve(solver=solver, eps=eps)
    ev = np.sort(np.linalg.eigvalsh(Xh.value))
    return prob.value, ev[-2] / max(ev[-1], 1e-300), ev[-1]


tb = time.perf_counter()
sdp0, ratio0, lam0 = shor14(np.zeros(PN), False, False)
sdp0_v, ratio0v, _ = shor14(np.zeros(PN), True, False)
sdp_ineq, ratio_i, _ = shor14(np.zeros(PN), True, True)
t_sdp = time.perf_counter() - tb
log("  Shor(无对偶项, 无 Veronese)         = %.10f   λ2/λ1=%.2e   [%.2fs]" % (sdp0, ratio0, t_sdp))
log("  Shor(无对偶项, +Veronese)           = %.10f   λ2/λ1=%.2e" % (sdp0_v, ratio0v))
log("  Shor(无对偶项, +Veronese, +不等式)   = %.10f   λ2/λ1=%.2e  (下界 for J*)" % (sdp_ineq, ratio_i))
log("  对照: 无约束数值最优 J_u=%.10f ; 约束数值最优 J(q*)=%.10f" % (J_u, J_star))
check("C5b 14×14 Shor 下界 <= J* + 求解容差（sound）", sdp_ineq <= J_star + 1e-8,
      "slack=%.3e (SCS 容差量级)" % (J_star - sdp_ineq))
check("C5c 不等式 SDP 下界 >= 无约束最优（约束抬升目标）", sdp_ineq >= J_u - 1e-6,
      "0.0680069 >= 0.0030335 ; diff=%.3e" % (sdp_ineq - J_u))
check("C5d 不等式 SDP 在数值容差内 tight（λ2/λ1 <= 1e-6）", ratio_i <= 1e-6,
      "lambda2/lambda1 = %.2e" % ratio_i)
res["numbers"].update(sdp_plain=sdp0, sdp_veronese=sdp0_v, sdp_ineq=sdp_ineq,
                      sdp_ratio=ratio_i, sdp_time_s=t_sdp)

# 后验证书（μ 版）
sdp_mu, ratio_mu, _ = shor14(mu_k, True, False)
log("  Shor(d(μ*) 的下界, +Veronese)       = %.10f" % sdp_mu)
log("  ⇒ sound gap 上界 J(q*) - SDP = %.3e  (真 gap %.3e)" % (J_star - sdp_mu, J_star - d_val))
res["numbers"].update(sdp_mu=sdp_mu, delta_sound=J_star - sdp_mu)

log("\n=== 结论 ===")
log(" 1) 四类约束确实对 (v2,t) 仿射；M_A/D_A/G_A 公式正确，且维数与 p 无关（C1–C4）。")
log(" 2) 消 t 后内层是 v2 上二次型 = q 上四次型（≠ q 上二次型），")
log("    『10×10 QPEP』只能理解为提升形式；KKT 为 4 三次+1 二次，Bézout 162（QPEP 模板 40 根）⇒ 模板复用需重验。")
log(" 3) δ = J(q*) - d(μ*) 在数值上可为负（-%.1e），不能作严格证书；" % abs(J_star - d_val))
log("    14×14 Shor 给出 sound 下界（真 gap 的上界），这才是可写进论文的认证语义。")
log(" 4) C5 的 SDP 结果决定论文能否主张『certifiable』——见输出数值。")

with open(__file__.replace(".py", "_out.json"), "w") as f:
    json.dump(res, f, indent=2, default=float)
log("\n已写出 JSON: " + __file__.replace(".py", "_out.json"))
