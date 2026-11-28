#!/usr/bin/env python
"""Constrained-QPEPLayer 实验 —— 代码正确性校验（先跑这个，再跑主实验）。

原则：先用已知真值 / 闭式 / 复现已发表的 verify 数字验证实现，再相信主表。
每一项都打印中间量；FAIL 不跳过、不改指标。

覆盖：
  V1  m(p) 精确公式 vs lstsq、v2 恒等式、齐次性
  V2  四类约束对 (v2,t) 仿射（逐点数值）
  V3  穿透度量：GT=0、平移闭式、密集网格 vs 4 角点解析
  V4  增广拉格朗日 M_A/D_A/G_A 展开；常数项只由非活跃约束贡献（含"写错版本"对照）
  V5  Schur 补 / G_A≻0 / t* 闭式
  V6  内层多起点 vs 200 起点全局参照
  V7  无噪无偏 ⇒ 无约束精确恢复 GT；桌面放低 ⇒ ALM = 无约束解
  V8  单约束平移问题闭式：ALM 精确命中投影解
  V9  **回归**：完整复现 verify/sanity_constraints_alm.py 的 p=12 数字
      (J*=0.06800693548, Shor=0.0680069355, 无约束 0.0030334724)
  V10 14×14 Shor：sound（Shor<=J*）、tight（λ2/λ1<=1e-6）、约束收紧
  V11 d(μ) 弱对偶下界（含 δ<0 的数值现象）
  V12 ALM 解的 KKT 残差 / 乘子非负 / G_A≻0 / 无 NaN / LICQ 秩
  V13 投影可行性、软惩罚 Pareto 单调性方向
  V14 活跃集 Jacobian 秩（四脚桌 LICQ 失效在真实实例上的对应）

运行：srun --partition=cpu --time=00:30:00 --ntasks=1 \
      /home/syk/miniconda3/envs/sdpr/bin/python validate_code.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpep_core as C
from certificate import shor_bound

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

res = {"checks": {}, "numbers": {}}


def log(*a):
    print(*a, flush=True)


def check(key, cond, detail=""):
    res["checks"][key] = bool(cond)
    log(("  [PASS] " if cond else "  [FAIL] ") + key + ("  " + detail if detail else ""))
    return bool(cond)


# ===========================================================================
log("\n=== V1: m(p) 精确公式 / v2 恒等式 / 齐次性 ===")
rng = np.random.default_rng(0)
Qs = C.rand_unit(12, rng)
V = np.array([C.v2(q) for q in Qs])
err_m = 0.0
for p in rng.normal(size=(8, 3)):
    rhs = np.array([C.R_of(q) @ p for q in Qs])
    sol, *_ = np.linalg.lstsq(V, rhs, rcond=None)
    for q in C.rand_unit(10, rng):
        err_m = max(err_m, np.abs(C.R_of(q) @ p - sol.T @ C.v2(q)).max(),
                    np.abs(C.R_of(q) @ p - C.m_rows(p) @ C.v2(q)).max())
check("V1a R(q)p = m(p)v2(q)（精确公式与 lstsq 都对）", err_m < 1e-12, "max err %.2e" % err_m)
res["numbers"]["v1_m_err"] = err_m
q = C.rand_unit(1, rng)[0]
check("V1b e_tr^T v2 = ||q||^2 = 1", abs(C.E_TR @ C.v2(q) - 1.0) < 1e-12)
check("V1c R(q) 正交 det=+1",
      np.abs(C.R_of(q) @ C.R_of(q).T - np.eye(3)).max() < 1e-12 and abs(np.linalg.det(C.R_of(q)) - 1) < 1e-12)
check("V1d q_from_R(R(q)) 还原 v2", np.abs(C.v2(C.q_from_R(C.R_of(q))) - C.v2(q)).max() < 1e-12)

# ===========================================================================
log("\n=== V2: 四类约束对 (v2,t) 仿射 ===")
err_g = 0.0
for _ in range(6):
    qq = C.rand_unit(1, rng)[0]
    tt = rng.normal(size=3)
    p = rng.normal(size=3)
    n = np.array([0.0, 0.0, 1.0])
    d = -0.05
    zmin, tau, al = 0.3, 0.7, 0.5
    X = C.R_of(qq) @ p + tt
    for (a, b), gt in [(C.c_plane(p, n, d), -(n @ X + d)),
                       (C.c_depth(p, zmin), zmin - X[2]),
                       (C.c_fov(p, tau, +1), X[0] - tau * X[2]),
                       (C.c_fov(p, tau, -1), -X[0] - tau * X[2]),
                       (C.c_tilt(al), qq[1] ** 2 + qq[2] ** 2 - np.sin(al) ** 2)]:
        err_g = max(err_g, abs(a @ C.v2(qq) + b @ tt - gt))
check("V2 depth/non-pen/FoV×2/tilt 仿射于 (v2,t)", err_g < 1e-11, "max err %.2e" % err_g)
res["numbers"]["v2_affine_err"] = err_g

# ===========================================================================
log("\n=== V3: 穿透度量 ===")
inst16 = C.make_instance(0, 16, noise=0.01, bias_z=0.08)
inst0 = C.make_instance(0, 16, noise=0.0, bias_z=0.0, bias_xy=0.0)
pen_gt = C.penetration_mm(inst16["R_gt"], inst16["t_gt"], inst16["V_dense"])
check("V3a GT 穿透 = 0（GT 恰好接触桌面）", pen_gt < 1e-9, "pen=%.3e mm" % pen_gt)
# 平移闭式：整体下移 7.3mm ⇒ 穿透 = 7.3mm（底面法向 z）
pen_shift = C.penetration_mm(inst16["R_gt"], inst16["t_gt"] - np.array([0, 0, 0.0073]), inst16["V_dense"])
check("V3b 下移 7.3mm ⇒ pen = 7.3mm", abs(pen_shift - 7.3) < 1e-9, "pen=%.6f mm" % pen_shift)
# 密集网格 vs 4 角解析（矩形底面的 z 最小值必在角点）
Vd = inst16["V_dense"]
Vc = np.array([[sx * C.BOX_HX, sy * C.BOX_HY, -C.BOX_HZ] for sx in (-1, 1) for sy in (-1, 1)])
for tz in [0.0, -0.013, 0.02]:
    R = inst16["R_gt"]; tt = inst16["t_gt"] + np.array([0, 0, tz])
    zd = (R @ Vd.T).T[:, 2].min() + tt[2]
    zc = (R @ Vc.T).T[:, 2].min() + tt[2]
    check("V3c 密集网格 min_z = 角点 min_z (tz=%.3f)" % tz, abs(zd - zc) < 1e-12, "%.2e" % abs(zd - zc))
# 上移 ⇒ 0
check("V3d 上移 2cm ⇒ pen=0", C.penetration_mm(inst16["R_gt"], inst16["t_gt"] + np.array([0, 0, 0.02]), Vd) < 1e-9)
res["numbers"].update(pen_gt=pen_gt, pen_shift=pen_shift)

# ===========================================================================
log("\n=== V4: 增广拉格朗日展开与常数项 ===")
M0, D0, G0 = inst16["M0"], inst16["D0"], inst16["G0"]
A, B = inst16["A"], inst16["B"]
p = A.shape[0]
rho = 3.0
mu = rng.uniform(0.0, 1.0, size=p)
q_t, t_t = C.rand_unit(1, rng)[0], rng.normal(size=3)
g_t = A @ C.v2(q_t) + B @ t_t
act = np.where(mu + rho * g_t > 0)[0]
L_direct = C.full_L(q_t, t_t, M0, D0, G0, A, B, mu, rho)
MA, DA, GA = C._alm_quadratic(M0, D0, G0, A, B, mu, rho, act)
const_ok = -np.sum(mu[np.setdiff1d(np.arange(p), act)] ** 2) / (2 * rho)
x = C.v2(q_t)
L_exp = x @ MA @ x + 2 * t_t @ (DA @ x) + t_t @ GA @ t_t + const_ok
check("V4a L_rho|_A = v2^T M_A v2 + 2t^T D_A v2 + t^T G_A t + kappa_A",
      abs(L_direct - L_exp) < 1e-10, "err %.2e |A|=%d" % (abs(L_direct - L_exp), len(act)))
# 故意写错的常数项（全部约束都减 mu^2/(2rho)）应该差 sum_{A} mu^2/(2rho)
const_wrong = -np.sum(mu ** 2) / (2 * rho)
L_wrong = L_exp - const_ok + const_wrong
err_wrong = abs(L_direct - L_wrong)
check("V4b 错误常数项（全部约束）确实对不上 ⇒ 该项是实现级 bug 源",
      err_wrong > 1e-6, "wrong-version err = %.3e（应等于 Σ_{j∈A} μ_j²/2ρ = %.3e）"
      % (err_wrong, np.sum(mu[act] ** 2) / (2 * rho)))
check("V4c 错误常数项误差 = Σ_{A} μ²/2ρ（逐项对账）",
      abs(err_wrong - np.sum(mu[act] ** 2) / (2 * rho)) < 1e-10)
res["numbers"].update(v4_err=float(abs(L_direct - L_exp)), v4_wrong_err=float(err_wrong),
                      v4_kappa=float(const_ok))

# ===========================================================================
log("\n=== V5: Schur 补 / G_A≻0 / t* 闭式 ===")
check("V5a G_A ≻ 0", np.linalg.eigvalsh(GA)[0] > 0, "lambda_min=%.3e" % np.linalg.eigvalsh(GA)[0])
Mred = C._rho_of(MA, DA, GA)


def Lmin_t(qq):
    xx = C.v2(qq)
    ts = -np.linalg.solve(GA, DA @ xx)
    return xx @ MA @ xx + 2 * ts @ (DA @ xx) + ts @ GA @ ts + const_ok


err_schur = max(abs((C.v2(qq) @ Mred @ C.v2(qq) + const_ok) - Lmin_t(qq)) for qq in C.rand_unit(20, rng))
check("V5b min_t L_rho = v2^T (M_A - D_A^T G_A^{-1} D_A) v2 + kappa", err_schur < 1e-9, "err %.2e" % err_schur)
res["numbers"]["v5_schur_err"] = err_schur

# ===========================================================================
log("\n=== V6: 内层多起点 vs 200 起点全局参照 ===")
insts = [C.make_instance(s, 16, noise=0.01, bias_z=0.08) for s in range(3)]
for k, ins in enumerate(insts):
    Mred0 = C.obj_min_t(ins["M0"], ins["D0"], ins["G0"])
    f8, q8 = C.solve_inner(Mred0, n_starts=8, rng=np.random.default_rng(10 + k))
    f200, q200 = C.solve_inner(Mred0, n_starts=200, rng=np.random.default_rng(20 + k))
    gap = abs(f8 - f200)
    check("V6 内层 8 起点 vs 200 起点实例 %d（Δf<=1e-9）" % k, gap <= 1e-9,
          "f8=%.10f f200=%.10f Δ=%.2e" % (f8, f200, gap))
res["numbers"]["v6_inner_gap"] = float(gap)

# ===========================================================================
log("\n=== V7: 无噪无偏 ⇒ 精确恢复 GT；桌面放低 ⇒ ALM 退化为无约束 ==")
J_gt = C.obj_value(inst0["q_gt"], inst0["t_gt"], inst0["M0"], inst0["D0"], inst0["G0"])
Ju, qu, tu = C.solve_qp_global(inst0["M0"], inst0["D0"], inst0["G0"], n_starts=80, rng=np.random.default_rng(3))
check("V7a 无噪无偏 GT 处 J=0", abs(J_gt) < 1e-12, "J_gt=%.2e" % J_gt)
check("V7b 无约束解恢复 GT（J<=1e-12, rot<=1e-3 deg, trans<=1e-3 mm）",
      Ju <= 1e-12 and C.rot_deg(qu, inst0["q_gt"]) <= 1e-3 and C.trans_mm(tu, inst0["t_gt"]) <= 1e-3,
      "J=%.2e rot=%.2e deg trans=%.2e mm" % (Ju, C.rot_deg(qu, inst0["q_gt"]), C.trans_mm(tu, inst0["t_gt"])))
# 桌面放低 30cm：所有非穿透约束不活跃
inst_low = C.make_instance(0, 16, noise=0.01, bias_z=0.08)
Ju_low, qu_low, tu_low = C.solve_qp_global(inst_low["M0"], inst_low["D0"], inst_low["G0"],
                                           n_starts=80, rng=np.random.default_rng(6))
A_low = inst_low["A"].copy()
for j, typ in enumerate(inst_low["actypes"]):
    if typ == "pen":
        A_low[j] = A_low[j] - 0.30 * C.E_TR        # H_TAB -> H_TAB-0.30（桌面下移）
out_low = C.alm_qpep(inst_low["M0"], inst_low["D0"], inst_low["G0"], A_low, inst_low["B"], qu_low, tu_low,
                     rho0=10.0, rng=np.random.default_rng(4))
check("V7c 桌面放低 30cm ⇒ ALM = 无约束解（ΔJ<=1e-9, 约束不活跃）",
      abs(out_low["J"] - Ju_low) <= 1e-9 and out_low["viol"] <= 1e-8 and np.all(out_low["mu"] <= 1e-12),
      "ΔJ=%.2e viol=%.2e ||mu||=%.2e" % (abs(out_low["J"] - Ju_low), out_low["viol"], np.abs(out_low["mu"]).max()))
res["numbers"].update(v7_Jgt=float(J_gt), v7_Ju=float(Ju))

# ===========================================================================
log("\n=== V8: 单约束平移问题闭式（ALM 解投影） ===")
# min ||t||² s.t. t_z >= h  ⇒  t = (0,0,h)（无约束解 t_u=(0,0,-0.1)）
M1 = np.eye(10); D1 = np.zeros((3, 10)); G1 = np.eye(3)
A1 = np.zeros((1, 10)); B1 = np.array([[0.0, 0.0, -1.0]])
h = 0.0
t_u = np.array([0.0, 0.0, -0.1])
q_any = np.array([1.0, 0, 0, 0])
out8 = C.alm_qpep(M1, D1, G1, A1, B1, q_any, t_u, rho0=1.0, n_starts=3, rng=np.random.default_rng(5))
check("V8 ALM 命中闭式投影 t_z=h", abs(out8["t"][2] - h) < 1e-8 and out8["viol"] <= 1e-9,
      "t_z=%.10f viol=%.2e" % (out8["t"][2], out8["viol"]))

# ===========================================================================
log("\n=== V9: 回归 —— 复现 verify/sanity_constraints_alm.py 的 p=12 实例 ===")
N = 12
Gen = np.random.default_rng(11)
P_obj = Gen.normal(size=(N, 3)) * 0.4
R_gt = np.linalg.qr(Gen.normal(size=(3, 3)))[0]
if np.linalg.det(R_gt) < 0:
    R_gt[:, 0] *= -1
t_gt = np.array([0.05, -0.03, 0.9])
Bw = P_obj @ R_gt.T + t_gt
BIAS = np.array([0.0, 0.0, -0.10])
NOISE = 0.015
Bw_obs = Bw + BIAS + NOISE * Gen.normal(size=Bw.shape)
U = Bw_obs / np.linalg.norm(Bw_obs, axis=1, keepdims=True)
H_TAB_V = t_gt[2] - 0.12
M0v, D0v, G0v = C.build_objective(P_obj, U)
A_l, B_l = [], []
for xx in (-0.15, 0.15):
    for yy in (-0.10, 0.10):
        for zz in (0.0, -0.12):
            a, b = C.c_plane(np.array([xx, yy, zz]), np.array([0.0, 0.0, 1.0]), -H_TAB_V)
            A_l.append(a); B_l.append(b)
for pp in P_obj[:4]:
    a, b = C.c_depth(pp, 0.05)
    A_l.append(a); B_l.append(b)
Av, Bv = np.array(A_l), np.array(B_l)
q_gt_v = C.q_from_R(R_gt)
J_gt_v = C.obj_value(q_gt_v, t_gt, M0v, D0v, G0v)
Ju_v, qu_v, tu_v = C.solve_qp_global(M0v, D0v, G0v, n_starts=80, rng=np.random.default_rng(1))
out_v = C.alm_qpep(M0v, D0v, G0v, Av, Bv, qu_v, tu_v, rho0=1.0, n_starts=8,
                   rng=np.random.default_rng(2))
sdp_v = shor_bound(M0v, D0v, G0v, Av, Bv, enforce_ineq=True)
sdp_v_un = shor_bound(M0v, D0v, G0v, Av, Bv, enforce_ineq=False)
log("  GT J=%.10f (verify: 0.023608)  g_gt max=%.4e (verify: 9.54e-2)" % (J_gt_v, (Av @ C.v2(q_gt_v) + Bv @ t_gt).max()))
log("  无约束 J=%.10f (verify: 0.0030334724) ; ALM J=%.10f (verify: 0.06800693548)" % (Ju_v, out_v["J"]))
log("  Shor(ineq)=%.10f (verify: 0.0680069355) ratio=%.2e ; Shor(uncons)=%.10f (verify: 0.0030334692)"
    % (sdp_v["value"], sdp_v["ratio"], sdp_v_un["value"]))
check("V9a 无约束值复现 verify（|Δ|<=1e-9）", abs(Ju_v - 0.0030334724) < 1e-9, "J_u=%.10f" % Ju_v)
check("V9b ALM 约束最优复现 verify（|Δ|<=1e-8）", abs(out_v["J"] - 0.06800693548) < 1e-8, "J*=%.10f" % out_v["J"])
check("V9c 14×14 Shor 复现 verify（|Δ|<=1e-8 且 tight）",
      abs(sdp_v["value"] - 0.0680069355) < 1e-8 and sdp_v["ratio"] <= 1e-6,
      "Shor=%.10f ratio=%.2e" % (sdp_v["value"], sdp_v["ratio"]))
check("V9d 不加不等式的 Shor 复现 verify 无约束值", abs(sdp_v_un["value"] - 0.0030334692) < 1e-8,
      "Shor_un=%.10f" % sdp_v_un["value"])
res["numbers"].update(v9_Jgt=float(J_gt_v), v9_Ju=float(Ju_v), v9_Jalm=float(out_v["J"]),
                      v9_shor=float(sdp_v["value"]), v9_shor_un=float(sdp_v_un["value"]))

# ===========================================================================
log("\n=== V10: 14×14 Shor 在 8 个随机实例上的 sound / tight ===")
gap_max, ratio_max, n_tight, n_sound = 0.0, 0.0, 0, 0
for k in range(8):
    ins = C.make_instance(k, 16, noise=0.01, bias_z=0.08)
    Jk, qk, tk = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=48, rng=np.random.default_rng(k + 1))
    ok = C.alm_qpep(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qk, tk,
                    rho0=10.0, n_starts=5, rng=np.random.default_rng(k + 2))
    sd = shor_bound(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], enforce_ineq=True)
    gap = ok["J"] - sd["value"]
    gap_max = max(gap_max, abs(gap)); ratio_max = max(ratio_max, sd["ratio"])
    n_tight += int(sd["ratio"] <= 1e-6)
    n_sound += int(gap >= -1e-7)
    log("  seed=%d J_ALM=%.10f Shor=%.10f gap=%.2e ratio=%.2e [%.2fs]"
        % (k, ok["J"], sd["value"], gap, sd["ratio"], sd["time_s"]))
check("V10a sound：Shor <= J_ALM + 1e-7（8/8）", n_sound == 8, "%d/8, max|gap|=%.2e" % (n_sound, gap_max))
check("V10b tight：λ2/λ1 <= 1e-6（8/8）", n_tight == 8, "%d/8, max ratio=%.2e" % (n_tight, ratio_max))

# ===========================================================================
log("\n=== V11: d(μ) 弱对偶下界（含 δ<0 数值现象） ===")


def dual_value(M0, D0, G0, A, B, mu, n_starts=32, seed=0):
    MD, DD = M0.copy(), D0.copy()
    for j in range(A.shape[0]):
        if mu[j] == 0.0:
            continue
        MD += mu[j] / 2.0 * (np.outer(C.E_TR, A[j]) + np.outer(A[j], C.E_TR))
        DD += mu[j] / 2.0 * np.outer(B[j], C.E_TR)
    f, q, t = C.solve_qp_global(MD, DD, G0, n_starts=n_starts, rng=np.random.default_rng(seed))
    return f


viol_d = []
for k in range(4):
    ins = C.make_instance(k, 16, noise=0.01, bias_z=0.08)
    Jk, qk, tk = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=48, rng=np.random.default_rng(k + 1))
    ok = C.alm_qpep(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qk, tk,
                    rho0=10.0, n_starts=5, rng=np.random.default_rng(k + 2))
    d = dual_value(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], ok["mu"], seed=k + 7)
    viol_d.append(ok["J"] - d)
    log("  seed=%d J*_ALM=%.10f d(mu)=%.10f delta=%.3e %s"
        % (k, ok["J"], d, ok["J"] - d, "(! δ<0 数值误差)" if ok["J"] - d < 0 else ""))
check("V11a d(μ) <= J*（4/4，容差 1e-8）", all(x >= -1e-8 for x in viol_d), "min δ=%.2e" % min(viol_d))
res["numbers"]["v11_delta"] = [float(x) for x in viol_d]

# ===========================================================================
log("\n=== V12: ALM 解的 KKT 残差 / μ>=0 / G_A≻0 / NaN / LICQ 秩 ===")
ins = C.make_instance(0, 16, noise=0.01, bias_z=0.08)
Jk, qk, tk = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=48, rng=np.random.default_rng(1))
ok = C.alm_qpep(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qk, tk,
                rho0=10.0, n_starts=6, rng=np.random.default_rng(2))
q, t, mu = ok["q"], ok["t"], ok["mu"]
A, B, M0, D0, G0 = ins["A"], ins["B"], ins["M0"], ins["D0"], ins["G0"]
check("V12a μ >= 0", np.all(mu >= 0))
# 最后一次内层的 G_A ≻ 0
g = A @ C.v2(q) + B @ t
act = np.where(mu + ok["rho"] * g > 0)[0]
MA, DA, GA = C._alm_quadratic(M0, D0, G0, A, B, mu, ok["rho"], act)
check("V12b G_A ≻ 0（末端活跃集）", np.linalg.eigvalsh(GA)[0] > 0, "lambda_min=%.3e" % np.linalg.eigvalsh(GA)[0])
# 200 组随机 (G0≻0, ρ, 活跃集) 的 G_A 正定性（G_A=G0+ρ/2 Σ b b^T 是定理 2.1 的结构）
rng_ga = np.random.default_rng(123)
bad_ga, ga_min = 0, np.inf
for _ in range(200):
    L = rng_ga.normal(size=(3, 3))
    G0r = L @ L.T + 0.1 * np.eye(3)
    Br = rng_ga.normal(size=(rng_ga.integers(1, 40), 3))
    rho_r = 10 ** rng_ga.uniform(-1, 4)
    act_r = rng_ga.choice(Br.shape[0], size=rng_ga.integers(0, Br.shape[0]), replace=False)
    GAr = G0r + rho_r / 2.0 * sum(np.outer(Br[j], Br[j]) for j in act_r)
    emin = float(np.linalg.eigvalsh(GAr)[0])
    ga_min = min(ga_min, emin)
    bad_ga += int(emin <= 0)
check("V12b2 200 组随机 (G0≻0,ρ,A) 的 G_A 均 ≻ 0", bad_ga == 0, "%d bad, min λ=%.3e" % (bad_ga, ga_min))
check("V12c 无 NaN/Inf", all(np.all(np.isfinite(v)) for v in (q, t, mu, ok["J"])), "J=%.8f" % ok["J"])
# 互补松弛
comp = np.abs(mu * g).max()
check("V12d 互补松弛 max|μ_j g_j| <= 1e-8", comp <= 1e-8, "%.2e" % comp)
# KKT 残差：L0 = J + Σ μ_j g_j 在 6 维切空间 (δω, δt) 的方向导数
L0 = lambda qq, tt: C.obj_value(qq, tt, M0, D0, G0) + float(mu @ (A @ C.v2(qq) + B @ tt))
eps = 1e-6
grad6 = np.zeros(6)
for k in range(3):
    w = np.zeros(3); w[k] = eps
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    Rk = C.R_of(q) @ (np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K))
    qp_ = C.q_from_R(Rk); qm_ = C.q_from_R(C.R_of(q) @ (np.eye(3) - np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K)))
    grad6[k] = (L0(qp_, t) - L0(qm_, t)) / (2 * eps)
for k in range(3):
    dt = np.zeros(3); dt[k] = eps
    grad6[3 + k] = (L0(q, t + dt) - L0(q, t - dt)) / (2 * eps)
check("V12e KKT 驻点残差 ||∇L0||_6 <= 1e-4", np.linalg.norm(grad6) <= 1e-4,
      "res=%.2e (%s)" % (np.linalg.norm(grad6), np.array2string(grad6, precision=2)))
rank, _ = C.active_jacobian_rank(q, t, A, B, act)
log("  活跃集 |A|=%d ; 6 维切空间 rank(J_A)=%d ⇒ %s"
    % (len(act), rank, "LICQ 失效" if rank < len(act) else "LICQ 成立"))
check("V12f 本实例活跃集秩亏（LICQ 失效，与四脚桌同结构）", rank < len(act),
      "|A|=%d rank=%d" % (len(act), rank))
res["numbers"].update(v12_comp=float(comp), v12_kkt=float(np.linalg.norm(grad6)),
                      v12_rank=int(rank), v12_nact=int(len(act)))

# ===========================================================================
log("\n=== V13: 投影可行性 / 软惩罚 Pareto 方向 ===")
ins = C.make_instance(0, 16, noise=0.01, bias_z=0.08)
Jk, qk, tk = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=48, rng=np.random.default_rng(1))
pr = C.project_feasible(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qk, tk,
                        n_starts=4, rng=np.random.default_rng(3))
check("V13a 投影返回可行点（viol<=1e-6）", pr["viol"] <= 1e-6, "viol=%.2e method=%s" % (pr["viol"], pr["method"]))
pen_list = []
for lam in [1, 10, 100, 1000]:
    sp = C.soft_penalty(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], lam, qk, tk,
                        n_starts=10, rng=np.random.default_rng(4))
    pen_list.append(C.penetration_mm(C.R_of(sp["q"]), sp["t"], ins["V_dense"]))
log("  软惩罚 λ=[1,10,100,1000] 穿透(mm) = %s" % np.array2string(np.array(pen_list), precision=3))
check("V13b 穿透随 λ 总体下降（末项 <= 首项 + 0.5mm）", pen_list[-1] <= pen_list[0] + 0.5,
      "pen(λ=1)=%.2f pen(λ=1e3)=%.2f" % (pen_list[0], pen_list[-1]))
res["numbers"]["v13_pen_lam"] = [float(x) for x in pen_list]

# ===========================================================================
log("\n=== V14: 活跃集 Jacobian 秩（四脚桌 LICQ 失效的真实实例对应） ===")
rows = []
for p_ in [8, 16, 32, 64]:
    for seed in range(4):
        ins = C.make_instance(seed, p_, noise=0.01, bias_z=0.08)
        Jk, qk, tk = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=40,
                                       rng=np.random.default_rng(seed + 1))
        ok = C.alm_qpep(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qk, tk,
                        rho0=10.0, n_starts=5, rng=np.random.default_rng(seed + 2))
        act = np.where(ok["mu"] > 1e-9)[0]
        r, _ = C.active_jacobian_rank(ok["q"], ok["t"], ins["A"], ins["B"], act)
        rows.append((p_, seed, len(act), r))
        log("  p=%2d seed=%d |A|=%2d rank=%d %s" % (p_, seed, len(act), r,
                                                    "LICQ fail" if r < len(act) else "LICQ ok"))
check("V14 至少一半实例 LICQ 失效（rank<|A|）", sum(1 for x in rows if x[3] < x[2]) >= len(rows) // 2,
      "%d/%d" % (sum(1 for x in rows if x[3] < x[2]), len(rows)))
res["numbers"]["v14_ranks"] = [[int(v) for v in x] for x in rows]

# ===========================================================================
log("\n=== 汇总 ===")
n_pass = sum(res["checks"].values())
n_all = len(res["checks"])
log("  %d/%d 项检查通过" % (n_pass, n_all))
if n_pass < n_all:
    log("  FAIL 项：" + ", ".join(k for k, v in res["checks"].items() if not v))
res["summary"] = {"n_pass": int(n_pass), "n_total": int(n_all)}
with open(os.path.join(RES, "validation.json"), "w") as f:
    json.dump(res, f, indent=2, default=float)
log("  已写出 results/validation.json")
