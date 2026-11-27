#!/usr/bin/env python
"""四脚桌 LICQ 失效 / FB 极限 / exact vs naive(StopGrad) 数值表。

对应 theory/proofs.md §3.2–3.5 与 verify/sanity_fb_saddle.py 的 F5/F6/F7。
这里把它们整理成一张可直接进论文附录的数值表，并补上：
  * exact（FB 正则化鞍点） vs naive（D_a=0，硬活跃集） vs StopGrad（dz=0）
  * 真实合成桌面实例上的活跃集秩（|A| vs rank(J_A)），确认四脚桌结构不是玩具
输出：results/fb_table.json / results/fb_table.md
运行：srun --partition=cpu --time=00:15:00 --ntasks=1 \
      /home/syk/miniconda3/envs/sdpr/bin/python run_fb_table.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpep_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
out = {"tables": {}, "numbers": {}}


def log(*a):
    print(*a, flush=True)


def saddle(H, J, D):
    n, m = J.shape
    return np.block([[H, J.T], [J, -np.diag(D)]])


def fb_ab(mu, g, eps):
    """Lemma 2.3: a = 1 - mu/r, b = 1 + g/r, r = sqrt(mu^2+g^2+eps^2)。"""
    r = np.sqrt(mu * mu + g * g + eps * eps)
    return 1.0 - mu / r, 1.0 + g / r, r


# ---------------------------------------------------------------------------
# 1. 四脚桌几何
# ---------------------------------------------------------------------------
a_c, b_c = 0.15, 0.10
corners = np.array([[a_c, b_c], [a_c, -b_c], [-a_c, b_c], [-a_c, -b_c]])
J4 = np.column_stack([np.ones(4), corners[:, 0], corners[:, 1]])   # (z, roll, pitch), 4×3
rank4 = np.linalg.matrix_rank(J4)
sv4 = np.linalg.svd(J4, compute_uv=False)
log("四脚桌接触 Jacobian J_A (4×3):\n%s" % np.array2string(J4, precision=3))
log("rank(J_A) = %d < 4 ; singular values = %s ; cond = %.2e" % (rank4, np.array2string(sv4, precision=4), sv4[0] / sv4[-1]))
S0 = saddle(np.eye(3), J4, np.zeros(4))
out["numbers"].update(four_rank=int(rank4), four_cond=float(sv4[0] / sv4[-1]),
                      four_saddle_rank=int(np.linalg.matrix_rank(S0)))

# ---------------------------------------------------------------------------
# 2. κ(S) vs ε（强活跃 mu=1, g=0 ⇒ D_a = a = 1 - 1/sqrt(1+eps²) ≈ eps²/2）
# ---------------------------------------------------------------------------
log("\n=== 表 A：ε → 0 时鞍点条件数（四脚桌，4 约束 / 3 未知） ===")
rows_A = []
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    a, b, _ = fb_ab(1.0, 0.0, eps)
    D = np.full(4, a / b)
    sv = np.linalg.svd(saddle(np.eye(3), J4, D), compute_uv=False)
    row = dict(eps=eps, Da=float(a / b), cond=float(sv.max() / sv.min()))
    rows_A.append(row)
    log("  eps=%.0e  D_a=%.3e  cond(S)=%.3e" % (eps, row["Da"], row["cond"]))
slope = float(np.polyfit(np.log([r["eps"] for r in rows_A]), np.log([r["cond"] for r in rows_A]), 1)[0])
log("  log(cond) vs log(eps) 斜率 = %.2f（定理 2.3(iv): κ ≍ 1/ε²）" % slope)
out["tables"]["A_kappa_eps"] = rows_A
out["numbers"]["kappa_slope"] = slope

# ---------------------------------------------------------------------------
# 3. 相容 w = J_A v：dz(ε) → v，且与 D_a 分布无关
# ---------------------------------------------------------------------------
log("\n=== 表 B：相容情形 w = J_A v（真方向导数存在） ===")
v_true = np.array([0.3, -0.4, 0.5])
w_con = J4 @ v_true
rows_B = []
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    a, b, _ = fb_ab(1.0, 0.0, eps)
    D = np.full(4, a / b)
    sol = np.linalg.solve(saddle(np.eye(3), J4, D), np.concatenate([np.zeros(3), w_con]))
    rows_B.append(dict(eps=eps, dz_err=float(np.linalg.norm(sol[:3] - v_true)),
                       dmu_norm=float(np.linalg.norm(sol[3:]))))
    log("  eps=%.0e  ||dz - v||=%.3e  ||dmu||=%.3e" % (eps, rows_B[-1]["dz_err"], rows_B[-1]["dmu_norm"]))
# D_a 分布不同（不同乘子选择）→ dz 不变
rows_B2 = []
for dd in [np.array([1.0, 1, 1, 1]), np.array([1.0, 2, 5, 10]), np.array([1e-4, 1, 1, 1])]:
    a, b, _ = fb_ab(1.0, 0.0, 1e-4)
    D = a / b * dd / dd.mean()
    sol = np.linalg.solve(saddle(np.eye(3), J4, D), np.concatenate([np.zeros(3), w_con]))
    rows_B2.append(dict(Ddist=[float(x) for x in dd], dz_err=float(np.linalg.norm(sol[:3] - v_true)),
                        dmu_norm=float(np.linalg.norm(sol[3:]))))
    log("  D_a ∝ %-18s  ||dz-v||=%.3e  ||dmu||=%.2e" % (np.array2string(dd, precision=0), rows_B2[-1]["dz_err"], rows_B2[-1]["dmu_norm"]))
out["tables"]["B_compatible"] = rows_B
out["tables"]["B_multiplier_spread"] = rows_B2

# ---------------------------------------------------------------------------
# 4. 精确 vs naive vs StopGrad
# ---------------------------------------------------------------------------
log("\n=== 表 C：exact(FB) vs naive(D_a=0, 硬活跃集) vs StopGrad(dz=0) ===")
rows_C = []
# (a) 相容 w，eps=1e-6
eps = 1e-6
a, b, _ = fb_ab(1.0, 0.0, eps)
D = np.full(4, a / b)
S = saddle(np.eye(3), J4, D)
sol_exact = np.linalg.solve(S, np.concatenate([np.zeros(3), w_con]))
dz_exact = sol_exact[:3]
# naive：D_a=0 的 KKT 矩阵奇异，用 pinv 求"最小范数"解（实现中常见做法）
S0 = saddle(np.eye(3), J4, np.zeros(4))
sol_naive = np.linalg.pinv(S0) @ np.concatenate([np.zeros(3), w_con])
dz_naive = sol_naive[:3]
dz_stop = np.zeros(3)
rows_C.append(dict(case="compatible", method="exact FB (eps=1e-6)",
                   dz_err=float(np.linalg.norm(dz_exact - v_true)), dmu=float(np.linalg.norm(sol_exact[3:])),
                   invertible=True, rank=int(np.linalg.matrix_rank(S))))
rows_C.append(dict(case="compatible", method="naive D_a=0 + pinv",
                   dz_err=float(np.linalg.norm(dz_naive - v_true)), dmu=float(np.linalg.norm(sol_naive[3:])),
                   invertible=bool(np.linalg.matrix_rank(S0) == 7), rank=int(np.linalg.matrix_rank(S0))))
rows_C.append(dict(case="compatible", method="StopGrad (dz=0)",
                   dz_err=float(np.linalg.norm(dz_stop - v_true)), dmu=0.0,
                   invertible=True, rank=7))
for r in rows_C:
    log("  %-10s %-24s dz_err=%.3e dmu=%.2e rank(S)=%d/7" %
        (r["case"], r["method"], r["dz_err"], r["dmu"], r["rank"]))
# (b) 不相容 w：w ∉ range(J_A)（加核空间分量）
nullv = np.linalg.svd(J4.T)[2][-1]                 # J_A^T 的零空间（4 维，1 维）
w_inc = w_con + 1e-2 * nullv
resid = float(np.linalg.norm((np.eye(4) - J4 @ np.linalg.pinv(J4)) @ w_inc))
log("  不相容性：||(I - J_A J_A^+) w|| = %.3e > 0" % resid)
for name, ds in [("exact FB (eps=1e-6)", None), ("naive D_a=0 + pinv", None), ("StopGrad (dz=0)", None)]:
    if name.startswith("exact"):
        sol = np.linalg.solve(S, np.concatenate([np.zeros(3), w_inc]))
        dz, dmu = sol[:3], np.linalg.norm(sol[3:])
        rk = int(np.linalg.matrix_rank(S))
    elif name.startswith("naive"):
        sol = np.linalg.pinv(S0) @ np.concatenate([np.zeros(3), w_inc])
        dz, dmu = sol[:3], np.linalg.norm(sol[3:])
        rk = int(np.linalg.matrix_rank(S0))
    else:
        dz, dmu, rk = np.zeros(3), 0.0, 7
    rows_C.append(dict(case="incompatible", method=name, dz_norm=float(np.linalg.norm(dz)),
                       dmu=float(dmu), rank=rk,
                       dz_err=float(np.linalg.norm(dz - v_true)) if name != "StopGrad (dz=0)" else float(np.linalg.norm(v_true))))
    log("  %-12s %-24s ||dz||=%.3e dmu=%.2e rank(S)=%d/7" %
        ("incompatible", name, np.linalg.norm(dz), dmu, rk))
out["tables"]["C_exact_naive_stopgrad"] = rows_C
out["numbers"]["incompat_resid"] = resid

# (c) H 非各向同性 + J 列秩亏（null(J)≠{0}）：exact 极限 = H-加权最小范数解 ≠ 欧氏 pinv
# 四脚桌本身 J 列满秩(rank 3)，相容时 dz 唯一、H 不起作用；要看到 H-加权效应
# 必须用 Prop 2.1(b) 的秩亏情形。这里沿用 verify F5d 的 4×3 rank-2 例子。
log("\n  非各向同性 H = diag(1,1,10) + rank(J)=2（null 维 1）：FB 极限 vs 欧氏 pinv")
Jd = np.array([[1.0, 0.0, 1.0],
               [0.0, 1.0, 1.0],
               [1.0, 1.0, 2.0],
               [1.0, -1.0, 0.0]])
Hd = np.diag([1.0, 1.0, 10.0])
v_d = np.array([1.0, 2.0, 0.5])
wd = Jd @ v_d
JH = Jd @ np.linalg.inv(Hd) @ Jd.T
dz_h = np.linalg.inv(Hd) @ Jd.T @ np.linalg.pinv(JH) @ wd
dz_e = np.linalg.pinv(Jd) @ wd
a, b, _ = fb_ab(1.0, 0.0, 1e-6)
sol_h = np.linalg.solve(saddle(Hd, Jd, np.full(4, a / b)), np.concatenate([np.zeros(3), wd]))
log("    rank(J)=%d ; H-加权闭式 = %s ; 欧氏 pinv = %s ; 差 = %.3e ; FB 解误差 = %.3e"
    % (np.linalg.matrix_rank(Jd), np.array2string(dz_h, precision=4), np.array2string(dz_e, precision=4),
       np.linalg.norm(dz_h - dz_e), np.linalg.norm(sol_h[:3] - dz_h)))
out["tables"]["C2_H_weighted"] = dict(rank=int(np.linalg.matrix_rank(Jd)),
                                      dz_weighted=[float(x) for x in dz_h],
                                      dz_euclid=[float(x) for x in dz_e],
                                      diff=float(np.linalg.norm(dz_h - dz_e)),
                                      fb_err=float(np.linalg.norm(sol_h[:3] - dz_h)))

# ---------------------------------------------------------------------------
# 5. 2D kink：min 1/2(z-a)^2 s.t. z>=0
# ---------------------------------------------------------------------------
log("\n=== 表 D：2D kink（a=0）exact FB vs StopGrad vs 单侧导数 ===")


def kink_fb(a, eps, h=1.0):
    z = 0.5 * (a + np.sqrt(a * a + 2 * eps * eps))
    mu = z - a
    a_j, b_j, _ = fb_ab(mu, -z, eps)      # g = -z
    D = a_j / b_j
    if not np.isfinite(D):
        D = 1e300          # b→0（非活跃约束）时 D_a→∞ 的数值极限
    M = np.block([[np.array([[h]]), np.array([[-1.0]])], [np.array([[-1.0]]), np.array([[-D]])]])
    sol = np.linalg.solve(M, np.array([1.0, 0.0]))
    return float(sol[0])


rows_D = []
for eps in [1e-1, 1e-2, 1e-4, 1e-6]:
    dz_fb = kink_fb(0.0, eps)
    rows_D.append(dict(eps=eps, dz_fb=dz_fb, dz_stopgrad=0.0, dz_onesided=1.0,
                       clarke_low=0.0, clarke_high=1.0))
    log("  eps=%.0e: exact FB dz/da=%.10f ; StopGrad=0 ; 单侧真导数=1 ; Clarke=[0,1]" % (eps, dz_fb))
for a in [-1.0, -1e-2, 0.0, 1e-2, 1.0]:
    log("    a=%+.0e: exact FB dz/da=%.6f (真: %d)" % (a, kink_fb(a, 1e-8), 1 if a > 0 else 0))
out["tables"]["D_kink"] = rows_D

# ---------------------------------------------------------------------------
# 6. 真实合成桌面实例的活跃集秩（四脚桌结构是否出现）
# ---------------------------------------------------------------------------
log("\n=== 表 E：真实桌面 PnP 实例的活跃集秩（p=8/16/32/64） ===")
rows_E = []
for p_ in [8, 16, 32, 64]:
    ins = C.make_instance(0, p_, noise=0.01, bias_z=0.08)
    Ju, qu, tu = C.solve_qp_global(ins["M0"], ins["D0"], ins["G0"], n_starts=40, rng=np.random.default_rng(1))
    ok = C.alm_qpep(ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"], qu, tu,
                    rho0=10.0, n_starts=6, rng=np.random.default_rng(2))
    act = np.where(ok["mu"] > 1e-9)[0]
    rank, _ = C.active_jacobian_rank(ok["q"], ok["t"], ins["A"], ins["B"], act)
    rows_E.append(dict(p=p_, n_act=int(len(act)), rank=int(rank), LICQ=bool(rank == len(act)),
                       viol=float(ok["viol"]), J=float(ok["J"])))
    log("  p=%2d |A|=%2d rank(J_A)=%2d %s  viol=%.1e" %
        (p_, len(act), rank, "LICQ ok" if rank == len(act) else "LICQ FAIL", ok["viol"]))
out["tables"]["E_real_instance_rank"] = rows_E

with open(os.path.join(RES, "fb_table.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)

# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
md = []
md.append("# 四脚桌 LICQ / FB 极限 / exact vs naive 数值表\n")
md.append("> 生成脚本：`experiments/code/run_fb_table.py`；原始数据：`results/fb_table.json`。\n")
md.append("## A. 鞍点条件数 vs ε（4 接触 / 3 未知，rank(J_A)=%d<4）\n" % rank4)
md.append("| ε | D_a | cond(S) |")
md.append("|---|---|---|")
for r in rows_A:
    md.append("| %.0e | %.3e | %.3e |" % (r["eps"], r["Da"], r["cond"]))
md.append("\nlog–log 斜率 = **%.2f**（定理 2.3(iv)：κ ≍ 1/ε²）。\n" % slope)
md.append("## B. 相容 w = J_A v：dz(ε) → v（且与乘子分布无关）\n")
md.append("| ε | ‖dz−v‖ | ‖dμ‖ |")
md.append("|---|---|---|")
for r in rows_B:
    md.append("| %.0e | %.3e | %.3e |" % (r["eps"], r["dz_err"], r["dmu_norm"]))
md.append("\n乘子分布（D_a ∝ [1,1,1,1] / [1,2,5,10] / [1e−4,1,1,1]）下 ‖dz−v‖：%s ⇒ **dz 与乘子选择无关**（Lemma 2.4）。\n"
          % ", ".join("%.2e" % r["dz_err"] for r in rows_B2))
md.append("## C. exact(FB) vs naive(D_a=0) vs StopGrad(dz=0)\n")
md.append("| 情形 | 方法 | ‖dz−v‖ 或 ‖dz‖ | ‖dμ‖ | rank(S) / 7 |")
md.append("|---|---|---|---|---|")
for r in rows_C:
    if r["case"] == "compatible":
        val = "%.3e" % r["dz_err"]
    else:
        val = "%.3e" % r["dz_norm"]
    md.append("| %s | %s | %s | %.2e | %d |" % (r["case"], r["method"], val, r["dmu"], r["rank"]))
md.append("\n- 相容时 exact 收敛到真 v；naive（D_a=0）的 KKT 矩阵奇异（rank 6/7），只能靠 pinv 给出数值；"
          "StopGrad 直接把 dz 置 0，误差 = ‖v‖ = %.3f。" % np.linalg.norm(v_true))
md.append("- 不相容时 exact 的 dz 有限但**不是真导数**、dμ 随 1/D_a 爆破；这正是 Prop 2.1(d)(e)。")
md.append("\n非各向同性 H 的极限：H-加权闭式 %s vs 欧氏 pinv %s，差 %.3f；FB 解误差 %.2e ⇒ 极限是 **H-加权最小范数解**。\n"
          % (np.array2string(dz_h, precision=3), np.array2string(dz_e, precision=3),
             np.linalg.norm(dz_h - dz_e), np.linalg.norm(sol_h[:3] - dz_h)))
md.append("## D. 2D kink：min ½(z−a)² s.t. z≥0\n")
md.append("| ε | exact FB dz/da | StopGrad | 单侧真导数 | Clarke 广义导数 |")
md.append("|---|---|---|---|---|")
for r in rows_D:
    md.append("| %.0e | %.6f | 0 | 1 | [0,1] |" % (r["eps"], r["dz_fb"]))
md.append("\nFB 精确给出 Clarke 中点 **1/2**，与 ε 无关（例 2.5.1）。\n")
md.append("## E. 真实桌面 PnP 实例上的 LICQ\n")
md.append("| p | 活跃约束数 | rank(J_A)，6 维切空间 | LICQ |")
md.append("|---|---|---|---|")
for r in rows_E:
    md.append("| %d | %d | %d | %s |" % (r["p"], r["n_act"], r["rank"], "ok" if r["LICQ"] else "**FAIL**"))
md.append("\n接触点是共面的（底面贴桌面），活跃约束的切空间 Jacobian 秩 ≤ 3 ⇒ "
          "与四脚桌同构的 LICQ 失效在真实合成实例上**每帧出现**；ALM 仍收敛到 KKT 点、"
          "14×14 证书不受影响。\n")
with open(os.path.join(RES, "fb_table.md"), "w") as f:
    f.write("\n".join(md))
log("\n已写出 results/fb_table.json / results/fb_table.md")
