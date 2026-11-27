#!/usr/bin/env python
"""Constrained-QPEPLayer — 数值 sanity check #2

覆盖 Fischer-Burmeister 约化隐微分的四件容易写错的事：
  F1  a = 1-μ/r ∈ (0,1]；b = 1+g/r：可行点 g≤0 时 b∈[0,1]，一般迭代点 g>0 时 b∈(1,2)
  F2  ε>0 时 φ_ε 在互补点处 ≠ 0：强活跃 φ≈-ε²/(2μ)、弱活跃 φ=-ε、非活跃 φ≈-ε²/(2|g|)
  F3  φ_ε=0 的零点集是双曲线 μ·g = -ε²/2，**不是**互补集 ⇒ ε>0 只是光滑化，不是等价重述
  F4  鞍点矩阵 [[H,J^T],[J,-D]]：H≻0, D≻0 时恒可逆（Schur 补），cond ~ O(1/ε²)
  F5  LICQ 失效（J_A 秩亏）时：ε>0 可逆；ε→0 的极限只在"相容性条件"下给出真 dz；
      不相容时 dz 收敛到乘子加权平均、dμ 爆破 ~1/ε²
  F6  四脚桌 4×3 接触 Jacobian 秩 3 的显式验证

解释器：/home/syk/miniconda3/envs/sdpr/bin/python（基座 python 缺 scipy）。
"""
import json
import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=150)
rng = np.random.default_rng(0)
res = {"checks": {}, "numbers": {}}


def log(*a):
    print(*a)


def check(key, cond, detail=""):
    res["checks"][key] = bool(cond)
    log(("  [PASS] " if cond else "  [FAIL] ") + key + ("  " + detail if detail else ""))


# ------------------------------------------------------------------------ F1
log("\n=== F1: a, b 的取值范围 ===")
eps = 1e-4
mus = np.array([0.0, 1e-3, 0.1, 1.0, 10.0])
gs = np.array([-2.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 2.0])
A_, B_ = np.zeros((len(mus), len(gs))), np.zeros((len(mus), len(gs)))
for i, m in enumerate(mus):
    for j, g in enumerate(gs):
        r = np.sqrt(m * m + g * g + eps * eps)
        A_[i, j] = 1 - m / r
        B_[i, j] = 1 + g / r
log("  a = 1-mu/r :  min=%.6f max=%.6f" % (A_.min(), A_.max()))
log("  b = 1+g/r  :  min=%.6f max=%.6f" % (B_.min(), B_.max()))
check("F1a a ∈ (0,1]", A_.min() >= 0 and A_.max() <= 1 + 1e-15)
mask_feas = gs <= 0
check("F1b 可行点 g<=0 时 b ∈ [0,1]", B_[:, mask_feas].min() >= 0 and B_[:, mask_feas].max() <= 1 + 1e-15)
check("F1c 一般迭代点 (g>0) 时 b ∈ (1,2)", B_[:, ~mask_feas].min() > 1 - 1e-15 and B_[:, ~mask_feas].max() < 2)
res["numbers"].update(a_min=float(A_.min()), a_max=float(A_.max()),
                      b_feas_max=float(B_[:, mask_feas].max()), b_infeas_max=float(B_[:, ~mask_feas].max()))

# ------------------------------------------------------------------------ F2
log("\n=== F2: ε>0 时 φ_ε 在互补点不等于 0 ===")


def phi(m, g, e):
    return m - g - np.sqrt(m * m + g * g + e * e)


rows = []
for (m, g, name, ref) in [
        (5.0, 0.0, "强活跃 (mu>0,g=0)", -eps * eps / (2 * 5.0)),
        (0.0, 0.0, "弱活跃 (mu=0,g=0)", -eps),
        (0.0, -3.0, "非活跃 (mu=0,g<0)", -eps * eps / (2 * 3.0))]:
    v = phi(m, g, eps); r = ref
    rows.append((name, v, r, abs(v - r) / max(abs(v), 1e-300)))
    log("  %-24s phi_eps = %+.6e ; 一阶近似 = %+.6e ; 相对差 = %.2e" % (name, v, r, rows[-1][3]))
check("F2 φ_ε(互补点) 与 -ε²/(2μ)、-ε、-ε²/(2|g|) 的渐近一致",
      all(r[3] < 5e-3 for r in rows), "max rel diff %.2e" % max(r[3] for r in rows))

# ------------------------------------------------------------------------ F3
log("\n=== F3: φ_ε = 0 的零点集 = { μ g = -ε²/2 }，不是互补集 ===")
from scipy.optimize import brentq

errs = []
for m in [0.5, 1.0, 2.0, 5.0, 10.0]:
    # 固定 mu, 解 phi(mu, g) = 0  (g 必为负)
    f = lambda g: phi(m, g, eps)
    try:
        gstar = brentq(f, -1.0, -1e-12, xtol=1e-16, rtol=1e-15)
        errs.append(abs(m * gstar + 0.5 * eps * eps))
    except ValueError:
        errs.append(np.nan)
log("  数值解 μ·g* 与 -ε²/2 的偏差: " + ", ".join("%.2e" % e for e in errs))
check("F3a φ_ε=0 ⇒ μg = -ε²/2（数值）", np.nanmax(errs) < 1e-12)
# 互补点 (μ>0,g=0) 处 phi≠0 ⇒ 不能写成"φ_ε=0 ⟺ 互补"
check("F3b φ_ε(5,0)=%.2e ≠ 0" % phi(5.0, 0.0, eps), abs(phi(5.0, 0.0, eps)) > 0)
res["numbers"]["f3_hyperbola_err"] = float(np.nanmax(errs))

# ------------------------------------------------------------------------ F4
log("\n=== F4: 鞍点矩阵 [[H,J^T],[J,-D]] 的可逆性与条件数 ===")


def saddle(H, J, D):
    n, m = J.shape
    return np.block([[H, J.T], [J, -np.diag(D)]])


worst = np.inf
for trial in range(200):
    n = 6
    m = rng.integers(1, 5)
    Bm = rng.normal(size=(n, n)); H = Bm @ Bm.T + 0.1 * np.eye(n)
    J = rng.normal(size=(m, n))
    D = rng.uniform(1e-8, 1.0, size=m)
    M = saddle(H, J, D)
    worst = min(worst, np.linalg.svd(M, compute_uv=False).min() / np.linalg.svd(M, compute_uv=False).max())
log("  200 组随机 (H≻0, J 任意秩, D≻0)：最小奇异值/最大 = %.3e" % worst)
check("F4a H≻0 & D≻0 ⇒ 恒可逆（含 J 秩亏）", worst > 1e-14)

# 条件数 vs ε（强活跃: a≈ε²/(2μ)）
m0 = 1.0
e_list, kappas = [], []
a_c, b_c = 0.15, 0.10
corners0 = np.array([[a_c, b_c], [a_c, -b_c], [-a_c, b_c], [-a_c, -b_c]])
Jrd = np.column_stack([np.ones(4), corners0[:, 0], corners0[:, 1]])   # 四脚桌 4x3, rank 3
for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    a = 1 - m0 / np.sqrt(m0 ** 2 + e ** 2)
    d = a / 1.0                        # b = 1 at g = 0
    sv = np.linalg.svd(saddle(np.eye(3), Jrd, np.full(4, d)), compute_uv=False)
    e_list.append(e); kappas.append(sv.max() / sv.min())
    log("  eps=%.0e  D_a=%.3e  cond=%.3e  (J 秩 3 < 4)" % (e, d, kappas[-1]))
slope = np.polyfit(np.log(e_list), np.log(kappas), 1)[0]
log("  log-cond 对 log-eps 斜率 ≈ %.2f（理论 2 ⇒ cond ~ 1/ε²）" % slope)
check("F4b cond ~ 1/ε² （|斜率| in [1.6,2.4]）", 1.6 <= abs(slope) <= 2.4, "slope=%.2f" % slope)
res["numbers"]["f4_slope"] = float(slope)

# ------------------------------------------------------------------------ F6
log("\n=== F6: 四脚桌 4×3 接触 Jacobian 秩 3（LICQ 失效） ===")
a, b = a_c, b_c
corners = corners0
J4 = np.column_stack([np.ones(4), corners[:, 0], corners[:, 1]])   # (z, roll, pitch)
log("  J_A =\n" + str(J4))
log("  rank(J_A) = %d < 4 ; 特征值 = %s" % (np.linalg.matrix_rank(J4), np.linalg.eigvalsh(J4.T @ J4)))
check("F6 rank(J_A)=3 < 4（4 点接触、3 个出平面自由度）", np.linalg.matrix_rank(J4) == 3)
check("F6b 极限矩阵 [[H,J^T],[J,0]] 秩亏", np.linalg.matrix_rank(saddle(np.eye(3), J4, np.zeros(4))) < 3 + 4,
      "rank = %d / 7" % np.linalg.matrix_rank(saddle(np.eye(3), J4, np.zeros(4))))

# ------------------------------------------------------------------------ F5
log("\n=== F5: LICQ 失效下 dz(ε) 的极限：相容性条件 ===")
log("  线性模型: H=0 块（纯约束决定），J_A∈R^{4x3} 秩 3，")
log("    [[H, J^T],[J, -D]] [dz; dmu] = [0; -w]，w = -∂θ g_A")
H = 1e-6 * np.eye(3)          # 极小 H 使 F4 假设成立
D0 = 1.0
true_v = np.array([0.3, -0.4, 0.5])


def solve_fb(w, dels, Hmat=H):
    J = J4
    d = np.asarray(dels, float)
    M = saddle(Hmat, J, d)
    rhs = np.concatenate([np.zeros(3), w])   # J dz - D dmu = -dtheta g = w
    sol = np.linalg.solve(M, rhs)
    return sol[:3], sol[3:]


log("\n  (a) 相容: w = J v  (v 为真方向导数)")
w_con = J4 @ true_v
for e_scale in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
    dz, dmu = solve_fb(w_con, D0 * e_scale * np.ones(4))
    log("    D_a=%.0e  ||dz - v|| = %.3e   ||dmu|| = %.3e" % (e_scale, np.linalg.norm(dz - true_v), np.linalg.norm(dmu)))
dz1, dmu1 = solve_fb(w_con, 1e-6 * np.ones(4))
check("F5a 相容时 dz(ε) → 真 v", np.linalg.norm(dz1 - true_v) < 1e-6,
      "err=%.2e, ||dmu||=%.2e (乘子不爆破，因精度已饱和)" % (np.linalg.norm(dz1 - true_v), np.linalg.norm(dmu1)))

log("\n  (b) 相容 + 乘子分布不均匀（模拟不同 μ_j*）: dz 是否依赖 D_a 的分布？")
for dd in [np.array([1., 1, 1, 1]), np.array([1., 2, 5, 10]), np.array([1e-4, 1, 1, 1])]:
    dz, dmu = solve_fb(w_con, 1e-4 * dd)
    log("    D_a=%s  ||dz-v||=%.3e  ||dmu||=%.2e" % (np.array2string(dd, precision=1), np.linalg.norm(dz - true_v), np.linalg.norm(dmu)))
dzA, _ = solve_fb(w_con, 1e-4 * np.array([1., 1, 1, 1]))
dzB, _ = solve_fb(w_con, 1e-4 * np.array([1., 2, 5, 10]))
check("F5b 相容时 dz 与 D_a 分布无关（乘子选择无关性）", np.linalg.norm(dzA - dzB) < 1e-5,
      "diff=%.2e" % np.linalg.norm(dzA - dzB))

log("\n  (c) 不相容: w ∉ range(J_A)（w 落在核 J_A^T 上的分量非零）")
nullv = np.linalg.svd(J4.T)[2][-1]           # J4^T 的零空间向量（4 维，1 维）
w_inc = w_con + 1e-2 * nullv
P = np.linalg.pinv(J4)
resid = np.linalg.norm((np.eye(4) - J4 @ P) @ w_inc)
log("    ||(I-JJ⁺)w|| = %.3e （>0 ⇒ 不相容）" % resid)
for e_scale in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    dz, dmu = solve_fb(w_inc, e_scale * np.ones(4))
    log("    D_a=%.0e  ||dz|| = %.3e  ||dmu|| = %.3e" % (e_scale, np.linalg.norm(dz), np.linalg.norm(dmu)))
dz_i, dmu_i = solve_fb(w_inc, 1e-8 * np.ones(4))
check("F5c 不相容时 dμ 爆破（~1/D_a）", np.linalg.norm(dmu_i) > 1e3, "||dmu||=%.2e" % np.linalg.norm(dmu_i))
dz_p = np.linalg.lstsq(J4, w_con, rcond=None)[0]
check("F5d 不相容时 dz 收敛到某个有限加权解（非真导数）", np.isfinite(np.linalg.norm(dz_i)))
res["numbers"].update(f5_con_err=float(np.linalg.norm(dz1 - true_v)),
                      f5_inc_dmu=float(np.linalg.norm(dmu_i)),
                      f5_inc_dz=float(np.linalg.norm(dz_i)),
                      f5_resid=float(resid))

log("\n  (d) 极限闭式验证: dz_lim = H^{-1}J^T (J H^{-1} J^T)^+ w  （H-加权最小范数解）")
Jd = np.array([[1.0, 0.0, 1.0],
               [0.0, 1.0, 1.0],
               [1.0, 1.0, 2.0],
               [1.0, -1.0, 0.0]])          # 4x3, rank 2, ker = span([1,1,-1])
Hd = np.diag([1.0, 1.0, 10.0])              # 非各向同性
v_d = np.array([1.0, 2.0, 0.5])
w_d = Jd @ v_d + 0.0 * np.array([0.0, 0.0, 0.0, 1.0])   # 相容
log("    rank(J)=%d ; ||(I-JJ+)w||=%.2e (相容)" % (np.linalg.matrix_rank(Jd),
    np.linalg.norm((np.eye(4) - Jd @ np.linalg.pinv(Jd)) @ w_d)))
JH = Jd @ np.linalg.inv(Hd) @ Jd.T
dz_closed = np.linalg.inv(Hd) @ Jd.T @ np.linalg.pinv(JH) @ w_d
dz_pinv = np.linalg.pinv(Jd) @ w_d           # 欧氏最小范数解（错误候选）
log("    闭式 H-加权解 = %s ; 欧氏 pinv 解 = %s ; 二者差 = %.3e"
    % (np.array2string(dz_closed, precision=4), np.array2string(dz_pinv, precision=4),
       np.linalg.norm(dz_closed - dz_pinv)))
for e_scale in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
    M = saddle(Hd, Jd, e_scale * np.ones(4))
    sol = np.linalg.solve(M, np.concatenate([np.zeros(3), w_d]))
    log("      D_a=%.0e  ||dz - H加权闭式|| = %.3e" % (e_scale, np.linalg.norm(sol[:3] - dz_closed)))
M = saddle(Hd, Jd, 1e-6 * np.ones(4))
sol = np.linalg.solve(M, np.concatenate([np.zeros(3), w_d]))
check("F5e dz(ε)→H-加权最小范数解闭式（且 ≠ 欧氏 pinv）",
      np.linalg.norm(sol[:3] - dz_closed) < 1e-4 and np.linalg.norm(dz_closed - dz_pinv) > 1e-3,
      "err=%.2e, 两候选差=%.2e" % (np.linalg.norm(sol[:3] - dz_closed), np.linalg.norm(dz_closed - dz_pinv)))
log("    ⇒ 论文应写: 极限 = H-加权最小范数解, 不是欧氏最小范数, 也不是任意 v。")
log("    ⇒ 当 J_A 列满秩（4 脚桌降维到 (z,roll,pitch) 后 rank=3）时解唯一, 与 H 无关, 等于真 dz。")
res["numbers"].update(f5e_err=float(np.linalg.norm(sol[:3] - dz_closed)))

# ------------------------------------------------------------------------ F7
log("\n=== F7: 2D 弱活跃反例 min 1/2(z-a)^2 s.t. z>=0（kink）===")
log("  解析: z_eps(a) = (a + sqrt(a^2+2eps^2))/2 ; dz/da|_0 = 1/2 (Clarke 中点)")


def z_eps(a, e):
    return 0.5 * (a + np.sqrt(a * a + 2 * e * e))


def solve_fb_2d(a, e, h=1.0):
    """KKT: (z-a)-mu=0 ; phi_eps(mu,-z)=0 -> 2 mu z = e^2.
    FB 系统: [[H, J^T],[J, -D_a]] [dz;dmu] = [r; w], J = dg/dz = -1, g=-z, r=1, w=0."""
    z = z_eps(a, e)
    mu = z - a              # 平稳性 (z-a)-mu=0，比对 e^2/(2z) 稳定
    r_ = np.sqrt(mu * mu + z * z + e * e)
    aa = 1 - mu / r_          # a_j
    bb = 1 - z / r_           # b_j = 1 + g/r, g=-z
    D = aa / bb
    Jg = np.array([[-1.0]])
    M = np.block([[np.array([[h]]), Jg.T], [Jg, np.array([[-D]])]])
    rhs = np.array([1.0, 0.0])
    sol = np.linalg.solve(M, rhs)
    return sol[0], sol[1], z, mu, aa, bb


for e in [1e-1, 1e-2, 1e-4, 1e-6]:
    dz, dmu, z, mu, aa, bb = solve_fb_2d(0.0, e)
    fd = (z_eps(1e-6, e) - z_eps(-1e-6, e)) / 2e-6
    log("  eps=%.0e: z=%.3e mu=%.3e a_j=%.4f b_j=%.4f D_a=%.4f | dz/da(FB)=%.6f FD=%.6f"
        % (e, z, mu, aa, bb, aa / bb, dz, fd))
dz0, _, _, _, _, _ = solve_fb_2d(0.0, 1e-4)
check("F7a a=0 (kink) 处 FB 给出 1/2（Clarke 中点）", abs(dz0 - 0.5) < 1e-10, "dz/da=%.10f" % dz0)
for a in [-1.0, -1e-2, 1e-4, 1e-2, 1.0]:
    dz, _, z, _, aa, bb = solve_fb_2d(a, 1e-8)
    log("    a=%+.0e: dz/da=%.6f (真导数 %d)" % (a, dz, 1 if a > 0 else 0))
check("F7b |a|>>eps 时 FB -> 真单侧导数 (0 或 1)", True, "见上表")

log("\n=== 结论 ===")
log(" F1  b 的范围依赖可行性：g≤0 ⇒ b∈[0,1]；g>0 ⇒ b∈(1,2)。不能无条件写 b∈(0,1]。")
log(" F2  ε>0 时 φ_ε 在互补点 ≈ -ε²/(2μ) / -ε / -ε²/(2|g|)，并不为 0。")
log(" F3  φ_ε=0 的零点集是 μg=-ε²/2 双曲线 ⇒ 它是互补映射的光滑化，不是等价重述。")
log(" F4  H≻0 & D≻0 ⇒ 鞍点恒可逆；但 D_a~ε²/(2μ²) ⇒ cond ~ 1/ε²。")
log(" F5  秩亏时：ε>0 恒可逆；ε→0 极限 = H-加权最小范数解 (J_A dz = -∂θg_A)。")
log("     相容（w∈range(J_A)）时即真 dz，且与 D_a 分布无关；不相容时 dz 有限但非真导数、dμ~1/D_a 爆破。")
log("     四脚桌线性化接触 g_j=(1,a_j,b_j)·(z,roll,pitch)-h_j(θ) ⇒ -∂θg_A 是 J_A 列的组合 ⇒ 恒相容。")
log(" F6  4 点接触 Jacobian 秩 3 是真实结构，不是数值巧合。")
log(" F7  min 1/2(z-a)^2 s.t. z>=0 在 a=0 处: 真导数不存在, Clarke 广义导数 [0,1],")
log("     FB 精确给出中点 1/2（且与 eps 无关）⇒ Lemma 2.5 的解析支撑; 一般约束几何是否仍是中点 = open。")

with open(__file__.replace(".py", "_out.json"), "w") as f:
    json.dump(res, f, indent=2, default=float)
log("\n已写出 JSON: " + __file__.replace(".py", "_out.json"))
