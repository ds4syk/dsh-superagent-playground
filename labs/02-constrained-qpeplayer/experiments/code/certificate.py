#!/usr/bin/env python
"""14×14 Veronese 增强 Shor 松弛（Lemma 2.2）—— sound 后验证书。

与 verify/sanity_constraints_alm.py 的 shor14 等价，但接口化：
  shor_bound(M0,D0,G0,A,B, mu=None, add_veronese=True, enforce_ineq=True)
返回 dict(value, ratio, lam1, status, time_s, ...)。

两种用法：
  * enforce_ineq=True, mu=None  : 直接对约束问题求 Shor 下界（主用法）
  * enforce_ineq=False, mu!=None: 对偶函数 d(mu) 的 Shor 下界（对照，见 proofs.md §2.2）

可行性口径（round-rlt）
----------------------
`value` = 求解器返回的 `<Q,X>`。**它不是严格下界**：只有当求解器返回的 X
对等式与 `g_j<=0` 都可行时，SDP 值才是松弛问题的可行值（进而是 J 的下界候选，
仍依赖求解器容差）。`status=optimal_inaccurate` 时 X 往往轻微违背 `g_j<=0`。

本文件新增的 **严格（rigorous）下界** 走 Lagrange 对偶：
    S = Q + Σ_k y_k A_k + Σ_m λ_m M_m ,  λ_m >= 0,  S ⪰ 0
  ⇒  对任意原始可行 X:  <Q,X> >= -Σ_k y_k b_k .
（cvxpy 的 `cons.dual_value` y 与标准乘子反号；实测确认 S 与 `X>>0` 的对偶
矩阵一致。见 `_dual_repair` 与 results/round_rlt.md 的符号校验。）
求解器给出的 `cons.dual_value` 直接取 (y, λ)；把 λ 截断到 >=0 后用
Schur 补求最小平移 c 使 S + c·P ⪰ 0（P = diag(1_{0..9},0,0,0,1)，它正好是
两个 b=1 的等式约束矩阵之和，因此平移使对偶目标减少 2c），得到
    bound_rigorous = -Σ y_k b_k - 2c .
该量与求解器收敛程度、原始可行性、容差**无关**（只依赖浮点舍入）。任何
(y, λ>=0) 都给出合法下界，所以此路径永不"伪证"。

切族（默认全关，flag 打开）：
  add_rlt       : g_i g_j >= 0（零维 RLT，二次；原实现）
  add_bounds    : 已证变量盒 (y_k - lo)(y_k - hi) <= 0 / 单侧界（线性于 X）
  add_bound_rlt : g_i · (y_k - lo_k) <= 0（g_i<=0 且 y_k>=lo_k；二次，线性于 X）
  add_box_rlt   : 两个已证单侧因子的乘积 >= 0（线性于 X）

参考：J 是 ALM 前向的可行点目标值（J* <= J），`gap = J - bound`。
"""
import time

import cvxpy as cp
import numpy as np

SQRT2 = np.sqrt(2.0)
N = 14

# ---- 已证变量盒（与 make_instance 的约束语义对应；对任意可行点成立） ----
V2_LO = np.zeros(10)
V2_HI = np.empty(10)
V2_LO[:4] = 0.0
V2_HI[:4] = 1.0                       # v2_i = q_i^2 in [0,1]
V2_LO[4:] = -1.0 / SQRT2              # v2_ab = sqrt2 q_a q_b, |q_a q_b|<=1/2
V2_HI[4:] = 1.0 / SQRT2


def _sym(M):
    return 0.5 * (M + M.T)


def _e(i, j, n=N):
    M = np.zeros((n, n))
    M[i, j] = 1.0
    return M


def _e14(n=N):
    e = np.zeros(n)
    e[13] = 1.0
    return e


def _veronese_specs(n=N):
    """2x2 子式线性化（v2 坐标 c_ab=1/sqrt2），返回 [(A_k, b_k=0)]。"""
    idx = {}
    for i in range(4):
        idx[(i, i)] = i
    k = 4
    for i in range(4):
        for j in range(i + 1, 4):
            idx[(i, j)] = k
            k += 1
    coef = lambda a, b: 1.0 if a == b else 1.0 / SQRT2
    specs = []
    for i in range(4):
        for j in range(i + 1, 4):
            for kk in range(4):
                for ll in range(kk + 1, 4):
                    a1 = idx[(min(i, kk), max(i, kk))]
                    b1 = idx[(min(j, ll), max(j, ll))]
                    a2 = idx[(min(i, ll), max(i, ll))]
                    b2 = idx[(min(j, kk), max(j, kk))]
                    c1 = coef(min(i, kk), max(i, kk)) * coef(min(j, ll), max(j, ll))
                    c2 = coef(min(i, ll), max(i, ll)) * coef(min(j, kk), max(j, kk))
                    M = np.zeros((n, n))
                    M[a1, b1] += c1
                    M[a2, b2] -= c2
                    specs.append((_sym(M), 0.0))
    return specs


def _base_eq_specs(n=N):
    """等式约束矩阵：X[13,13]=1, trace(X[:10,:10])=1, e_tr^T X e14 = 1。"""
    e14 = _e14(n)
    e_tr = np.zeros(n)
    e_tr[:10] = np.array([1.0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    E1 = 0.5 * (np.outer(e_tr, e14) + np.outer(e14, e_tr))
    D10 = np.zeros((n, n))
    for i in range(10):
        D10[i, i] = 1.0
    return [(D10, 1.0), (_e(13, 13, n), 1.0), (E1, 1.0)], e14, e_tr


def _v2_row(A, B, j, n=N):
    c = np.zeros(n)
    c[:10] = A[j]
    c[10:13] = B[j]
    return c


def _g_matrix(c14, e14):
    """g_j = c^T y 的对称矩阵（<C,X> = g_j）。"""
    return 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))


def _v2_iv(a, lo=None, hi=None):
    """a^T v2 的区间（默认用已证 v2 盒）。"""
    lo = V2_LO if lo is None else lo
    hi = V2_HI if hi is None else hi
    l = np.where(a > 0, lo, hi)
    h = np.where(a > 0, hi, lo)
    return float(np.sum(a * l)), float(np.sum(a * h))


# 交叉项 v2_ab 对应的对角下标（v2 顺序: 0..3 = q_i^2, 4..9 = (01,02,03,12,13,23)）
_CROSS_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def box_bounds(A, B, n=N, sweeps=8):
    """对 y=(v2,t) 做 **sound 区间传播**，返回 (lo, hi) 各长 14（y_13=1 固定）。

    * 用 E_TR·v2 = 1 精确消去 v2_0：v2_0 = 1 - v2_1 - v2_2 - v2_3；
    * 基础盒：v2_i=q_i^2>=0，v2_i<=1；|v2_ab|<=1/sqrt2；
    * 追加 **sound** 行 v2_1+v2_2+v2_3 <= 1（因 v2_0>=0），否则盒与
      v2_0 = 1-Σ 矛盾、传播会推出 infeasible 的 ±inf；
    * 对角线/交叉项互相收紧：|v2_ab| <= sqrt(2 v2_a v2_b) <= sqrt(2 hi_a hi_b)；
    * 对每个约束 a_j·v2 + b_j·t <= 0 反复做区间传播（含 t 的单侧界）。
    未证方向为 ±inf。所有界只依赖约束本身，故对所有可行点成立。
    """
    # z = [v2_1, v2_2, v2_3, v2_4..v2_9, t_x, t_y, t_z]  (3+6+3 = 12)
    # 传播行：原始约束 + v2_0>=0 行；tilt 等行由 A 自带，不必硬编码 alpha
    nrow = A.shape[0] + 1
    Cm = np.zeros((nrow, 12))
    c0 = np.zeros(nrow)
    for j in range(A.shape[0]):
        a = np.asarray(A[j], float)
        c0[j] = a[0]
        Cm[j, 0] = a[1] - a[0]
        Cm[j, 1] = a[2] - a[0]
        Cm[j, 2] = a[3] - a[0]
        Cm[j, 3:9] = a[4:10]
        Cm[j, 9:12] = np.asarray(B[j], float)
    # v2_1+v2_2+v2_3 <= 1  <=>  (e1+e2+e3-E_TR)·v2 <= 0
    Cm[-1, 0:3] = 1.0
    c0[-1] = -1.0

    zlo = np.full(12, -np.inf)
    zhi = np.full(12, np.inf)
    zlo[0], zhi[0] = 0.0, 1.0
    zlo[1], zhi[1] = 0.0, 1.0
    zlo[2], zhi[2] = 0.0, 1.0
    for m in range(6):
        zlo[3 + m], zhi[3 + m] = -1.0 / SQRT2, 1.0 / SQRT2

    def _propagate():
        changed = False
        for j in range(nrow):
            c = Cm[j]
            for k in range(12):
                ck = c[k]
                if ck == 0.0:
                    continue
                # RHS = -c0 - Σ_{l≠k} c_l z_l 的区间。端点可能无穷，必须分别
                # 记录"是否 -inf / +inf"，否则 -inf + (+inf) 会得到 nan。
                base = -c0[j]
                rlo = base
                rhi = base
                neg_inf = False
                pos_inf = False
                for l in range(12):
                    cl = c[l]
                    if l == k or cl == 0.0:
                        continue
                    t1 = -cl * zlo[l]
                    t2 = -cl * zhi[l]
                    lo_t, hi_t = (t1, t2) if t1 <= t2 else (t2, t1)
                    if lo_t == -np.inf:
                        neg_inf = True
                    else:
                        rlo += lo_t
                    if hi_t == np.inf:
                        pos_inf = True
                    else:
                        rhi += hi_t
                rlo = -np.inf if neg_inf else rlo
                rhi = np.inf if pos_inf else rhi
                if ck > 0.0:
                    # ck*z_k <= RHS, ck>0 => z_k <= RHS/ck；对一切 RHS 用其上端
                    cand = rhi / ck
                    if np.isfinite(cand) and cand < zhi[k]:
                        zhi[k] = cand
                        changed = True
                else:
                    # ck<0 => z_k >= RHS/ck；除以负数反号，RHS/ck 随 RHS 递减，取下端 rhi/ck
                    cand = rhi / ck
                    if np.isfinite(cand) and cand > zlo[k]:
                        zlo[k] = cand
                        changed = True
        return changed

    # 阶段 1：只传播对角/仿射行；阶段 2：用对角界收紧交叉项后再传播
    for _ in range(sweeps):
        if not _propagate():
            break
    # 交叉项 |v2_ab| <= sqrt(2 v2_a v2_b)；v2_0 的下界用 1-(hi1+hi2+hi3)
    dh = {0: min(1.0, 1.0 - (zlo[0] + zlo[1] + zlo[2])), 1: zhi[0], 2: zhi[1], 3: zhi[2]}
    for m, (a, b) in enumerate(_CROSS_PAIRS):
        h = min(SQRT2 * np.sqrt(max(dh[a], 0.0) * max(dh[b], 0.0)), 1.0 / SQRT2)
        h = max(h, 0.0)
        zlo[3 + m] = max(zlo[3 + m], -h)
        zhi[3 + m] = min(zhi[3 + m], h)
    for _ in range(sweeps):
        if not _propagate():
            break

    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    lo[1], hi[1] = zlo[0], zhi[0]
    lo[2], hi[2] = zlo[1], zhi[1]
    lo[3], hi[3] = zlo[2], zhi[2]
    lo[4:10], hi[4:10] = zlo[3:9], zhi[3:9]
    lo[10:13], hi[10:13] = zlo[9:12], zhi[9:12]
    # v2_0 = 1 - v2_1 - v2_2 - v2_3，再与 [0,1] 取交
    lo[0] = max(0.0, 1.0 - (hi[1] + hi[2] + hi[3]))
    hi[0] = min(1.0, 1.0 - (lo[1] + lo[2] + lo[3]))
    lo[13], hi[13] = 1.0, 1.0
    for k in range(4):
        lo[k] = max(lo[k], 0.0)
        hi[k] = min(hi[k], 1.0)
    return lo, hi


def t_lower_bounds(A, B, n=N):
    """t_k 的已证单侧界（来自 box_bounds 的区间传播）。"""
    lo, hi = box_bounds(A, B, n)
    return lo[10:13].copy(), hi[10:13].copy()


def proven_t_lo(A, B, n=N):
    lo, _ = box_bounds(A, B, n)
    return lo[10:13].copy()


def _var_bounds(A, B, n=N):
    """已证变量单侧界列表：[(k, lo_k)] / [(k, hi_k)]（box_bounds 的非平凡界）。"""
    lo, hi = box_bounds(A, B, n)
    lower = [(k, float(lo[k])) for k in range(13) if np.isfinite(lo[k])]
    upper = [(k, float(hi[k])) for k in range(13) if np.isfinite(hi[k])]
    return lower, upper


def _bound_specs(A, B, n=N):
    """已证变量盒切（线性于 X），全部来自 box_bounds 的单侧界。

    (y_k - lo)(y_k - hi) <= 0  <=>  X_kk - (lo+hi) X_{k,13} + lo*hi <= 0，
    单侧界退化为 X_{k,13} <= hi 或 X_{k,13} >= lo。返回 [(M_m, 0.0)]。
    """
    specs = []
    lo, hi = box_bounds(A, B, n)
    for k in range(13):
        lo_k, hi_k = float(lo[k]), float(hi[k])
        if not (np.isfinite(lo_k) or np.isfinite(hi_k)):
            continue
        M = _e(k, k, n)
        if np.isfinite(lo_k) and np.isfinite(hi_k):
            M = M - (lo_k + hi_k) * _sym(_e(k, 13, n)) + lo_k * hi_k * _e(13, 13, n)
            specs.append((M, 0.0))
        elif np.isfinite(hi_k):
            M = _sym(_e(k, 13, n)) - hi_k * _e(13, 13, n)
            specs.append((M, 0.0))
        else:
            M = lo_k * _e(13, 13, n) - _sym(_e(k, 13, n))
            specs.append((M, 0.0))
    return specs


def proven_t_lo(A, B, n=N):
    lo, _ = box_bounds(A, B, n)
    return lo[10:13].copy()


def _var_bounds(A, B, n=N):
    """已证变量单侧界列表：[(k, lo_k)] / [(k, hi_k)]（box_bounds 的非平凡界）。"""
    lo, hi = box_bounds(A, B, n)
    lower = [(k, float(lo[k])) for k in range(13) if np.isfinite(lo[k])]
    upper = [(k, float(hi[k])) for k in range(13) if np.isfinite(hi[k])]
    return lower, upper


def _bound_specs(A, B, n=N):
    """已证变量盒切（线性于 X），返回 [(M_m, 0.0)] 表示 <M_m,X> <= 0。"""
    specs = []
    e14 = _e14(n)
    # (a) v2_i in [0,1]  =>  X_ii <= X_{i,13}  （y_i^2 <= y_i，因 y_13=1）
    for i in range(4):
        M = _e(i, i, n) - _sym(_e(i, 13, n))
        specs.append((M, 0.0))
    # (b) |v2_ab| <= 1/sqrt2  =>  X_ii <= 1/2
    for i in range(4, 10):
        M = _e(i, i, n) - 0.5 * _e(13, 13, n)
        specs.append((M, 0.0))
    # (c) tilt 已证: v2_1+v2_2 <= sin^2(alpha/2)（ALPHA=12deg, 半角 6deg）
    s2 = float(np.sin(np.deg2rad(12.0) / 2.0) ** 2)
    for i in (1, 2):
        M = _e(i, i, n) - s2 * _sym(_e(i, 13, n))
        specs.append((M, 0.0))
    # (d) t_z 的已证下界（穿透/深度约束 + v2 盒区间传播）
    lo_t, _hi_t = t_lower_bounds(A, B, n)
    if np.isfinite(lo_t[2]):
        M = lo_t[2] * _e(13, 13, n) - _sym(_e(12, 13, n))
        specs.append((M, 0.0))
    return specs


def _bound_rlt_specs(A, B, n=N):
    """g_i * (y_k - lo_k) <= 0（g_i<=0 且 y_k>=lo_k）以及
    g_i * (hi_k - y_k) <= 0（g_i<=0 且 y_k<=hi_k），二次、线性于 X。"""
    specs = []
    e14 = _e14(n)
    lower, upper = _var_bounds(A, B, n)
    for j in range(A.shape[0]):
        c14 = _v2_row(A, B, j, n)
        Cj = _g_matrix(c14, e14)
        for k, lo in lower:
            ek = np.zeros(n)
            ek[k] = 1.0
            # g_i*y_k - lo*g_i <= 0
            M = _sym(np.outer(c14, ek)) - lo * Cj
            specs.append((M, 0.0))
        for k, hi in upper:
            ek = np.zeros(n)
            ek[k] = 1.0
            # hi*g_i - g_i*y_k <= 0
            M = hi * Cj - _sym(np.outer(c14, ek))
            specs.append((M, 0.0))
    return specs


def _box_rlt_specs(A, B, n=N):
    """两个已证单侧因子的乘积 >= 0（线性于 X）：
    (y_k - lo_k)(y_l - lo_l) >= 0,  (hi_k - y_k)(hi_l - y_l) >= 0,  以及
    (y_k - lo_k)(hi_l - y_l) >= 0（k,l 可相同）。"""
    specs = []
    e14 = _e14(n)
    lower, upper = _var_bounds(A, B, n)
    ev = np.zeros((n, n))
    for k in range(n):
        ev[k, k] = 1.0

    def _fac(sign, k, c):
        # f = sign*y_k - sign*c 的系数向量（sign=+1: y_k-c; sign=-1: c-y_k）
        f = sign * ev[k].copy()
        f[13] = -sign * c
        return f

    lower_f = [(k, _fac(+1.0, k, c)) for k, c in lower]
    upper_f = [(k, _fac(-1.0, k, c)) for k, c in upper]
    for a in range(len(lower_f)):
        for b in range(a, len(lower_f)):
            M = _sym(np.outer(lower_f[a][1], lower_f[b][1]))
            specs.append((-M, 0.0))       # <M,X> >= 0
    for a in range(len(upper_f)):
        for b in range(a, len(upper_f)):
            M = _sym(np.outer(upper_f[a][1], upper_f[b][1]))
            specs.append((-M, 0.0))
    for a in range(len(lower_f)):
        for b in range(len(upper_f)):
            M = _sym(np.outer(lower_f[a][1], upper_f[b][1]))
            specs.append((-M, 0.0))
    return specs


def _schur_shift(S, delta, vh_idx):
    """求最小 c>=0 使 S + c*P ⪰ 0，P=diag(1 on vh_idx)。利用 t 块与 P 无交。

    S 分块为 [[A0, B],[B^T, G0]]（t 块索引 10..12）。Schur 补
    A0 - B G0^{-1} B^T 在 (v,h) 子空间上，P 在该子空间上恰为单位阵，
    故 c_min = max(0, -lambda_min(Schur)) + delta。若 G0 非 PD 则回退到
    全局 lambda_min（可能不足，调用方需再验证）。
    """
    t_idx = [10, 11, 12]
    vh = list(vh_idx)
    A0 = S[np.ix_(vh, vh)]
    Bm = S[np.ix_(vh, t_idx)]
    G0 = S[np.ix_(t_idx, t_idx)]
    try:
        Ginv = np.linalg.inv(G0)
        ev = np.linalg.eigvalsh(G0)
        if ev[0] <= 0:
            raise np.linalg.LinAlgError("G0 not PD")
        Schur = A0 - Bm @ Ginv @ Bm.T
        Schur = _sym(Schur)
        lmin = float(np.linalg.eigvalsh(Schur)[0])
        return max(0.0, -lmin) + delta, "schur"
    except np.linalg.LinAlgError:
        lmin = float(np.linalg.eigvalsh(_sym(S))[0])
        return max(0.0, -lmin) + delta, "full"


def _dual_repair(Qtot, eq_specs, ineq_specs, cons, delta=1e-8):
    """从求解器对偶构造严格可行的对偶点，返回严格下界。

    符号约定（实测于 cvxpy 1.5.4 + SCS/CLARABEL，见 results/round_rlt.md）：
    cvxpy 的 `cons.dual_value` y 与标准 Lagrange 乘子 ỹ 反号（ỹ = -y），
    而 `<=` 约束的 λ 与标准乘子同号。于是
        S = Q + Σ y_k A_k + Σ λ_m M_m ⪰ 0   （与 cons[0].dual_value 一致）
        bound = Σ ỹ_k b_k = -Σ y_k b_k
    严格性：对任意原始可行 X，<Q,X> >= <S,X> + Σ ỹ b - Σ λ <M,X> >= -y·b。
    只依赖 (y, λ>=0, S⪰0)，与求解器容差/原始可行性无关。
    """
    m_eq = len(eq_specs)
    m_in = len(ineq_specs)
    n = Qtot.shape[0]
    try:
        y = np.array([float(cons[1 + k].dual_value) for k in range(m_eq)])
        lam_raw = np.array([float(cons[1 + m_eq + k].dual_value) for k in range(m_in)])
    except (TypeError, ValueError):
        return dict(rigorous=float("nan"), dual_obj=float("nan"),
                    dual_shift=float("nan"), dual_lam_min=float("nan"),
                    dual_n_negative_lam=-1, dual_method="no-duals")
    lam = np.maximum(lam_raw, 0.0)
    S = Qtot.copy()
    for (Amat, b), yy in zip(eq_specs, y):
        S = S + yy * Amat
    for (M, _), ll in zip(ineq_specs, lam):
        if ll != 0.0:
            S = S + ll * M
    S = _sym(S)
    d_dual = float(-sum(yy * b for (Amat, b), yy in zip(eq_specs, y)))
    vh_idx = list(range(10)) + [13]
    P = np.zeros((n, n))
    for i in vh_idx:
        P[i, i] = 1.0
    c, method = _schur_shift(S, delta, vh_idx)
    # 校验：若 Schur 平移后仍不 PSD（例如加切后 t 块不再 ≻0），逐步加大平移；
    # 仍不行则退回全局 lambda_min 并如实标记（此时界可能无效，调用方需检查
    # dual_lam_min >= -1e-10）。
    for _ in range(6):
        S_rep = S + c * P
        lam_rep_min = float(np.linalg.eigvalsh(S_rep)[0])
        if lam_rep_min >= -1e-10:
            break
        c = max(2.0 * c, -float(np.linalg.eigvalsh(_sym(S))[0]) + delta)
    if lam_rep_min < -1e-10:
        method = method + "+unrepaired"
    # 平移 c*P 改变两个 b=1 的等式乘子（D10, E1313），对偶目标 -2c
    rigorous = d_dual - 2.0 * c
    return dict(rigorous=float(rigorous), dual_obj=d_dual,
                dual_shift=float(c), dual_lam_min=float(lam_rep_min),
                dual_n_negative_lam=int(np.sum(lam_raw < 0)), dual_method=method)


def shor_bound(M0, D0, G0, A, B, mu=None, add_veronese=True, enforce_ineq=True,
               add_rlt=False, add_bounds=False, add_bound_rlt=False,
               add_box_rlt=False,
               repair_dual=False, repair_delta=1e-8,
               solver="SCS", eps=1e-9, max_iters=20000, time_limit=None,
               verbose=False):
    """min <Qh,X> s.t. X⪰0, X[13,13]=1, e_tr^T v2=1, ||v2||^2=1, Veronese, g_j<=0。

    Qh 来自 J（mu=None）或 J+Σμ_j g_j（mu!=None，此时 enforce_ineq 应为 False）。
    返回 dict；关键字段：
      value     : 求解器报告值（不可直接当严格下界）
      rigorous  : 严格对偶下界（仅 repair_dual=True；否则 nan）
      status    : 求解器状态
      ratio     : λ2/λ1（X 的次大/最大特征值，越小越接近秩 1）
      feas_ineq : max_m <M_m,X>（<=0 才原始可行）
      eq_resid  : 等式最大残差
    """
    n = N
    Qm = np.zeros((13, 13))
    Qm[:10, :10] = M0
    Qm[:10, 10:] = D0.T
    Qm[10:, :10] = D0
    Qm[10:, 10:] = G0
    Qh = np.zeros((n, n))
    Qh[:13, :13] = Qm
    e14 = _e14(n)

    lhs_lin = np.zeros((n, n))
    if mu is not None:
        for j in range(A.shape[0]):
            if mu[j] == 0.0:
                continue
            c14 = _v2_row(A, B, j, n)
            lhs_lin += mu[j] * 0.5 * (np.outer(c14, e14) + np.outer(e14, c14))
    Qtot = Qh + 2.0 * lhs_lin

    eq_specs, _, _ = _base_eq_specs(n)
    if add_veronese:
        eq_specs = eq_specs + _veronese_specs(n)

    ineq_specs = []
    c14_list = []
    if enforce_ineq:
        for j in range(A.shape[0]):
            c14 = _v2_row(A, B, j, n)
            c14_list.append(c14)
            ineq_specs.append((_g_matrix(c14, e14), 0.0))
    if add_rlt and enforce_ineq:
        # 零维 RLT：g_i g_j >= 0（对 g<=0 恒成立），线性于 X
        for i in range(len(c14_list)):
            for j in range(i, len(c14_list)):
                M = _sym(np.outer(c14_list[i], c14_list[j]))
                ineq_specs.append((-M, 0.0))     # -M <= 0  <=>  <M,X> >= 0
    # 所有切都对「原始可行集」成立；enforce_ineq=False 时 SDP 是对偶函数的
    # 球面松弛（不含 g<=0），此时加切会抬高 min 从而破坏下界性质，故一并关闭。
    if add_bounds and enforce_ineq:
        ineq_specs += _bound_specs(A, B, n)
    if add_bound_rlt and enforce_ineq:
        ineq_specs += _bound_rlt_specs(A, B, n)
    if add_box_rlt and enforce_ineq:
        ineq_specs += _box_rlt_specs(A, B, n)

    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0]
    for Amat, b in eq_specs:
        cons.append(cp.trace(Amat @ X) == b)
    for M, _ in ineq_specs:
        cons.append(cp.trace(M @ X) <= 0.0)
    prob = cp.Problem(cp.Minimize(cp.trace(Qtot @ X)), cons)
    t0 = time.perf_counter()
    kw = dict(solver=solver)
    if solver == "SCS":
        kw["eps"] = eps
        kw["max_iters"] = max_iters
    elif solver == "CLARABEL":
        kw["tol_gap_abs"] = eps
        kw["tol_gap_rel"] = eps
        kw["tol_feas"] = eps
    if time_limit is not None:
        kw["time_limit_secs"] = time_limit
    try:
        prob.solve(**kw)
    except Exception as exc:              # 求解器失败也要返回可诊断信息
        return dict(value=float("nan"), ratio=float("nan"), lam1=float("nan"),
                    status="error:%s" % exc, time_s=time.perf_counter() - t0,
                    rigorous=float("nan"), solver=solver, eps=eps,
                    n_eq=len(eq_specs), n_ineq=len(ineq_specs),
                    feas_ineq=float("nan"), eq_resid=float("nan"), rank1=float("nan"))
    dt = time.perf_counter() - t0
    out = dict(status=str(prob.status), time_s=dt, solver=solver, eps=eps,
               n_eq=len(eq_specs), n_ineq=len(ineq_specs))
    if X.value is None or prob.value is None:
        out.update(value=float("nan"), ratio=float("nan"), lam1=float("nan"),
                   rigorous=float("nan"), feas_ineq=float("nan"),
                   eq_resid=float("nan"), rank1=float("nan"))
        return out

    Xv = np.asarray(X.value)
    ev = np.sort(np.linalg.eigvalsh(Xv))
    lam1 = float(ev[-1])
    # 原始可行性残差（value 只有在这些 <= tol 时才是"松弛问题可行值"）
    feas_ineq = float(max([np.sum(M * Xv) for (M, _) in ineq_specs], default=-np.inf))
    eq_resid = float(max([abs(np.sum(Amat * Xv) - b) for (Amat, b) in eq_specs], default=0.0))
    out.update(value=float(prob.value), lam1=lam1,
               ratio=float(ev[-2] / max(lam1, 1e-300)),
               feas_ineq=feas_ineq, eq_resid=eq_resid,
               x_lam_min=float(ev[0]),
               rank1=bool(ev[-2] <= 1e-6 * max(lam1, 1e-300) and ev[-1] > 1e-9))
    if repair_dual:
        out.update(_dual_repair(Qtot, eq_specs, ineq_specs, cons, repair_delta))
    else:
        out.update(rigorous=float("nan"), dual_obj=float("nan"),
                   dual_shift=float("nan"), dual_lam_min=float("nan"),
                   dual_n_negative_lam=-1, dual_method="off")
    if verbose:
        print(out, flush=True)
    return out
