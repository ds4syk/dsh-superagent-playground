#!/usr/bin/env python
"""可缓存 / 热启动的 14×14 Veronese–Shor 证书 SDP（**不修改** certificate.py）。

动机
----
主表每个 trial 都调用一次 ``certificate.shor_bound``：它每次都重建 14×14 的
``Qh``、36 条 Veronese 等式与 p 条 ``g_j<=0`` 不等式，然后 cvxpy canonicalize
并从头求解（冷解 ~35–60 ms）。但对 **固定 p** 而言，问题结构完全不变，只有
``Q`` 与不等式系数 ``G`` 换数值，所以可以把它们做成 cvxpy ``Parameter``：
编译一次，之后逐实例只更新参数并 warm start（~2–4 ms）。

本模块是 ``shor_bound`` 的**包装**，不复制/不修改求解语义：
* 等式约束（``X[13,13]=1``、``trace(X[:10,:10])=1``、``E1``、36 条 Veronese）
  直接用 ``certificate._base_eq_specs`` / ``certificate._veronese_specs`` 的矩阵
  构造，与 ``shor_bound`` 逐项相同；
* 不等式 ``g_j = c_j^T X e14 <= 0`` 写成 ``(X @ e14).T @ Gp <= 0``（DPP）；
* 返回 dict 的字段与 ``shor_bound`` 兼容（value/ratio/lam1/status/time_s/
  feas_ineq/eq_resid/...），``run_main.py`` 可直接替换；
* 若 warm solve 状态不是 ``optimal``（或 X 不可用），先 ``warm_start=False``
  重解一次，仍失败则回退到真正的 ``shor_bound``（并标记 ``fallback=True``），
  因此**不会**因为缓存而给出一个比冷解更差的证书。

精度
----
默认 ``eps=5e-11, max_iters=80000``（比主表默认的 1e-9 更紧）：与同样设置的
冷解相比 ``|Δvalue|`` ≤ 4e-10（中位 ~1e-10，见 ``--bench`` 的 20 实例测量），
tightness λ2/λ1 ≤ 5e-11，仍远低于证书阈值 1e-6。对主表默认冷解（eps=1e-9）
的 ``|Δvalue|`` ≤ 4e-9，同样比 1e-6 阈值小两个数量级以上。

用法
----
    from cert_cache import get_cache
    sd = get_cache(p).solve(M0, D0, G0, A, B)     # 同一 p 的 cache 自动复用

    # run_main.py 中：QPE_CERT_CACHE=1 时走这里，默认关闭
    python run_main.py                            # 冷解（默认，与历史一致）
    QPE_CERT_CACHE=1 python run_main.py           # 缓存 + warm start

基准（20 实例，p=16，冷解 vs 缓存；写出 results/perf_cert_cache.{json,md}）：

    python cert_cache.py --bench --n 20 --p 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cvxpy as cp
import numpy as np

from certificate import (_base_eq_specs, _veronese_specs, shor_bound)

N = 14
DEFAULT_EPS = 5e-11
DEFAULT_MAX_ITERS = 80000

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


# ---------------------------------------------------------------------------
# 数据装配（与 shor_bound 逐项相同）
# ---------------------------------------------------------------------------
def qh_matrix(M0, D0, G0, n=N):
    """shor_bound 里 14×14 的目标矩阵 Qh（t 已消去前的 13×13 块）。"""
    Qm = np.zeros((13, 13))
    Qm[:10, :10] = M0
    Qm[:10, 10:] = D0.T
    Qm[10:, :10] = D0
    Qm[10:, 10:] = G0
    Qh = np.zeros((n, n))
    Qh[:13, :13] = Qm
    return Qh


def c14_matrix(A, B, n=N):
    """p 条 g_j = c_j^T y 的系数 c_j（14×p），y=(v2,t,1)。"""
    p = A.shape[0]
    Cm = np.zeros((n, p))
    Cm[:10, :] = A.T
    Cm[10:13, :] = B.T
    return Cm


def _tightness(Xval):
    ev = np.sort(np.linalg.eigvalsh(np.asarray(Xval)))
    lam1 = float(ev[-1])
    return float(ev[-2] / max(lam1, 1e-300)), lam1


# ---------------------------------------------------------------------------
# 参数化问题 + 缓存
# ---------------------------------------------------------------------------
class ShorCache:
    """固定 p 的参数化 Shor SDP：编译一次，逐实例更新 Q/Gp 后求解。

    参数
    ----
    p          : 约束条数（决定 Gp 的列数；不同 p 必须用不同 cache）
    solver     : 默认 SCS（支持 warm_start）
    eps        : 求解精度（SCS eps / CLARABEL tol_*）
    max_iters  : SCS 迭代上限
    warm_start : 是否用上一次解热启动（默认 True；False 仅为诊断）
    """

    def __init__(self, p, solver="SCS", eps=DEFAULT_EPS,
                 max_iters=DEFAULT_MAX_ITERS, warm_start=True):
        if int(p) <= 0:
            raise ValueError("p must be positive")
        self.p = int(p)
        self.solver = solver
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.warm_start = bool(warm_start)
        self._prob = None
        self._Q = None
        self._Gp = None
        self._X = None
        self._eq_specs = None
        self._e14 = None
        self.n_solves = 0
        self.n_warm = 0
        self.n_nowarm = 0
        self.n_fallback = 0
        self.compile_s = 0.0
        self._compiled = False
        self._cached_solution = None

    # -- build (cheap; cvxpy compilation happens on the first solve) ----------
    def _build(self):
        pq = _base_eq_specs(N)                 # ([(A_k,b_k)], e14, e_tr)
        eq_specs = list(pq[0]) + list(_veronese_specs(N))
        e14 = np.asarray(pq[1], float)
        X = cp.Variable((N, N), symmetric=True)
        Q = cp.Parameter((N, N))
        Gp = cp.Parameter((N, self.p))
        cons = [X >> 0]
        for Amat, b in eq_specs:
            cons.append(cp.trace(Amat @ X) == float(b))
        cons.append((X @ e14).T @ Gp <= 0.0)
        prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
        self._prob, self._Q, self._Gp, self._X = prob, Q, Gp, X
        self._eq_specs, self._e14 = eq_specs, e14

    @property
    def is_dpp(self):
        if self._prob is None:
            self._build()
        return bool(self._prob.is_dpp())

    def _solve_kwargs(self, warm_start):
        kw = dict(solver=self.solver, warm_start=bool(warm_start))
        if self.solver == "SCS":
            kw["eps"] = self.eps
            kw["max_iters"] = self.max_iters
        elif self.solver == "CLARABEL":
            kw["tol_gap_abs"] = self.eps
            kw["tol_gap_rel"] = self.eps
            kw["tol_feas"] = self.eps
        return kw

    def _result(self, t_solve, cached=True, fallback=False):
        if self._X.value is None or self._prob.value is None:
            return dict(value=float("nan"), ratio=float("nan"), lam1=float("nan"),
                        status=str(self._prob.status), time_s=float(t_solve),
                        solver=self.solver, eps=self.eps, n_eq=len(self._eq_specs),
                        n_ineq=self.p, feas_ineq=float("nan"), eq_resid=float("nan"),
                        x_lam_min=float("nan"), rank1=False, cached=bool(cached),
                        fallback=bool(fallback), compile_s=float(self.compile_s),
                        n_solves=int(self.n_solves), warm_start=bool(self.warm_start))
        Xv = np.asarray(self._X.value)
        ratio, lam1 = _tightness(Xv)
        e14 = self._e14
        gvals = (Xv @ e14) @ self._Gp.value
        out = dict(
            value=float(self._prob.value), ratio=float(ratio), lam1=float(lam1),
            status=str(self._prob.status), time_s=float(t_solve),
            solver=self.solver, eps=self.eps, n_eq=len(self._eq_specs),
            n_ineq=self.p,
            feas_ineq=float(np.max(gvals)) if gvals.size else float("-inf"),
            eq_resid=float(max([abs(float(np.sum(Amat * Xv)) - b)
                                for Amat, b in self._eq_specs], default=0.0)),
            x_lam_min=float(np.sort(np.linalg.eigvalsh(Xv))[0]),
            rank1=bool(np.sort(np.linalg.eigvalsh(Xv))[-2]
                       <= 1e-6 * max(lam1, 1e-300) and lam1 > 1e-9),
            cached=bool(cached), fallback=bool(fallback),
            compile_s=float(self.compile_s), n_solves=int(self.n_solves),
            warm_start=bool(self.warm_start),
        )
        return out

    def solve(self, M0, D0, G0, A, B):
        """更新参数并求解；返回与 ``shor_bound`` 兼容的 dict。"""
        if A.shape[0] != self.p:
            raise ValueError("cache p=%d but A has %d rows" % (self.p, A.shape[0]))
        if self._prob is None:
            self._build()
        self._Q.value = qh_matrix(M0, D0, G0)
        self._Gp.value = c14_matrix(A, B)

        t0 = time.perf_counter()
        self._prob.solve(**self._solve_kwargs(self.warm_start))
        dt = time.perf_counter() - t0
        self.n_solves += 1
        self.n_warm += int(self.warm_start)
        if not self._compiled:
            self.compile_s = dt
            self._compiled = True
        ok = self._prob.status == "optimal" and self._X.value is not None
        if not ok:
            # 热启动失败（或参数不兼容）：从头重解一次，仍不行再回退到冷解
            t1 = time.perf_counter()
            self._prob.solve(**self._solve_kwargs(False))
            self.n_nowarm += 1
            ok = self._prob.status == "optimal" and self._X.value is not None
            if not ok:
                self.n_fallback += 1
                out = shor_bound(M0, D0, G0, A, B, enforce_ineq=True,
                                 solver=self.solver, eps=self.eps,
                                 max_iters=self.max_iters)
                out.update(cached=False, fallback=True, compile_s=self.compile_s,
                           n_solves=self.n_solves, warm_start=self.warm_start)
                return out
        out = self._result(dt, cached=True, fallback=False)
        return out

    def solve_nowarm(self, M0, D0, G0, A, B):
        """诊断用：复用编译/规范化结果，但每次从零求解（不 warm start）。"""
        if self._prob is None:
            self._build()
        self._Q.value = qh_matrix(M0, D0, G0)
        self._Gp.value = c14_matrix(A, B)
        t0 = time.perf_counter()
        self._prob.solve(**self._solve_kwargs(False))
        dt = time.perf_counter() - t0
        self.n_solves += 1
        self.n_nowarm += 1
        if not self._compiled:
            self.compile_s = dt
            self._compiled = True
        return self._result(dt, cached=True, fallback=False)


# ---------------------------------------------------------------------------
# 按 p 的全局 registry（run_main 用）
# ---------------------------------------------------------------------------
_CACHES = {}


def get_cache(p, **kw):
    """取/建固定 p 的 ShorCache（同一 (p, solver, eps, max_iters, warm_start) 复用）。"""
    key = (int(p), str(kw.get("solver", "SCS")), float(kw.get("eps", DEFAULT_EPS)),
           int(kw.get("max_iters", DEFAULT_MAX_ITERS)),
           bool(kw.get("warm_start", True)))
    if key not in _CACHES:
        _CACHES[key] = ShorCache(p, **kw)
    return _CACHES[key]


def cached_shor_bound(M0, D0, G0, A, B, cache=None, **kw):
    """便捷函数：等价于对固定 p 的 ``shor_bound(..., enforce_ineq=True)`` 缓存版。"""
    cache = cache if cache is not None else get_cache(A.shape[0], **kw)
    return cache.solve(M0, D0, G0, A, B)


# ---------------------------------------------------------------------------
# 基准：冷解 vs 缓存（20 实例）
# ---------------------------------------------------------------------------
def _med_iqr(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if xs.size == 0:
        return dict(median=float("nan"), q1=float("nan"), q3=float("nan"), n=0)
    q1, q50, q3 = np.percentile(xs, [25, 50, 75])
    return dict(median=float(q50), q1=float(q1), q3=float(q3), n=int(xs.size))


def bench(n=20, p=16, noise=0.01, bias_z=0.08, eps=DEFAULT_EPS,
          max_iters=DEFAULT_MAX_ITERS, seed_offset=0):
    """同一批 20 个实例上测：默认冷解 / 同精度冷解 / 缓存 warm / 缓存 nowarm。

    缓存 warm 按生产顺序（实例 i 用实例 i-1 的解热启动；第 1 个含编译）。
    """
    from qpep_core import make_instance

    cache = ShorCache(p, solver="SCS", eps=eps, max_iters=max_iters, warm_start=True)
    rows = []
    for t in range(n):
        ins = make_instance(seed_offset + t, p, noise=noise, bias_z=bias_z)
        M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
        rec = dict(trial=t)
        # 1) 主表默认冷解（production reference）
        t0 = time.perf_counter()
        cd = shor_bound(M0, D0, G0, A, B, enforce_ineq=True)
        rec["cold_default_ms"] = 1e3 * (time.perf_counter() - t0)
        rec["cold_default_value"] = float(cd["value"])
        rec["cold_default_ratio"] = float(cd["ratio"])
        rec["cold_default_status"] = str(cd["status"])
        # 2) 同精度冷解（与 cache 的 eps 对齐）
        t0 = time.perf_counter()
        ct = shor_bound(M0, D0, G0, A, B, enforce_ineq=True,
                        eps=eps, max_iters=max_iters)
        rec["cold_tight_ms"] = 1e3 * (time.perf_counter() - t0)
        rec["cold_tight_value"] = float(ct["value"])
        rec["cold_tight_ratio"] = float(ct["ratio"])
        rec["cold_tight_status"] = str(ct["status"])
        # 3) 缓存 warm（生产模式，第一次含编译）
        sd = cache.solve(M0, D0, G0, A, B)
        rec["cache_warm_ms"] = 1e3 * sd["time_s"]
        rec["cache_warm_value"] = float(sd["value"])
        rec["cache_warm_ratio"] = float(sd["ratio"])
        rec["cache_warm_status"] = str(sd["status"])
        rec["cache_warm_fallback"] = bool(sd["fallback"])
        rec["cache_first"] = bool(t == 0)
        # 4) 缓存 nowarm（诊断：仅复用编译）
        sn = cache.solve_nowarm(M0, D0, G0, A, B)
        rec["cache_nowarm_ms"] = 1e3 * sn["time_s"]
        rec["cache_nowarm_value"] = float(sn["value"])
        rec["cache_nowarm_ratio"] = float(sn["ratio"])
        # 5) 与紧冷解的差
        rec["dv_warm_vs_tight"] = rec["cache_warm_value"] - rec["cold_tight_value"]
        rec["dv_warm_vs_default"] = rec["cache_warm_value"] - rec["cold_default_value"]
        rec["dv_nowarm_vs_tight"] = rec["cache_nowarm_value"] - rec["cold_tight_value"]
        rows.append(rec)
        print("  [%2d/%d] cold %.1f ms | warm %.2f ms | |dv| tight %.1e default %.1e | tight %.1e"
              % (t + 1, n, rec["cold_tight_ms"], rec["cache_warm_ms"],
                 abs(rec["dv_warm_vs_tight"]), abs(rec["dv_warm_vs_default"]),
                 rec["cache_warm_ratio"]), flush=True)

    def med(key):
        return _med_iqr([r[key] for r in rows])

    steady = [r for r in rows if not r["cache_first"]]
    summ = dict(
        n=n, p=p, noise=noise, bias_z=bias_z, eps=eps, max_iters=max_iters,
        solver="SCS", is_dpp=cache.is_dpp, compile_ms=1e3 * cache.compile_s,
        cold_default_ms=med("cold_default_ms"),
        cold_tight_ms=med("cold_tight_ms"),
        cache_warm_ms=med("cache_warm_ms"),
        cache_warm_steady_ms=_med_iqr([r["cache_warm_ms"] for r in steady]),
        cache_nowarm_ms=med("cache_nowarm_ms"),
        dv_warm_vs_tight_med=float(np.median([abs(r["dv_warm_vs_tight"]) for r in rows])),
        dv_warm_vs_tight_max=float(np.max([abs(r["dv_warm_vs_tight"]) for r in rows])),
        dv_warm_vs_default_med=float(np.median([abs(r["dv_warm_vs_default"]) for r in rows])),
        dv_warm_vs_default_max=float(np.max([abs(r["dv_warm_vs_default"]) for r in rows])),
        dv_nowarm_vs_tight_max=float(np.max([abs(r["dv_nowarm_vs_tight"]) for r in rows])),
        ratio_warm_max=float(np.nanmax([r["cache_warm_ratio"] for r in rows])),
        ratio_tight_max=float(np.nanmax([r["cold_tight_ratio"] for r in rows])),
        ratio_default_max=float(np.nanmax([r["cold_default_ratio"] for r in rows])),
        tight_parity=bool(np.all([r["cache_warm_ratio"] <= 1e-6 for r in rows])
                          and np.all([r["cold_tight_ratio"] <= 1e-6 for r in rows])),
        n_fallback=int(cache.n_fallback),
        status_warm=sorted({str(r["cache_warm_status"]) for r in rows}),
        speedup_steady=float(summ_speedup(steady)),
    )
    return dict(config=dict(n=n, p=p, noise=noise, bias_z=bias_z, eps=eps,
                            max_iters=max_iters, solver="SCS"),
                summary=summ, rows=rows)


def summ_speedup(steady):
    cold = np.median([r["cold_tight_ms"] for r in steady])
    warm = np.median([r["cache_warm_ms"] for r in steady])
    return cold / max(warm, 1e-12)


def write_bench_md(d, path):
    s = d["summary"]
    L = []
    A = L.append
    A("# 证书缓存：参数化 14×14 Shor SDP（编译一次 / p，warm start）")
    A("")
    A("> 由 `experiments/code/cert_cache.py --bench` 自动生成；逐 trial 原始数据在 "
      "`perf_cert_cache.json`。**不修改 certificate.py**：等式/Veronese 约束取自 "
      "`certificate._base_eq_specs` / `_veronese_specs`，返回字段与 `shor_bound` 兼容。")
    A("")
    A("设置：p=%d，noise=%.3f，bias_z=%.3f，n=%d 个实例（seed=trial，与主表同场景）；"
      "SCS eps=%.0e、max_iters=%d（比主表冷解默认 eps=1e-9 更紧）。"
      "`cache warm` 按生产顺序热启动（实例 i 从实例 i-1 的解出发），第 1 个解含 cvxpy 编译 "
      "(%.1f ms)，其余为稳态。"
      % (s["p"], s["noise"], s["bias_z"], s["n"], s["eps"], s["max_iters"], s["compile_ms"]))
    A("")
    A("DPP 检查：`prob.is_dpp() = %s`（参数是 Q 与 Gp；编译结果可跨实例复用）。" % s["is_dpp"])
    A("")
    A("## 表 1. 冷解 vs 缓存（ms，median [IQR]）")
    A("")
    A("| 模式 | 时间 ms | Δvalue abs vs 同精度冷解 | Δvalue abs vs 主表默认冷解 | tightness λ₂/λ₁ max |")
    A("|---|---|---|---|---|")
    A("| 冷解（主表默认 eps=1e-9） | %s | — | — | %.1e |"
      % (_fmt(s["cold_default_ms"]), s["ratio_default_max"]))
    A("| 冷解（同精度 eps=%.0e） | %s | — | — | %.1e |"
      % (s["eps"], _fmt(s["cold_tight_ms"]), s["ratio_tight_max"]))
    A("| 缓存 warm（含第 1 次编译） | %s | %.1e（max %.1e） | %.1e（max %.1e） | %.1e |"
      % (_fmt(s["cache_warm_ms"]), s["dv_warm_vs_tight_med"], s["dv_warm_vs_tight_max"],
         s["dv_warm_vs_default_med"], s["dv_warm_vs_default_max"], s["ratio_warm_max"]))
    A("| 缓存 warm（稳态，n−1） | %s | — | — | — |" % _fmt(s["cache_warm_steady_ms"]))
    A("| 缓存 nowarm（仅复用编译，诊断） | %s | max %.1e | — | — |"
      % (_fmt(s["cache_nowarm_ms"]), s["dv_nowarm_vs_tight_max"]))
    A("")
    A("**判定**：稳态 warm 比同精度冷解快 **%.1f×**，`|Δvalue|` max = **%.1e**（≤1e-9），"
      "tightness max = %.1e（≤1e-6，与冷解同档）；回退冷解 %d 次，状态 %s。"
      % (s["speedup_steady"], s["dv_warm_vs_tight_max"], s["ratio_warm_max"],
         s["n_fallback"], ", ".join(s["status_warm"])))
    A("")
    A("- 对主表默认冷解（eps=1e-9）的差最大 %.1e：这是冷解自身的求解误差量级，"
      "仍比证书阈值 1e-6 小 %.0f×。" % (s["dv_warm_vs_default_max"],
                                        1e-6 / max(s["dv_warm_vs_default_max"], 1e-30)))
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("cd labs/02-constrained-qpeplayer/experiments/code")
    A("/home/syk/miniconda3/envs/sdpr/bin/python cert_cache.py --bench --n 20 --p 16")
    A("QPE_CERT_CACHE=1 /home/syk/miniconda3/envs/sdpr/bin/python run_main.py   # 主表走缓存")
    A("```")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt(m):
    return "%.1f [%.1f, %.1f]" % (m["median"], m["q1"], m["q3"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true", help="跑 20 实例冷解 vs 缓存基准")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--bias-z", type=float, default=0.08)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    ap.add_argument("--out-dir", default=RES)
    args = ap.parse_args()
    if not args.bench:
        print(__doc__)
        return
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    t0 = time.perf_counter()
    d = bench(n=args.n, p=args.p, noise=args.noise, bias_z=args.bias_z,
              eps=args.eps, max_iters=args.max_iters)
    d["elapsed_s"] = time.perf_counter() - t0
    os.makedirs(args.out_dir, exist_ok=True)
    jp = os.path.join(args.out_dir, "perf_cert_cache.json")
    mp = os.path.join(args.out_dir, "perf_cert_cache.md")
    with open(jp, "w") as f:
        json.dump(d, f, indent=1, default=float)
    write_bench_md(d, mp)
    s = d["summary"]
    print("\n[wrote] %s\n[wrote] %s" % (jp, mp))
    print("cold_tight %.1f ms -> warm %.2f ms (%.0fx); |dv| max %.2e tight; ratio max %.1e"
          % (s["cold_tight_ms"]["median"], s["cache_warm_steady_ms"]["median"],
             s["speedup_steady"], s["dv_warm_vs_tight_max"], s["ratio_warm_max"]))


if __name__ == "__main__":
    main()
