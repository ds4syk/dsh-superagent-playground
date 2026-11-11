"""SDPR 前向的 SDP 规模-时间标定：随机 Shor 型 SDP，SCS 求解。

用于给 02 页提供"实测的 SDP 规模 vs 时间"曲线（与 QPEP 的常数模板代价对照）。
注意：这是**随机稠密实例**的标定，不是 SDPR 论文的具体实验设置；结论只用于
数量级/增长趋势。
"""

import json
import time

import numpy as np
import cvxpy as cp


def build(n, m, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    Q = B.T @ B + 0.01 * np.eye(n)   # PSD ⇒ 目标有下界
    Q = (Q + Q.T) / 2
    As = []
    for _ in range(m):
        A = rng.normal(size=(n, n))
        A = (A + A.T) / 2
        As.append(A)
    A0 = np.zeros((n, n))
    A0[0, 0] = 1.0
    # 用一个已知可行点构造一致的右端项，保证 SDP 可行
    y0 = rng.normal(size=n)
    X0 = np.outer(y0, y0) + 0.1 * np.eye(n)
    b = [float(np.trace(A @ X0)) for A in As]
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, cp.trace(A0 @ X) == 1]
    for A, bi in zip(As, b):
        cons.append(cp.trace(A @ X) == bi)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    return prob


def time_solve(prob, reps=8):
    # 第一次求解包含编译/缓存构建，丢弃
    prob.solve(solver=cp.SCS, eps=1e-9, verbose=False)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        prob.solve(solver=cp.SCS, eps=1e-9, verbose=False, warm_start=True)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(prob.value)


def main():
    out = []
    for n in [8, 13, 20, 30, 40]:
        m = max(2, n // 3)
        prob = build(n, m, seed=n)
        t, val = time_solve(prob)
        out.append(dict(n=n, m=m, sdp_var=n * (n + 1) // 2, sec=t, val=val))
        print(f"n={n:3d} m={m:3d} N={n*(n+1)//2:4d}  {t*1e3:8.2f} ms  val={val:.4g}", flush=True)
    json.dump(out, open("results/sdp_scaling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
