#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值实验：用 6000 位十进制 pi 的连分数前缀做统计。

只用整数运算（Chudnovsky + Decimal 开方），不依赖 mpmath。
目的：给出可复现的“有限前缀”数据，说明 μ(pi)=2 的证明不能靠有限计算；
数据与“部分商随机、limsup 未知”的图像一致。
"""
import json
import math
import os
from decimal import Decimal, getcontext

D = 6000  # 十进制位数


def pi_digits(digits: int) -> int:
    """返回 floor(pi * 10**digits)。"""
    getcontext().prec = digits + 50
    C = 426880 * Decimal(10005).sqrt()
    M = 1
    L = 13591409
    X = 1
    K = 6
    S = Decimal(L)
    for i in range(1, digits // 14 + 3):
        M = (K ** 3 - 16 * K) * M // (i ** 3)
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12
    val = C / S
    return int(val * (Decimal(10) ** digits))


def continued_fraction(num: int, den: int, max_terms: int, q_limit: int):
    """num/den 的连分数；返回 (a 列表, p 列表, q 列表)，q 超过 q_limit 即停。"""
    a, p, q = [], [], []
    p_m2, p_m1 = 0, 1
    q_m2, q_m1 = 1, 0
    while den and len(a) < max_terms:
        ai = num // den
        a.append(ai)
        pn = ai * p_m1 + p_m2
        qn = ai * q_m1 + q_m2
        p.append(pn)
        q.append(qn)
        if qn > q_limit:
            break
        num, den = den, num - ai * den
        p_m2, p_m1 = p_m1, pn
        q_m2, q_m1 = q_m1, qn
    return a, p, q


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "research", "pi-exponent", "data")
    os.makedirs(out, exist_ok=True)

    P = pi_digits(D)
    tenD = 10 ** D
    # 安全界：q_n < 10^(D/2)，保证连分数前缀与真实 pi 一致
    q_limit = 10 ** (D // 2)
    a, p, q = continued_fraction(P, tenD, 20000, q_limit)
    # 最后一项可能因截断不可靠，去掉
    a, p, q = a[:-1], p[:-1], q[:-1]

    # 只用 q_n >= 1000 的项统计 log 比值（小 n 的比值没有信息量）
    ratios = [
        (math.log(q[n + 1]) / math.log(q[n]), n)
        for n in range(len(q) - 1)
        if q[n] >= 1000
    ]
    rmax = max(ratios)
    apos = a[1:]
    amax = max(apos)
    amax_idx = apos.index(amax) + 1
    ratio_at_amax = (
        math.log(q[amax_idx + 1]) / math.log(q[amax_idx])
        if amax_idx + 1 < len(q)
        else float("nan")
    )

    stats = {
        "decimal_digits": D,
        "terms_used": len(a),
        "q_limit_digits": D // 2,
        "first_20_partial_quotients": a[:20],
        "max_partial_quotient": amax,
        "max_partial_quotient_index": amax_idx,
        "max_log_ratio_q_ge_1000": rmax[0],
        "max_log_ratio_index": rmax[1],
        "log_ratio_at_max_partial_quotient": ratio_at_amax,
        "log10_q_last": math.log10(q[-1]),
        "average_log_q_over_n": math.log(q[-1]) / (len(q) - 1),
        "levy_constant": math.pi ** 2 / (12 * math.log(2)),
    }
    with open(os.path.join(out, "pi_cf_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out, "pi_cf_prefix.csv"), "w", encoding="utf-8") as f:
        f.write("n,a_n,log10_q_n,log_ratio\n")
        for n in range(len(a)):
            lr = ""
            if n + 1 < len(q) and q[n] > 1:
                lr = f"{math.log(q[n+1])/math.log(q[n]):.6f}"
            f.write(f"{n},{a[n]},{math.log10(q[n]):.6f},{lr}\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
