#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical experiment for the independent attempt on mu(pi)=2.

Computes pi to D decimal digits via Machin's formula
    pi = 16 arctan(1/5) - 4 arctan(1/239)
in fixed-point integer arithmetic, then extracts the continued fraction
    pi = [a_0; a_1, a_2, ...]
and records statistics relevant to the identity
    mu(x) = 2 + limsup_n log(a_{n+1}) / log(q_n).

This file is a NUMERICAL EXPERIMENT ONLY.  It is not part of any proof.
"""

import sys
import time
import json
import math


# ---------------------------------------------------------------------------
# 1. Fixed point pi
# ---------------------------------------------------------------------------

def arctan_inv(x: int, one: int) -> int:
    """arctan(1/x) in fixed point with scale `one` (x >= 2 integer)."""
    x2 = x * x
    term = one // x
    total = term
    k = 0
    sign = -1
    while term:
        term //= x2
        k += 1
        if term == 0:
            break
        total += sign * (term // (2 * k + 1))
        sign = -sign
    return total


def pi_fixed(D: int, guard: int = 60) -> int:
    """floor(pi * 10^D)."""
    one = 10 ** (D + guard)
    a = arctan_inv(5, one)
    b = arctan_inv(239, one)
    return (16 * a - 4 * b) // 10 ** guard


# ---------------------------------------------------------------------------
# 2. Continued fraction
# ---------------------------------------------------------------------------

def cf_terms(pi_scaled: int, D: int, q_limit_digits: int | None = None):
    """Return (a_list, q_list) where q_list[n] is the denominator of the
    n-th convergent.  Stops when the remaining denominator would exceed
    the available precision, or at the exact termination of the Euclidean
    algorithm (which for truncated pi happens near the precision limit)."""
    p, q = pi_scaled, 10 ** D
    a_list = []
    q_list = []
    q_prev, q_cur = 0, 1          # q_{-1}, q_0
    # Reliability threshold.  If pi is known with absolute error eps
    # (one unit in the last of D decimal digits), then the remainder
    # r_n = |q_n*pi_scaled - p_n| carries an error of order q_n*eps while
    # r_n itself is of order 10^D/q_n.  Hence r_n is reliable as long as
    # q_n^2 << 10^D, i.e. q_n < 10^(D/2).  We stop well before that.
    if q_limit_digits is None:
        q_limit_digits = D // 2 - 10
    limit = 10 ** q_limit_digits
    while q:
        a = p // q
        if a < 0:
            break
        a_list.append(a)
        q_next = a * q_cur + q_prev
        q_list.append(q_next)
        if q_next > limit:
            break
        p, q = q, p - a * q
        q_prev, q_cur = q_cur, q_next
    return a_list, q_list


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------

def stats(a, q):
    n_terms = len(a)
    logs = []
    ratios = []
    for n in range(1, n_terms - 1):
        la = math.log(a[n])
        lq = math.log(q[n - 1])
        r = la / lq if lq > 0 else 0.0
        logs.append((n, a[n], lq, r))
        ratios.append((r, n, a[n], q[n - 1]))

    ratios_sorted = sorted(ratios, reverse=True)
    max_a = max(a)
    argmax_a = a.index(max_a)

    # Levy constant: (1/n) log q_n -> pi^2/(12 ln 2)
    levy_est = [(n, math.log(q[n - 1]) / n) for n in range(1, n_terms)]
    tail = [x[1] for x in levy_est if x[0] > n_terms // 2]

    # Khinchin geometric mean of a_1..a_N
    s = sum(math.log(x) for x in a[1:])
    geo = math.exp(s / (n_terms - 1))

    # distribution of partial quotients
    from collections import Counter
    cnt = Counter(a[1:])

    out = {
        "n_terms": n_terms,
        "D_digits": None,
        "max_a": max_a,
        "argmax_a": argmax_a,
        "max_ratio": ratios_sorted[0][0] if ratios_sorted else None,
        "max_ratio_info": ratios_sorted[0] if ratios_sorted else None,
        "top10_ratios": ratios_sorted[:10],
        "levy_mean_tail": sum(tail) / len(tail) if tail else None,
        "levy_expected": math.pi ** 2 / (12 * math.log(2)),
        "khinchin_geo": geo,
        "khinchin_expected": 2.6854520010,
        "count_a_gt_1000": sum(1 for x in a[1:] if x > 1000),
        "count_a_gt_100": sum(1 for x in a[1:] if x > 100),
        "count_a_gt_10": sum(1 for x in a[1:] if x > 10),
        "largest_20_a": sorted(a[1:], reverse=True)[:20],
    }
    return out, logs


# ---------------------------------------------------------------------------
# 4. Minimal SVG scatter plot (no third party libraries)
# ---------------------------------------------------------------------------

def write_svg(path, points, xlabel, ylabel, title,
              width=1000, height=560, xr=None, yr=None):
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = (min(xs), max(xs)) if xr is None else xr
    ymin, ymax = (min(ys), max(ys)) if yr is None else yr
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    ml, mr, mt, mb = 80, 30, 50, 60
    W, H = width - ml - mr, height - mt - mb

    def X(x):
        return ml + (x - xmin) / (xmax - xmin) * W

    def Y(y):
        return mt + H - (y - ymin) / (ymax - ymin) * H

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, sans-serif">')
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>')
    # axes
    parts.append(f'<line x1="{ml}" y1="{mt+H}" x2="{ml+W}" y2="{mt+H}" stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+H}" stroke="#333" stroke-width="1.2"/>')
    # grid
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        yy = Y(y)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+W}" y2="{yy:.2f}" stroke="#ddd" stroke-width="0.6"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-size="11">{y:.3g}</text>')
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        xx = X(x)
        parts.append(f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+H}" stroke="#eee" stroke-width="0.6"/>')
        parts.append(f'<text x="{xx:.2f}" y="{mt+H+20}" text-anchor="middle" font-size="11">{x:.3g}</text>')
    # points
    step = max(1, len(points) // 20000)
    for i in range(0, len(points), step):
        x, y = points[i]
        parts.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="0.8" fill="#1a5fb4" opacity="0.55"/>')
    parts.append(f'<text x="{ml+W/2}" y="{height-18}" text-anchor="middle" font-size="13">{xlabel}</text>')
    parts.append(f'<text x="18" y="{mt+H/2}" text-anchor="middle" font-size="13" '
                 f'transform="rotate(-90 18 {mt+H/2})">{ylabel}</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    tag = sys.argv[2] if len(sys.argv) > 2 else f"d{D}"
    sys.set_int_max_str_digits(2 * D + 1000)

    t0 = time.time()
    pi_scaled = pi_fixed(D)
    t1 = time.time()
    # sanity check the first 50 digits against a known prefix
    known = ("314159265358979323846264338327950288419716939937510582097494"
             "45923078164062862089986280348253421170679")
    got = str(pi_scaled)[:len(known)]
    assert got == known, "pi sanity check failed"

    a, q = cf_terms(pi_scaled, D)
    t2 = time.time()

    st, logs = stats(a, q)
    st["D_digits"] = D
    st["time_pi_s"] = round(t1 - t0, 3)
    st["time_cf_s"] = round(t2 - t1, 3)

    # write CSV: n, a_n, log10(q_n), ratio log(a_n)/log(q_{n-1})
    with open(f"cf_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("n,a_n,log10_q_n,log_a_n_over_log_q_prev\n")
        for n in range(1, len(a) - 1):
            f.write(f"{n},{a[n]},{math.log10(q[n-1]):.6f},"
                    f"{math.log(a[n])/math.log(q[n-1]):.8f}\n")

    # SVG 1: ratio vs log10 q
    pts1 = [(math.log10(q[n - 1]), math.log(a[n]) / math.log(q[n - 1]))
            for n in range(1, len(a) - 1)]
    # exclude a_n = 1 (ratio 0) for visibility? keep all
    write_svg(f"ratio_{tag}.svg", pts1,
              "log10(q_n)", "log(a_n) / log(q_n)",
              "Partial quotients of pi: growth ratio (numerical, D=%d digits)" % D,
              xr=(0, math.log10(q[-1])), yr=(0, 0.25))

    # SVG 2: log10 a_n vs n
    pts2 = [(n, math.log10(a[n])) for n in range(1, len(a) - 1)]
    write_svg(f"logapart_{tag}.svg", pts2,
              "n", "log10(a_n)",
              "Partial quotients of pi (numerical, D=%d digits)" % D)

    # SVG 3: |pi - p_n/q_n| q_n^2 = approx 1/(q_{n+1}/q_n + q_{n-1}/q_n)
    pts3 = []
    for n in range(1, len(a) - 2):
        # exact rational value of q_n * |pi - p_n/q_n| is not needed; use
        # the standard estimate q_n^2 |pi-p_n/q_n| in (1/(a_{n+1}+2), 1/a_{n+1})
        pts3.append((math.log10(q[n - 1]), 1.0 / a[n]))
    write_svg(f"q2err_{tag}.svg", pts3,
              "log10(q_n)", "1 / a_{n+1}  (approx q_n^2 |pi-p_n/q_n|)",
              "Normalised convergent error (numerical, D=%d digits)" % D,
              yr=(0, 1))

    # JSON summary
    st_json = dict(st)
    st_json["top10_ratios"] = [(r, n, an) for (r, n, an, qq) in st["top10_ratios"]]
    with open(f"summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(st_json, f, indent=2, ensure_ascii=False)

    with open(f"summary_{tag}.txt", "w", encoding="utf-8") as f:
        for k, v in st.items():
            f.write(f"{k}: {v}\n")

    print(json.dumps({k: v for k, v in st.items() if k != "top10_ratios"},
                     indent=2, ensure_ascii=False))
    print("top10 ratios (ratio, n, a_n):")
    for row in st["top10_ratios"]:
        print("  ", row)


if __name__ == "__main__":
    main()
