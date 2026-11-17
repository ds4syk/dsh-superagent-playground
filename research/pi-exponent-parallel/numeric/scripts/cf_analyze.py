#!/usr/bin/env python3
"""Statistics of the continued fraction of pi.

Input: a file of partial quotients a_0, a_1, ..., a_{n-1} (space separated),
certified by scripts/certified_cf.py.

The quotients are exact integers.  Everything derived below uses the exact
recursions, evaluated in double precision (relative error ~1e-16 per step, and
the recursions are contracting in the stable direction):

    q_k = a_k q_{k-1} + q_{k-2},     rho_k := q_{k-1}/q_k  in (0,1],
    rho_k = 1/(a_k + rho_{k-1}),
    log q_k = log q_{k-1} + log(a_k + rho_{k-1}),

    R_k := [a_k; a_{k+1}, ...],      R_k = a_k + 1/R_{k+1}   (backward pass)
    |pi - p_k/q_k| = 1/(q_k (q_k R_{k+1} + q_{k-1})),
    Q_k := q_k^2 ||q_k pi|| = 1/(R_{k+1} + rho_k)             (exact identity)

THE mu-STATISTIC
----------------
For any irrational x,  mu(x) = 1 + limsup_{k} log q_{k+1}/log q_k.
Write r_k := log q_{k+1}/log q_k = 1 + log(a_{k+1}+rho_k)/log q_k.
Therefore  mu(pi) <= 2  <=>  T_K := sup_{k>=K} r_k  tends to 1, i.e.
log a_{k+1} = o(log q_k).  T_K is monotone decreasing in K; the data below
reports T_K over the certified range, which is the quantity that matters
(not the running max over k<=N, which is polluted by the tiny k's).

Usage:
  python3 cf_analyze.py --quotients ../data/cf_pi_d1000000_quotients.txt \
      --tag d1e6 --outdir ../data
"""
import argparse
import json
import math
import os

LÉVY = math.pi ** 2 / (12 * math.log(2))          # 1.1865691104... (natural log)
LÉVY10 = LÉVY / math.log(10)                      # 0.51532041... (log10)
KHINCHIN = 2.6854520010653064453097148354817956938203822939944629530511523
LOCHS = math.pi ** 2 / (6 * math.log(2) * math.log(10))  # 1.0306408341...


def load(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def gauss_kuzmin_pmf(k):
    return math.log2(1.0 + 1.0 / (k * (k + 2)))


def gauss_kuzmin_cdf(k):
    return 1.0 - math.log2(1.0 + 1.0 / (k + 1))


def analyse(a):
    n = len(a)
    logq = [0.0] * n
    rho = [0.0] * n
    rho_prev = 0.0
    for k in range(n):
        if k == 0:
            logq[0] = 0.0
            rho[0] = 0.0
        else:
            logq[k] = logq[k - 1] + math.log10(a[k] + rho_prev)
            rho[k] = 1.0 / (a[k] + rho_prev)
        rho_prev = rho[k]
    R = [0.0] * n
    R[n - 1] = float(a[n - 1])
    for k in range(n - 2, -1, -1):
        R[k] = a[k] + 1.0 / R[k + 1]
    Q = [0.0] * (n - 1)
    r = [0.0] * (n - 1)
    for k in range(n - 1):
        Q[k] = 1.0 / (R[k + 1] + rho[k])
        r[k] = logq[k + 1] / logq[k] if logq[k] > 0 else float("nan")
    return dict(n=n, logq=logq, rho=rho, R=R, Q=Q, r=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotients", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="../data")
    args = ap.parse_args()
    a = load(args.quotients)
    d = analyse(a)
    n, logq, rho, R, Q, r = d["n"], d["logq"], d["rho"], d["R"], d["Q"], d["r"]
    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag
    st = {"tag": tag, "n_quotients": n, "first_30": a[:30]}

    known = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2,
             1, 84, 2, 1, 1, 15, 3, 13, 1, 4]
    st["matches_known_first_30"] = a[:30] == known

    # ---------------- distribution vs Gauss-Kuzmin ----------------
    maxk = min(200, n)
    obs = [0] * (maxk + 2)
    for x in a:
        obs[min(x, maxk + 1)] += 1
    exp = [0.0] * (maxk + 2)
    for k in range(1, maxk + 1):
        exp[k] = n * gauss_kuzmin_pmf(k)
    exp[maxk + 1] = n * (1.0 - gauss_kuzmin_cdf(maxk))
    chi2, df = 0.0, 0
    for k in range(1, maxk + 2):
        if exp[k] >= 5:
            chi2 += (obs[k] - exp[k]) ** 2 / exp[k]
            df += 1
    cum, ks = 0, 0.0
    for k in range(1, maxk + 1):
        cum += obs[k]
        ks = max(ks, abs(cum / n - gauss_kuzmin_cdf(k)))
        if k > 1:
            ks = max(ks, abs((cum - obs[k]) / n - gauss_kuzmin_cdf(k - 1)))
    st["gauss_kuzmin"] = dict(chi2=round(chi2, 3), dof=df - 1,
                              chi2_per_dof=round(chi2 / max(df - 1, 1), 4),
                              ks_D=ks, ks_critical_5pct=1.36 / math.sqrt(n))
    # tail counts
    tails = {}
    for X in (10, 100, 1000, 10_000, 100_000):
        o = sum(1 for x in a if x > X)
        e = n * (1.0 - gauss_kuzmin_cdf(X))
        tails[f"a>{X}"] = dict(observed=o, gauss_kuzmin_expected=round(e, 1),
                               ratio=round(o / e, 3) if e else None)
    st["tail_counts"] = tails

    # ---------------- Lévy / Khinchin / Lochs ----------------
    st["levy"] = dict(observed_log10_qn_over_n=logq[n - 1] / (n - 1),
                      predicted_log10=LÉVY10, log10_q_n_last=logq[n - 1])
    st["khinchin"] = dict(geometric_mean=math.exp(sum(math.log(x) for x in a) / n),
                          predicted=KHINCHIN)
    # Lochs: decimal digits of accuracy of the convergent p_k/q_k, per term,
    # for k = n-2 (the last index for which R_{k+1} is known).
    kk = n - 2
    acc = 2 * logq[kk] - math.log10(Q[kk])
    st["lochs"] = dict(accuracy_digits_per_term=acc / kk, predicted=LOCHS)

    # ---------------- records of partial quotients ----------------
    recs, best = [], 0
    for k, x in enumerate(a):
        if x > best:
            best = x
            recs.append(dict(index=k, a=x,
                             log10_q_prev=logq[k - 1] if k else 0.0))
    st["n_records"] = len(recs)
    st["records"] = recs
    st["records_top15"] = sorted(recs, key=lambda t: -t["a"])[:15]

    # ---------------- top-20 largest quotients ----------------
    idx = sorted(range(n), key=lambda k: -a[k])[:20]
    st["top20"] = [dict(index=k, a=a[k],
                        log10_q_prev=(logq[k - 1] if k else 0.0),
                        r_prev=(r[k - 1] if k >= 1 else None),
                        r_excess_prev=(r[k - 1] - 1.0 if k >= 1 else None))
                   for k in idx]

    # ---------------- T_K := sup_{k>=K} r_k   (the mu statistic) ----------
    T = [0.0] * (n - 1)
    am = [0] * (n - 1)
    run, arg = -1e9, -1
    for k in range(n - 2, -1, -1):
        if r[k] > run:
            run, arg = r[k], k
        T[k], am[k] = run, arg
    grid = []
    K = 2
    while K < n - 2:
        grid.append(K)
        K = int(K * 1.25) + 1
    rows = []
    for K in grid:
        k = K
        rows.append(dict(K=K, T_K=T[k], T_minus_1=T[k] - 1.0,
                         argmax_index=am[k],
                         a_at_argmax=a[am[k] + 1],
                         log10_q_at_argmax=logq[am[k]],
                         excess_times_K_over_log10K=(T[k] - 1.0) * K / math.log10(K)))
    st["T_K_curve"] = rows
    st["T_K_final"] = rows[-1] if rows else None
    st["mu_bound_finite"] = 1.0 + T[2]
    st["mu_bound_finite_note"] = (
        "1 + T_2 is the bound obtained from the certified prefix alone: "
        "mu(pi) <= 1 + sup_{k>=2} r_k; the supremum here is over the certified "
        "range only and is attained at small k, so this number is NOT an "
        "estimate of mu, it is an upper bound valid for the certified range.")

    # ---------------- best-quality convergents (Q_k near 1) --------------
    # A_k = q_k^2|pi-p_k/q_k| in (1/(a_{k+1}+2), 1/a_{k+1}): SMALL means a
    # particularly good approximation (the dangerous direction for mu), LARGE
    # (close to 1) means a comparatively poor convergent.
    qidx = sorted(range(len(Q)), key=lambda k: Q[k])[:10]
    st["smallest_A_most_dangerous"] = [
        dict(index=k, A_k=Q[k], log10_q_k=logq[k], next_a=a[k + 1],
             exponent_2_minus_log10A_over_log10q=2.0 - math.log10(Q[k]) / logq[k],
             lower_bound_exponent=2.0 + math.log10(a[k + 1] + 2) / logq[k])
        for k in qidx]
    qidx2 = sorted(range(len(Q)), key=lambda k: -Q[k])[:10]
    st["largest_A_worst_convergents"] = [
        dict(index=k, A_k=Q[k], log10_q_k=logq[k], next_a=a[k + 1]) for k in qidx2]

    # ---------------- the largest r_k (most dangerous for mu) -------------
    ridx = sorted(range(len(r)), key=lambda k: -r[k])[:10]
    st["largest_r_k"] = [dict(index=k, r_k=r[k], excess=r[k] - 1.0,
                              a_next=a[k + 1], log10_q_k=logq[k]) for k in ridx]

    # ---------------- write files ----------------
    with open(os.path.join(args.outdir, f"cf_{tag}_stats.json"), "w") as f:
        json.dump(st, f, indent=2)
    with open(os.path.join(args.outdir, f"cf_{tag}_T_K.csv"), "w") as f:
        f.write("K,T_K,T_minus_1,argmax_index,a_at_argmax,log10_q_at_argmax,"
                "excess_x_K_over_log10K\n")
        for w in rows:
            f.write(f"{w['K']},{w['T_K']:.15g},{w['T_minus_1']:.6g},"
                    f"{w['argmax_index']},{w['a_at_argmax']},"
                    f"{w['log10_q_at_argmax']:.12g},"
                    f"{w['excess_times_K_over_log10K']:.6g}\n")
    with open(os.path.join(args.outdir, f"cf_{tag}_records.csv"), "w") as f:
        f.write("index,a_k,log10_q_{k-1}\n")
        for rr in recs:
            f.write(f"{rr['index']},{rr['a']},{rr['log10_q_prev']:.12g}\n")
    with open(os.path.join(args.outdir, f"cf_{tag}_a.txt"), "w") as f:
        f.write(" ".join(map(str, a)) + "\n")

    short = {k: v for k, v in st.items()
             if k in ("tag", "n_quotients", "matches_known_first_30",
                      "gauss_kuzmin", "tail_counts", "levy", "khinchin",
                      "lochs", "n_records", "T_K_final", "mu_bound_finite")}
    print(json.dumps(short, indent=2))
    print("\nT_K (tail sup of r_k): K, T_K, argmax, a_at_argmax, log10 q")
    for w in rows[::max(1, len(rows) // 12)] + [rows[-1]]:
        print(f"  K={w['K']:8d}  T_K={w['T_K']:.9f}  (T_K-1)={w['T_minus_1']:.3e}"
              f"  argmax={w['argmax_index']:8d}  a={w['a_at_argmax']:8d}"
              f"  log10q={w['log10_q_at_argmax']:.1f}")
    print("\nfull stats:", os.path.join(args.outdir, f"cf_{tag}_stats.json"))


if __name__ == "__main__":
    main()
