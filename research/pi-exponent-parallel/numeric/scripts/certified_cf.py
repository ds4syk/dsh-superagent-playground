#!/usr/bin/env python3
"""Extract the continued fraction of pi from a decimal expansion, with a
rigorous *certificate* for the extracted prefix.

Idea (endpoint certification)
-----------------------------
Let N = floor(pi * 10^d), so pi lies in the interval I = [N/10^d, (N+1)/10^d].
Compute the (finite, terminating) continued fraction expansions of both
rational endpoints in lockstep.  Claim: if they agree in the first j partial
quotients, then *every* real x in I has those same first j partial quotients
as its continued fraction.

Proof sketch: put psi_k(R) = (p_{k-1} R + p_{k-2})/(q_{k-1} R + q_{k-2}),
so that x = psi_k([a_k; a_{k+1}, ...]).  psi_k is a fractional linear map with
positive integer coefficients, hence monotone and pole-free on the region where
the previous quotients are fixed (x > p_{k-1}/q_{k-1}).  Hence the k-th
"tail" map is monotone; the image of the interval is the interval between the
endpoint values.  By induction, if the two endpoints have the same quotient
a_k = floor(tail) then all interior points do too.  []

Consequently all quotients in the common prefix are the true quotients of pi.
The prefix length is limited by the resolution: the expansion of a rational
with denominator 10^d is stable while q_k^2 << 10^d, i.e. q_k <~ 10^{d/2}.
(That is exactly what the implementation reports.)

Usage:
  python3 certified_cf.py --digits-file ../data/pi_chud_d1000000.txt --d 1000000 \
      --out ../data/cf_pi_d1000000 [--max-quotients K]
"""
import argparse
import json
import sys
import time

sys.set_int_max_str_digits(0)


def read_digits(path, d):
    """Read decimal digits of the fractional part, return int(floor(pi*10^d))."""
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(line.strip())
    frac = "".join(chunks)
    assert len(frac) >= d, (len(frac), d)
    frac = frac[:d]
    return int("3" + frac)          # = floor(pi * 10^d)


def cf_stream(a, b):
    """Yield partial quotients of a/b (a >= 0, b > 0), finite."""
    while b:
        q, r = divmod(a, b)
        yield q
        a, b = b, r


def common_prefix(a1, b1, a2, b2, cap=None):
    """Lockstep comparison of CF(a1/b1) and CF(a2/b2).

    Returns (prefix, diverge_index, taillens)."""
    prefix = []
    g1 = cf_stream(a1, b1)
    g2 = cf_stream(a2, b2)
    idx = 0
    while True:
        if cap is not None and idx >= cap:
            return prefix, None, ("capped", idx)
        try:
            q1 = next(g1)
        except StopIteration:
            return prefix, idx, ("end1", idx)
        try:
            q2 = next(g2)
        except StopIteration:
            return prefix, idx, ("end2", idx)
        if q1 != q2:
            return prefix, idx, ("mismatch", q1, q2)
        prefix.append(q1)
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits-file", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-quotients", type=int, default=None)
    ap.add_argument("--skip-second", action="store_true",
                    help="only run one endpoint (faster, no certificate)")
    args = ap.parse_args()
    d = args.d

    t0 = time.time()
    N = read_digits(args.digits_file, d)
    t1 = time.time()
    print(f"read digits, N has {len(str(N)) - 1} decimal digits ({t1-t0:.1f}s)", flush=True)

    den = 10 ** d
    t0 = time.time()
    if args.skip_second:
        prefix = []
        for q in cf_stream(N, den):
            prefix.append(q)
            if args.max_quotients and len(prefix) >= args.max_quotients:
                break
        info = ("skipped",)
        j = len(prefix)
    else:
        prefix, j, info = common_prefix(N, den, N + 1, den, args.max_quotients)
    t2 = time.time()

    print(f"common prefix length j = {j}  ({info})", flush=True)
    print(f"elapsed for CF: {t2-t0:.1f}s", flush=True)

    with open(args.out + "_quotients.txt", "w") as f:
        f.write(" ".join(map(str, prefix)) + "\n")
    meta = dict(d=d, certified_quotients=j, status=str(info),
                cf_seconds=round(t2 - t0, 2),
                first20=prefix[:20],
                tail10=prefix[-10:] if prefix else [],
                digits_file=args.digits_file)
    with open(args.out + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
