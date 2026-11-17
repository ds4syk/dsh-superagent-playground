#!/usr/bin/env python3
"""Continued fraction of pi: partial quotients, convergent quality exponents,
   the 'delta' of the CF construction, and the Legendre barrier.
   Uses mpmath only.
"""
from mpmath import mp, mpf, pi, log, nint, floor
import json, sys

mp.dps = 4000

def cf_expansion(x, N):
    a = []
    for _ in range(N):
        ai = int(floor(x))
        a.append(ai)
        x = x - ai
        if x == 0:
            break
        x = 1/x
    return a

N = 1200
a = cf_expansion(pi, N)
print("first 40 partial quotients of pi:", a[:40])

# convergents
p = [a[0], a[0]*a[1]+1]
q = [1, a[1]]
for n in range(2, N):
    p.append(a[n]*p[-1] + p[-2])
    q.append(a[n]*q[-1] + q[-2])

# exact errors e_n = q_n*pi - p_n  (sign (-1)^n)
e = [q[n]*pi - p[n] for n in range(N)]

print("\n n      a_{n+1}   log10 q_n     quality nu_n = -log|f|/log q   delta_n = nu_n-1   q_n*|e_n|")
for n in range(1, min(N-2, 60)):
    f = abs(e[n])/q[n]
    nu = -log(f)/log(q[n])
    print(f"{n:3d} {a[n+1]:10d} {float(log(q[n])/log(10)):10.3f} {float(nu):22.6f} {float(nu)-1:18.6f} {float(q[n]*abs(e[n])):14.6f}")

# statistics
import statistics
mx = max(a[1:600]); print("\nmax partial quotient among first 600:", mx)
print("mean:", float(statistics.mean(a[1:600])))
# Khinchin constant check
from mpmath import euler, exp
print("Khinchin constant:", 2.6854520010653064453)

# empirical mu = 1 + limsup log q_{n+1}/log q_n
best = 0
for n in range(1, N-3):
    r = float(log(q[n+1])/log(q[n]))
    best = max(best, r)
print("max over n<1200 of log q_{n+1}/log q_n =", best, " => empirical mu =", 1+best)
print("last 10 such ratios:", [round(float(log(q[n+1])/log(q[n])),6) for n in range(N-11,N-1)])
