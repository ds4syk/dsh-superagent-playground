#!/usr/bin/env python3
"""Asymptotics of the Salikhov / Zeilberger-Zudilin period integral (test bunny)

   I_n = -i int_{4-2i}^{4+2i} prod_j (x-z_j)^{2n} / (x^{3n+1}(x-10)^{3n+1}) dx ,
   z_j = 4+2i, 4-2i, 5, 6+2i, 6-2i .

Phase phi(x) = 2 sum_j log(x-z_j) - 3 log x - 3 log(x-10);  |I_n| ~ rho^n, rho = |e^{phi(x_s)}|.

Outputs
 (a) all saddles of phi with Re phi  (four complex ones are degenerate: same Re);
 (b) high precision complex quadrature along the contour, with *local slopes*
     Delta log|I_n| / Delta n, which approach Re phi(x_s) = -3.5247729.
     The approach is irregular because the four saddles have equal Re and their
     phases interfere; the slopes nevertheless settle at -3.52...-3.57.
"""
from mpmath import mp, mpf, mpc, log, exp, fabs, quad, findroot, nstr
mp.dps = 200

Z = [mpc(4,2), mpc(4,-2), mpc(5,0), mpc(6,2), mpc(6,-2)]

def phi(x):
    s = mpc(0)
    for z in Z: s += 2*log(x - z)
    return s - 3*log(x) - 3*log(x - 10)

def dphi(x):
    s = mpc(0)
    for z in Z: s += 2/(x - z)
    return s - 3/x - 3/(x - 10)

def integrand(x, n):
    return -1j*exp(n*phi(x))/(x*(x - 10))

def I_quad(n):
    return quad(lambda t: integrand(mpc(4, t), n)*1j,
                [-2, -1.6, -1, -0.5, 0, 0.5, 1, 1.6, 2])

print("(a) saddles of phi:")
roots = []
for re in [x/2 for x in range(-8, 32)]:
    for im in [x/2 for x in range(-12, 13)]:
        try:
            r = findroot(dphi, mpc(re, im))
            if fabs(dphi(r)) < mpf(10)**(-25) and abs(r) < 10**4:
                if all(fabs(r - o) > 1e-9 for o in roots): roots.append(r)
        except Exception:
            pass
for r in sorted(roots, key=lambda s: s.real):
    print(f"    x* = {nstr(r,12):>32}   Re phi = {nstr(phi(r).real,12):>14}   |e^phi| = {nstr(exp(phi(r).real),10)}")

xs = findroot(dphi, mpc('4.646','1.430'), tol=mpf(10)**(-40))
L = -phi(xs).real
print(f"\n    relevant saddle x* = {nstr(xs,18)}")
print(f"    log(1/rho) = -Re phi(x*) = {nstr(L,18)}    rho = {nstr(exp(-L),18)}")

print("\n(b) quadrature and local slopes:")
prev = None
for n in [8, 10, 12, 14, 16, 20, 24, 28]:
    v = log(fabs(I_quad(n)))
    sl = "" if prev is None else f"   local slope = {nstr((v-prev[1])/(n-prev[0]),9)}"
    print(f"    n={n:3d}  log|I_n| = {nstr(v,14):>20}{sl}")
    prev = (n, v)
print(f"\n    saddle prediction  -Re phi(x*) = {nstr(L,12)}")
