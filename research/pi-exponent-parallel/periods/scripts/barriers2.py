#!/usr/bin/env python3
from mpmath import mp, mpf, pi, log, fabs, nint, floor, nstr
from math import gcd
mp.dps = 500

N = 900
x = pi; a = []
for _ in range(N):
    ai = int(floor(x)); a.append(ai)
    x -= ai
    if x == 0: break
    x = 1/x
p=[a[0], a[0]*a[1]+1]; q=[1, a[1]]
for n in range(2,N):
    p.append(a[n]*p[-1]+p[-2]); q.append(a[n]*q[-1]+q[-2])

print("(A) Legendre barrier  |pi-p/q| < 1/(2q^2) => p/q convergent; also |pi-p/q|<1/q^2 =>")
print("    convergent or semiconvergent.  Scanning q <= 10^5.")
conv = {}
for n in range(N):
    if q[n] <= 100000:
        conv[(p[n],q[n])] = n
# semiconvergents: (p_n + j p_{n-1})/(q_n + j q_{n-1})
semi = {}
for n in range(1,N):
    for j in range(0, a[n+1]+1 if n+1<len(a) else 1):
        qq = q[n] + j*q[n-1]; pp = p[n] + j*p[n-1]
        if qq <= 100000:
            semi[(pp,qq)] = (n,j)
bad1=[]; bad2=[]
for qq in range(1, 100001):
    pp = int(nint(qq*pi))
    g = gcd(abs(pp),qq); pr,qr = pp//g, qq//g
    f = fabs(pi - mpf(pp)/qq)
    if f < mpf(1)/(2*qq*qq) and (pr,qr) not in conv: bad1.append((pp,qq,pr,qr))
    if f < mpf(1)/(qq*qq) and (pr,qr) not in conv and (pr,qr) not in semi: bad2.append((pp,qq,pr,qr))
print("    violations of Legendre (1/2q^2):", len(bad1), bad1[:3])
print("    violations of 1/q^2 => (semi)convergent:", len(bad2), bad2[:3])

print("\n(B) quality/height of convergents:  eps_n*q_n = q_n|q_n pi - p_n|  (two-sided 'type')")
print("      n   a_{n+1}   log10 q_n     nu_n=quality   eps_n q_n    1/a_{n+1}")
for n in [1,2,3,10,20,50,100,200,400,700]:
    if n+1 >= len(a): break
    eps = q[n]*pi - p[n]
    nu = -log(fabs(eps)/q[n])/log(q[n])
    print(f"    {n:4d} {a[n+1]:8d} {float(log(q[n])/log(10)):11.3f} {float(nu):13.6f} {float(fabs(eps)*q[n]):12.7f} {1/a[n+1]:12.7f}")

print("\n(C) quadratic route (d=2 Nesterenko degeneration): minimal polynomial of the")
print("    quadratic irrational with periodic CF (a_1..a_n):")
a0 = a[0]
print("      n   log10 H(P)   -log|P(pi)|/log H(P)    |P(pi)|*H(P)      |pi-r_n|")
for n in [10,20,40,80,160,320,600]:
    if n+1 >= len(q): break
    pn_ = p[n] - a0*q[n]; qn_ = q[n]
    pn1 = p[n-1] - a0*q[n-1]; qn1 = q[n-1]
    A2 = qn_; B2 = qn1 - pn_; C2 = -pn1
    co = [A2, B2 - 2*a0*A2, C2 - B2*a0 + A2*a0*a0]
    H = max(abs(c) for c in co)
    val = co[0]*pi*pi + co[1]*pi + co[2]
    disc = mpf(co[1])**2 - 4*mpf(co[0])*co[2]
    r = (-mpf(co[1]) + mp.sqrt(disc))/(2*mpf(co[0]))
    print(f"    {n:4d} {float(log(H)/log(10)):11.3f} {float(-log(fabs(val))/log(H)):21.5f} {float(fabs(val)*H):16.4f} {float(fabs(pi-r)):12.3e}")
print("    => |P(pi)| ~ H^{-1}: any degree-2 measure must have kappa >= 1; the route needs kappa=1." )
print("    factor form P=x(qx-p): |P(pi)|*H = pi*eps_n*q_n (same column above) -> kappa >= 1 again.")

print("\n(D) Zeilberger-Zudilin reconstruction:  tau = log(1/rho)/lambda ,  mu <= 1+1/tau")
LR = mpf('3.524772923965284')          # = -Re phi(x*) from saddles2.py / salikhov_saddle.py
for lab, d in [("crude  (mu<=10.747747)", mpf('0.102587803338')),
               ("refined(mu<=7.103205)  ", mpf('0.163848329730'))]:
    print(f"    {lab}:  tau={nstr(d,12)}  =>  implied height growth lambda = {nstr(LR/d,10)}")
print("    (the crude bookkeeping has the LARGER height growth 34.36^n; the extra divisor")
print("     F(n) divides it down to 21.51^n.  This arithmetic input is Zeilberger-Zudilin's,")
print("     not re-derived here; what is verified here is log(1/rho)=3.524772923965284.)")
