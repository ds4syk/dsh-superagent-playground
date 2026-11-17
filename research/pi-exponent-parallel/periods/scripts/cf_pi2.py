#!/usr/bin/env python3
"""High precision CF of pi: partial quotients, type tau_n = log(1/eps_n)/log H_n,
   and the comparison with the analytic constructions (tau ~ 0.164)."""
from mpmath import mp, mpf, pi, log, fabs, floor, nint
mp.dps = 2600
N = 2000
x = pi; a=[]
for _ in range(N):
    ai=int(floor(x)); a.append(ai); x-=ai
    if x==0: break
    x=1/x
p=[a[0], a[0]*a[1]+1]; q=[1,a[1]]
for n in range(2,N):
    p.append(a[n]*p[-1]+p[-2]); q.append(a[n]*q[-1]+q[-2])
print("first 30 partial quotients:", a[:30])
print("max a_n, n<=2000:", max(a[1:2000]), " at n=", a[1:2000].index(max(a[1:2000]))+1)
print("max a_n, n<=1990:", max(a[1:1990]))
# tau_n for convergents
taus=[]
for n in range(1,1900):
    eps = q[n]*pi - p[n]
    tau = -log(fabs(eps))/log(q[n])
    taus.append((n,float(tau),a[n+1],float(log(q[n])/log(10))))
print("\n n     a_{n+1}   log10 q_n    tau_n=log(1/eps)/log H   (1+1/tau_n)")
for n,t,an,lq in taus[:12]:
    print(f"{n:4d} {an:9d} {lq:11.3f} {t:20.8f} {1+1/t:14.6f}")
print("...")
for n,t,an,lq in taus[-6:]:
    print(f"{n:4d} {an:9d} {lq:11.3f} {t:20.8f} {1+1/t:14.6f}")
ts=[t for _,t,_,_ in taus]
print("\nmin tau over computed range:", min(ts), " (at n=", taus[ts.index(min(ts))][0],")")
print("max tau:", max(ts))
print("mean tau:", sum(ts)/len(ts))
print("\nthe CRITICAL value is tau=1 (would give mu<=2).")
print("analytic constructions: tau = 0.0438 (Beukers zeta(3)), 0.157 (Salikhov), 0.1638 (ZZ-pi).")
print("ratio needed: 1/0.1638 =", 1/0.1638)
