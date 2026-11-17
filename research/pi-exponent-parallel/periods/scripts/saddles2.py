from mpmath import mp, mpc, mpf, log, exp, fabs, polyroots, findroot
mp.dps=30
Z=[mpc(4,2),mpc(4,-2),mpc(5,0),mpc(6,2),mpc(6,-2)]
# phase phi(x) = 2*sum log(x-z_j) - 3 log x - 3 log(x-10)   (test bunny, n -> order)
def phi(x):
    s=mpc(0)
    for z in Z: s+=2*log(x-z)
    return s-3*log(x)-3*log(x-10)
def dphi(x):
    s=mpc(0)
    for z in Z: s+=2/(x-z)
    return s-3/x-3/(x-10)
# find all roots with many starts
roots=[]
for re in [x/2 for x in range(-8,30)]:
    for im in [x/2 for x in range(-12,13)]:
        g=mpc(re,im)
        try:
            r=findroot(dphi,g)
            if fabs(dphi(r))<1e-20 and abs(r)<1e6:
                if all(fabs(r-o)>1e-10 for o in roots): roots.append(r)
        except Exception: pass
print("saddles of phi (test-bunny phase):")
for r in roots:
    print("  x*=",mp.nstr(r,10),"  Re phi=",mp.nstr(phi(r).real,10),"  |exp phi|=",mp.nstr(exp(phi(r).real),8))
