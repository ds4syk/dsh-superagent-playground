import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=os.path.join(os.path.dirname(__file__),'..','results'); FIG=os.path.join(os.path.dirname(__file__),'..','figures'); os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)
def J(x,th): return (x*x-1)**2+th*x
def dJ(x,th): return 4*x*(x*x-1)+th
def d2J(x): return 12*x*x-4
def branches(th):
    roots=np.roots([4,0,-4,th]); out=[]
    for r in roots:
        if abs(r.imag)<1e-10:
            x=r.real; h=d2J(x)
            if h>0: out.append((x,J(x,th),h))
    return sorted(out,key=lambda z:z[0])
th_fold=8/(3*np.sqrt(3))
ths=np.linspace(1.20,th_fold-1e-6,120)
rows=[]
for th in ths:
    b=branches(th)
    if len(b)<2: continue
    x1=b[0][0]; x2=b[1][0]; h1=b[0][2]; h2=b[1][2]
    rows.append(dict(theta=float(th),x1=float(x1),x2=float(x2),h1=float(h1),h2=float(h2),
                     deriv=float(-1/h2),gap=float(abs(b[1][1]-b[0][1]))))
# FD vs analytic near the fold
fd=[]
for d in [1e-1,3e-2,1e-2,3e-3,1e-3,1e-4]:
    th=th_fold-d
    b1=branches(th); b2=branches(th-d/2)
    if len(b1)<2 or len(b2)<2: continue
    x1=b1[1][0]; xp=b2[1][0]
    fd.append(dict(dist=float(d),fd=float((x1-xp)/(d/2)),deriv=float(-1/b1[1][2]),h=float(b1[1][2])))
with open(os.path.join(OUT,'fold.json'),'w') as f: json.dump(dict(fold_theta=float(th_fold),rows=rows,fd=fd),f,indent=2)
plt.figure(figsize=(5.2,3.2))
plt.semilogy([r['theta'] for r in rows],[abs(r['deriv']) for r in rows if abs(r['deriv'])<1e12],color='#2b5d8a')
plt.axvline(th_fold,color='#b56576',ls=':'); plt.xlabel(r'$\theta$'); plt.ylabel(r'$|dx^\star/d\theta|$')
plt.title('fold: derivative diverges',fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_fold.png'),dpi=200); plt.close()
print('fold',th_fold,'last deriv',rows[-1]['deriv'],'fd',fd)
