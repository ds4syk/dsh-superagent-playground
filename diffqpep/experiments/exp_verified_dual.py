import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R, gram_W, monomials_deg_le
from diffqpep.rform import build_Q
from diffqpep.moment import solve_moment, MON2
import cvxpy as cp
OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)
P=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def normalize(q): return q/np.linalg.norm(q)
def gen(seed,noise=5e-4):
    rng=np.random.default_rng(seed); q=normalize(rng.normal(size=4)); R=q2R(q)
    t=np.array([0.,0.,3.0])+rng.normal(size=3)*0.2
    Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(4,2))
    return q,uv,R,t
def orth12():
    out=[]
    for a in range(3):
        for b in range(a,3):
            A=np.zeros((12,12))
            for k in range(3):
                A[3*k+a,3*k+b]+=0.5; A[3*k+b,3*k+a]+=0.5
            out.append((A,1.0 if a==b else 0.0))
    return out
def solve_r12(Q):
    X=cp.Variable((12,12),symmetric=True); cons=[X>>0]
    for A,b in orth12(): cons.append(cp.trace(A@X)==b)
    prob=cp.Problem(cp.Minimize(cp.trace(Q@X)),cons)
    prob.solve(solver=cp.SCS,eps=1e-11,max_iters=200000)
    H=np.zeros((12,12))
    for con,A in zip(cons[1:],orth12()):
        H -= float(con.dual_value) * A[0]
    H += Q
    # dual of the PSD constraint is Z>=0; dual of equality adds H; the SDP dual is H (PSD block)
    # The equality duals are unconstrained; project the equality-dual part:
    Hp=H
    w,V=np.linalg.eigh(Hp); Hpsd=V@np.diag(np.maximum(w,0))@V.T
    Xv=X.value
    return dict(val=float(prob.value),xrank=int(np.sum(np.linalg.eigvalsh(Xv)>1e-6*max(np.linalg.eigvalsh(Xv)[-1],1e-30))),
                hmin=float(w.min()),hpsd_min=float(np.linalg.eigvalsh(Hpsd).min()),
                trHX=float(np.trace(Hpsd@Xv)),gap=float(prob.value-prob.value),status=str(prob.status))
rows=[]
for seed in range(20):
    g=gen(seed)
    if g is None: continue
    q,uv,R,t=g
    J=J_from_res(qpep_residuals(P,uv)); Q=build_Q(P,uv)[:12,:12]
    r=solve_r12(Q); r['seed']=seed; rows.append(r)
with open(os.path.join(OUT,'verified_dual.json'),'w') as f: json.dump(rows,f,indent=2)
print('n',len(rows))
for k in ['xrank','hmin','hpsd_min','trHX','status']:
    print(k, [r[k] for r in rows[:5]])
print('median hmin', np.median([r['hmin'] for r in rows]), 'median trHX', np.median([r['trHX'] for r in rows]))
