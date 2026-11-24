import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res, q2R, monomials_deg_le, peval
from diffqpep.moment import solve_moment, MON2
from diffqpep.rform import build_Q
from diffqpep.instances import make_pnp
from diffqpep.local import multistart
import cvxpy as cp

OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)

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
    X=cp.Variable((12,12),symmetric=True)
    cs=[X>>0]
    for A,b in orth12(): cs.append(cp.trace(A@X)==b)
    prob=cp.Problem(cp.Minimize(cp.trace(Q@X)),cs)
    prob.solve(solver=cp.SCS,eps=1e-10,max_iters=200000)
    return X.value,prob.value,prob.status

def main():
    rows=[]
    for noise in [0.0,1e-3,1e-2]:
        for seed in range(3):
            d=make_pnp(noise=noise,seed=seed); p,uv=d['p'],d['uv']
            Q=build_Q(p,uv); X,val,st=solve_r12(Q[:12,:12])
            w,V=np.linalg.eigh(X); x=V[:,-1]*np.sqrt(max(w[-1],0))
            R=x[:9].reshape(3,3); t=x[9:12]
            if np.linalg.det(R)<0:  # report sign separately, do not flip
                pass
            orth=float(np.linalg.norm(R.T@R-np.eye(3))); det=float(np.linalg.det(R))
            xt=np.r_[d['R'].flatten(),d['t']]
            e_plus=float(np.linalg.norm(x-xt)); e_minus=float(np.linalg.norm(x+xt))
            rs=qpep_residuals(p,uv); W=gram_W(rs,MON2); J=J_from_res(rs)
            ew=np.linalg.eigvalsh(W); dimker=int(np.sum(np.abs(ew)<1e-9))
            M,bound,stm,mon,dt=solve_moment(W,order=2,eps=1e-8,max_iters=50000)
            eM=np.linalg.eigvalsh(M); rM=int(np.sum(eM>1e-6*max(eM[-1],1e-30)))
            ratio=float(eM[-1]/max(eM[-2],1e-30))
            rows.append(dict(noise=noise,seed=seed,
                rform12=dict(val=float(val),rank=int(np.sum(w>1e-6*max(w[-1],1e-30))),
                            orth=orth,det=det,e_plus=e_plus,e_minus=e_minus,status=st),
                qform=dict(dim_ker_W=dimker,rank_M=rM,lambda_ratio=ratio,
                           bound=float(bound),gap=float(peval(J,d['q'])-bound),status=stm)))
            print(json.dumps(rows[-1])[:500],flush=True)
    with open(os.path.join(OUT,'correct.json'),'w') as f: json.dump(rows,f,indent=2)
    print('saved')
main()
