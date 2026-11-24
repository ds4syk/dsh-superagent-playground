import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res, q2R, monomials_deg_le, peval, qpep_residuals
from diffqpep.moment import solve_moment, MON2, IDX2
from diffqpep.rform import build_Q, solve as solve_r, orth_constraints, E13, depth_forms
from diffqpep.instances import make_pnp, pose_error
from diffqpep.local import multistart

OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)

def rot_from_x(x):
    R=x[:9].reshape(3,3); t=x[9:12]
    if abs(x[12])>1e-12: R=R/x[12]; t=t/x[12]
    return R,t

def atom_extract(X, Q, p, uv, nscan=2000):
    w,V=np.linalg.eigh(X)
    order=np.argsort(w)[::-1]; w=w[order]; V=V[:,order]
    V2=V[:,:2]; lam=np.sqrt(np.maximum(w[:2],0))
    best=None
    for phi in np.linspace(0,2*np.pi,nscan,endpoint=False):
        c,s=np.cos(phi),np.sin(phi)
        Qm=np.array([[c,-s],[s,c]])
        C=np.diag(lam)@Qm
        xs=[V2@C[:,0], V2@C[:,1]]
        rec=[]
        for x in xs:
            R,t=rot_from_x(x)
            orth=np.linalg.norm(R.T@R-np.eye(3))
            cost=float(x@Q@x)
            rec.append((orth,abs(abs(np.linalg.det(R))-1),cost,x,R,t))
        score=sum(r[0] for r in rec)+sum(r[1] for r in rec)
        if best is None or score<best[0]: best=(score,phi,rec)
    score,phi,rec=best
    out=[]
    for orth,det1,cost,x,R,t in rec:
        depth=(R@p.T).T[:,2]+t[2]
        out.append(dict(orth=float(orth),det=float(np.linalg.det(R)),cost=cost,
                        depth=[float(z) for z in depth], x=[float(z) for z in x]))
    return dict(phi=float(phi), atoms=out, eig=[float(z) for z in w[:4]])

def main():
    rows=[]
    for noise in [0.0,1e-3,1e-2]:
        for seed in range(3):
            d=make_pnp(noise=noise,seed=seed); p,uv=d['p'],d['uv']
            Q=build_Q(p,uv); X,val,st=solve_r(Q)
            face=atom_extract(X,Q,p,uv)
            x_true=np.r_[d['R'].flatten(),d['t'],1.0]
            # q-form M_true
            rs=qpep_residuals(p,uv); W=gram_W(rs,MON2); J=J_from_res(rs)
            mon=monomials_deg_le(2); m=np.array([peval({k:1.0},d['q']) for k in mon]) # not correct
            # build m(q)=monomial values
            m=np.array([np.prod([d['q'][i]**k[i] for i in range(4)]) for k in mon])
            Mtrue=np.outer(m,m)
            constr=(abs(Mtrue[IDX2[(0,0,0,0)],IDX2[(0,0,0,0)]]-1),
                    abs(sum(Mtrue[IDX2[tuple(1 if j==i else 0 for j in range(4))],IDX2[tuple(1 if j==i else 0 for j in range(4))]] for i in range(4))-1),
                    float(np.trace(W@Mtrue)))
            M,bound,stm,mon2,dt=solve_moment(W,order=2,eps=1e-8,max_iters=50000)
            wM=np.linalg.eigvalsh(M)
            rows.append(dict(noise=noise,seed=seed,rform=dict(obj=float(val),status=st,
                true_cost=float(x_true@Q@x_true),face=face),
                qform=dict(bound=float(bound),gap=float(peval(J,d['q'])-bound),
                           Mtrue_obj=constr[2],Mtrue_constraints=[float(constr[0]),float(constr[1])],
                           returned_eig=[float(z) for z in wM[-5:]])))
            print(json.dumps(rows[-1])[:600],flush=True)
    with open(os.path.join(OUT,'face.json'),'w') as f: json.dump(rows,f,indent=2)
    print('saved')
main()
