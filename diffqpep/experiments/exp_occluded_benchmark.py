import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from scipy.spatial.transform import Rotation as Rot
import cv2
P4=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def norm(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def q_from_R(R):
    q=Rot.from_matrix(R.T).as_quat(); q=np.array([q[3],q[0],q[1],q[2]])
    return q/np.linalg.norm(q)
def gen(seed,noise=5e-4):
    rng=np.random.default_rng(seed); q=norm(rng.normal(size=4)); R=q2R(q)
    t=np.array([0.,0.,3.0])+rng.normal(size=3)*0.2
    keep=rng.choice(4,size=3,replace=False)
    P=P4[keep]; Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(3,2))
    return q,uv,R,t,P
def opencv_p3p(P,uv):
    K=np.eye(3); d=np.zeros(5)
    try:
        n,rvecs,tvecs,errs=cv2.solvePnPGeneric(P.astype(np.float64),uv.astype(np.float64),K,d,flags=cv2.SOLVEPNP_P3P)
        outs=[]
        for rv,tv in zip(rvecs,tvecs):
            R,_=cv2.Rodrigues(rv); outs.append(R)
        return outs
    except Exception: return []
starts=branch_starts(0); rows=[]
for seed in range(500):
    g=gen(seed)
    if g is None: continue
    q,uv,R,t,P=g
    J=J_from_res(qpep_residuals(P,uv)); sols=solve_branches(J,starts,keep=4)
    if len(sols)<2: continue
    qs=[qq for f,qq in sols]; fs=[f for f,qq in sols]
    ql,fl,_=local_solve(J,np.array([1.,0,0,0]))
    # cue heading
    rng=np.random.default_rng(seed+999); h_true=R@np.array([1.,0,0]); h_prior=h_true+0.25*rng.normal(size=3); h_prior/=np.linalg.norm(h_prior)
    hs=np.array([q2R(qq)@np.array([1.,0,0]) for qq in qs]); ch=int(np.argmax(hs@h_prior)); q_atom=qs[ch]
    # cue-init local
    n=q2R(qs[0])@np.array([0.,0.,1.]); x=h_prior/np.linalg.norm(h_prior); y=np.cross(n,x); y/=np.linalg.norm(y); x=np.cross(y,n)
    Rc=np.stack([x,y,n],axis=1); qc=q_from_R(Rc); qci,_,_=local_solve(J,qc)
    # opencv p3p: choose solution with min reprojection? all zero; choose first
    cops=opencv_p3p(P,uv); e_oc=[rot_err(q_from_R(Rc2),q) for Rc2 in cops]
    rows.append(dict(seed=seed,local=rot_err(ql,q),global_=rot_err(qs[0],q),cue_local=rot_err(qci,q),atom_cue=rot_err(q_atom,q),
                     opencv=float(min(e_oc)) if e_oc else np.nan, n_branch=len(qs), best=float(min(rot_err(qq,q) for qq in qs))))
with open(os.path.join(os.path.dirname(__file__),'..','results','occluded_benchmark.json'),'w') as f: json.dump(rows,f,indent=2)
def s(k):
    a=np.array([r[k] for r in rows if np.isfinite(r[k])]); return dict(n=int(len(a)),success=float(np.mean(a<0.1)),med=float(np.median(a)),mean=float(np.mean(a)))
print('n',len(rows))
for k in ['local','global_','cue_local','opencv','atom_cue','best']: print(k,s(k))
print('branches median',np.median([r['n_branch'] for r in rows]))
