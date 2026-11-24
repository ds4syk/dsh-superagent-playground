import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from scipy.spatial.transform import Rotation as Rot
import cv2
OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)
def norm(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def q_from_R(R):
    q=Rot.from_matrix(R.T).as_quat(); q=np.array([q[3],q[0],q[1],q[2]])
    return q/np.linalg.norm(q)
def gen(seed,noise=1e-3,outlier=False):
    rng=np.random.default_rng(seed); P=rng.uniform(-1,1,size=(8,3))
    q=norm(rng.normal(size=4)); R=q2R(q); t=np.array([0.,0.,3.0])+rng.normal(size=3)*0.2
    Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(8,2))
    if outlier:
        i=rng.integers(0,8); uv[i]+=rng.normal(size=2)*0.05
    return q,uv,R,t,P
def opencv(P,uv,noise):
    K=np.eye(3); d=np.zeros(5); outs=[]
    for flag in [cv2.SOLVEPNP_ITERATIVE,cv2.SOLVEPNP_EPNP,cv2.SOLVEPNP_SQPNP]:
        try:
            ok,rvec,tvec=cv2.solvePnP(P.astype(np.float64),uv.astype(np.float64),K,d,flags=flag)
            if ok:
                R,_=cv2.Rodrigues(rvec); outs.append(R)
        except Exception: pass
    try:
        ok,rvec,tvec,inl=cv2.solvePnPRansac(P.astype(np.float64),uv.astype(np.float64),K,d,reprojectionError=4.0,iterationsCount=100,flags=cv2.SOLVEPNP_EPNP)
        if ok:
            R,_=cv2.Rodrigues(rvec); outs.append(R)
    except Exception: pass
    return outs
starts=branch_starts(0); rows=[]
for outlier in [False,True]:
  for seed in range(300):
    g=gen(seed,outlier=outlier)
    if g is None: continue
    q,uv,R,t,P=g
    J=J_from_res(qpep_residuals(P,uv)); sols=solve_branches(J,starts,keep=4)
    if len(sols)<2: continue
    qs=[qq for f,qq in sols]
    ql,_,_=local_solve(J,np.array([1.,0,0,0]))
    rng=np.random.default_rng(seed+999); h_true=R@np.array([1.,0,0]); h_prior=h_true+0.25*rng.normal(size=3); h_prior/=np.linalg.norm(h_prior)
    hs=np.array([q2R(qq)@np.array([1.,0,0]) for qq in qs]); ch=int(np.argmax(hs@h_prior)); qa=qs[ch]
    n=q2R(qs[0])@np.array([0.,0.,1.]); x=h_prior/np.linalg.norm(h_prior); y=np.cross(n,x); y/=np.linalg.norm(y); x=np.cross(y,n)
    Rc=np.stack([x,y,n],axis=1); qc=q_from_R(Rc); qci,_,_=local_solve(J,qc)
    cops=opencv(P,uv,1e-3); e_oc=min([rot_err(q_from_R(Rc2),q) for Rc2 in cops]) if cops else np.nan
    rows.append(dict(outlier=outlier,seed=seed,local=rot_err(ql,q),global_=rot_err(qs[0],q),cue_local=rot_err(qci,q),atom_cue=rot_err(qa,q),opencv=e_oc,best=min(rot_err(qq,q) for qq in qs)))
with open(os.path.join(OUT,'benchmark3d.json'),'w') as f: json.dump(rows,f,indent=2)
def s(key,out):
    a=np.array([r[key] for r in rows if r['outlier']==out and np.isfinite(r[key])]); return dict(n=int(len(a)),success=float(np.mean(a<0.1)),med=float(np.median(a)))
for out in [False,True]:
    print('outlier',out)
    for k in ['local','global_','cue_local','opencv','atom_cue','best']: print(k,s(k,out))
