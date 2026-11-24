import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
try:
    import cv2
except Exception:
    cv2=None
OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)
P=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def norm(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def gen(seed, noise=5e-4):
    rng=np.random.default_rng(seed); q=norm(rng.normal(size=4)); R=q2R(q)
    t=np.array([0.,0.,3.0])+rng.normal(size=3)*0.2
    Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(4,2))
    return q,uv,R,t
def opencv_pose(uv):
    if cv2 is None: return None
    K=np.eye(3); dist=np.zeros(5)
    try:
        ok,rvec,tvec=cv2.solvePnP(P.astype(np.float64), uv.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok: return None
        R,_=cv2.Rodrigues(rvec); return R,tvec.ravel()
    except Exception:
        return None
def q_from_R(R):
    return rot_from_R(R)

def rot_from_R(R):
    # quaternion from rotation matrix (Shepperd)
    tr=np.trace(R)
    if tr>0:
        s=np.sqrt(tr+1.0)*2; q=np.array([0.25*s,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s])
    else:
        i=np.argmax(np.diag(R))
        if i==0:
            s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q=np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
        elif i==1:
            s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q=np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
        else:
            s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q=np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])
    return norm(q)
rows=[]; starts=branch_starts(0)
for seed in range(400):
    g=gen(seed)
    if g is None: continue
    q,uv,R,t=g
    J=J_from_res(qpep_residuals(P,uv)); sols=solve_branches(J,starts,keep=2)
    if len(sols)<2: continue
    (f1,q1),(f2,q2)=sols[0],sols[1]
    ql,fl,_=local_solve(J,np.array([1.,0,0,0]))
    # heading cue
    rng=np.random.default_rng(seed+999); h_true=R@np.array([1.,0,0]); h=np.zeros((2,3))
    for j,qq in enumerate([q1,q2]): h[j]=q2R(qq)@np.array([1.,0,0])
    h_prior=h_true+0.25*rng.normal(size=3); h_prior/=np.linalg.norm(h_prior)
    scores=[float(h_prior@h[0]),float(h_prior@h[1])]
    choose=int(np.argmax(scores)); q_atom=[q1,q2][choose]
    # no-cue global (lowest cost q1), local, opencv
    cop=opencv_pose(uv)
    e_local=rot_err(ql,q); e_glob=rot_err(q1,q); e_atom_cue=rot_err(q_atom,q)
    e_oc=rot_err(rot_from_R(cop[0]),q) if cop is not None else np.nan
    # cue-initialized local baseline: heading cue + branch normal (same for both)
    n=q2R(q1)@np.array([0.,0.,1.]); x=h_prior/np.linalg.norm(h_prior); y=np.cross(n,x); y/=np.linalg.norm(y); x=np.cross(y,n)
    Rc=np.stack([x,y,n],axis=1); qc=rot_from_R(Rc)
    qci,fci,_=local_solve(J,qc)
    e_cue_local=rot_err(qci,q)
    rows.append(dict(seed=seed,local=e_local,global_=e_glob,cue_local=e_cue_local,atom_cue=e_atom_cue,opencv=e_oc,
                     f1=float(f1),f2=float(f2),score_gap=abs(scores[0]-scores[1])))
with open(os.path.join(OUT,'cue_benchmark.json'),'w') as f: json.dump(rows,f,indent=2)
def summ(key):
    a=np.array([r[key] for r in rows if np.isfinite(r[key])])
    return dict(n=int(len(a)),success=float(np.mean(a<0.1)),err_med=float(np.median(a)),err_mean=float(np.mean(a)))
print('n',len(rows))
for k in ['local','global_','cue_local','opencv','atom_cue']: print(k,summ(k))
print('score gap median',np.median([r['score_gap'] for r in rows]))
