import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from scipy.spatial.transform import Rotation as Rot
import cv2
P=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def norm(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def q_from_R(R):
    q=Rot.from_matrix(R.T).as_quat(); q=np.array([q[3],q[0],q[1],q[2]])
    return q/np.linalg.norm(q)
def gen(seed,noise=5e-4):
    rng=np.random.default_rng(seed); q=norm(rng.normal(size=4)); R=q2R(q)
    t=np.array([0.,0.,3.0])+rng.normal(size=3)*0.2
    Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(4,2))
    return q,uv,R,t
def opencv(uv):
    K=np.eye(3); d=np.zeros(5)
    try:
        ok,rvec,tvec=cv2.solvePnP(P.astype(np.float64),uv.astype(np.float64),K,d,flags=cv2.SOLVEPNP_IPPE)
        if not ok: return None
        R,_=cv2.Rodrigues(rvec); return R
    except Exception: return None
starts=branch_starts(0); rows=[]; rng=np.random.default_rng(12345)
for seed in range(400):
    g=gen(seed)
    if g is None: continue
    q,uv,R,t=g; J=J_from_res(qpep_residuals(P,uv)); sols=solve_branches(J,starts,keep=2)
    if len(sols)<2: continue
    (f1,q1),(f2,q2)=sols[0],sols[1]
    ql,_,_=local_solve(J,np.array([1.,0,0,0]))
    cop=opencv(uv); e_oc=rot_err(q_from_R(cop),q) if cop is not None else np.nan
    n=q2R(q1)@np.array([0.,0.,1.]); h_true=R@np.array([1.,0,0])
    rec=dict(seed=seed,local=rot_err(ql,q),global_=rot_err(q1,q),opencv=e_oc,
             true_in=min(rot_err(q1,q),rot_err(q2,q)),
             f1=float(f1),f2=float(f2))
    # cue noise sweep + shuffle
    for sig in [0.0,0.25,0.5,0.8,1.5]:
        h_prior=h_true+sig*rng.normal(size=3); h_prior/=np.linalg.norm(h_prior)
        h=np.array([q2R(q1)@np.array([1.,0,0]),q2R(q2)@np.array([1.,0,0])])
        ch=int(np.argmax([h_prior@h[0],h_prior@h[1]])); qa=[q1,q2][ch]
        rec[f'atom_{sig}']=rot_err(qa,q)
        x=h_prior/np.linalg.norm(h_prior); y=np.cross(n,x); y/=np.linalg.norm(y); x=np.cross(y,n)
        Rc=np.stack([x,y,n],axis=1); qc=q_from_R(Rc); qci,_,_=local_solve(J,qc)
        rec[f'cue_local_{sig}']=rot_err(qci,q)
        rec[f'gap_{sig}']=abs(float(h_prior@h[0]-h_prior@h[1]))
    hrand=rng.normal(size=3); hrand/=np.linalg.norm(hrand)
    h=np.array([q2R(q1)@np.array([1.,0,0]),q2R(q2)@np.array([1.,0,0])])
    ch=int(np.argmax([hrand@h[0],hrand@h[1]])); rec['atom_shuffle']=rot_err([q1,q2][ch],q)
    rows.append(rec)
with open(os.path.join(os.path.dirname(__file__),'..','results','audit_cue.json'),'w') as f: json.dump(rows,f,indent=2)
def s(key):
    a=np.array([r[key] for r in rows if np.isfinite(r[key])]); return (len(a),float(np.mean(a<0.1)),float(np.median(a)))
print('n',len(rows))
for k in ['local','global_','opencv','true_in','atom_0.0','atom_0.25','atom_0.5','atom_0.8','atom_1.5','atom_shuffle',
          'cue_local_0.0','cue_local_0.25','cue_local_0.5','cue_local_0.8','cue_local_1.5']:
    print(k,s(k))
print('cue gap 0.25 median',np.median([r['gap_0.25'] for r in rows]),'gap 0.8',np.median([r['gap_0.8'] for r in rows]))
