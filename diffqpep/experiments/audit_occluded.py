import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from scipy.spatial.transform import Rotation as Rot
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
    keep=rng.choice(4,size=3,replace=False); P=P4[keep]; Xc=(R@P.T).T+t
    if np.any(Xc[:,2]<1): return None
    uv=Xc[:,:2]/Xc[:,2:3]+noise*rng.normal(size=(3,2))
    return q,uv,R,t,P
rng0=np.random.default_rng(7)
extra=np.array([norm(rng0.normal(size=4)) for _ in range(4)])
starts4=branch_starts(0); starts8=np.vstack([starts4,extra])
rows=[]
for seed in range(500):
    g=gen(seed)
    if g is None: continue
    q,uv,R,t,P=g; J=J_from_res(qpep_residuals(P,uv))
    ql,_,_=local_solve(J,np.array([1.,0,0,0]))
    def run(starts,keep):
        sols=solve_branches(J,starts,keep=keep); qs=[qq for f,qq in sols]
        return qs
    qs4=run(starts4,4); qs8=run(starts8,8)
    rng=np.random.default_rng(seed+999); h_true=R@np.array([1.,0,0]); h_prior=h_true+0.25*rng.normal(size=3); h_prior/=np.linalg.norm(h_prior)
    def sel(qs):
        hs=np.array([q2R(qq)@np.array([1.,0,0]) for qq in qs]); ch=int(np.argmax(hs@h_prior)); return qs[ch]
    qa4=sel(qs4); qa8=sel(qs8)
    n=q2R(qs4[0])@np.array([0.,0.,1.]); x=h_prior/np.linalg.norm(h_prior); y=np.cross(n,x); y/=np.linalg.norm(y); x=np.cross(y,n)
    Rc=np.stack([x,y,n],axis=1); qc=q_from_R(Rc); qci,_,_=local_solve(J,qc)
    hsh=rng.normal(size=3); hsh/=np.linalg.norm(hsh); hs4=np.array([q2R(qq)@np.array([1.,0,0]) for qq in qs4]); qsh=qs4[int(np.argmax(hs4@hsh))]
    rows.append(dict(seed=seed,local=rot_err(ql,q),global_=rot_err(qs8[0],q),cue_local=rot_err(qci,q),atom4=rot_err(qa4,q),atom8=rot_err(qa8,q),shuffle=rot_err(qsh,q),
                     best4=min(rot_err(qq,q) for qq in qs4),best8=min(rot_err(qq,q) for qq in qs8),n4=len(qs4),n8=len(qs8)))
with open(os.path.join(os.path.dirname(__file__),'..','results','audit_occluded.json'),'w') as f: json.dump(rows,f,indent=2)
def s(k):
    a=np.array([r[k] for r in rows if np.isfinite(r[k])]); return (len(a),float(np.mean(a<0.1)),float(np.median(a)),float(np.mean(a)))
print('n',len(rows))
for k in ['local','global_','cue_local','atom4','atom8','shuffle','best4','best8']: print(k,s(k))
print('branch med',np.median([r['n4'] for r in rows]),np.median([r['n8'] for r in rows]))
