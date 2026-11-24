import sys, os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R, gram_W, monomials_deg_le
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from diffqpep.moment import solve_moment, MON2
from diffqpep.rform import build_Q
from diffqpep.implicit import kkt_matrix
import cvxpy as cp

OUT=os.path.join(os.path.dirname(__file__),'..','results'); FIG=os.path.join(os.path.dirname(__file__),'..','figures'); os.makedirs(OUT,exist_ok=True); os.makedirs(FIG,exist_ok=True)
P=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def normalize(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
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
    X=cp.Variable((12,12),symmetric=True); cs=[X>>0]
    for A,b in orth12(): cs.append(cp.trace(A@X)==b)
    prob=cp.Problem(cp.Minimize(cp.trace(Q@X)),cs); prob.solve(solver=cp.SCS,eps=1e-11,max_iters=200000)
    return X.value,prob.value,prob.status
q=normalize(np.array([0.95,0.2,-0.15,0.1])); R=q2R(q); t=np.array([0.1,-0.05,3.0])
Xc=(R@P.T).T+t; uv=Xc[:,:2]/Xc[:,2:3]
J=J_from_res(qpep_residuals(P,uv)); starts=branch_starts(0)
sols=solve_branches(J,starts,keep=3)
rows=[]
for f,qq in sols:
    rows.append(dict(f=float(f),err=rot_err(qq,q),q=[float(z) for z in qq]))
ql,fl,_=local_solve(J,np.array([1.0,0,0,0]))
# 12x12 R-form
Q12=build_Q(P,uv)[:12,:12]; X,val,st=solve_r12(Q12)
w,V=np.linalg.eigh(X); x=V[:,-1]*np.sqrt(max(w[-1],0)); Rr=x[:9].reshape(3,3); tr=x[9:12]
xt=np.r_[R.flatten(),t]
out=dict(branches=rows,local=dict(f=float(fl),err=rot_err(ql,q)),
         rform12=dict(val=float(val),rank=int(np.sum(w>1e-6*max(w[-1],1e-30))),
                     orth=float(np.linalg.norm(Rr.T@Rr-np.eye(3))),det=float(np.linalg.det(Rr)),
                     e_plus=float(np.linalg.norm(x-xt)),e_minus=float(np.linalg.norm(x+xt)),status=st),
         qform={})
# q-form face
rs=qpep_residuals(P,uv); W=gram_W(rs,MON2); ew=np.linalg.eigvalsh(W)
M,bound,stm,mon,dt=solve_moment(W,order=2,eps=1e-8,max_iters=50000); eM=np.linalg.eigvalsh(M)
out['qform']=dict(dim_ker_W=int(np.sum(np.abs(ew)<1e-9)),rank_M=int(np.sum(eM>1e-6*max(eM[-1],1e-30))),
                  lambda_ratio=float(eM[-1]/max(eM[-2],1e-30)),bound=float(bound),status=stm)
with open(os.path.join(OUT,'double_cover.json'),'w') as f: json.dump(out,f,indent=2)
print(json.dumps(out)[:1200])
# figure: branch costs/pose errors and local/global
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7))
ax=axs[0]
ax.bar(['branch 1','branch 2','local'], [rows[0]['f'],rows[1]['f'],fl], color=['#2b5d8a','#7aa6c2','#d08c60'])
ax.set_ylabel('QPEP cost'); ax.set_title('(a) planar double cover: equal-cost branches',fontsize=8)
ax=axs[1]
ax.bar(['branch 1','branch 2','local'], [rows[0]['err'],rows[1]['err'],rot_err(ql,q)], color=['#2b5d8a','#7aa6c2','#d08c60'])
ax.set_ylabel('rotation error to truth (rad)'); ax.set_title('(b) same cost, different poses',fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_double_cover.png'),dpi=300); plt.close()
print('saved')
