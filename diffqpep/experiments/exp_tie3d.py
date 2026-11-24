import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R, peval, pgrad
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from diffqpep.implicit import kkt_matrix, angular_derivative, vee

OUT=os.path.join(os.path.dirname(__file__),'..','results'); os.makedirs(OUT,exist_ok=True)
P=np.array([[-1.0,-0.9,0.0],[1.05,-0.95,0.0],[1.0,1.1,0.0],[-0.95,1.0,0.0]])
def normalize(q): return q/np.linalg.norm(q)
def rot_err(q1,q2):
    R1,R2=q2R(q1),q2R(q2); return float(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def base():
    q=normalize(np.array([0.95,0.2,-0.15,0.1])); R=q2R(q); t=np.array([0.1,-0.05,3.0])
    Xc=(R@P.T).T+t; uv=Xc[:,:2]/Xc[:,2:3]
    return uv,q,t
uv0,qtrue,ttrue=base()
starts=branch_starts(0)
def branches(uv):
    J=J_from_res(qpep_residuals(P,uv)); sols=solve_branches(J,starts,keep=3); return J,sols
# find a signed noise direction that flips the global branch at delta=0
rng=np.random.default_rng(0); e=rng.normal(size=(4,2)); e=e/np.linalg.norm(e)
for sign in [1,-1]:
    rows=[]
    for d in [-0.02,-0.01,-0.002,0.0,0.002,0.01,0.02]:
        uv=uv0+sign*d*e; J,sols=branches(uv)
        (f1,q1),(f2,q2)=sols[0],sols[1]
        rows.append((d,float(f1),float(f2),rot_err(q1,qtrue),rot_err(q2,qtrue)))
    print('sign',sign)
    for r in rows: print('  d %+.3f f1 %.9e f2 %.9e q1err %.3f q2err %.3f'%r)
