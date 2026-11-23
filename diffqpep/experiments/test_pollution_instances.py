import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import multistart, local_solve
from diffqpep.instances import make_planar_pnp, make_symmetric_pnp, pose_error

def analyze(inst, name):
    p, uv = inst['p'], inst['uv']
    rs = qpep_residuals(p, uv); J = J_from_res(rs)
    glob = multistart(J, 30, seed=0)
    fg, qg = glob[0]
    ql, fl, _ = local_solve(J, np.array([1.0, 0, 0, 0]))
    ang_g, _ = pose_error(q2R(qg), inst['t'], inst['R'], inst['t'])
    ang_l, _ = pose_error(q2R(ql), inst['t'], inst['R'], inst['t'])
    return fg, fl, ang_g, ang_l, qg, ql

print('=== planar PnP, identity init')
for seed in range(10):
    inst = make_planar_pnp(noise=1e-3, seed=seed)
    fg, fl, ag, al, qg, ql = analyze(inst, 'planar')
    print('seed %d glob %.3e local %.3e gap %+.2e | ang glob %.3f local %.3f' % (seed, fg, fl, fl-fg, ag, al))

print('=== symmetric PnP, tilt sweep, identity init')
for tilt in [-0.05, -0.01, 0.0, 0.01, 0.05]:
    inst = make_symmetric_pnp(noise=1e-3, seed=0, tilt=tilt)
    fg, fl, ag, al, qg, ql = analyze(inst, 'sym')
    print('tilt %+.2f glob %.3e local %.3e gap %+.2e | ang glob %.3f local %.3f' % (tilt, fg, fl, fl-fg, ag, al))
