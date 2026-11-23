import sys, os, json
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
planar = []
for seed in range(10):
    inst = make_planar_pnp(noise=1e-3, seed=seed)
    fg, fl, ag, al, qg, ql = analyze(inst, 'planar')
    planar.append(dict(seed=seed, f_global=float(fg), f_local=float(fl), gap=float(fl-fg),
                       ang_global=float(ag), ang_local=float(al)))
    print('seed %d glob %.3e local %.3e gap %+.2e | ang glob %.3f local %.3f' % (seed, fg, fl, fl-fg, ag, al))

print('=== symmetric PnP, tilt sweep, identity init')
symmetric = []
for tilt in [-0.05, -0.01, 0.0, 0.01, 0.05]:
    inst = make_symmetric_pnp(noise=1e-3, seed=0, tilt=tilt)
    fg, fl, ag, al, qg, ql = analyze(inst, 'sym')
    symmetric.append(dict(tilt=tilt, f_global=float(fg), f_local=float(fl), gap=float(fl-fg),
                          ang_global=float(ag), ang_local=float(al)))
    print('tilt %+.2f glob %.3e local %.3e gap %+.2e | ang glob %.3f local %.3f' % (tilt, fg, fl, fl-fg, ag, al))
with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'planar_pollution.json'), 'w') as f:
    json.dump(dict(planar=planar, symmetric=symmetric), f, indent=2)
print('saved planar_pollution.json')
