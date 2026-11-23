import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.rform import build_Q, solve, recover, pose_from_x
from diffqpep.instances import make_pnp, pose_error

for noise in [0.0, 1e-3, 1e-2]:
    print('=== noise', noise)
    for seed in range(3):
        d = make_pnp(noise=noise, seed=seed)
        p, uv, R, t = d['p'], d['uv'], d['R'], d['t']
        Q = build_Q(p, uv)
        x_true = np.r_[R.flatten(), t, 1.0]
        X, val, st = solve(Q)
        x, rr = recover(X); Rr, tr = pose_from_x(x)
        ang, terr = pose_error(Rr, tr, R, t)
        print('seed', seed, 'true cost %.2e obj %.2e rank1 %.1e det %+.2f orth %.1e ang %.2e t %.2e' % (
            x_true @ Q @ x_true, val, rr, np.linalg.det(Rr), np.linalg.norm(Rr.T @ Rr - np.eye(3)), ang, terr))
