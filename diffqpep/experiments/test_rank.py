import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res, q2R
from diffqpep.moment import solve_moment, rank_ratio, recover_q, MON2
from diffqpep.local import multistart
from diffqpep.instances import make_pnp, pose_error
from diffqpep.rform import build_Q, solve as solve_r, recover as rec_r, pose_from_x

for noise in [0.0, 1e-3]:
    d = make_pnp(noise=noise, seed=0)
    p, uv = d['p'], d['uv']
    rs = qpep_residuals(p, uv); W = gram_W(rs, MON2); J = J_from_res(rs)
    fg, qg = multistart(J, 30, seed=0)[0]
    print('noise', noise, 'local best %.3e' % fg)
    M, b, st, mon, dt = solve_moment(W, order=2)
    rr, eff, ev = rank_ratio(M); qr = recover_q(M, mon)
    er = pose_error(q2R(qr), d['t'], d['R'], d['t'])[0] if qr is not None else -1
    print('  moment bound %.3e gap %.3e rank %.1e eff %d qerr %.3f %s %.1fs' % (b, fg-b, rr, eff, er, st, dt))
    Q = build_Q(p, uv); X, val, st2 = solve_r(Q)
    x, rr2 = rec_r(X); Rr, tr = pose_from_x(x)
    print('  rform obj %.3e true %.3e eff? lastdiag %.3f orth %.2e det %+.2f ang %.3f' % (
        val, np.r_[d['R'].flatten(), d['t'], 1.0] @ Q @ np.r_[d['R'].flatten(), d['t'], 1.0],
        X[12,12], np.linalg.norm(Rr.T@Rr-np.eye(3)), np.linalg.det(Rr), pose_error(Rr,tr,d['R'],d['t'])[0]))
