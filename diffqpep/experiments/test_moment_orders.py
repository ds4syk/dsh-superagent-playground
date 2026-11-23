import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res
from diffqpep.moment import solve_moment, rank_ratio, MON2
from diffqpep.local import multistart
from diffqpep.instances import make_pnp

inst = make_pnp(noise=1e-3, seed=0)
p, uv = inst['p'], inst['uv']
rs = qpep_residuals(p, uv)
W = gram_W(rs, MON2)
J = J_from_res(rs)
fs = multistart(J, 30, seed=0)[0][0]
print('local best %.12e' % fs, flush=True)
for order in [2, 3]:
    M, bound, status, mon, dt = solve_moment(W, order=order, eps=1e-8, max_iters=50000)
    rr, ev = rank_ratio(M)
    print('order %d bound %.12e gap %.3e rank1 %.2e %.1fs %s' % (order, bound, fs-bound, rr, dt, status), flush=True)
