"""Consistent-tolerance face diagnostics for Table 2.

Both ranks use the same relative tolerance: an eigenvalue counts as nonzero
when |lambda| > TOL * max|lambda|.  This removes the inconsistency in
exp_correct.py (1e-9 absolute for W, 1e-6 relative for M), which produced the
apparent rank(M) > dim ker W violation.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res, monomials_deg_le, peval
from diffqpep.moment import solve_moment, MON2
from diffqpep.instances import make_pnp

TOL = 1e-8
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')

rows = []
for noise in [0.0, 1e-3, 1e-2]:
    for seed in range(3):
        d = make_pnp(noise=noise, seed=seed)
        p, uv = d['p'], d['uv']
        rs = qpep_residuals(p, uv)
        W = gram_W(rs, MON2)
        J = J_from_res(rs)
        ew = np.linalg.eigvalsh(W)
        scaleW = max(abs(ew[-1]), 1e-30)
        dimker = int(np.sum(np.abs(ew) <= TOL * scaleW))
        M, bound, st, mon, dt = solve_moment(W, order=2, eps=1e-8, max_iters=50000)
        eM = np.linalg.eigvalsh(M)
        scaleM = max(abs(eM[-1]), 1e-30)
        rankM = int(np.sum(np.abs(eM) > TOL * scaleM))
        # gap = true cost - lower bound (can be slightly negative from SCS)
        jgt = float(peval(J, d['q']))
        gap = jgt - float(bound)
        rows.append(dict(noise=noise, seed=seed,
                         dim_ker_W=dimker, rank_M=rankM,
                         eig_W=[float(x) for x in ew],
                         eig_M=[float(x) for x in eM],
                         J_gt=jgt, bound=float(bound), gap=gap, status=st))
        print(f"noise={noise} seed={seed} dimker={dimker} rankM={rankM} gap={gap:.3e}", flush=True)

with open(os.path.join(OUT, 'face_diag.json'), 'w') as f:
    json.dump(rows, f, indent=2)
print('saved', len(rows))
