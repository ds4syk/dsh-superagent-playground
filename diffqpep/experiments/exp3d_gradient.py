import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import multistart, local_solve
from diffqpep.instances import make_pnp
from diffqpep.implicit import kkt_derivative, angular_derivative, vee

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)
rows = []
for seed in range(6):
    d = make_pnp(noise=0.0, seed=seed)
    p, uv = d['p'], d['uv']
    def build(theta):
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pth = p @ Rz.T
        return J_from_res(qpep_residuals(pth, uv))
    J = build(0.0)
    fg, q0 = multistart(J, 30, seed=0)[0]
    for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        Jp, Jm = build(h), build(-h)
        dJ = {}
        for k in set(Jp) | set(Jm):
            dJ[k] = (Jp.get(k, 0.0) - Jm.get(k, 0.0)) / (2 * h)
        dq, dlam, K, lam, g = kkt_derivative(J, q0, dJ)
        omega = angular_derivative(q0, dq)
        q1, f1, _ = local_solve(Jp, q0)
        q2, f2, _ = local_solve(Jm, q0)
        if q1 @ q0 < 0: q1 = -q1
        if q2 @ q0 < 0: q2 = -q2
        R1, R2 = q2R(q1), q2R(q2)
        omega_fd = vee((R1 - R2) / (2 * h) @ R1.T)
        dq_fd = (q1 - q2) / (2 * h)
        e_omega = float(np.linalg.norm(omega - omega_fd) / max(np.linalg.norm(omega_fd), 1e-12))
        e_q = float(np.linalg.norm(dq - dq_fd) / max(np.linalg.norm(dq_fd), 1e-12))
        rows.append(dict(seed=seed, h=h, cond_K=float(np.linalg.cond(K)),
                         err_omega=e_omega, err_q=e_q, f_plus=f1, f_minus=f2))
        print('seed %d h %.0e err_omega %.2e err_q %.2e cond %.1f' % (seed, h, e_omega, e_q, np.linalg.cond(K)), flush=True)
with open(os.path.join(OUT, 'exp3d_gradient.json'), 'w') as f:
    json.dump(rows, f, indent=2)
print('saved')
