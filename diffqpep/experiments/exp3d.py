import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, gram_W, J_from_res, q2R
from diffqpep.moment import solve_moment, rank_ratio, MON2
from diffqpep.local import multistart
from diffqpep.instances import make_pnp, pose_error
from diffqpep.rform import build_Q, solve as solve_r, recover as rec_r, pose_from_x
from diffqpep.implicit import kkt_derivative, angular_derivative, vee

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)

def certificate():
    rows = []
    for noise in [0.0, 1e-3, 1e-2]:
        for seed in range(3):
            d = make_pnp(noise=noise, seed=seed)
            p, uv = d['p'], d['uv']
            rs = qpep_residuals(p, uv); W = gram_W(rs, MON2); J = J_from_res(rs)
            fg, qg = multistart(J, 30, seed=0)[0]
            rec = dict(noise=noise, seed=seed, local_best=float(fg))
            for order in [2, 3]:
                M, bound, st, mon, dt = solve_moment(W, order=order, eps=1e-8, max_iters=50000)
                rr, eff, ev = rank_ratio(M)
                rec['moment_order%d' % order] = dict(bound=float(bound), gap=float(fg-bound),
                                                     eff_rank=int(eff), status=st, sec=float(dt))
            Q = build_Q(p, uv); X, val, st2 = solve_r(Q)
            w = np.linalg.eigvalsh(X); effX = int(np.sum(w > 1e-6 * max(w[-1], 1e-30)))
            x, rr2 = rec_r(X); Rr, tr = pose_from_x(x)
            x_true = np.r_[d['R'].flatten(), d['t'], 1.0]
            ang, terr = pose_error(Rr, tr, d['R'], d['t'])
            rec['rform'] = dict(obj=float(val), true_cost=float(x_true @ Q @ x_true),
                                eff_rank=int(effX), last_diag=float(X[12, 12]),
                                orth=float(np.linalg.norm(Rr.T @ Rr - np.eye(3))),
                                det=float(np.linalg.det(Rr)), ang_err=float(ang), t_err=float(terr),
                                status=st2)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    return rows

def gradient():
    rows = []
    for seed in range(5):
        d = make_pnp(noise=0.0, seed=seed)
        p, uv = d['p'], d['uv']
        def rotated(theta):
            c, s = np.cos(theta), np.sin(theta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            return p @ Rz.T
        def build(theta):
            return J_from_res(qpep_residuals(rotated(theta), uv))
        J = build(0.0)
        fg, q0 = multistart(J, 30, seed=0)[0]
        for h in [1e-4, 1e-5, 1e-6]:
            Jp, Jm = build(h), build(-h)
            dJ = {}
            for k in set(Jp) | set(Jm):
                dJ[k] = (Jp.get(k, 0.0) - Jm.get(k, 0.0)) / (2 * h)
            dq, dlam, K, lam, g = kkt_derivative(J, q0, dJ)
            omega = angular_derivative(q0, dq)
            def gsolve(theta):
                Jt = build(theta); ft, qt = multistart(Jt, 20, seed=0)[0]
                return qt, q2R(qt), ft
            q1, R1, f1 = gsolve(h); q2, R2, f2 = gsolve(-h)
            if q1 @ q0 < 0: q1, R1 = -q1, R1
            if q2 @ q0 < 0: q2, R2 = -q2, R2
            omega_fd = vee((R1 - R2) / (2 * h) @ R1.T)
            dq_fd = (q1 - q2) / (2 * h)
            e_omega = float(np.linalg.norm(omega - omega_fd) / max(np.linalg.norm(omega_fd), 1e-12))
            e_q = float(np.linalg.norm(dq - dq_fd) / max(np.linalg.norm(dq_fd), 1e-12))
            rows.append(dict(seed=seed, h=h, cond_K=float(np.linalg.cond(K)),
                             err_omega=e_omega, err_q=e_q,
                             f_plus=f1, f_minus=f2))
            print('grad seed %d h %.0e err_omega %.2e err_q %.2e cond %.1f' % (seed, h, e_omega, e_q, np.linalg.cond(K)), flush=True)
    return rows

if __name__ == '__main__':
    cert = certificate()
    grad = gradient()
    with open(os.path.join(OUT, 'exp3d.json'), 'w') as f:
        json.dump(dict(certificate=cert, gradient=grad), f, indent=2)
    print('saved')
