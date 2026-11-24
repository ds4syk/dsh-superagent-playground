"""Minimal 3-point benchmark v3 (final table).

Protocol
--------
500 instances, normalized image noise 5e-4, three of four planar points,
heading cue h ~ N(h_true, 0.25^2) (a noisy measurement of the true heading).
Candidate generators: 8 QPEP local starts (4 deterministic + 4 fixed random)
and OpenCV P3P (up to 4 solutions).  Selection rules: identity, lowest cost
(multi-start best-found), and heading-cue argmax.  The oracle is the best
branch in the generator's candidate set.

Baselines / ablations
---------------------
local           1 identity start
global-8        lowest cost over 8 starts (no cue)
cue-local-1     one local solve from the cue
cue-local-8     cue start + 7 extra starts, lowest cost (same start budget as atomic-8)
atomic-4        cue selection among 4-start candidates
atomic-8        cue selection among 8-start candidates
shuffle-8       random unit cue selection among 8-start candidates
p3p-cue         cue selection among P3P solutions
p3p-best        oracle over P3P solutions
oracle-8        oracle over 8-start QPEP candidates
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffqpep.poly import qpep_residuals, J_from_res, q2R
from diffqpep.local import local_solve
from diffqpep.torch_layer import branch_starts, solve_branches
from scipy.spatial.transform import Rotation as Rot
import cv2

P4 = np.array([[-1.0, -0.9, 0.0], [1.05, -0.95, 0.0], [1.0, 1.1, 0.0], [-0.95, 1.0, 0.0]])
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
NOISE = 5e-4


def norm(q):
    return q / np.linalg.norm(q)


def rot_err(q1, q2):
    R1, R2 = q2R(q1), q2R(q2)
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def q_from_R(R):
    q = Rot.from_matrix(R.T).as_quat()
    q = np.array([q[3], q[0], q[1], q[2]])
    return q / np.linalg.norm(q)


def t_from_R(p, uv, R):
    A = np.zeros((2 * len(p), 3))
    b = np.zeros(2 * len(p))
    for i in range(len(p)):
        ui, vi = uv[i]
        Ai = np.array([[ui, 0, -1.0], [0, vi, -1.0]])
        A[2 * i:2 * i + 2] = Ai
        b[2 * i:2 * i + 2] = Ai @ (R @ p[i])
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def t_err(p, uv, q, t_true):
    return float(np.linalg.norm(t_from_R(p, uv, q2R(q)) - t_true))


def gen(seed, noise=NOISE):
    rng = np.random.default_rng(seed)
    q = norm(rng.normal(size=4))
    R = q2R(q)
    t = np.array([0., 0., 3.0]) + rng.normal(size=3) * 0.2
    keep = rng.choice(4, size=3, replace=False)
    P = P4[keep]
    Xc = (R @ P.T).T + t
    if np.any(Xc[:, 2] < 1):
        return None
    uv = Xc[:, :2] / Xc[:, 2:3] + noise * rng.normal(size=(3, 2))
    return q, uv, R, t, P


def p3p_solutions(P, uv):
    K = np.eye(3)
    d = np.zeros(5)
    try:
        n, Rs, ts = cv2.solveP3P(P.astype(np.float64), uv.astype(np.float64), K, d,
                                 flags=cv2.SOLVEPNP_P3P)
        out = []
        for rv, tv in zip(Rs, ts):
            Rmat, _ = cv2.Rodrigues(np.array(rv).reshape(3, 1))
            out.append((q_from_R(Rmat), np.array(tv).reshape(3)))
        return out
    except Exception:
        return []


def cue_quat(h_prior, q_ref):
    n = q2R(q_ref) @ np.array([0., 0., 1.])
    x = h_prior / np.linalg.norm(h_prior)
    y = np.cross(n, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    return q_from_R(np.stack([x, y, n], axis=1))


def main():
    rng0 = np.random.default_rng(7)
    extra = np.array([norm(rng0.normal(size=4)) for _ in range(4)])
    starts8 = np.vstack([branch_starts(0), extra])
    rows = []
    for seed in range(500):
        g = gen(seed)
        if g is None:
            continue
        q, uv, R, t_true, P = g
        J = J_from_res(qpep_residuals(P, uv))
        ql, _, _ = local_solve(J, np.array([1., 0, 0, 0]))
        qs8 = [qq for f, qq in solve_branches(J, starts8, keep=8)]
        qs4 = [qq for f, qq in solve_branches(J, branch_starts(0), keep=4)]
        if len(qs8) < 2 or len(qs4) < 2:
            continue
        # cue: noisy measurement of the true heading
        rng = np.random.default_rng(seed + 999)
        h_true = R @ np.array([1., 0, 0])
        h_prior = h_true + 0.25 * rng.normal(size=3)
        h_prior /= np.linalg.norm(h_prior)

        def pick_cue(qs):
            hs = np.array([q2R(qq) @ np.array([1., 0, 0]) for qq in qs])
            return qs[int(np.argmax(hs @ h_prior))]

        hsh = rng.normal(size=3)
        hsh /= np.linalg.norm(hsh)

        def pick_shuffle(qs):
            hs = np.array([q2R(qq) @ np.array([1., 0, 0]) for qq in qs])
            return qs[int(np.argmax(hs @ hsh))]

        q_atom8 = pick_cue(qs8)
        q_atom4 = pick_cue(qs4)
        q_shuf = pick_shuffle(qs4)
        q_first8 = qs8[0]  # lowest-cost branch
        # cue-initialized local solve, 1 start
        qc = cue_quat(h_prior, qs4[0])
        qci, _, _ = local_solve(J, qc)
        # cue start + 7 extra starts, cue-selected among distinct solutions
        sols = []
        for q0 in [qc] + list(starts8[1:]):
            qq, ff, _ = local_solve(J, q0)
            sols.append((ff, qq))
        uniq = []
        for ff, qq in sorted(sols, key=lambda z: z[0]):
            if all(min(np.linalg.norm(qq - u), np.linalg.norm(qq + u)) > 1e-4 for _, u in uniq):
                uniq.append((ff, qq))
        best_q = pick_cue([qq for _, qq in uniq])
        # P3P
        p3 = p3p_solutions(P, uv)
        if p3:
            p3_best = min(rot_err(qq, q) for qq, _ in p3)
            hs = np.array([q2R(qq) @ np.array([1., 0, 0]) for qq, _ in p3])
            i_cue = int(np.argmax(hs @ h_prior))
            p3_cue = rot_err(p3[i_cue][0], q)
            p3_t = float(np.linalg.norm(p3[i_cue][1] - t_true))
        else:
            p3_best = p3_cue = p3_t = np.nan
        rows.append(dict(
            seed=seed, n_branch=len(qs8),
            local=rot_err(ql, q), local_t=t_err(P, uv, ql, t_true),
            global_=rot_err(q_first8, q), global_t=t_err(P, uv, q_first8, t_true),
            cue_local=rot_err(qci, q), cue_local_t=t_err(P, uv, qci, t_true),
            cue_local8=rot_err(best_q, q), cue_local8_t=t_err(P, uv, best_q, t_true),
            atom4=rot_err(q_atom4, q), atom4_t=t_err(P, uv, q_atom4, t_true),
            atom8=rot_err(q_atom8, q), atom8_t=t_err(P, uv, q_atom8, t_true),
            shuffle=rot_err(q_shuf, q), shuffle_t=t_err(P, uv, q_shuf, t_true),
            p3p_cue=float(p3_cue), p3p_cue_t=float(p3_t),
            p3p_best=float(p3_best), p3p_n=len(p3),
            oracle8=float(min(rot_err(qq, q) for qq in qs8)),
            oracle8_t=float(min(t_err(P, uv, qq, t_true) for qq in qs8)),
        ))
        if seed % 100 == 0:
            print('seed', seed, flush=True)

    with open(os.path.join(OUT, 'benchmark_3pt_final.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    def s(k):
        a = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return dict(n=int(len(a)), success=round(float(np.mean(a < 0.1)), 4),
                    med=round(float(np.median(a)), 4), mean=round(float(np.mean(a)), 4))

    print('n', len(rows))
    for k in ['local', 'global_', 'cue_local', 'cue_local8', 'atom4', 'atom8',
              'shuffle', 'p3p_cue', 'p3p_best', 'oracle8']:
        print(k, s(k))
    print('median branches', np.median([r['n_branch'] for r in rows]),
          'p3p nsol', np.median([r['p3p_n'] for r in rows]))


if __name__ == '__main__':
    main()
