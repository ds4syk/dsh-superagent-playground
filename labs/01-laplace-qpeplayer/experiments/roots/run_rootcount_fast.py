#!/usr/bin/env python
"""Fast numerical root count for the QPEP stationarity (KKT) system.

No external CAS is installed on this host (M2 / msolve / sage / phcpy /
bertini / Singular all missing), so this script implements one fast numerical
method: a **total-degree homotopy** for the *projective* stationarity
condition, run in all four affine charts, plus a residual-verified affine
lift back to the complex sphere.

Polynomial system (q in C^4, lambda in C):

    J(q)  = v2(q)^T M v2(q),
    v2(q) = [q0^2, q1^2, q2^2, q3^2, sqrt2 q0q1, sqrt2 q0q2, sqrt2 q0q3,
             sqrt2 q1q2, sqrt2 q1q3, sqrt2 q2q3],
    Jv    = d v2 / d q                       (10 x 4, linear in q),
    grad J(q)_i = 2 sum_a Jv[a,i] (M v2(q))_a          (cubic in q)

    F_i(q,lambda) = grad J(q)_i - lambda q_i = 0,   i = 0..3        (deg 3)
    F_4(q,lambda) = q^T q - 1 = 0                                   (deg 2)
    => Bezout number 3^4 * 2 = 162.

Because J is homogeneous of degree 4 and even in q, the stationarity
condition is scale invariant: grad J(cq) = c^3 grad J(q) = c^2 lambda (cq),
so projectivising q is legitimate; and q and -q are the same projective point.
In the chart q_k = 1 the condition becomes the three polynomial equations
(degree 4)

    P_i(q) = grad J(q)_i - grad J(q)_k * q_i = 0,   i != k,

whose Bezout number is 4^3 = 64.  We track all 64 total-degree paths with a
batched complex predictor-corrector (natural-parameter homotopy
H = (1-t)G + t*F, G_i = x_i^4 - c_i, complex random c), with an endgame step
control near t = 1, and collect the residual-verified finite roots.  Each
projective root has at most one coordinate chart in which the tracker can
lose it (q_k = 0), so the union over the four charts is the projective root
set.  For a root with q^T q != 0 the affine lift q/sqrt(q^T q) satisfies the
5-equation system; a real projective root lifts to the real unit sphere iff
its (algebraic) q^T q > 0.

Run:
    /home/syk/miniconda3/envs/WM/bin/python \
        labs/01-laplace-qpeplayer/experiments/roots/run_rootcount_fast.py

Outputs: ../results/rootcount_fast.{json,md}
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
EXP = os.path.dirname(HERE)
RESULTS = os.path.join(EXP, "results")
sys.path.insert(0, CODE)

import instance  # noqa: E402

SQ = np.sqrt(2.0)
DEG_AFF = (3, 3, 3, 3, 2)       # affine 5-equation system
BEZOUT_AFF = int(np.prod(DEG_AFF))          # 162
DEG_CH = 4                       # chart system
BEZOUT_CH = DEG_CH ** 3          # 64
TOL_DEDUPE = 1e-5
TOL_REAL = 1e-7


# ------------------------------------------------------------------ complex algebra
def _hv():
    H = np.zeros((10, 4, 4))
    for i in range(4):
        H[i, i, i] = 2.0
    for i, j, a in [(0, 1, 4), (0, 2, 5), (0, 3, 6),
                    (1, 2, 7), (1, 3, 8), (2, 3, 9)]:
        H[a, i, j] = H[a, j, i] = SQ
    return H


HV = _hv()


def v2c(q):
    q = np.asarray(q, dtype=complex)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([q0 * q0, q1 * q1, q2 * q2, q3 * q3,
                     SQ * q0 * q1, SQ * q0 * q2, SQ * q0 * q3,
                     SQ * q1 * q2, SQ * q1 * q3, SQ * q2 * q3], axis=-1)


def jvc(q):
    q = np.asarray(q, dtype=complex)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    z = np.zeros_like(q0)
    rows = [
        (2 * q0, z, z, z), (z, 2 * q1, z, z), (z, z, 2 * q2, z),
        (z, z, z, 2 * q3),
        (SQ * q1, SQ * q0, z, z), (SQ * q2, z, SQ * q0, z),
        (SQ * q3, z, z, SQ * q0), (z, SQ * q2, SQ * q1, z),
        (z, SQ * q3, z, SQ * q1), (z, z, SQ * q3, SQ * q2),
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def gradj(q, M):
    Mv = v2c(q) @ M.T
    return 2.0 * np.einsum('...ai,...a->...i', jvc(q), Mv)


def hessj(q, M):
    Jq = jvc(q)
    Mv = v2c(q) @ M.T
    MJ = np.matmul(M, Jq)
    H = 2.0 * np.einsum('...ai,...aj->...ij', Jq, MJ)
    return H + 2.0 * np.einsum('...a,aij->...ij', Mv, HV)


def F_aff(x, M):
    """5-equation affine KKT residual; x = (q0..q3, lambda)."""
    q, lam = x[..., :4], x[..., 4]
    r = gradj(q, M) - lam[..., None] * q
    return np.concatenate([r, (np.sum(q * q, axis=-1) - 1.0)[..., None]], axis=-1)


def JF_aff(x, M):
    q, lam = x[..., :4], x[..., 4]
    n = x.shape[:-1]
    H = hessj(q, M) - lam[..., None, None] * np.eye(4, dtype=complex)
    top = np.concatenate([H, -q[..., None]], axis=-1)
    bot = np.concatenate([2.0 * q, np.zeros(n + (1,), dtype=complex)], axis=-1)
    return np.concatenate([top, bot[..., None, :]], axis=-2)


# ------------------------------------------------------------------ chart system
def _q_of(x, chart):
    n = x.shape[:-1]
    q = np.zeros(n + (4,), dtype=complex)
    q[..., chart] = 1.0
    idx = [i for i in range(4) if i != chart]
    q[..., idx] = x
    return q, idx


def F_ch(x, M, chart):
    q, idx = _q_of(x, chart)
    g = gradj(q, M)
    out = [g[..., i] - g[..., chart] * x[..., a] for a, i in enumerate(idx)]
    return np.stack(out, axis=-1)


def J_ch(x, M, chart):
    q, idx = _q_of(x, chart)
    g = gradj(q, M)
    H = hessj(q, M)
    Jx = np.zeros(x.shape[:-1] + (3, 3), dtype=complex)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            Jx[..., a, b] = (H[..., i, j] - H[..., chart, j] * x[..., a]
                             - (g[..., chart] if a == b else 0.0))
    return Jx


def G_start(x, c):
    return np.stack([x[..., i] ** 4 - c[i] for i in range(3)], axis=-1)


def JG_start(x):
    o = np.zeros(x.shape[:-1] + (3, 3), dtype=complex)
    for i in range(3):
        o[..., i, i] = 4.0 * x[..., i] ** 3
    return o


def start_roots(c):
    roots = [[c[i] ** 0.25 * np.exp(2j * np.pi * k / 4.0) for k in range(4)]
             for i in range(3)]
    X = np.array(np.meshgrid(*roots, indexing='ij')).reshape(3, -1).T
    assert X.shape == (BEZOUT_CH, 3)
    return X.astype(complex)


# ------------------------------------------------------------------ batched tracker
def track_chart(M, chart, seed=0, dt0=1e-3, dt_max=1e-2, grow=1.2,
                newton_tol=1e-11, newton_iters=8, endgame=0.9,
                max_steps=400000):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=3) + 1j * rng.normal(size=3)
    c /= np.abs(c)
    X = start_roots(c)
    m = BEZOUT_CH
    t = np.zeros(m)
    dt = np.full(m, dt0)
    done = np.zeros(m, bool)
    fail = np.zeros(m, bool)
    div = np.zeros(m, bool)
    Xfin = np.full((m, 3), np.nan + 1e-300j)
    steps = 0
    max_norm = 0.0

    def Hv(X, tt):
        return (1 - tt)[..., None] * G_start(X, c) + tt[..., None] * F_ch(X, M, chart)

    def Jv(X, tt):
        return ((1 - tt)[..., None, None] * JG_start(X)
                + tt[..., None, None] * J_ch(X, M, chart))

    for _ in range(max_steps):
        act = ~(done | fail | div)
        if not np.any(act):
            break
        ia = np.flatnonzero(act)
        Xa, ta, dta = X[ia], t[ia], dt[ia]
        # endgame: geometric approach to t = 1 so close roots separate
        dta = np.where(ta > endgame,
                       np.minimum(dta, np.maximum(0.1 * (1.0 - ta), 1e-7)), dta)
        dxt = -np.linalg.solve(Jv(Xa, ta),
                               (F_ch(Xa, M, chart) - G_start(Xa, c))[..., None])[..., 0]
        tn = np.minimum(ta + dta, 1.0)
        Xn = Xa + (tn - ta)[:, None] * dxt
        for _it in range(newton_iters):
            Ha = Hv(Xn, tn)
            rn = np.max(np.abs(Ha), axis=1)
            if np.all(rn < newton_tol):
                break
            try:
                dx = np.linalg.solve(Jv(Xn, tn), -Ha[..., None])[..., 0]
            except np.linalg.LinAlgError:
                rn = np.full(len(ia), np.inf)
                break
            Xn = Xn + dx
        Ha = Hv(Xn, tn)
        rn = np.max(np.abs(Ha), axis=1)
        ok = np.isfinite(rn) & (rn < 1e-10)
        X[ia[ok]] = Xn[ok]
        t[ia[ok]] = tn[ok]
        steps += int(ok.sum())
        dt[ia[ok]] = np.minimum(dt[ia[ok]] * grow, dt_max)
        bad = ~ok
        if np.any(bad):
            ib = ia[bad]
            dt[ib] *= 0.4
            fail[ib[dt[ib] < 1e-14]] = True
        ns = np.linalg.norm(X[ia], axis=1)
        max_norm = max(max_norm, float(np.max(ns)))
        big = ns > 1e12
        if np.any(big):
            div[ia[big]] = True
        fin = (~done) & (~fail) & (~div) & (t >= 1.0 - 1e-16)
        if np.any(fin):
            ifin = np.flatnonzero(fin)
            Xf = X[ifin]
            for _it in range(12):
                Hf = F_ch(Xf, M, chart)
                rf = np.max(np.abs(Hf), axis=1)
                if np.all(rf < 1e-13):
                    break
                try:
                    Xf = Xf + np.linalg.solve(J_ch(Xf, M, chart),
                                              -Hf[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    rf = np.full(len(ifin), np.inf)
                    break
            Hf = F_ch(Xf, M, chart)
            rf = np.max(np.abs(Hf), axis=1)
            good = np.isfinite(rf) & (rf < 1e-8)
            Xfin[ifin[good]] = Xf[good]
            done[ifin[good]] = True
            lo = ifin[~good]
            if len(lo):
                large = np.linalg.norm(Xf[~good], axis=1) > 1e8
                div[lo[large]] = True
                fail[lo[~large]] = True
    return Xfin[done], dict(chart=chart, n_paths=BEZOUT_CH,
                            n_converged=int(done.sum()),
                            n_failed=int(fail.sum()),
                            n_diverged=int(div.sum()),
                            n_steps=int(steps), max_norm=max_norm)


# ------------------------------------------------------------------ root set utilities
def canon(x, chart):
    """Canonical projective representative: q / q_k, k = argmax |q|."""
    q = np.zeros(4, dtype=complex)
    q[chart] = 1.0
    idx = [i for i in range(4) if i != chart]
    q[idx] = x
    k = int(np.argmax(np.abs(q)))
    return q / q[k]


def dedupe(vs, tol=TOL_DEDUPE):
    vs = np.asarray(vs, complex)
    used = np.zeros(len(vs), bool)
    reps = []
    for i in range(len(vs)):
        if used[i]:
            continue
        same = np.max(np.abs(vs - vs[i]), axis=1) < tol
        used |= same
        reps.append(vs[same].mean(axis=0))
    return np.array(reps) if reps else np.zeros((0, 4), complex)


def analyse_root(v, M, tol_real=TOL_REAL):
    """Residual-verify one projective representative v (canonical)."""
    qq = complex(v @ v)                      # algebraic q^T q
    imag = float(np.max(np.abs(v.imag)))
    real = imag < tol_real
    # scale-invariant verification: best chart residual over the charts in
    # which this projective point is finite (|v_k| not tiny)
    chart_res = np.inf
    for k in range(4):
        if abs(v[k]) < 1e-12:
            continue
        idx = [i for i in range(4) if i != k]
        xk = np.array([v[i] / v[k] for i in idx])
        chart_res = min(chart_res, float(np.max(np.abs(F_ch(xk, M, k)))))
    rec = dict(q=[[float(z.real), float(z.imag)] for z in v],
               qq=[qq.real, qq.imag],
               is_real=real, imag_max=imag,
               chart_residual=float(chart_res))
    if abs(qq) > 1e-8:
        qn = v / np.sqrt(qq)
        lam = 4.0 * complex(v2c(qn) @ M @ v2c(qn))
        r = float(np.max(np.abs(F_aff(np.concatenate([qn, [lam]]), M))))
        rec['affine_lift_residual'] = r
        rec['affine_lift'] = dict(q=[[float(z.real), float(z.imag)] for z in qn],
                                  lam=[lam.real, lam.imag],
                                  residual=r)
        if real and qq.real > 0:
            rec['real_sphere_lift'] = [float(x) for x in qn.real]
    rec['lifts_to_real_sphere'] = bool(real and qq.real > 0)
    rec['real_on_hyperboloid'] = bool(real and qq.real < 0)
    return rec


def minima_count(I, M, n_starts=512):
    t0 = time.time()
    md = I.find_modes(M, n_starts=n_starts, polish=True)
    return dict(n_minima=len(md['q']),
                n_stationary_found=len(md['stationary']),
                J_min=[float(x) for x in md['J']],
                grad_norm=[float(x) for x in md['grad_norm']],
                H_eig=[[float(y) for y in x] for x in md['H_eig']],
                runtime=time.time() - t0)


def random_symmetric_M(seed):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(10, 10))
    return 0.5 * (R + R.T)


# ------------------------------------------------------------------ per-instance driver
def run_case(name, M, I, chart_seed=7):
    t0 = time.time()
    chart_infos = []
    allv = []
    for chart in range(4):
        X, info = track_chart(M, chart, seed=chart_seed + chart)
        chart_infos.append(info)
        allv += [canon(x, chart) for x in dedupe(X)]
    roots = dedupe(allv, tol=TOL_DEDUPE)
    recs = [analyse_root(v, M) for v in roots]
    n_proj = len(recs)
    n_real = sum(r['is_real'] for r in recs)
    n_real_sph = sum(r['lifts_to_real_sphere'] for r in recs)
    n_real_hyp = sum(r['real_on_hyperboloid'] for r in recs)
    n_complex = n_proj - n_real
    mm = None
    try:
        mm = minima_count(I, M)
    except Exception as e:                       # pragma: no cover
        mm = dict(error=repr(e))
    rec = dict(name=name,
               M=[[float(x) for x in row] for row in M],
               n_projective_roots=n_proj,
               n_affine_roots=2 * n_proj,        # +/- lift, verified below
               n_real_projective=n_real,
               n_complex_projective=n_complex,
               n_real_sphere_projective=n_real_sph,
               n_real_sphere_affine=2 * n_real_sph,
               n_real_hyperboloid_projective=n_real_hyp,
               max_chart_residual=max(r['chart_residual'] for r in recs),
               max_affine_lift_residual=max(
                   r.get('affine_lift_residual', 0.0) for r in recs),
               n_affine_lifts_ok=sum(
                   r.get('affine_lift_residual', 1.0) < 1e-9 for r in recs),
               minima=mm,
               charts=chart_infos,
               runtime=time.time() - t0,
               roots=recs)
    print(f"[{name}] projective={n_proj} affine={2 * n_proj} "
          f"real(proj)={n_real} real_on_sphere(proj)={n_real_sph} "
          f"minima={mm.get('n_minima')} "
          f"charts={[c['n_converged'] for c in chart_infos]}/{BEZOUT_CH} "
          f"t={rec['runtime']:.2f}s")
    return rec


def main():
    os.makedirs(RESULTS, exist_ok=True)
    I = instance.Instance()
    cases = [(f"random_symmetric_seed{s}", random_symmetric_M(1000 + s))
             for s in (1, 2, 3)]
    cases.append(("instance_M_theta0", I.M_sym))

    out = dict(
        method="total-degree homotopy on the projective stationarity condition "
               "in all 4 affine charts (batched complex predictor-corrector, "
               "endgame step control), residual-verified union",
        external_tools_available=False,
        degrees_affine=list(DEG_AFF),
        bezout_affine=BEZOUT_AFF,
        bezout_chart=BEZOUT_CH,
        system_affine="F_i = 2*sum_a Jv[a,i]*(M v2(q))_a - lambda*q_i (i=0..3); "
                      "F_4 = sum_i q_i^2 - 1",
        system_chart="P_i(q) = gradJ(q)_i - gradJ(q)_k*q_i = 0, i != k, q_k = 1",
        v2="[q0^2,q1^2,q2^2,q3^2,sqrt2 q0q1,sqrt2 q0q2,sqrt2 q0q3,"
           "sqrt2 q1q2,sqrt2 q1q3,sqrt2 q2q3]",
        claim="P1-Q3: at most 40 complex projective roots (generic M)",
        command="/home/syk/miniconda3/envs/WM/bin/python "
                "labs/01-laplace-qpeplayer/experiments/roots/run_rootcount_fast.py",
        cases=[],
    )
    for name, M in cases:
        out['cases'].append(run_case(name, M, I))

    with open(os.path.join(RESULTS, 'rootcount_fast.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(RESULTS, 'rootcount_fast.json'))
    return out


if __name__ == '__main__':
    main()
