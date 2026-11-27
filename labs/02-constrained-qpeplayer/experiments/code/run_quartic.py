#!/usr/bin/env python
"""Validation harness for quartic_roots.py (P2 exact inner enumeration).

Phases
------
generic   : 5 random well-conditioned symmetric 10x10 M; projective/affine
            root counts, real counts, path status, runtime.
p2        : P2 instances from qpep_core.make_instance(p, noise) -> Mred;
            root counts, exact-vs-multistart objective gaps (K=8/64/256),
            argmin comparison, timing.
perturb   : does a tiny generic perturbation of the (PSD, rank-deficient)
            Mred restore the root count?  eps * ||Mred||  sweep.
robust    : PSD/rank-deficient Mred at noise=0, ALM-assembled M_A, and
            synthetic singular M (rank-1, zero) -- fallback flag behaviour.
alm       : small ALM exact-vs-multistart comparison (p=16, 8 trials).

Usage
-----
    python run_quartic.py --phases generic p2 perturb robust alm
Results are merged into ../results/quartic_validation.json (one key per phase).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import qpep_core as C
import quartic_roots as Q

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "quartic_validation.json")


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _stats_pick(st):
    keys = ["n_paths", "n_finite_paths", "n_diverged", "n_newton_fail",
            "n_nonfinite", "n_lost", "n_rescued", "n_finite_projective",
            "n_finite_affine", "n_finite_affine_raw", "n_real_affine",
            "n_unique", "n_batch_evals", "t_track", "t_polish", "t_total",
            "paths_per_s", "max_finite_xnorm"]
    return {k: st[k] for k in keys}


# ---------------------------------------------------------------------------
def phase_generic(n_gamma=2):
    recs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(10, 10))
        M = 0.5 * (M + M.T)
        M = M / np.linalg.norm(M)
        t0 = time.perf_counter()
        roots, st = Q.critical_points(M, return_stats=True, n_gamma=n_gamma)
        wall = time.perf_counter() - t0
        rec = dict(seed=seed, n_gamma=n_gamma, wall=wall, f_min=roots[0]["f"],
                   f_second=roots[1]["f"] if len(roots) > 1 else None,
                   n_real_geometric=len(roots), **_stats_pick(st))
        # single-phase cost / throughput
        t0 = time.perf_counter()
        r1, s1 = Q.critical_points(M, return_stats=True, n_gamma=1)
        rec["single_phase"] = dict(wall=time.perf_counter() - t0,
                                   f_min=r1[0]["f"], **_stats_pick(s1))
        # union over phases must not add roots: phase0 count == union count
        rec["phase_union_extra"] = st["n_finite_projective"] - s1["n_finite_projective"]
        recs.append(rec)
        print("  generic seed=%d proj=%d affine=%d real=%d lost=%d  %.2fs"
              % (seed, st["n_finite_projective"], st["n_finite_affine"],
                 st["n_real_affine"], st["n_lost"], wall), flush=True)
    return dict(n_gamma=n_gamma, cases=recs)


def _ms(Mred, K, seed):
    rng = np.random.default_rng(4000 + seed)
    t0 = time.perf_counter()
    f, q = C.solve_inner(Mred, n_starts=K, rng=rng)
    dt = time.perf_counter() - t0
    return dict(f=_f(f), q=q, time=dt)


def phase_p2(ps=(8, 16, 32, 64), noises=(0.0, 0.01, 0.05), seeds=(0, 1), n_gamma=1):
    cases = []
    for p in ps:
        for noise in noises:
            for seed in seeds:
                ins = C.make_instance(seed, p, noise=noise)
                Mred = C.obj_min_t(ins["M0"], ins["D0"], ins["G0"])
                Mred = 0.5 * (Mred + Mred.T)
                ev = np.linalg.eigvalsh(Mred)
                t0 = time.perf_counter()
                ex = Q.solve_inner_exact(Mred, seed=0, n_gamma=n_gamma)
                dt_ex = time.perf_counter() - t0
                rec = dict(p=p, noise=noise, seed=seed, n_gamma=n_gamma,
                           eig_min=float(ev.min()), eig_max=float(ev.max()),
                           cond=float(ev.max() / max(abs(ev.min()), 1e-300)),
                           f_exact=ex["f"], f_enum=ex["f_enum"],
                           fallback=ex["fallback"], exact=ex["exact"],
                           degenerate=ex["degenerate"],
                           sigma_min_best=ex["sigma_min_best"],
                           n_lost=ex["n_lost"], n_unique=ex["n_unique"],
                           n_finite_projective=ex["n_finite_projective"],
                           n_finite_affine=ex["n_finite_affine"],
                           n_finite_affine_raw=ex["n_finite_affine"],
                           t_exact=dt_ex, q_exact=ex["q"].tolist(),
                           **{k: ex["stats"][k] for k in
                              ("n_finite_paths", "n_diverged", "n_newton_fail",
                               "n_batch_evals", "t_track", "t_polish",
                               "paths_per_s")})
                for K in (8, 64, 256):
                    ms = _ms(Mred, K, seed)
                    rec["ms_%d" % K] = dict(f=ms["f"], time=ms["time"],
                                            q=ms["q"].tolist())
                    rec["gap_%d" % K] = ms["f"] - ex["f_enum"]
                    rec["rot_deg_%d" % K] = C.rot_deg(ex["q"], ms["q"])
                    rec["qdist_%d" % K] = float(min(
                        np.linalg.norm(ex["q"] - ms["q"]),
                        np.linalg.norm(ex["q"] + ms["q"])))
                cases.append(rec)
                print("  p=%2d noise=%.2f seed=%d proj=%d real=%d f=%.12g "
                      "gap8=%+.2e gap64=%+.2e gap256=%+.2e t=%.2fs"
                      % (p, noise, seed, rec["n_finite_projective"],
                         rec["n_unique"], rec["f_exact"], rec["gap_8"],
                         rec["gap_64"], rec["gap_256"], dt_ex), flush=True)
    return dict(cases=cases)


def phase_perturb(cases=((0, 16, 0.01), (0, 32, 0.01), (0, 8, 0.05)),
                  eps_list=(0.0, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.2), n_gamma=2):
    recs = []
    for (seed, p, noise) in cases:
        ins = C.make_instance(seed, p, noise=noise)
        Mred = C.obj_min_t(ins["M0"], ins["D0"], ins["G0"])
        Mred = 0.5 * (Mred + Mred.T)
        sc = np.linalg.norm(Mred)
        rng = np.random.default_rng(1)
        E = rng.normal(size=(10, 10))
        E = 0.5 * (E + E.T)
        E = E / np.linalg.norm(E)
        for eps in eps_list:
            M = Mred + eps * sc * E
            t0 = time.perf_counter()
            roots, st = Q.critical_points(M, return_stats=True, n_gamma=n_gamma)
            recs.append(dict(seed=seed, p=p, noise=noise, eps=eps,
                             n_finite_projective=st["n_finite_projective"],
                             n_finite_affine=st["n_finite_affine"],
                             n_real_affine=st["n_real_affine"],
                             n_unique=st["n_unique"], n_lost=st["n_lost"],
                             f_min=roots[0]["f"],
                             wall=time.perf_counter() - t0))
            print("  perturb p=%d eps=%.0e proj=%d real=%d lost=%d"
                  % (p, eps, st["n_finite_projective"], st["n_unique"],
                     st["n_lost"]), flush=True)
    return dict(cases=recs)


def _alm_Mred(M0, D0, G0, A, B, mu, rho, act):
    MA, DA, GA = C._alm_quadratic(M0, D0, G0, A, B, mu, rho, act)
    return C._rho_of(MA, DA, GA), (MA, DA, GA)


def phase_robust(n_gamma=1):
    out = dict(psd=[], alm_M=[], synthetic=[])
    # (a) PSD / rank-deficient Mred at noise = 0
    for p in (8, 16):
        for seed in (0, 1):
            ins = C.make_instance(seed, p, noise=0.0)
            Mred = C.obj_min_t(ins["M0"], ins["D0"], ins["G0"])
            Mred = 0.5 * (Mred + Mred.T)
            ev = np.linalg.eigvalsh(Mred)
            ex = Q.solve_inner_exact(Mred, seed=0, n_gamma=n_gamma)
            ms = _ms(Mred, 256, seed)
            out["psd"].append(dict(
                p=p, seed=seed, noise=0.0, eig_min=float(ev.min()),
                rank=int(np.sum(np.abs(ev) > 1e-9 * abs(ev).max())),
                n_finite_projective=ex["n_finite_projective"],
                n_unique=ex["n_unique"], f_exact=ex["f"],
                f_enum=ex["f_enum"], f_ms256=ms["f"], gap=ms["f"] - ex["f_enum"],
                fallback=ex["fallback"], exact_flag=ex["exact"],
                degenerate=ex["degenerate"], n_lost=ex["n_lost"],
                t_exact=ex["stats"]["t_total"]))
            print("  robust PSD p=%d seed=%d proj=%d fallback=%s gap=%+.2e"
                  % (p, seed, ex["n_finite_projective"], ex["fallback"],
                     ms["f"] - ex["f_enum"]), flush=True)
    # (b) ALM-assembled M_A
    for seed in (0, 1, 2):
        ins = C.make_instance(seed, 16, noise=0.01)
        M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
        Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48,
                                       rng=np.random.default_rng(1000 + seed))
        r = C.alm_qpep(M0, D0, G0, A, B, qu, tu, rho0=10.0, n_starts=6,
                       rng=np.random.default_rng(4000 + seed))
        g = C._gfun(A, B, r["q"], r["t"])
        act = np.where(r["mu"] + r["rho"] * g > 0.0)[0]
        M_A, _ = _alm_Mred(M0, D0, G0, A, B, r["mu"], r["rho"], act)
        M_A = 0.5 * (M_A + M_A.T)
        ev = np.linalg.eigvalsh(M_A)
        ex = Q.solve_inner_exact(M_A, seed=0, n_gamma=n_gamma)
        ms = _ms(M_A, 256, seed)
        out["alm_M"].append(dict(
            seed=seed, n_act=int(len(act)), eig_min=float(ev.min()),
            n_finite_projective=ex["n_finite_projective"],
            n_unique=ex["n_unique"], f_exact=ex["f"], f_enum=ex["f_enum"],
            f_ms256=ms["f"], gap=ms["f"] - ex["f_enum"],
            fallback=ex["fallback"], degenerate=ex["degenerate"],
            n_lost=ex["n_lost"], rot_deg=C.rot_deg(ex["q"], ms["q"]),
            t_exact=ex["stats"]["t_total"]))
        print("  robust ALM seed=%d act=%d proj=%d fallback=%s gap=%+.2e"
              % (seed, len(act), ex["n_finite_projective"], ex["fallback"],
                 ms["f"] - ex["f_enum"]), flush=True)
    # (c) synthetic singular M: rank-1 and identically zero
    rand = np.random.default_rng(0).normal(size=(10, 10))
    rand = 0.5 * (rand + rand.T)
    specs = {"rank1_diag": np.diag([0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]),
             "zero": np.zeros((10, 10)),
             "rank1_random": np.outer(rand[:, 0], rand[:, 0])}
    for name, M in specs.items():
        ex = Q.solve_inner_exact(M, seed=0, n_gamma=n_gamma)
        ms = _ms(M, 64, 0)
        out["synthetic"].append(dict(
            name=name, n_finite_projective=ex["n_finite_projective"],
            n_unique=ex["n_unique"], f_exact=ex["f"], f_enum=ex["f_enum"],
            f_ms64=ms["f"], fallback=ex["fallback"],
            degenerate=ex["degenerate"], n_lost=ex["n_lost"],
            sigma_min_best=ex["sigma_min_best"]))
        print("  robust synthetic %s proj=%d fallback=%s degenerate=%s"
              % (name, ex["n_finite_projective"], ex["fallback"],
                 ex["degenerate"]), flush=True)
    return out


def phase_alm(trials=8, p=16, noise=0.01, bias_z=0.08, n_gamma=1):
    rows = []
    for t in range(trials):
        ins = C.make_instance(t, p, noise=noise, bias_z=bias_z)
        M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
        Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48,
                                       rng=np.random.default_rng(1000 + t))
        rec = dict(trial=t, J_unc=Ju)
        for mode in ("multistart", "exact"):
            t0 = time.perf_counter()
            r = C.alm_qpep(M0, D0, G0, A, B, qu, tu, rho0=10.0, n_starts=6,
                           rng=np.random.default_rng(4000 + t), inner=mode,
                           inner_kw=dict(n_gamma=n_gamma) if mode == "exact" else None)
            dt = time.perf_counter() - t0
            pen = C.penetration_mm(C.R_of(r["q"]), r["t"], ins["V_dense"])
            rec[mode] = dict(
                J=_f(r["J"]), viol=float(r["viol"]), iters=int(r["iters"]),
                time=dt, fallback_count=int(r["fallback_count"]),
                feasible=bool(C.max_viol(A, B, r["q"], r["t"]) <= 1e-8),
                penetration_mm=pen,
                rot_deg=C.rot_deg(r["q"], ins["q_gt"]),
                trans_mm=C.trans_mm(r["t"], ins["t_gt"]),
                exact_calls=int(r.get("exact_calls", 0)),
                exact_fallbacks=int(r.get("exact_fallbacks", 0)),
                exact_time=float(r.get("exact_time", 0.0)),
                n_act=int(np.sum(r["mu"] > 1e-9)))
        rec["dJ"] = rec["multistart"]["J"] - rec["exact"]["J"]
        rec["speed_ratio"] = (rec["exact"]["time"] / rec["multistart"]["time"]
                              if rec["multistart"]["time"] > 0 else None)
        rows.append(rec)
        print("  ALM trial=%d Jms=%.10f Jex=%.10f dJ=%+.2e t_ms=%.2fs t_ex=%.2fs "
              "ratio=%.1f fb=%d/%d"
              % (t, rec["multistart"]["J"], rec["exact"]["J"], rec["dJ"],
                 rec["multistart"]["time"], rec["exact"]["time"],
                 rec["speed_ratio"], rec["exact"]["fallbacks"]
                 if "fallbacks" in rec["exact"] else rec["exact"]["exact_fallbacks"],
                 rec["exact"]["exact_calls"]), flush=True)
    return dict(trials=rows, p=p, noise=noise, bias_z=bias_z, n_gamma=n_gamma)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["generic", "p2", "perturb",
                                                    "robust", "alm"])
    ap.add_argument("--n-gamma-generic", type=int, default=2)
    ap.add_argument("--n-gamma-p2", type=int, default=1)
    ap.add_argument("--n-gamma-alm", type=int, default=1)
    ap.add_argument("--alm-trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    data = {}
    if os.path.exists(OUT):
        try:
            with open(OUT) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data.setdefault("meta", {}).update(dict(
        numpy=np.__version__, started=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=os.uname().nodename, git_branch=None))

    t_all = time.perf_counter()
    if "generic" in args.phases:
        print("[phase generic]", flush=True)
        data["generic"] = phase_generic(n_gamma=args.n_gamma_generic)
    if "p2" in args.phases:
        print("[phase p2]", flush=True)
        data["p2"] = phase_p2(n_gamma=args.n_gamma_p2)
    if "perturb" in args.phases:
        print("[phase perturb]", flush=True)
        data["perturb"] = phase_perturb(n_gamma=2)
    if "robust" in args.phases:
        print("[phase robust]", flush=True)
        data["robust"] = phase_robust(n_gamma=1)
    if "alm" in args.phases:
        print("[phase alm]", flush=True)
        data["alm"] = phase_alm(trials=args.alm_trials,
                                n_gamma=args.n_gamma_alm)
    data["meta"]["wall_total"] = time.perf_counter() - t_all
    data["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", OUT, "in %.1fs" % data["meta"]["wall_total"], flush=True)


if __name__ == "__main__":
    main()
