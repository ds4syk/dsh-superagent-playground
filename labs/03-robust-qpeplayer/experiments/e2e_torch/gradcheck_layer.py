#!/usr/bin/env python3
"""Validation of the torch robust-pose layer backward.

Three checks:

1. **Port fidelity**: the torch forward pose and the torch analytic dL/dy are
   compared elementwise with the NumPy reference in
   ``experiments/synthetic/common.py`` (``newton_polish``,
   ``robust_hessian_exact/naive``, ``dphi_dy_exact/naive``).
2. **torch.autograd.gradcheck**: central finite differences of the layer map
   y -> (R, t), float64, N in {6,8,10}.  ``exact`` must pass; ``naive`` is run
   through the same protocol to show it does not.
3. **Max relative error**: analytic vs central-difference gradient of a fixed
   random scalar loss (the same quantity ``gradcheck`` checks, reported as
   numbers).

Run:
  python gradcheck_layer.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic"))

import robust_layer as rl  # noqa: E402

DT = rl.DT


def _instance(n, seed, p=0.3):
    import common as cm
    rng = np.random.default_rng(seed)
    x, y, Rgt, tgt, meta = cm.make_instance(rng, p, n=n,
                                            outlier_type="boundary",
                                            return_meta=True)
    return x, y, Rgt, tgt


def _scalar_loss_grad(y, xt, At, Wr, Wt, mode):
    yt = torch.as_tensor(y, dtype=DT).clone().requires_grad_(True)
    R, t = rl.robust_pose(yt, xt, At, mode=mode)
    L = (R * Wr).sum() + (t * Wt).sum()
    (g,) = torch.autograd.grad(L, yt)
    return g.detach(), float(L.detach())


def fd_grad(y, xt, At, Wr, Wt, mode, h=1e-6):
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    g = np.zeros_like(y)
    for i in range(n):
        for k in range(3):
            yp, ym = y.copy(), y.copy()
            yp[i, k] += h
            ym[i, k] -= h
            _, lp = _scalar_loss_grad(yp, xt, At, Wr, Wt, mode)
            _, lm = _scalar_loss_grad(ym, xt, At, Wr, Wt, mode)
            g[i, k] = (lp - lm) / (2.0 * h)
    return g


def common_reference(x, y, Rgt, tgt, mode):
    """NumPy reference pose + dL/dy from experiments/synthetic/common.py."""
    import common as cm
    A = [cm.build_A(x[i]) for i in range(len(x))]
    Rg, tg = cm.solve_gnc(x, y)
    q0, t0, H, pn, nit = cm.newton_polish(x, y, cm.quat_from_R(Rg), tg, A_list=A)
    Bq = cm.tangent_basis(q0)
    terms = cm.robust_terms(x, y, q0, t0, A_list=A)
    dL = cm.dL_dz(q0, t0, Rgt, tgt)
    B = np.zeros((7, 6))
    B[:4, :3] = Bq
    B[4:, 3:] = np.eye(3)
    if mode == "exact":
        Hm, lam, gtot, _ = cm.robust_hessian_exact(terms, q0, Bq)
        dphi = cm.dphi_dy_exact
    else:
        Hm, lam = cm.robust_hessian_naive(terms, q0, Bq)
        dphi = cm.dphi_dy_naive
    g = np.zeros_like(np.asarray(y, dtype=float))
    for i in range(len(x)):
        J = -np.linalg.solve(Hm, dphi(terms, i, B))
        g[i] = J.T @ dL
    return cm.Rmat(q0), t0, g, pn


def check_case(n, seed, p=0.3, mode="exact", fd_h=1e-6, do_gradcheck=True):
    x, y, Rgt, tgt = _instance(n, seed, p=p)
    xt = torch.as_tensor(x, dtype=DT)
    At = rl.make_A(xt)
    Wr = torch.randn(3, 3, dtype=DT, generator=torch.Generator().manual_seed(seed))
    Wt = torch.randn(3, dtype=DT, generator=torch.Generator().manual_seed(seed + 1))

    out = dict(n=n, seed=seed, p=p, mode=mode)
    t0 = time.time()
    if do_gradcheck:
        yt = torch.as_tensor(y, dtype=DT).clone().requires_grad_(True)

        def f(yy):
            return rl.robust_pose(yy, xt, At, mode=mode)

        out["gradcheck_pass"] = bool(torch.autograd.gradcheck(
            f, (yt,), eps=fd_h, atol=1e-7, rtol=1e-4,
            raise_exception=False, fast_mode=False))
    else:
        out["gradcheck_pass"] = None
    out["gradcheck_s"] = time.time() - t0

    ga, _ = _scalar_loss_grad(y, xt, At, Wr, Wt, mode)
    gfd = torch.as_tensor(fd_grad(y, xt, At, Wr, Wt, mode, h=fd_h), dtype=DT)
    diff = torch.abs(ga - gfd)
    rel = diff / torch.clamp(torch.maximum(torch.abs(ga), torch.abs(gfd)),
                             min=1e-12)
    out["max_abs_err"] = float(diff.max())
    out["max_rel_err"] = float(rel.max())
    out["median_rel_err"] = float(rel.median())
    out["grad_scale"] = float(ga.abs().max())
    out["rel_to_scale"] = float(diff.max()) / max(out["grad_scale"], 1e-12)
    return out


def port_fidelity(n=10, seed=0, p=0.3):
    x, y, Rgt, tgt = _instance(n, seed, p=p)
    xt = torch.as_tensor(x, dtype=DT)
    yt = torch.as_tensor(y, dtype=DT)
    At = rl.make_A(xt)
    q, t, info = rl.solve_robust(xt, yt, At)
    R = rl.Rmat(q)
    out = dict(n=n, seed=seed, newton_phi=info["pn"])
    for mode in ("exact", "naive"):
        Rref, tref, gref, pn = common_reference(x, y, Rgt, tgt, mode)
        gR = 2.0 * (R - torch.as_tensor(Rgt, dtype=DT))
        gt = 2.0 * (t - torch.as_tensor(tgt, dtype=DT))
        gtorch = rl.backward_y(q, t, xt, yt, At, gR, gt, mode=mode)
        out[f"pose_R_diff_{mode}"] = float(np.abs(R.detach().numpy() - Rref).max())
        out[f"pose_t_diff_{mode}"] = float(np.abs(t.detach().numpy() - tref).max())
        out[f"grad_y_diff_{mode}"] = float(np.abs(gtorch.numpy() - gref).max())
        out[f"common_phi_{mode}"] = float(pn)
    return out


def main(ns=(6, 8, 10), seeds=(0,), p=0.3):
    cases = []
    for n in ns:
        for seed in seeds:
            for mode in ("exact", "naive"):
                c = check_case(n, seed, p=p, mode=mode)
                cases.append(c)
                print(f"N={n} seed={seed} {mode:5s} gradcheck="
                      f"{c['gradcheck_pass']} max_abs={c['max_abs_err']:.2e} "
                      f"max_rel={c['max_rel_err']:.2e} "
                      f"med_rel={c['median_rel_err']:.2e}", flush=True)
    fid = port_fidelity()
    print("port fidelity vs common.py:", json.dumps(fid, indent=2))
    data = dict(
        protocol=dict(eps=1e-6, atol=1e-7, rtol=1e-4, dtype="float64",
                      outlier="boundary", p=p, ns=list(ns), seeds=list(seeds)),
        cases=cases,
        port_fidelity=fid,
        summary=dict(
            exact_gradcheck_pass=all(c["gradcheck_pass"] for c in cases if c["mode"] == "exact"),
            naive_gradcheck_pass=all(c["gradcheck_pass"] for c in cases if c["mode"] == "naive"),
            exact_max_rel_err=max(c["max_rel_err"] for c in cases if c["mode"] == "exact"),
            exact_median_rel_err=float(np.median([c["median_rel_err"] for c in cases if c["mode"] == "exact"])),
            naive_max_rel_err=max(c["max_rel_err"] for c in cases if c["mode"] == "naive"),
            naive_median_rel_err=float(np.median([c["median_rel_err"] for c in cases if c["mode"] == "naive"])),
        ),
    )
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "gradcheck.json")
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    d = main(ns=args.ns, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(d, indent=2))
    print("wrote", args.out)
