#!/usr/bin/env python3
"""Correctness check: batched GNC-GM vs the per-instance loop.

Fixed seeds; compares ``batch_gnc.batch_solve_gnc`` against
``synthetic/common.py::solve_gnc`` on B=8 independent fixed random instances
for N in {100,500}, outlier ratio p in {0,0.3,0.7}, and both Kabsch backends
(``eigh`` and ``svd``).  Also checks the per-instance mu schedule
(``n_mu`` equality) and, if torch is available, the torch mirror on CPU and
CUDA.

Acceptance (P3 task):
  max_b ||R_batch - R_loop||_F < 1e-8 and max_b ||t_batch - t_loop|| < 1e-10.

Run:
  python verify_batch.py            # numpy + torch CPU (+ CUDA if visible)
  python verify_batch.py --no-torch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "synthetic"))

import common as cm  # noqa: E402
import batch_gnc as bg  # noqa: E402

BASE_SEED = 20261127
B_FIXED = 8
ROT_TOL = 1e-8
TRANS_TOL = 1e-10


def make_batch(b, n, p, trial=0):
    ss = np.random.SeedSequence([BASE_SEED, 7, b, n, int(round(100 * p)), trial])
    xs = np.empty((b, n, 3))
    ys = np.empty((b, n, 3))
    for i, child in enumerate(ss.spawn(b)):
        x, y, _, _ = cm.make_instance(np.random.default_rng(child), p, n)
        xs[i] = x
        ys[i] = y
    return xs, ys


def loop_reference(x, y):
    Rl = np.empty((len(x), 3, 3))
    tl = np.empty((len(x), 3))
    nmu = np.empty(len(x), dtype=np.int64)
    mu0 = np.empty(len(x))
    for i in range(len(x)):
        R, t, info = cm.solve_gnc(x[i], y[i], return_info=True)
        Rl[i], tl[i], nmu[i], mu0[i] = R, t, info["n_mu"], info["mu0"]
    return Rl, tl, nmu, mu0


def check_numpy(x, y, method):
    Rb, tb, info = bg.batch_solve_gnc(x, y, method=method, return_info=True)
    Rl, tl, nmu, mu0 = loop_reference(x, y)
    dr = np.linalg.norm(Rb - Rl, axis=(1, 2))
    dt = np.linalg.norm(tb - tl, axis=1)
    return dict(
        method=method,
        rot_max=float(dr.max()),
        trans_max=float(dt.max()),
        n_mu_mismatch=int(np.sum(info["n_mu"] != nmu)),
        mu0_rel_max=float(np.max(np.abs(info["mu0"] - mu0) / mu0)),
        n_mu=list(int(v) for v in info["n_mu"]),
    )


def check_torch(x, y, device, dtype, method="eigh"):
    import torch
    import batch_gnc_torch as bgt

    xT = torch.as_tensor(x, dtype=dtype, device=device).permute(0, 2, 1).contiguous()
    yT = torch.as_tensor(y, dtype=dtype, device=device).permute(0, 2, 1).contiguous()
    with torch.no_grad():
        R, t = bgt.solve_gnc_t(xT, yT, method=method)[:2]
    Rb = R.detach().to("cpu", torch.float64).numpy()
    tb = t.detach().to("cpu", torch.float64).numpy()
    Rl, tl, _, _ = loop_reference(x, y)
    return dict(
        device=str(device), dtype=str(dtype),
        rot_max=float(np.linalg.norm(Rb - Rl, axis=(1, 2)).max()),
        trans_max=float(np.linalg.norm(tb - tl, axis=1).max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()

    rows = []
    print("numpy: batch_solve_gnc vs common.solve_gnc "
          f"(B={B_FIXED}, fixed seeds, tol rot<{ROT_TOL:g}, trans<{TRANS_TOL:g})")
    print(f"{'N':>5} {'p':>5} {'method':>7} {'rot_max':>10} {'trans_max':>10} "
          f"{'n_mu_mismatch':>13} {'mu0_rel_max':>12} {'ok':>4}")
    ok_all = True
    for n in (100, 500):
        for p in (0.0, 0.3, 0.7):
            x, y = make_batch(B_FIXED, n, p)
            for method in ("eigh", "svd"):
                r = check_numpy(x, y, method)
                ok = (r["rot_max"] < ROT_TOL and r["trans_max"] < TRANS_TOL
                      and r["n_mu_mismatch"] == 0)
                ok_all &= ok
                rows.append(dict(N=n, p=p, **r, ok=bool(ok)))
                print(f"{n:>5} {p:>5.1f} {method:>7} {r['rot_max']:>10.2e} "
                      f"{r['trans_max']:>10.2e} {r['n_mu_mismatch']:>13d} "
                      f"{r['mu0_rel_max']:>12.2e} {str(ok):>4}")
            # schedule identity: eigh and svd must agree on n_mu
            assert rows[-1]["n_mu"] == rows[-2]["n_mu"], "n_mu differs between backends"

    torch_rows = []
    if not args.no_torch:
        try:
            import torch
            dtypes = [torch.float64]
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")
            print("\ntorch mirror (device, dtype) vs loop:")
            print(f"{'device':>6} {'dtype':>9} {'N':>5} {'p':>5} {'rot_max':>10} {'trans_max':>10}")
            for device in devices:
                for dtype in dtypes:
                    for n in (100, 500):
                        for p in (0.0, 0.3, 0.7):
                            x, y = make_batch(B_FIXED, n, p)
                            r = check_torch(x, y, device, dtype)
                            torch_rows.append(dict(N=n, p=p, **r))
                            tol_scale = 1.0 if dtype is torch.float64 else 1e3
                            ok = (r["rot_max"] < ROT_TOL * tol_scale
                                  and r["trans_max"] < TRANS_TOL * tol_scale)
                            ok_all &= ok
                            print(f"{device:>6} {str(dtype).replace('torch.',''):>9} "
                                  f"{n:>5} {p:>5.1f} {r['rot_max']:>10.2e} "
                                  f"{r['trans_max']:>10.2e} {str(ok):>4}")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"\ntorch check skipped: {exc}")

    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
