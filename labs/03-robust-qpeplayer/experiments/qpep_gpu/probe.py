#!/usr/bin/env python3
"""Throwaway probe: capture safety / accuracy / small-B latency of the backends."""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "systems"))
sys.path.insert(0, str(HERE.parent / "synthetic"))
import common as cm
import gpu_solver as gs
import batch_gnc as bg

BASE_SEED = 20261127


def make_batch(b, n, p, trial):
    ss = np.random.SeedSequence([BASE_SEED, 11, int(b), int(n),
                                 int(round(100 * p)), int(trial)])
    xs = np.empty((b, n, 3)); ys = np.empty((b, n, 3))
    for i, child in enumerate(ss.spawn(b)):
        x, y, _, _ = cm.make_instance(np.random.default_rng(child), p, n)
        xs[i] = x; ys[i] = y
    return xs, ys


torch.manual_seed(0)
dev = "cuda"
for (b, n) in [(1, 100), (8, 100), (32, 500)]:
    x, y = make_batch(b, n, 0.3, 0)
    xT = torch.tensor(x.transpose(0, 2, 1).copy(), dtype=torch.float64, device=dev)
    yT = torch.tensor(y.transpose(0, 2, 1).copy(), dtype=torch.float64, device=dev)
    Rref, tref, inf = bg.batch_solve_gnc(x, y, method="eigh", return_info=True)
    print(f"\n== B={b} N={n}")
    # eager dynamic
    Rd, td, infd = gs.solve_gnc_t(xT, yT)
    dR = (Rd.cpu().numpy() - Rref); print("eager dyn rot", np.abs(dR).max(), "nmu match", (infd["n_mu"].cpu().numpy() == inf["n_mu"]).all())
    for eigen in ("eigh", "jacobi"):
        t0 = time.perf_counter(); Rs, ts, infs = gs.solve_gnc_static(xT, yT, eigen=eigen); torch.cuda.synchronize(); t1 = time.perf_counter()
        dR = np.abs(Rs.cpu().numpy() - Rref).max(); dT = np.abs(ts.cpu().numpy() - tref).max()
        nm = (infs["n_mu"].cpu().numpy() == inf["n_mu"]).all()
        print(f"static {eigen:7s} maxdR={dR:.2e} maxdT={dT:.2e} nmu={nm} eager_ms={1000*(t1-t0):.1f}")
        # graph capture
        try:
            g = gs.CudaGraphSolver(lambda a, b_: gs.solve_gnc_static(a, b_, eigen=eigen), xT, yT, warmup=2)
            for _ in range(3): g.replay(xT, yT)
            torch.cuda.synchronize()
            ts_ = []
            for _ in range(20):
                torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(xT, yT); torch.cuda.synchronize(); ts_.append(time.perf_counter() - t0)
            Rg, tg, infg = g.replay(xT, yT)
            dR = np.abs(Rg.cpu().numpy() - Rref).max(); dT = np.abs(tg.cpu().numpy() - tref).max()
            print(f"  graph {eigen:7s} p50={1000*np.median(ts_):.2f}ms maxdR={dR:.2e} maxdT={dT:.2e} nmu={(infg['n_mu'].cpu().numpy()==inf['n_mu']).all()}")
        except Exception as exc:
            print(f"  graph {eigen:7s} FAILED: {type(exc).__name__}: {str(exc)[:300]}")
print("\ndone")
