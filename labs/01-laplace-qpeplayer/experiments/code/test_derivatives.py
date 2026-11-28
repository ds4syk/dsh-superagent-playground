#!/usr/bin/env python3
"""Regression tests for the P1 derivative stack.

Motivated by a real bug found on 2026-11-27: `qpep_core.dE` (the partial-derivative
tensor of the tangent basis) had wrong entries for b=1,2,3, so the closed-form
`dlogdetH_dtheta` (used by the Laplace weight derivative) was wrong by O(1) even
though `dlogdet_exact` (which uses the directional `E(dq)`) and the numeric
fallback were fine. This test pins the fix.

Run:
  /home/syk/miniconda3/envs/WM/bin/python code/test_derivatives.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "theory"))
import qpep_core as c  # noqa: E402
from verify_q6 import dlogdet_exact  # noqa: E402


def numeric_branch(q, M, dM, eps=1e-6):
    def branch(Mx):
        qq = q.copy()
        for _ in range(80):
            g = c.proj_grad(qq, Mx)
            H = c.H_sphere(qq, Mx)
            E = c.E(qq)
            qq = qq + E @ (-np.linalg.solve(H, g))
            qq /= np.linalg.norm(qq)
        return qq
    return (c.logdetH(branch(M + eps * dM), M + eps * dM)
            - c.logdetH(branch(M - eps * dM), M - eps * dM)) / (2 * eps)


def main():
    rng = np.random.default_rng(0)
    errs = {"dE": [], "dlogdet_core_vs_exact": [], "dlogdet_core_vs_fd": [],
            "dH_fd": [], "dq_fd": []}
    for _ in range(6):
        A = rng.normal(size=(10, 10)); M = A @ A.T
        B = rng.normal(size=(10, 10)); dM = 0.5 * (B + B.T)
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        for _ in range(100):
            g = c.proj_grad(q, M); H = c.H_sphere(q, M); E = c.E(q)
            q = q + E @ (-np.linalg.solve(H, g)); q /= np.linalg.norm(q)
        eps = 1e-6
        for b in range(4):
            e = np.zeros(4); e[b] = eps
            errs["dE"].append(float(np.max(np.abs(
                (c.E(q + e) - c.E(q - e)) / (2 * eps) - c.dE(q)[b]))))
        dq = c.ift_dq(q, M, dM)
        ex = dlogdet_exact(q, M, dM, dq)
        co = c.dlogdetH_dtheta(q, M, dM, dq)[0]
        fd = numeric_branch(q, M, dM)
        errs["dlogdet_core_vs_exact"].append(abs(co - ex))
        errs["dlogdet_core_vs_fd"].append(abs(co - fd))
        # dH along the branch
        fdH = (c.H_sphere(q + eps * dq, M + eps * dM)
               - c.H_sphere(q - eps * dq, M - eps * dM)) / (2 * eps)
        anH = c.dHsphere_dtheta_fixedq(q, M, dM) + np.einsum(
            "b,bac->ac", dq, c.dHsphere_dq(q, M))
        errs["dH_fd"].append(float(np.max(np.abs(fdH - anH))))
        # dq along the branch
        fq = (numeric_branch_q(q, M, dM, eps) )
        errs["dq_fd"].append(float(np.max(np.abs(fq - dq))))

    print("max abs errors over 6 random instances:")
    ok = True
    for k, v in errs.items():
        print(f"  {k:28s} {max(v):.3e}")
    ok = (max(errs["dE"]) < 1e-8 and max(errs["dlogdet_core_vs_exact"]) < 1e-8
          and max(errs["dlogdet_core_vs_fd"]) < 1e-6 and max(errs["dH_fd"]) < 1e-6)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def numeric_branch_q(q, M, dM, eps):
    def branch(Mx):
        qq = q.copy()
        for _ in range(80):
            g = c.proj_grad(qq, Mx); H = c.H_sphere(qq, Mx); E = c.E(qq)
            qq = qq + E @ (-np.linalg.solve(H, g)); qq /= np.linalg.norm(qq)
        return qq
    qp = branch(M + eps * dM); qm = branch(M - eps * dM)
    # align antipodal signs before differencing
    if qp @ qm < 0:
        qp = -qp
    return (qp - qm) / (2 * eps)


if __name__ == "__main__":
    raise SystemExit(main())
