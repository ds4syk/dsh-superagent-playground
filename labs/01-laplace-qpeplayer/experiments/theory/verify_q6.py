#!/usr/bin/env python3
"""P1-Q6: closed-form d log det H vs branch-following finite differences."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
import qpep_core as c  # noqa: E402


def dlogdet_exact(q, M, dM, dq):
    J = c.Jval(q, M); H = c.hess_J(q, M); E = c.E(q)
    A = H - 4.0 * J * np.eye(4)
    Jv = c.Jv(q); dJv = c.Jv(dq); dE = c.E(dq)
    v = c.v2(q); df = float(v @ dM @ v)
    dH = 2.0 * (dJv.T @ M @ Jv + Jv.T @ dM @ Jv + Jv.T @ M @ dJv)
    for a in range(10):
        dH += 2.0 * float((dM @ v + M @ Jv @ dq)[a]) * c.HV[a]
    dHs = dE.T @ A @ E + E.T @ A @ dE + E.T @ (dH - 4.0 * df * np.eye(4)) @ E
    Hs = E.T @ A @ E
    return float(np.trace(np.linalg.solve(Hs, dHs)))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for trial in range(5):
        A = rng.normal(size=(10, 10)); M = A @ A.T
        B = rng.normal(size=(10, 10)); dM = 0.5 * (B + B.T)
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        for _ in range(100):
            g = c.proj_grad(q, M); H = c.H_sphere(q, M); E = c.E(q)
            q = q + E @ (-np.linalg.solve(H, g)); q /= np.linalg.norm(q)
        dq = c.ift_dq(q, M, dM)
        exact = dlogdet_exact(q, M, dM, dq)
        eps = 1e-6
        def branch(Mx):
            qq = q.copy()
            for _ in range(60):
                g = c.proj_grad(qq, Mx); H = c.H_sphere(qq, Mx); E = c.E(qq)
                qq = qq + E @ (-np.linalg.solve(H, g)); qq /= np.linalg.norm(qq)
            return qq
        qp = branch(M + eps * dM); qm = branch(M - eps * dM)
        fd = (c.logdetH(qp, M + eps * dM) - c.logdetH(qm, M - eps * dM)) / (2 * eps)
        rows.append(dict(trial=trial, exact=exact, fd=fd, abs_err=abs(exact - fd),
                         rel_err=abs(exact - fd) / max(abs(fd), 1e-12)))
        print(rows[-1], flush=True)
    json.dump(rows, open(HERE / "results" / "q6_verify.json", "w"), indent=2)
    lines = ["# P1-Q6 closed-form d log det H vs branch FD\n",
             "| trial | exact | FD | abs err | rel err |", "|-|-|-|-|-|"]
    for r in rows:
        lines.append(f"| {r['trial']} | {r['exact']:.6e} | {r['fd']:.6e} | {r['abs_err']:.2e} | {r['rel_err']:.2e} |")
    (HERE / "results" / "q6_verify.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
