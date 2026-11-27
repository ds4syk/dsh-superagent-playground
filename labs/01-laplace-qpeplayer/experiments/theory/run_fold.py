#!/usr/bin/env python3
"""P1 Theorem P1-5 regime II: Fold (A2) merging and the tau^(1/3) regularizer.

Model J_a(x)=x^4/4 - a x^2/2 (symmetric double well for a>0).  As a -> 0 the
two minima +-sqrt(a) merge with the saddle at 0: the barrier and the Hessian
eigenvalue both vanish.  Standard Laplace has 1/sqrt(det H) -> infinity,
while the exact partition function is finite; the tau^(1/3) regularized
volume factor is compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def J(x, a):
    return x ** 4 / 4.0 - a * x * x / 2.0


def d2J(x, a):
    return 3.0 * x * x - a


def exact(a, tau, n=2_000_001):
    x = np.linspace(-3.5, 3.5, n)
    w = np.exp(-J(x, a) / tau)
    Z = np.trapezoid(w, x)
    Ex2 = np.trapezoid(x * x * w, x) / Z
    return float(Z), float(Ex2)


def laplace(a, tau, eta=0.0):
    if a <= 0:
        return float("nan"), float("nan")
    x0 = np.sqrt(a)
    H = d2J(x0, a)
    Z = 2.0 * np.sqrt(2.0 * np.pi * tau) / np.sqrt(H + eta) * np.exp(-J(x0, a) / tau)
    return float(Z), float(x0 * x0)


def main():
    taus = [0.02, 0.01, 0.005]
    avec = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    rows = []
    for tau in taus:
        for a in avec:
            Zex, Ex2 = exact(a, tau)
            Zlap, _ = laplace(a, tau, 0.0)
            # Fold scaling: eta = tau^(1/3) (with a fixed O(1) constant).
            eta = tau ** (1.0 / 3.0)
            Zreg, _ = laplace(a, tau, eta)
            rows.append(dict(a=a, tau=tau, H=2.0 * a, barrier=a * a / 4.0,
                             Z_exact=Zex, Z_lap=Zlap, Z_reg=Zreg,
                             err_lap=abs(Zlap - Zex) / Zex,
                             err_reg=abs(Zreg - Zex) / Zex))
            print(f"tau {tau} a {a} H {2*a:.3e} err_lap {abs(Zlap-Zex)/Zex:.3e} err_reg {abs(Zreg-Zex)/Zex:.3e}", flush=True)
    json.dump(rows, open(OUT / "fold.json", "w"), indent=2)
    lines = ["# P1 regime II: Fold merging vs tau^(1/3) regularization\n",
             "Model $J=x^4/4-a x^2/2$. Exact partition by quadrature; standard Laplace "
             "$Z\\propto 1/\\sqrt{H}$; regularized $H\\to H+\\tau^{1/3}$.\n",
             "| tau | a | H | barrier | Z exact | Z Laplace | rel err Laplace | Z regularized | rel err regularized |",
             "|-|-|-|-|-|-|-|-|-|"]
    for r in rows:
        lines.append(f"| {r['tau']:.3f} | {r['a']:.3f} | {r['H']:.3e} | {r['barrier']:.3e} | "
                     f"{r['Z_exact']:.4e} | {r['Z_lap']:.4e} | {r['err_lap']:.3e} | {r['Z_reg']:.4e} | {r['err_reg']:.3e} |")
    (OUT / "fold_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
