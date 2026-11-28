#!/usr/bin/env python3
"""P1 Theorem P1-5 crossing with *unequal* curvatures.

J(x) = (x^2-1)^2 (1+c x) + g x.  At g=0 the two minima are exactly x=+-1,
both with energy 0 (Delta J=0), but curvatures 4(1+c) and 4(1-c) differ.
This is the clean separated-basin crossing test (not a fold).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def J(x, c, g):
    return (x * x - 1.0) ** 2 * (1.0 + c * x) + g * x


def dJ(x, c, g):
    return 4.0 * x * (x * x - 1.0) * (1.0 + c * x) + c * (x * x - 1.0) ** 2 + g


def d2J(x, c, g):
    return (12.0 * x * x - 4.0) * (1.0 + c * x) + 2.0 * c * 4.0 * x * (x * x - 1.0) + 2.0 * c * c * (x * x - 1.0) * 0.0


def d2J_exact(x, c, g):
    # derivative of dJ
    return (12.0 * x * x - 4.0) * (1.0 + c * x) + 8.0 * c * x * (x * x - 1.0)


def minima(c, g):
    out = []
    for lo, hi in [(0.5, 2.0), (-2.0, -0.5)]:
        try:
            r = brentq(lambda z: dJ(z, c, g), lo, hi)
            if d2J_exact(r, c, g) > 0:
                out.append(r)
        except ValueError:
            pass
    return np.array(out)


def exact(c, g, tau, n=2_000_001):
    x = np.linspace(-3.0, 3.0, n)
    w = np.exp(-J(x, c, g) / tau)
    Z = np.trapezoid(w, x)
    Ex = np.trapezoid(x * w, x) / Z
    Ex2 = np.trapezoid(x * x * w, x) / Z
    var = Ex2 - Ex * Ex
    return float(Ex), float(-var / tau), float(var)


def laplace(c, g, tau):
    ms = minima(c, g)
    Js = J(ms, c, g)
    Hs = d2J_exact(ms, c, g)
    w = np.exp(-Js / tau) / np.sqrt(Hs)
    w = w / w.sum()
    Ex = float(w @ ms)
    dx = -1.0 / Hs
    gd = -ms / tau
    dw = w * (gd - float(w @ gd))
    dEx = float(w @ dx + dw @ ms)
    return Ex, dEx, w, ms, Js, Hs


def main():
    c = 0.25
    rows = []
    for label, g in [("crossing", 0.0), ("separated", 0.2)]:
        for tau in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            ex, dex, var = exact(c, g, tau)
            lx, ldx, w, ms, Js, Hs = laplace(c, g, tau)
            rows.append(dict(regime=label, g=g, c=c, tau=tau, deltaJ=float(Js.max() - Js.min()),
                             E_exact=ex, E_lap=lx, errE=abs(lx - ex), errE_tau=abs(lx - ex) / tau,
                             dE_exact=dex, dE_lap=ldx, errG=abs(ldx - dex),
                             relG=abs(ldx - dex) / max(abs(dex), 1e-12),
                             H=Hs.tolist(), x=ms.tolist()))
            print(label, tau, "dJ", Js.max() - Js.min(), "errE", abs(lx - ex), "relG", rows[-1]["relG"], flush=True)
    json.dump(rows, open(OUT / "uniform_unequal.json", "w"), indent=2)
    lines = ["# P1 crossing with unequal curvatures\n",
             "Model $J=(x^2-1)^2(1+cx)+gx$, $c=0.25$; at $g=0$ both minima are at $x=\\pm1$ "
             "with $\\Delta J=0$ and curvatures $5$ and $3$.\n",
             "| regime | tau | Delta J | E exact | E Laplace | err E | err E / tau | rel grad err |",
             "|-|-|-|-|-|-|-|-|"]
    for r in rows:
        lines.append(f"| {r['regime']} | {r['tau']:.3f} | {r['deltaJ']:.3e} | {r['E_exact']:.6f} | {r['E_lap']:.6f} | {r['errE']:.3e} | {r['errE_tau']:.3e} | {r['relG']:.2e} |")
    for label in ["crossing", "separated"]:
        rs = [r for r in rows if r["regime"] == label]
        t = np.array([r["tau"] for r in rs]); e = np.array([r["errE"] for r in rs])
        slope = np.polyfit(np.log(t), np.log(e), 1)[0]
        lines.append(f"\n- {label}: log-log slope of err E vs tau = **{slope:.3f}**")
    (OUT / "uniform_unequal_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
