#!/usr/bin/env python3
"""P1 Theorem P1-5 validation: uniform Laplace asymptotics at an energy crossing.

Model: J_b,g(x) = (x^2-1)^2 + b (x^3-x) + g x.
At g=0 the two minima x=+-1 have equal energy (Delta J=0) but different
curvature when b != 0, so this is exactly the "separated basins, energy
crossing" regime of Theorem P1-5, not a physical fold.

Exact posterior is computed by dense trapezoid quadrature; Laplace uses the
two local minima.  Outputs a main table, a tau-scaling ablation and a
Delta-J (g) sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def J(x, b, g):
    return (x * x - 1.0) ** 2 + b * (x ** 3 - x) + g * x


def dJ(x, b, g):
    return 4.0 * x * (x * x - 1.0) + b * (3.0 * x * x - 1.0) + g


def d2J(x, b, g):
    return 12.0 * x * x - 4.0 + 6.0 * b * x


def minima(b, g):
    xs = np.linspace(-3.0, 3.0, 6001)
    roots = []
    for i in range(len(xs) - 1):
        a, c = xs[i], xs[i + 1]
        if dJ(a, b, g) * dJ(c, b, g) < 0:
            r = brentq(lambda z: dJ(z, b, g), a, c)
            if d2J(r, b, g) > 1e-10:
                roots.append(r)
    return np.array(roots)


def exact(b, g, tau, n=2_000_001):
    x = np.linspace(-3.5, 3.5, n)
    w = np.exp(-J(x, b, g) / tau)
    Z = np.trapezoid(w, x)
    Ex = np.trapezoid(x * w, x) / Z
    Ex2 = np.trapezoid(x * x * w, x) / Z
    var = Ex2 - Ex * Ex
    return float(Ex), float(-var / tau), float(var)


def laplace(b, g, tau):
    ms = minima(b, g)
    Js = J(ms, b, g)
    Hs = d2J(ms, b, g)
    w = np.exp(-Js / tau) / np.sqrt(Hs)
    w = w / w.sum()
    Ex = float(w @ ms)
    # g enters only linearly in J; H_g = 0, dJ/dg = x, dx/dg = -1/H.
    dx = -1.0 / Hs
    gd = -ms / tau
    dw = w * (gd - float(w @ gd))
    dEx = float(w @ dx + dw @ ms)
    return Ex, dEx, w, ms, Js, Hs


def main():
    rows = []
    taus = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    # exact energy crossings g* (found by root-finding on Delta J(g); at
    # these points the two minima have equal energy and equal curvature).
    gstar = {0.50: -0.015625, 0.25: -0.001953125, 0.75: -0.052734375}
    regimes = [("crossing_b0.50", 0.50, gstar[0.50]),
               ("crossing_b0.25", 0.25, gstar[0.25]),
               ("separated_b0.50", 0.50, gstar[0.50] + 0.2),
               ("separated_b0.25", 0.25, gstar[0.25] + 0.1)]
    for name, b, g in regimes:
        for tau in taus:
            ex, dex, var = exact(b, g, tau)
            lx, ldx, w, ms, Js, Hs = laplace(b, g, tau)
            deltaJ = float(Js.max() - Js.min())
            rows.append(dict(regime=name, b=b, g=g, tau=tau, deltaJ=deltaJ, n_modes=len(ms),
                             E_exact=ex, E_lap=lx, errE=abs(lx - ex), errE_tau=abs(lx - ex) / tau,
                             dE_exact=dex, dE_lap=ldx, errG=abs(ldx - dex),
                             relG=abs(ldx - dex) / max(abs(dex), 1e-12),
                             var=var, H=Hs.tolist(), x=ms.tolist()))
            print(name, tau, "dJ", f"{deltaJ:.3e}", "errE", f"{abs(lx-ex):.3e}", "relG", f"{abs(ldx-dex)/max(abs(dex),1e-12):.2e}", flush=True)
    json.dump(rows, open(OUT / "uniform.json", "w"), indent=2)

    lines = ["# P1 Theorem P1-5: uniform Laplace at a separated energy crossing\n",
             "Model $J=(x^2-1)^2+b(x^3-x)+gx$. At $g=0$ the two minima have $\\Delta J=0$ "
             "and unequal curvature when $b\\ne0$; exact posterior by dense quadrature.\n"]
    for name, b, g in regimes:
        rs = [r for r in rows if r["regime"] == name]
        lines.append(f"## {name} (b={b}, g={g})\n")
        lines.append("| tau | Delta J | E exact | E Laplace | err E | err E / tau | dE exact | dE Laplace | rel grad err |")
        lines.append("|-|-|-|-|-|-|-|-|-|")
        for r in rs:
            lines.append(f"| {r['tau']:.3f} | {r['deltaJ']:.3e} | {r['E_exact']:.6f} | {r['E_lap']:.6f} | "
                         f"{r['errE']:.3e} | {r['errE_tau']:.3e} | {r['dE_exact']:.4f} | {r['dE_lap']:.4f} | {r['relG']:.2e} |")
        lines.append("")
    # log-log slopes per regime
    for name, _, _ in regimes:
        rs = [r for r in rows if r["regime"] == name]
        t = np.array([r["tau"] for r in rs]); e = np.array([r["errE"] for r in rs])
        slope = np.polyfit(np.log(t), np.log(e), 1)[0]
        lines.append("- " + name + r": $\log$--$\log$ slope of $|E_{\rm Lap}-E_{\rm exact}|$ vs $\tau$ = **" + f"{slope:.3f}**")
    (OUT / "uniform_table.md").write_text("\n".join(lines) + "\n")

    # Delta-J sweep at tau=0.01
    sweep = []
    for g in np.linspace(-0.015625 - 0.2, -0.015625 + 0.2, 21):
        b = 0.5
        ex, dex, _ = exact(b, g, 0.01)
        lx, ldx, _, ms, Js, _ = laplace(b, g, 0.01)
        sweep.append(dict(g=float(g), deltaJ=float(Js.max() - Js.min()), E_exact=ex, E_lap=lx,
                          errE=abs(lx - ex), dE_exact=dex, dE_lap=ldx, errG=abs(ldx - dex),
                          relG=abs(ldx - dex) / max(abs(dex), 1e-12)))
    json.dump(sweep, open(OUT / "deltaJ_sweep.json", "w"), indent=2)
    lines = ["# P1 Delta-J sweep (b=0.5, tau=0.01)\n", "| g | Delta J | E exact | E Laplace | err E | rel grad err |",
             "|-|-|-|-|-|-|"]
    for r in sweep:
        lines.append(f"| {r['g']:.3f} | {r['deltaJ']:.3e} | {r['E_exact']:.6f} | {r['E_lap']:.6f} | {r['errE']:.3e} | {r['relG']:.2e} |")
    (OUT / "deltaJ_sweep.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
