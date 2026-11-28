#!/usr/bin/env python3
"""Minimal numerical sanity check for the 1-D sign-flip proposition.

Model:  F(t) = sum_i rho((t - y_i)^2),  rho(e) = c^2 e / (c^2 + e)  (Geman-McClure)

At a strict local minimum t* (H_rob = sum_i lambda_parallel(e_i) > 0):
    dt*/dy_j = lambda_parallel(e_j) / H_rob          (exact IFT)
    dt*/dy_j = 2 rho'(e_j) / sum_i 2 rho'(e_i)       (frozen-weight / naive IRLS-StopGrad)
with
    lambda_parallel(e) = 2 (rho'(e) + 2 e rho''(e))
                       = 2 c^4 (c^2 - 3 e) / (c^2 + e)^3.

Checks:
  (1) lambda_parallel closed form, zero at e=c^2/3, minimum -1/2 at e=c^2;
  (2) exact vs naive vs central finite difference at a local minimum with e_2 > c^2/3;
  (3) log-log slope of |dt*/dy_2| versus the residual norm ||r_2||, expected -4.

No scipy is assumed (bisection only).
Run:  /home/syk/miniconda3/bin/python ift_1d_signflip.py
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np

np.set_printoptions(precision=6, suppress=False)

C = 0.5
C2 = C * C
C4 = C2 * C2


def rho(e):
    return C2 * e / (C2 + e)


def drho(e):
    return C4 / (C2 + e) ** 2


def d2rho(e):
    return -2.0 * C4 / (C2 + e) ** 3


def lam(e):
    e = np.asarray(e, dtype=float)
    return 2.0 * (drho(e) + 2.0 * e * d2rho(e))


def lam_closed(e):
    e = np.asarray(e, dtype=float)
    return 2.0 * C4 * (C2 - 3.0 * e) / (C2 + e) ** 3


def Phi(t, y):
    e0 = (t - y[0]) ** 2
    e1 = (t - y[1]) ** 2
    return 2.0 * drho(e0) * (t - y[0]) + 2.0 * drho(e1) * (t - y[1])


def Hrob(t, y):
    return lam((t - y[0]) ** 2) + lam((t - y[1]) ** 2)


def bisect(fun, a, b, n=200):
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("no sign change on bracket")
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = fun(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def local_min_near(y, guess, half=0.2, samples=41):
    """Find a root of Phi near `guess` by scanning a small bracket and bisection."""
    ts = np.linspace(guess - half, guess + half, samples)
    phis = np.array([Phi(t, y) for t in ts])
    roots = []
    for k in range(len(ts) - 1):
        if phis[k] == 0.0:
            roots.append(ts[k])
        elif phis[k] * phis[k + 1] < 0:
            roots.append(bisect(lambda t: Phi(t, y), ts[k], ts[k + 1]))
    # keep roots with H_rob > 0 (strict local minima)
    mins = [r for r in roots if Hrob(r, y) > 0.0]
    if not mins:
        raise RuntimeError("no local minimum found")
    return min(mins, key=lambda r: abs(r - guess))


def main():
    out = []
    out.append("=== 1-D Geman-McClure sign-flip / finite-difference check ===")
    out.append(f"c = {C}, c^2/3 = {C2/3:.12g}, c^2 = {C2:.12g}")

    # (0) Black-Rangarajan dual: min_w [w e + psi(w)] = rho(e), w* = rho'(e)
    wgrid = np.logspace(-9, 0, 200001)
    out.append("[0] Black-Rangarajan dual check for GM: psi(w)=c^2 (sqrt(w)-1)^2")
    for e0 in [0.0, 0.01, 0.25, 1.0, 10.0, 1e3]:
        vals = wgrid * e0 + C2 * (np.sqrt(wgrid) - 1.0) ** 2
        k = int(np.argmin(vals))
        out.append(
            f"    e={e0:8.3f}  min_w[we+psi]={vals[k]:.12f}  rho(e)={rho(e0):.12f}  "
            f"|diff|={abs(vals[k]-rho(e0)):.2e}  w*={wgrid[k]:.8f}  rho'(e)={drho(e0):.8f}"
        )

    # (1) closed form and landmarks
    e_grid = np.concatenate([np.linspace(0.0, 4.0 * C2, 2001), np.logspace(-3, 4, 2000) * C2])
    e_grid = np.unique(e_grid)
    err_closed = np.max(np.abs(lam(e_grid) - lam_closed(e_grid)))
    out.append(f"[1] max |lambda_num - lambda_closed| = {err_closed:.3e}")
    e_zero = C2 / 3.0
    out.append(f"    lambda(c^2/3) = {float(lam_closed(e_zero)):.3e}  (expected 0)")
    # minimum on e>=0 is at e=c^2, value -1/2
    e_min = C2
    out.append(f"    lambda(c^2)   = {float(lam_closed(e_min)):.6f}  (expected -0.5)")
    out.append(f"    lambda(0)     = {float(lam_closed(0.0)):.6f}  (expected  2.0)")

    # (2) two-observation local minimum with e_2 > c^2/3
    y = np.array([0.0, 1.5])
    t_star = local_min_near(y, guess=0.0, half=0.15)
    e = (t_star - y) ** 2
    Hs = Hrob(t_star, y)
    out.append("")
    out.append(f"[2] y = {y}, t* = {t_star:.12f}, e = {e}, H_rob = {Hs:.12f}")
    out.append(f"    e_2 > c^2/3 ? {bool(e[1] > C2/3)}   H_rob > 0 ? {bool(Hs > 0)}")

    exact = lam(e[1]) / Hs
    naive = 2.0 * drho(e[1]) / (2.0 * drho(e[0]) + 2.0 * drho(e[1]))
    h = 1e-4
    tp = local_min_near(np.array([y[0], y[1] + h]), guess=t_star, half=0.01)
    tm = local_min_near(np.array([y[0], y[1] - h]), guess=t_star, half=0.01)
    fd = (tp - tm) / (2.0 * h)
    out.append(f"    exact  dt*/dy_2 = {exact:+.12f}")
    out.append(f"    naive  dt*/dy_2 = {naive:+.12f}")
    out.append(f"    FD     dt*/dy_2 = {fd:+.12f}")
    out.append(f"    |exact-FD|/|FD| = {abs(exact-fd)/abs(fd):.3e}")
    out.append(f"    |naive-FD|/|FD| = {abs(naive-fd)/abs(fd):.3e}")
    out.append(f"    sign(exact) = {np.sign(exact):+.0f}, sign(naive) = {np.sign(naive):+.0f} "
               f"(opposite: {bool(np.sign(exact) * np.sign(naive) < 0)})")

    # (3) log-log slope of |dt*/dy_2| versus ||r_2||
    gaps = np.array([4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0])
    sens = []
    res = []
    for gap in gaps:
        yy = np.array([0.0, gap])
        tt = local_min_near(yy, guess=0.0, half=0.2)
        ee = (tt - yy) ** 2
        hh = Hrob(tt, yy)
        sens.append(abs(lam(ee[1]) / hh))
        res.append(np.sqrt(ee[1]))
    sens = np.array(sens)
    res = np.array(res)
    # fit the tail (last 5 points)
    k = 5
    slope, intercept = np.polyfit(np.log(res[-k:]), np.log(sens[-k:]), 1)
    out.append("")
    out.append("[3] tail log-log slope of |dt*/dy_2| vs ||r_2||:")
    for r, s in zip(res, sens):
        out.append(f"    ||r_2|| = {r:9.4f}   |dt*/dy_2| = {s:.6e}")
    out.append(f"    fitted slope (last {k} points) = {slope:.6f}  (expected -4)")

    text = "\n".join(out)
    print(text)
    with open(HERE / "ift_1d_signflip_output.txt", "w") as f:
        f.write(text + "\n")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
        ee = np.linspace(0.0, 4.0 * C2, 800)
        ax[0].plot(ee / C2, lam_closed(ee), lw=2)
        ax[0].axhline(0.0, color="k", lw=0.8)
        ax[0].axvline(1.0 / 3.0, color="r", ls="--", lw=1)
        ax[0].axvline(1.0, color="g", ls=":", lw=1)
        ax[0].set_xlabel(r"$e/c^2$")
        ax[0].set_ylabel(r"$\lambda_\parallel(e)$")
        ax[0].set_title(r"$\lambda_\parallel$: zero at $e=c^2/3$, min $-1/2$ at $e=c^2$")
        ax[0].grid(alpha=0.3)
        ax[1].loglog(res, sens, "o-", lw=1.5)
        ax[1].set_xticks([4, 8, 16, 32, 64])
        ax[1].set_xticklabels(["4", "8", "16", "32", "64"])
        ax[1].minorticks_off()
        ax[1].set_xlabel(r"$\|r_2\|$")
        ax[1].set_ylabel(r"$|dt^*/dy_2|$")
        ax[1].set_title(f"tail slope = {slope:.3f} (expected -4)")
        ax[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(HERE / "ift_1d_signflip.png", dpi=160)
        print("saved", HERE / "ift_1d_signflip.png")
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
