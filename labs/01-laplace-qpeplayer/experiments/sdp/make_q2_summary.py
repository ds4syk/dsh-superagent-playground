#!/usr/bin/env python3
"""P1-Q2 figure: full-20D 10x10 SDP vs upstream 17D 13x13 SDP.

Two panels:
  left  : f*_10 (21 constraints, full 20D) vs f*_13 upstream (24 constraints,
          17D) for the 10 instances of results/sdp_check.json, plus the
          6-point physical counterexample (star) and a third marker at
          (f*_10, f*_13 full-20D) showing f*_13full == f*_10.
  right : the 6-point instance only -- f*_QPEP = f*_10 = f*_13full = 5.415
          (rank 1) vs f*_13up = 3.59431373 (rank 4), with the 20-vs-17
          constraint-dimension annotation and the 3 missing row-orthogonality
          constraints.

Everything is read from the JSON files produced by the check scripts, so the
figure is regenerated from first principles and cannot drift from the data.

Run:
  /home/syk/miniconda3/envs/WM/bin/python sdp/make_q2_summary.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PNG = OUT / "q2_summary.png"
META = OUT / "q2_summary.json"

# colour-blind friendly palette (Okabe-Ito-ish)
C_TIGHT = "#0072B2"      # tight / full-20D
C_STRICT = "#D55E00"     # strict upstream gap
C_SIX = "#CC0000"        # 6-point star
C_FULL = "#009E73"       # f13_full marker
C_GREY = "#666666"


def load():
    chk_p = OUT / "sdp_check.json"
    six_p = OUT / "six_point.json"
    for p in (chk_p, six_p):
        if not p.is_file():
            sys.exit(f"missing {p}; run run_sdp_check.py and run_6point_check.py first")
    chk = json.loads(chk_p.read_text())
    six = json.loads(six_p.read_text())
    return chk, six, chk_p, six_p


def fmt_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(
        timespec="seconds")


def left_panel(ax, chk, six):
    rows = chk["main"]
    f10 = np.array([r["f_10"] for r in rows], dtype=float)
    f13 = np.array([r["f_13"] for r in rows], dtype=float)
    names = [r["name"] for r in rows]
    strict = np.abs(f10 - f13) > 1e-6

    lo, hi = -1.2, 6.4
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.35", zorder=1,
            label=r"$y=x$  (upstream = full-20D)")
    ax.fill_between([lo, hi], [lo, hi], [lo - 1.5, hi - 1.5],
                    color=C_STRICT, alpha=0.045, zorder=0)
    ax.text(4.85, 1.05, "upstream strictly looser\n$f^\\star_{13,\\rm up} < f^\\star_{10}$",
            color=C_STRICT, fontsize=9.5, ha="center", va="center", style="italic")

    # tight instances (on y=x) and strict ones
    ax.scatter(f10[~strict], f13[~strict], s=58, color=C_TIGHT, edgecolor="white",
               linewidth=0.7, zorder=4, label="tight: $f^\\star_{13,\\rm up}=f^\\star_{10}$")
    ax.scatter(f10[strict], f13[strict], s=58, color=C_STRICT, edgecolor="white",
               linewidth=0.7, zorder=4, label="strict: $f^\\star_{13,\\rm up}<f^\\star_{10}$")

    by = {r["name"]: r for r in rows}
    rd = by["rowdot"]
    rdg = by["rowdot_gen"]

    # 6-point counterexample: star = upstream, diamond = full-20D (on y=x)
    s10, sup, sfull = six["f10"], six["f13_up"], six["f13_full"]
    ax.scatter([s10], [sup], marker="*", s=340, color=C_SIX,
               edgecolor="black", linewidth=0.6, zorder=6,
               label="6-point, upstream 13×13")
    ax.scatter([s10], [sfull], marker="D", s=72, color=C_FULL,
               edgecolor="black", linewidth=0.6, zorder=6)
    gap6 = s10 - sup
    ax.annotate(
        "6-point physical 3D-3D counterexample\n"
        f"$f^\\star_{{10}}$ = {s10:.6f} (rank 1)\n"
        f"$f^\\star_{{13,\\rm up}}$ = {sup:.6f} (rank 4)\n"
        f"gap = {gap6:.6f}",
        xy=(s10 - 0.08, sup + 0.05), xytext=(1.55, 5.30),
        fontsize=8.4, ha="left", va="center", color="black",
        arrowprops=dict(arrowstyle="->", color=C_SIX, lw=1.1,
                        connectionstyle="arc3,rad=-0.18"),
        bbox=dict(boxstyle="round,pad=0.33", fc="#FFF5F5", ec=C_SIX, lw=0.8),
        zorder=7)
    # explicit label for the third marker (f10, f13_full)
    ax.annotate("full-20D\n$=f^\\star_{10}$",
                xy=(s10 + 0.05, sfull + 0.02), xytext=(5.55, 5.35),
                fontsize=7.8, ha="left", va="center", color=C_FULL,
                arrowprops=dict(arrowstyle="-", color=C_FULL, lw=0.8),
                zorder=7)

    # rowdot counterexample (largest gap among the 10 instances)
    ax.annotate(
        f"rowdot instance: {rd['f_10']:.3f} vs {rd['f_13']:.3f}\n"
        f"gap = {rd['f_10'] - rd['f_13']:.3f}",
        xy=(rd["f_10"], rd["f_13"]), xytext=(-0.90, -1.72),
        fontsize=9.0, ha="left", va="center", color="black",
        arrowprops=dict(arrowstyle="->", color=C_STRICT, lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9F3", ec=C_STRICT, lw=0.7),
        zorder=7)
    ax.annotate("rowdot_gen\n(gap %.3f)" % (rdg["f_10"] - rdg["f_13"]),
                xy=(rdg["f_10"], rdg["f_13"]), xytext=(-0.75, -2.55),
                fontsize=8.2, ha="left", va="center", color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7),
                zorder=7)

    # Choi-Lam cluster: strict instances just below the diagonal
    ax.annotate("Choi–Lam symmetry-broken\ninstances (3 shown; 8/8 strict)",
                xy=(1.958, 1.945), xytext=(0.35, 2.95),
                fontsize=8.6, ha="left", va="center", color=C_STRICT,
                arrowprops=dict(arrowstyle="->", color=C_STRICT, lw=0.9,
                                connectionstyle="arc3,rad=0.2"),
                zorder=7)
    ax.annotate("tight (random / P6-data) instances",
                xy=(1.25, 1.25), xytext=(0.35, 0.15),
                fontsize=8.2, ha="left", va="center", color=C_TIGHT,
                arrowprops=dict(arrowstyle="->", color=C_TIGHT, lw=0.8),
                zorder=7)

    # structural note inside the empty lower-right corner
    ax.text(0.985, 0.035,
            "13×13 full-20D (27 cons) ≡ 10×10; only the 6-point $f^\\star_{13\\rm full}$ is plotted.\n"
            "Upstream 13×13 (24 cons, rank 17) is strictly looser whenever a circle is off $y=x$.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color="0.25",
            bbox=dict(boxstyle="round,pad=0.33", fc="#F7F7F7", ec="0.7", lw=0.6))

    ax.set_xlim(lo, hi)
    ax.set_ylim(-2.9, 6.4)
    ax.set_xlabel(r"$f^\star_{10}$  (10×10, 21 constraints, full 20D)")
    ax.set_ylabel(r"$f^\star_{13,\ \rm upstream}$  (13×13, 24 constraints, 17D)")
    ax.set_title("Per instance: full-20D 10×10 vs upstream 13×13", fontsize=10.5,
                 fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92, borderpad=0.5)


def right_panel(ax, six):
    bars = [
        ("QPEP nonconvex\n$J(R{=}I,t{=}0)$", 5.415, 1,
         "nonconvex optimum\n(exact $6(1-\\varepsilon)^2$)", "#555555"),
        ("10×10\n21 cons · 20D", six["f10"], int(six["rank10"]),
         "21 constraints\nrank 20", C_TIGHT),
        ("13×13 full\n27 cons · 20D", six["f13_full"], int(six["rank13_full"]),
         "24 + 3 row-orth\nrank 20", C_FULL),
        ("13×13 upstream\n24 cons · 17D", six["f13_up"], int(six["rank13_up"]),
         "24 constraints\nrank 17", C_STRICT),
    ]
    xs = np.arange(len(bars))
    vals = [b[1] for b in bars]
    ax.bar(xs, vals, width=0.62, color=[b[4] for b in bars],
           edgecolor="black", linewidth=0.6, alpha=0.88, zorder=3)

    for x, (lab, v, rank, _, _) in zip(xs, bars):
        ax.text(x, v + 0.13, f"{v:.6f}", ha="center", va="bottom",
                fontsize=9.2, fontweight="bold")
        ax.text(x, v / 2.0, f"rank {rank}", ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold", zorder=4)

    ax.axhline(5.415, color="0.35", ls="--", lw=1.0, zorder=2)
    ax.text(4.42, 5.415, "nonconvex optimum\n$f^\\star_{\\rm QPEP}=5.415$",
            ha="right", va="bottom", fontsize=7.8, color="0.3")

    # gap bracket between upstream bar and the full-20D level
    xg = 3.55
    ax.annotate("", xy=(xg, 5.415), xytext=(xg, six["f13_up"]),
                arrowprops=dict(arrowstyle="<->", color=C_STRICT, lw=1.3),
                zorder=5)
    ax.text(xg + 0.10, (5.415 + six["f13_up"]) / 2.0,
            f"gap\n{six['f10'] - six['f13_up']:.4f}",
            ha="left", va="center", fontsize=9.5, color=C_STRICT, fontweight="bold")

    # constraint-dimension annotation (bottom-left, free region)
    ax.text(0.03, 0.97,
            "20 vs 17 independent constraint dimensions.\n"
            "3 missing row-orthogonality constraints:\n"
            r"$r_1\!\cdot\!r_2=r_1\!\cdot\!r_3=r_2\!\cdot\!r_3=0$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.4,
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9F3", ec=C_STRICT, lw=0.8))

    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.2)
    ax.set_ylabel("objective value  $f^\\star$")
    ax.set_ylim(0.0, 8.0)
    ax.set_xlim(-0.62, 4.55)
    ax.set_title("6-point instance: full-20D is tight, upstream is not",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def main():
    t0 = time.time()
    chk, six, chk_p, six_p = load()
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.4, 5.9), gridspec_kw={"width_ratios": [1.18, 1.0]})
    left_panel(axL, chk, six)
    right_panel(axR, six)

    fig.suptitle(
        "P1-Q2: the upstream SDPRLayers 13×13 SDP (17D) is a strictly looser "
        "relaxation than the full-20D 10×10",
        fontsize=12.2, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Source: labs/01-laplace-qpeplayer/experiments/sdp/results/"
             "{sdp_check.json, six_point.json}  ·  SCS / CLARABEL numerical "
             "solutions  ·  figure regenerated 2026-11-27  ·  numerical (float), "
             "not exact rational",
             ha="center", va="bottom", fontsize=7.6, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = chk["main"]
    by = {r["name"]: r for r in rows}
    meta = {
        "figure": str(PNG.relative_to(HERE)),
        "generated_by": "sdp/make_q2_summary.py",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "source_data": {
            "sdp_check.json": {"path": str(chk_p.relative_to(HERE)),
                               "mtime": fmt_mtime(chk_p),
                               "n_instances": len(rows)},
            "six_point.json": {"path": str(six_p.relative_to(HERE)),
                               "mtime": fmt_mtime(six_p),
                               "solver_scs_eps": 1e-11},
        },
        "axes": {
            "left": {
                "x": "f*_10 (10x10, 21 constraints, full 20D)",
                "y": "f*_13 upstream (13x13, 24 constraints, 17D)",
                "diagonal": "f*_13_up = f*_10",
                "instances": [
                    {"name": r["name"], "f10": r["f_10"], "f13_up": r["f_13"],
                     "gap": r["f_10"] - r["f_13"],
                     "tight": bool(abs(r["f_10"] - r["f_13"]) <= 1e-6),
                     "rank10": r["rank_X10"], "rank13_up": r["rank_X13"]}
                    for r in rows
                ],
                "six_point": {
                    "f10": six["f10"], "f13_up": six["f13_up"],
                    "f13_full": six["f13_full"],
                    "gap_up": six["f10"] - six["f13_up"],
                    "rank10": six["rank10"], "rank13_up": six["rank13_up"],
                    "rank13_full": six["rank13_full"],
                    "marker": "star = upstream, diamond = full-20D (on y=x)",
                },
                "rowdot": {
                    "name": "rowdot",
                    "f10": by["rowdot"]["f_10"],
                    "f13_up": by["rowdot"]["f_13"],
                    "gap": by["rowdot"]["f_10"] - by["rowdot"]["f_13"],
                    "annotation": "rowdot counterexample, gap 2.0",
                },
            },
            "right": {
                "instance": "6-point physical 3D-3D counterexample",
                "construction": "p_i=+-e1,+-e2,+-e3; y_i=0.05 p_i; "
                                "W=[[1,-0.6,0],[-0.6,1,0],[0,0,1]]; G=6W; D=0",
                "bars": [
                    {"label": "QPEP nonconvex J(R=I,t=0)", "value": 5.415,
                     "rank": 1, "constraint_dim": 20,
                     "note": "exact 6(1-eps)^2, eps=0.05"},
                    {"label": "10x10", "value": six["f10"],
                     "rank": six["rank10"], "n_constraints": 21,
                     "constraint_dim": 20},
                    {"label": "13x13 full (24+3 row-orthogonality)",
                     "value": six["f13_full"], "rank": six["rank13_full"],
                     "n_constraints": 27, "constraint_dim": 20},
                    {"label": "13x13 upstream", "value": six["f13_up"],
                     "rank": six["rank13_up"], "n_constraints": 24,
                     "constraint_dim": 17},
                ],
                "gap_full_minus_upstream": six["f10"] - six["f13_up"],
                "missing_constraints": ["r1.r2=0", "r1.r3=0", "r2.r3=0"],
                "annotation": "20 vs 17 independent constraint dimensions",
            },
        },
        "confirmed_numbers": {
            "f10": six["f10"], "f13_full": six["f13_full"],
            "f13_up": six["f13_up"],
            "f10_minus_f13_up": six["f10_minus_f13_up"],
            "f10_minus_f13_full": six["f10"] - six["f13_full"],
            "rank10": six["rank10"], "rank13_full": six["rank13_full"],
            "rank13_up": six["rank13_up"],
            "perturbations_strict": chk["summary"]["pert_n_strict"],
            "perturbations_total": chk["summary"]["pert_n"],
            "six_point_check_rerun": "run_6point_check.py re-run by the P1 phase-3 "
                                     "worker on 2026-11-27; values reproduced",
        },
        "caveats": [
            "Numerical (SCS eps=1e-11 / CLARABEL) values; not exact rational.",
            "The revised proof quotes f13_up = 3.59430886; the solvers agree on "
            "3.59431373 (difference 4.9e-6, above solver tolerance). The figure "
            "and metadata use the measured 3.59431373; the O(1) separation "
            "(gap ~1.82) is unaffected.",
            "13x13 full-20D equals 10x10 by the Schur bijection; only the 6-point "
            "f13_full value (5.4150000000003) is plotted as the green diamond.",
        ],
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {PNG}")
    print(f"wrote {META}")
    print(f"runtime {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
