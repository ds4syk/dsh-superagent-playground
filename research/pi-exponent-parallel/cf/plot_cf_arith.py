#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal SVG plots for CF stage-3 arithmetic statistics (no third-party libs)."""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def write_bar_svg(path, labels, values, title, ylabel, logy=True,
                  width=900, height=480, color="#1a5fb4"):
    ml, mr, mt, mb = 80, 30, 50, 80
    W, H = width - ml - mr, height - mt - mb
    if logy:
        vals = [math.log10(max(v, 1e-3)) for v in values]
    else:
        vals = list(values)
    vmin = min(0.0, min(vals))
    vmax = max(vals) if max(vals) > vmin else vmin + 1

    def Y(v):
        return mt + H - (v - vmin) / (vmax - vmin) * H

    n = len(labels)
    bw = W / n * 0.62
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-size="17">{title}</text>',
             f'<line x1="{ml}" y1="{mt+H}" x2="{ml+W}" y2="{mt+H}" stroke="#333"/>',
             f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+H}" stroke="#333"/>']
    for i in range(6):
        v = vmin + (vmax - vmin) * i / 5
        yy = Y(v)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+W}" y2="{yy:.2f}" stroke="#ddd" stroke-width="0.6"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-size="11">{v:.2g}</text>')
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = ml + (i + 0.5) * W / n
        y = Y(v)
        parts.append(f'<rect x="{x-bw/2:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{mt+H-y:.2f}" fill="{color}" opacity="0.8"/>')
        parts.append(f'<text x="{x:.2f}" y="{mt+H+18}" text-anchor="middle" font-size="11">{lab}</text>')
    parts.append(f'<text x="20" y="{mt+H/2}" text-anchor="middle" font-size="13" '
                 f'transform="rotate(-90 20 {mt+H/2})">{ylabel}</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_lines_svg(path, series, title, xlabel, ylabel, width=900, height=480):
    """series: list of (label, xs, ys, color)"""
    ml, mr, mt, mb = 80, 140, 50, 70
    W, H = width - ml - mr, height - mt - mb
    allx = [x for _, xs, _, _ in series for x in xs]
    ally = [y for _, _, ys, _ in series for y in ys]
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = 0, max(ally) * 1.1
    if xmax == xmin:
        xmax = xmin + 1

    def X(x):
        return ml + (x - xmin) / (xmax - xmin) * W

    def Y(y):
        return mt + H - (y - ymin) / (ymax - ymin) * H

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, sans-serif">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-size="17">{title}</text>',
             f'<line x1="{ml}" y1="{mt+H}" x2="{ml+W}" y2="{mt+H}" stroke="#333"/>',
             f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+H}" stroke="#333"/>']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        yy = Y(v)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+W}" y2="{yy:.2f}" stroke="#ddd" stroke-width="0.6"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-size="10">{v:.0f}</text>')
    for i in range(6):
        x = xmin + (xmax - xmin) * i / 5
        xx = X(x)
        parts.append(f'<text x="{xx:.2f}" y="{mt+H+18}" text-anchor="middle" font-size="10">{x:.0f}</text>')
    for idx, (label, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        lx = ml + W + 12
        ly = mt + 18 + idx * 22
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+18}" y2="{ly-4}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+24}" y="{ly}" font-size="11">{label}</text>')
    parts.append(f'<text x="{ml+W/2}" y="{height-14}" text-anchor="middle" font-size="12">{xlabel}</text>')
    parts.append(f'<text x="18" y="{mt+H/2}" text-anchor="middle" font-size="12" '
                 f'transform="rotate(-90 18 {mt+H/2})">{ylabel}</text>')
    parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    with open(os.path.join(HERE, "cf_arith_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    # runs
    h1 = {int(k): v for k, v in S["runs_of_1"]["hist"].items()}
    he = {int(k): v for k, v in S["equal_runs"]["hist"].items()}
    L = max(max(h1), max(he))
    xs = list(range(1, L + 1))
    write_lines_svg(os.path.join(HERE, "cf_runs.svg"),
                    [("runs of 1", xs, [h1.get(x, 0) for x in xs], "#1a5fb4"),
                     ("runs of equal values", xs, [he.get(x, 0) for x in xs], "#c01c28")],
                    "Run-length distribution of pi partial quotients (97189 terms)",
                    "run length", "count")
    # mod p chi2
    ps = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    vals = [S["modstats"][str(p)]["chi2_over_dof"] for p in ps]
    write_bar_svg(os.path.join(HERE, "cf_modchi.svg"),
                  [str(p) for p in ps], vals,
                  "q_n mod p: chi-square / dof (uniform = 1)", "log10(chi2/dof)")
    print("wrote cf_runs.svg, cf_modchi.svg")


if __name__ == "__main__":
    main()
