"""Figures for the SDPR-as-V-JEPA-2-output-bridge report."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "..", "artifacts", "results")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#3b5b92", fs=9, bold=False):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, ls="-", color="#333333", fs=8, rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}"),
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom",
                fontsize=fs, color=color)


def arch():
    fig, ax = plt.subplots(figsize=(12.8, 3.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    y, h = 1.45, 0.66
    specs = [
        (0.10, 1.12, "16-frame clip\n256×256", "#f5f5f5"),
        (1.48, 1.58, "V-JEPA 2 ViT-L\n(frozen, 304M)", "#e8f1e8"),
        (3.32, 1.42, "per-node features\n$f_t\\in\\mathbb{R}^{1024}$", "#eef3fb"),
        (5.00, 1.46, "edge head $g_\\phi$\n$(R_{ij}, w_{ij})$", "#eef3fb"),
        (6.72, 1.10, "cost matrix\n$Q(\\theta)$", "#fff2e0"),
        (8.08, 1.42, "SDPR: Shor SDP\n+ certificate $H$", "#ffe3e3"),
        (9.76, 1.40, "global rotations\n$\\hat R_1\\ldots\\hat R_8$", "#eef3fb"),
        (11.42, 1.10, "geodesic\nloss", "#f5f5f5"),
    ]
    for x, w, t, fc in specs:
        box(ax, x, y, w, h, t, fc=fc)
    xs = [(1.22, 1.48), (3.06, 3.32), (4.74, 5.00), (6.46, 6.72), (7.82, 8.08), (9.50, 9.76), (11.16, 11.42)]
    for x1, x2 in xs:
        arrow(ax, x1, y + h / 2, x2, y + h / 2)

    # backward path: loss -> adapter -> encoder, routed above the boxes
    arrow(ax, 11.97, y + h + 0.02, 5.70, y + h + 0.02, color="#c0392b", ls="--", rad=0.06)
    arrow(ax, 5.15, y + h + 0.02, 2.30, y + h + 0.02, color="#c0392b", ls="--", rad=0.06)
    ax.text(8.95, 2.86, "backward: implicit differentiation of the QCQP KKT system\n"
            "$dL/d\\theta$ = gradient of the certified global optimum (SDPR Thm. 2)",
            color="#c0392b", fontsize=8.5, ha="center", va="bottom")
    ax.text(3.65, 2.86, "gradient reaches the adapter\n(and, if unfrozen, the encoder)",
            color="#c0392b", fontsize=8.5, ha="center", va="bottom")

    ax.text(0.10, 0.55, "Key point: SDPR is a structured geometric readout, not a pixel/video decoder.\n"
            "It applies only when the output is the globally optimal solution of a homogeneous QCQP.",
            fontsize=9, color="#444444", va="center")
    ax.text(10.05, 0.38, "no decoder (by design)", fontsize=9, color="#777777", ha="center")

    fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=200)
    plt.close(fig)


def samples():
    from panorama import build_panorama, make_sequence, geodesic_deg

    img = os.path.join(HERE, "..", "artifacts", "assets", "pano_1.jpg")
    if os.path.exists(img):
        pano = build_panorama(image_path=img)
    else:
        pano = build_panorama(
            os.environ.get("UCF101_ROOT", "/mnt/storage1/syk/jepa/datasets/ucf101_alt/UCF-101"),
            seed=7,
        )
    frames, R = make_sequence(pano, np.random.default_rng(42))
    # 8 node frames
    tiles = [frames[2 * i] for i in range(8)]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.9))
    for k, (ax, t) in enumerate(zip(axes.ravel(), tiles)):
        ax.imshow(t)
        ax.set_title(f"node {k}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Rotating-camera sequence on a synthetic panorama (16 frames, 8 nodes)", fontsize=10)
    fig.savefig(os.path.join(OUT, "fig_samples.png"), dpi=200)
    plt.close(fig)

    # 3D camera-orientation trajectory
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(R[:, 0, 2], R[:, 1, 2], R[:, 2, 2], "o-", color="#c0392b", label="camera z-axis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"GT camera orientations\n(total {geodesic_deg(R[0], R[-1]):.1f}°)", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_trajectory.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="arch,samples")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for w in args.which.split(","):
        globals()[w]()
        print("done", w)
