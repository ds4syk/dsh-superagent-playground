"""Aggregate the experiment results into figures and Markdown tables."""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

METHOD_LABEL = {
    "direct": "direct MLP",
    "twostage": "two-stage\n+ SDPR",
    "twostage_chordal": "two-stage\n+ chordal",
    "e2e": "end-to-end\n+ SDPR",
    "e2e_chordal": "end-to-end\n+ chordal",
    "oracle": "oracle SDPR",
}
COLORS = ["#9aa5b1", "#4c78a8", "#a4c2e4", "#c0392b", "#e8a0a0", "#2e8b57"]


def load_metrics(results_dir: str, tag: str):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"metrics_*_{tag}.json"))):
        method = os.path.basename(path)[len("metrics_") : -len(f"_{tag}.json")]
        with open(path) as f:
            data = json.load(f)
        out[method] = data
        # chordal results live under <split>_chordal in the same metrics file
        chord = {k[: -len("_chordal")]: v for k, v in data.items() if k.endswith("_chordal")}
        if chord:
            out[method + "_chordal"] = chord
    return out


def fig_errors(metrics, out):
    rows = []
    for m in ["direct", "twostage", "twostage_chordal", "e2e", "e2e_chordal", "oracle"]:
        if m not in metrics:
            continue
        rows.append((m, metrics[m]["test_id"]["mean_deg"], metrics[m]["test_ood"]["mean_deg"]))
    if not rows:
        return
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="test (seen panoramas)", color=COLORS[1])
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="test OOD (unseen panorama)", color=COLORS[3])
    for i, (m, a, b) in enumerate(rows):
        ax.text(i - w / 2, a + 1.0, f"{a:.1f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, b + 1.0, f"{b:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL.get(r[0], r[0]) for r in rows], fontsize=7.5)
    ax.set_ylabel("mean geodesic error (deg)")
    ax.set_title("Global rotation error (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(os.path.join(out, "fig_errors.png"))
    plt.close(fig)


def fig_training(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    if "twostage" in metrics and metrics["twostage"]["history"]:
        h = metrics["twostage"]["history"]
        axes[0].plot([r["epoch"] for r in h], [r["train_rel_loss_deg"] for r in h], label="train (relative)", color="#4c78a8")
        axes[0].plot([r["epoch"] for r in h], [r["val_rel_deg"] for r in h], label="val (relative)", color="#c0392b")
        axes[0].set_title("Stage 1: relative-rotation head", fontsize=9)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("geodesic error (deg)")
        axes[0].legend(frameon=False, fontsize=7)
    if "e2e" in metrics and metrics["e2e"]["history"]:
        h = metrics["e2e"]["history"]
        axes[1].plot([r["epoch"] for r in h], [r["train_global_loss_deg"] for r in h], label="train (global)", color="#4c78a8")
        axes[1].plot([r["epoch"] for r in h], [r["val_mean_deg"] for r in h], label="val (global)", color="#c0392b")
        axes[1].set_title("Stage 2: end-to-end through SDPR", fontsize=9)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("geodesic error (deg)")
        axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_training.png"))
    plt.close(fig)


def fig_cert(results_dir, tag, out):
    path = os.path.join(results_dir, f"pred_oracle_test_id_{tag}.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    rank, eig = d["rank"], d["eig"]
    if rank.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].hist(np.log10(np.maximum(rank, 1e-12)), bins=20, color="#4c78a8")
    axes[0].axvline(np.log10(1e-6), color="#c0392b", ls="--", lw=1)
    axes[0].set_xlabel("log10 rank gap (λ₂/λ₁)")
    axes[0].set_ylabel("sequences")
    axes[0].set_title("SDP tightness", fontsize=9)
    axes[1].hist(eig, bins=20, color="#4c78a8")
    axes[1].axvline(-1e-5, color="#c0392b", ls="--", lw=1)
    axes[1].set_xlabel("min eigenvalue of certificate H")
    axes[1].set_title("Certificate PSD margin", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_certificate.png"))
    plt.close(fig)


def fig_bridge(results_dir, out):
    path = os.path.join(results_dir, "bridge_check.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    names = ["head", "encoder\nlast block", "encoder\npatch embed", "encoder\nnorm", "input\nfeatures"]
    vals = [
        d["head_grad_norm"],
        d["encoder_last_block_grad_norm"],
        d["encoder_patch_embed_grad_norm"],
        d["encoder_norm_grad_norm"],
        d["feature_grad_norm"],
    ]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(names, vals, color=["#4c78a8", "#c0392b", "#c0392b", "#c0392b", "#2e8b57"])
    ax.set_yscale("log")
    ax.set_ylabel("gradient norm (log)")
    ax.set_title("Gradients from the SDPR output reach V-JEPA 2", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(os.path.join(out, "fig_bridge_grad.png"))
    plt.close(fig)


def fig_pernode(metrics, out):
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    for m, color in [("oracle", "#2e8b57"), ("e2e", "#c0392b"), ("twostage", "#4c78a8"), ("direct", "#9aa5b1")]:
        if m not in metrics:
            continue
        y = metrics[m]["test_id"]["per_node_mean_deg"]
        ax.plot(range(len(y)), y, "o-", label=METHOD_LABEL.get(m, m), color=color, ms=3)
    ax.set_xlabel("node index (node 0 is anchored by the prior)")
    ax.set_ylabel("mean geodesic error (deg)")
    ax.set_title("Per-node error on the in-distribution test split")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(out, "fig_per_node.png"))
    plt.close(fig)


def write_tables(metrics, results_dir, out):
    lines = []
    header = (
        "| method | split | mean° | median° | p90° | rel. meas.° | cert rate | SDP fwd (s/sample) |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines.append(header)
    for m in ["direct", "twostage", "twostage_chordal", "e2e", "e2e_chordal", "oracle"]:
        if m not in metrics:
            continue
        for split in ["test_id", "test_ood"]:
            r = metrics[m].get(split)
            if not r:
                continue
            cert = r.get("cert_rate", float("nan"))
            spd = r.get("sdp_fwd_s", float("nan"))
            lines.append(
                f"| {m} | {split} | {r['mean_deg']:.2f} | {r['median_deg']:.2f} | {r['p90_deg']:.2f} | "
                f"{r.get('rel_mean_deg', 0):.2f} | {cert:.2f} | {spd:.3f} |"
            )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "results"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    out = args.results_dir
    metrics = load_metrics(out, args.tag)
    print("found methods:", sorted(metrics))
    fig_errors(metrics, out)
    fig_training(metrics, out)
    fig_cert(out, args.tag, out)
    fig_bridge(out, out)
    fig_pernode(metrics, out)
    write_tables(metrics, out, out)
    summary = {m: {s: metrics[m].get(s) for s in ("test_id", "test_ood")} for m in metrics}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
