"""Plot training curves from metrics.jsonl / episodes.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rewards.items import MILESTONE_ORDER


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rolling(x, window):
    if len(x) < window or window <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="valid")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args(argv)

    run = args.run
    out_dir = args.out or os.path.join(run, "curves")
    os.makedirs(out_dir, exist_ok=True)
    metrics = read_jsonl(os.path.join(run, "metrics.jsonl"))
    episodes = read_jsonl(os.path.join(run, "episodes.jsonl"))
    if not metrics and not episodes:
        print("nothing to plot in", run)
        return 1

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    ax = axes.ravel()
    steps = [m["step"] for m in metrics]

    # 1) episode return
    if episodes:
        xs = [e["step"] for e in episodes]
        ys = [e["return"] for e in episodes]
        ax[0].scatter(xs, ys, s=12, alpha=0.45, label="episode")
        if len(ys) >= 3:
            r = rolling(ys, min(args.window, len(ys)))
            ax[0].plot(xs[len(xs) - len(r):], r, lw=2, label=f"mean{args.window}")
        ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title("episode return (raw reward)")
    ax[0].set_xlabel("env steps")

    # 2) score
    if episodes:
        xs = [e["step"] for e in episodes]
        ys = [e["score"] for e in episodes]
        ax[1].step(xs, np.maximum.accumulate(ys), where="post", lw=2, label="max score")
        ax[1].scatter(xs, ys, s=8, alpha=0.4)
        ax[1].set_ylim(0, 13)
        ax[1].set_yticks(range(0, 13))
        ax[1].set_yticklabels(["-"] + MILESTONE_ORDER[:12], fontsize=7)
        ax[1].legend(loc="best", fontsize=8)
    if metrics and any(m.get("milestone_events") for m in metrics):
        ax1b = ax[1].twinx()
        ax1b.plot([m["step"] for m in metrics],
                  [m.get("milestone_events", 0) for m in metrics],
                  color="tab:orange", lw=1.0, alpha=0.7, label="milestone events/update")
        ax1b.set_ylabel("events/update", color="tab:orange", fontsize=8)
        ax1b.legend(loc="upper left", fontsize=7)
    ax[1].set_title("tech-tree score (first-reaching step)")
    ax[1].set_xlabel("env steps")

    # 3) losses
    for key, label in (("loss_policy_loss", "policy"), ("loss_value_loss", "value")):
        vals = [m.get(key) for m in metrics]
        if any(v is not None for v in vals):
            ax[2].plot(steps, vals, label=label, lw=1.2)
    ax[2].set_title("PPO losses")
    ax[2].legend(loc="best", fontsize=8)

    # 4) entropy / kl / clip
    ax3 = ax[3]
    for key, label in (("loss_entropy", "entropy"), ("loss_approx_kl", "approx KL"),
                       ("loss_clip_frac", "clip frac")):
        vals = [m.get(key) for m in metrics]
        if any(v is not None for v in vals):
            if key == "loss_approx_kl":
                ax3 = ax3.twinx()
                ax3.plot(steps, vals, label=label, lw=1.0, ls="--")
            else:
                ax[3].plot(steps, vals, label=label, lw=1.2)
    ax[3].set_title("entropy / KL / clip fraction")
    h1, l1 = ax[3].get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    if h1 or h2:
        ax[3].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    # 5) throughput
    if metrics:
        ax[4].plot(steps, [m.get("sps") for m in metrics], label="policy steps/s", lw=1.2)
        ax[4].plot(steps, [m.get("ticks_per_s") for m in metrics],
                   label="env ticks/s", lw=1.2)
        ax[4].legend(loc="best", fontsize=8)
    ax[4].set_title("throughput")

    # 6) episode reward components
    if episodes:
        names = sorted(episodes[0].get("components", {}).keys())
        xs = [e["step"] for e in episodes]
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(names))))
        bottom = np.zeros(len(episodes))
        comps = np.asarray([[e.get("components", {}).get(n, 0.0) for n in names]
                            for e in episodes])
        for j, name in enumerate(names):
            vals = comps[:, j]
            ax[5].plot(xs, vals, label=name, lw=1.0)
        ax[5].legend(loc="best", fontsize=6, ncol=2)
    ax[5].set_title("episode reward components")
    ax[5].set_xlabel("env steps")

    # 7) episode length
    if episodes:
        ax[6].plot([e["step"] for e in episodes], [e["length"] for e in episodes], lw=1.2)
    ax[6].set_title("episode length (policy steps)")

    # 8) tensorboard-style summary text
    ax[7].axis("off")
    last = metrics[-1] if metrics else {}
    text = [
        f"run: {os.path.basename(os.path.abspath(run))}",
        f"steps: {last.get('step', 0):,}",
        f"updates: {last.get('update', 0)}",
        f"episodes: {len(episodes)}",
        f"sps (last): {last.get('sps', 0):.1f}",
        f"max score: {max([e.get('score', 0) for e in episodes], default=0)}",
        f"best return: {max([e.get('return', 0) for e in episodes], default=0):.1f}",
    ]
    ax[7].text(0.02, 0.95, "\n".join(text), va="top", fontsize=10, family="monospace")

    fig.suptitle("MC-PPO training curves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = os.path.join(out_dir, "curves.png")
    pdf = os.path.join(out_dir, "curves.pdf")
    fig.savefig(png, dpi=130)
    fig.savefig(pdf)
    plt.close(fig)
    for name in ("curves.png", "curves.pdf"):
        latest = os.path.join(out_dir, "latest_" + name)
        try:
            shutil.copyfile(os.path.join(out_dir, name), latest)
        except OSError:
            pass
    print("wrote", png, pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
