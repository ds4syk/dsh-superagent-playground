"""Turn result JSONs into report tables and figures."""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def short(name: str) -> str:
    return {
        "persistence": "persistence",
        "tacit-Tacit-4B": "Tacit-4B (typed)",
        "decider-Qwen3-4B-Instruct-2507-raw": "Qwen3-4B (typed)",
        "generative-Qwen3-4B-Instruct-2507": "Qwen3-4B (gen.)",
    }.get(name, name)


def main():
    os.makedirs("results/figures", exist_ok=True)
    exp1 = {}
    for path in sorted(glob.glob("results/exp1_*.json")):
        if path.endswith("_preds.jsonl"):
            continue
        d = load(path)
        exp1[d["backend"]] = d

    print("== Exp1 ==")
    print("| model | exact | exec | macro var | observe | simple | derive | s/sample |")
    print("|-|-|-|-|-|-|-|-|")
    for name, d in exp1.items():
        kinds = d.get("exact_by_action_kind", {})
        print(f"| {short(name)} | {d['exact_match']:.3f} | {d['exec_accuracy']:.3f} | "
              f"{d['macro_variable_accuracy']:.3f} | {kinds.get('observe')} | {kinds.get('simple')} | "
              f"{kinds.get('derive')} | {d.get('mean_latency_s', 0):.4f} |")

    # accuracy by kind
    if exp1:
        names = list(exp1)
        kinds = ["observe", "simple", "derive"]
        x = np.arange(len(names))
        w = 0.26
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for i, k in enumerate(kinds):
            vals = [exp1[n].get("exact_by_action_kind", {}).get(k) or 0 for n in names]
            ax.bar(x + (i - 1) * w, vals, w, label=k)
        ax.set_xticks(x)
        ax.set_xticklabels([short(n) for n in names], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("exact state match")
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/figures/fig1_exp1_by_kind.png", dpi=160)
        plt.close(fig)

    # rollout
    if exp1:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for name, d in exp1.items():
            r = d.get("rollout")
            if not r:
                continue
            ks = sorted(int(k) for k in r["exact_by_k"])
            ax.plot(ks, [r["exact_by_k"][str(k)] for k in ks], marker="o", label=short(name))
        ax.set_xlabel("rollout step k")
        ax.set_ylabel("exact state match")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("results/figures/fig2_exp1_rollout.png", dpi=160)
        plt.close(fig)

    # exp2
    path = "results/exp2_eval.json"
    if os.path.exists(path):
        d = load(path)
        base = ["random", "oracle", "choice", "score", "wm"]
        names = [n for n in base if n in d] + [k for k in d if k.startswith("student")]
        print("\n== Exp2 ==")
        print("| policy | success | steps | decisions/ep | invalid |")
        print("|-|-|-|-|-|")
        for n in names:
            r = d[n]
            print(f"| {n} | {r['success_rate']:.3f} | {r['mean_steps']:.1f} | "
                  f"{r.get('decisions_per_episode', 0):.1f} | {r.get('invalid_rate', 0):.3f} |")
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        vals = [d[n]["success_rate"] for n in names]
        palette = ["#999999", "#444444", "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
        ax.bar(names, vals, color=palette[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("closed-loop success")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/figures/fig3_exp2_success.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        for n in names:
            r = d[n]
            ax.scatter(max(r.get("decisions_per_episode", 1), 0.1), r["success_rate"], s=60)
            ax.annotate(n, (max(r.get("decisions_per_episode", 1), 0.1), r["success_rate"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("JEV decisions per episode (log)")
        ax.set_ylabel("closed-loop success")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/figures/fig4_exp2_cost.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
