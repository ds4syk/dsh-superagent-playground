"""Compose the one-page visual for the SuperMimic reproduction.

Panels: (A) training loop, (B) training return vs wall-clock,
(C) Appendix-F evaluation per clip, (D) key numbers.

Run with the project venv (matplotlib + tensorboard available):
  uv run python repro/make_onepager.py --sm-csv ... --ppo-logdir ... \
      --sm-eval ... --ppo-eval ... --out ../../docs/visuals/2026-11-12-supermimic-repro.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plot_results as pr

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color):
  ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                              linewidth=1.2, edgecolor=color, facecolor=color + "22"))
  ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="#222")


def arrow(ax, x1, y1, x2, y2, text="", color="#555"):
  ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                               linewidth=1.1, color=color))
  if text:
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, text, ha="center", fontsize=7.5,
            color=color)


def method_panel(ax):
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis("off")
  ax.set_title("A. SuperMimic 训练循环（Algorithm A.1）", loc="left", fontsize=11, weight="bold")
  box(ax, 0.02, 0.62, 0.24, 0.18, "模拟器\nMuJoCo Warp", "#4c72b0")
  box(ax, 0.38, 0.62, 0.24, 0.18, "回放缓冲区 B\n(s,a,ξ)", "#4c72b0")
  box(ax, 0.72, 0.62, 0.26, 0.18, "世界模型 fω\nk=8 自回归", "#1b9e77")
  arrow(ax, 0.26, 0.71, 0.38, 0.71, "带噪动作")
  arrow(ax, 0.62, 0.71, 0.72, 0.71, "采样窗口")
  box(ax, 0.72, 0.20, 0.26, 0.18, "想象 rollout\nH=32（冻结 ω）", "#1b9e77")
  box(ax, 0.38, 0.20, 0.24, 0.18, "策略 πθ + critic Vψ\nBPTT / GAE", "#c44e52")
  box(ax, 0.02, 0.20, 0.24, 0.18, "部署\nproprio 10 帧 + goal", "#8172b2")
  arrow(ax, 0.85, 0.62, 0.85, 0.38)
  ax.text(0.855, 0.50, "梯度不回传模拟器", rotation=90, fontsize=7, color="#555", va="center")
  arrow(ax, 0.72, 0.29, 0.62, 0.29)
  ax.text(0.67, 0.245, "∇θ", ha="center", fontsize=8, color="#555")
  arrow(ax, 0.38, 0.29, 0.26, 0.29, "")
  ax.text(0.5, 0.06, "每步：收集 → 拟合 WM（冻结）→ BPTT 更新策略与 critic",
          ha="center", fontsize=8.5, color="#555")


def curves_panel(ax, sm, ppo):
  ax.set_title("B. 训练 return（proxy 归一化）vs 墙钟", loc="left", fontsize=11, weight="bold")
  for rows, label, color in ((sm, "SuperMimic (BPTT, WM)", "#1b9e77"),
                             (ppo, "PPO-Sim (rsl_rl)", "#d95f02")):
    if not rows:
      continue
    x, y = pr.proxy(rows)
    ax.plot(x, pr.smooth(y), label=label, lw=1.6, color=color)
  ax.set_xlabel("wall-clock training time (h)")
  ax.set_ylabel("return / (0.1 × episode length)")
  ax.grid(alpha=0.3)
  ax.legend(fontsize=9)


def eval_panel(ax, sm_eval, ppo_eval):
  ax.set_title("C. 40 条 LAFAN1 动作的 Appendix-F 评测", loc="left", fontsize=11, weight="bold")
  for data, label, color in ((ppo_eval, "PPO-Sim", "#d95f02"),
                             (sm_eval, "SuperMimic", "#1b9e77")):
    if not data:
      continue
    vals = sorted(r["normalized_return"] for r in data["rows"])
    ax.plot(range(len(vals)), vals, marker=".", ms=3, lw=1.0, color=color,
            label=f"{label}: mean {sum(vals)/len(vals):.3f}, fail {data['failed']}")
  ax.set_xlabel("clip (sorted)")
  ax.set_ylabel("normalized return")
  ax.set_ylim(0, 1)
  ax.grid(alpha=0.3)
  ax.legend(fontsize=9)


def numbers_panel(ax, summary):
  ax.axis("off")
  ax.set_title("D. 关键数字", loc="left", fontsize=11, weight="bold")
  lines = []
  for key, value in summary.items():
    lines.append((key, value))
  y = 0.92
  for key, value in lines:
    ax.text(0.02, y, key, fontsize=9, color="#444", va="top")
    ax.text(0.52, y, value, fontsize=9, color="#111", va="top", weight="bold")
    y -= 1 / (len(lines) + 1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sm-csv")
  ap.add_argument("--ppo-logdir")
  ap.add_argument("--sm-eval")
  ap.add_argument("--ppo-eval")
  ap.add_argument("--summary", help="JSON of key/value strings for panel D")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  sm = pr.read_sm(args.sm_csv) if args.sm_csv else []
  ppo = pr.read_tb(args.ppo_logdir) if args.ppo_logdir else []
  sm_eval = pr.read_eval(args.sm_eval) if args.sm_eval else None
  ppo_eval = pr.read_eval(args.ppo_eval) if args.ppo_eval else None
  summary = json.load(open(args.summary)) if args.summary else {}

  fig = plt.figure(figsize=(16, 10), dpi=170)
  gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
  method_panel(fig.add_subplot(gs[0, 0]))
  curves_panel(fig.add_subplot(gs[0, 1]), sm, ppo)
  eval_panel(fig.add_subplot(gs[1, 0]), sm_eval, ppo_eval)
  numbers_panel(fig.add_subplot(gs[1, 1]), summary)
  fig.suptitle("SuperMimic 复现（2026-11）｜官方实现 · 单卡 RTX 5090 · LAFAN1 40 条", fontsize=14,
               weight="bold")
  fig.savefig(args.out, bbox_inches="tight")
  print("wrote", args.out)


if __name__ == "__main__":
  main()
