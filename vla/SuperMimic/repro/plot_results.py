"""Plot the reproduction results.

Inputs (all optional; missing ones are skipped):
  --sm-csv      results/supermimic_train.csv  (from parse_logs.py)
  --ppo-logdir  logs/rsl_rl/g1_tracking/<run> (tensorboard event files)
  --sm-eval     results/eval_supermimic.json
  --ppo-eval    results/eval_ppo.json

Outputs: results/training_curves.png, results/eval_bars.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_sm(path):
  rows = []
  with open(path, newline="") as fh:
    for r in csv.DictReader(fh):
      if not r.get("mean_reward"):
        continue
      try:
        elapsed = sum(int(x) * m for x, m in zip(r["elapsed"].split(":"), (3600, 60, 1)))
      except Exception:
        elapsed = 0.0
      rows.append({
        "iteration": int(r["iteration"]),
        "mean_reward": float(r["mean_reward"]),
        "mean_episode_length": float(r["mean_episode_length"]),
        "elapsed": elapsed,
      })
  return rows


def read_tb(logdir):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  files = sorted(glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"),
                          recursive=True))
  if not files:
    return []
  ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags().get("scalars", [])
  rew_tag = next((t for t in tags if t.endswith("mean_reward")), None)
  len_tag = next((t for t in tags if t.endswith("mean_episode_length")), None)
  if rew_tag is None:
    return []
  lens = {e.step: e.value for e in ea.Scalars(len_tag)} if len_tag else {}
  t0 = min(e.wall_time for e in ea.Scalars(rew_tag))
  out = []
  for e in ea.Scalars(rew_tag):
    out.append({
      "iteration": e.step,
      "mean_reward": e.value,
      "mean_episode_length": lens.get(e.step, float("nan")),
      "elapsed": e.wall_time - t0,
    })
  return out


def proxy(rows):
  x = np.array([r["elapsed"] for r in rows]) / 3600.0
  rew = np.array([r["mean_reward"] for r in rows])
  length = np.array([r["mean_episode_length"] for r in rows])
  with np.errstate(divide="ignore", invalid="ignore"):
    y = rew / (0.1 * length)
  return x, y


def smooth(y, target=200):
  if len(y) <= 8:
    return y
  k = max(1, len(y) // target)
  pad = k // 2
  padded = np.pad(y, (pad, pad), mode="edge")
  return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(y)]


def plot_curves(sm, ppo, out):
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
  for rows, label, color in ((sm, "SuperMimic (BPTT, WM)", "#1b9e77"),
                             (ppo, "PPO-Sim (rsl_rl)", "#d95f02")):
    if not rows:
      continue
    x, y = proxy(rows)
    y_s = smooth(y)
    axes[0].plot(x, y_s, label=label, lw=1.4, color=color)
    axes[1].plot(np.array([r["iteration"] for r in rows]), y_s, label=label,
                 lw=1.4, color=color)
  axes[0].set_xlabel("wall-clock training time (h)")
  axes[1].set_xlabel("iteration")
  for ax in axes:
    ax.set_ylabel("training return / (0.1 x episode length)")
    ax.grid(alpha=0.3)
    ax.legend()
  axes[0].set_title("training return vs wall-clock")
  axes[1].set_title("training return vs iteration")
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print("wrote", out)


def read_eval(path):
  with open(path) as fh:
    return json.load(fh)


def plot_eval(sm_eval, ppo_eval, out):
  fig, ax = plt.subplots(figsize=(11, 3.8))
  for data, label, color in ((ppo_eval, "PPO-Sim", "#d95f02"),
                             (sm_eval, "SuperMimic", "#1b9e77")):
    if not data:
      continue
    vals = np.array([r["normalized_return"] for r in data["rows"]])
    order = np.argsort(vals)
    ax.plot(np.arange(len(vals)), vals[order], marker=".", ms=3, lw=0.9,
            label=f"{label} (mean {vals.mean():.3f}, fail {data['failed']})",
            color=color)
  ax.set_xlabel("clip (sorted)")
  ax.set_ylabel("normalized return")
  ax.set_ylim(0, 1)
  ax.grid(alpha=0.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig(out, dpi=160)
  print("wrote", out)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sm-csv")
  ap.add_argument("--ppo-logdir")
  ap.add_argument("--sm-eval")
  ap.add_argument("--ppo-eval")
  ap.add_argument("--outdir", default="results")
  args = ap.parse_args()
  os.makedirs(args.outdir, exist_ok=True)
  sm = read_sm(args.sm_csv) if args.sm_csv else []
  ppo = read_tb(args.ppo_logdir) if args.ppo_logdir else []
  if sm or ppo:
    plot_curves(sm, ppo, os.path.join(args.outdir, "training_curves.png"))
  sm_eval = read_eval(args.sm_eval) if args.sm_eval else None
  ppo_eval = read_eval(args.ppo_eval) if args.ppo_eval else None
  if sm_eval or ppo_eval:
    plot_eval(sm_eval, ppo_eval, os.path.join(args.outdir, "eval_bars.png"))


if __name__ == "__main__":
  main()
