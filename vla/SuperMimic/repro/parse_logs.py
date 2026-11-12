"""Parse SuperMimic's console log into a CSV of training metrics.

The runner prints one block per ``log_interval`` iterations:

    ########################################
     Learning iteration 12/6000
    Run name: lafan1
    Total steps: 49152
    Steps per second: 8123
    Mean reward: 0.42
    Mean episode length: 133.7
    ...

Example:
  python repro/parse_logs.py logs/slurm/sm-train-1234.log results/supermimic.csv
"""

from __future__ import annotations

import csv
import re
import sys

FIELDS = {
  "iteration": re.compile(r"Learning iteration (\d+)/(\d+)"),
  "steps_per_second": re.compile(r"Steps per second: ([\d.]+)"),
  "mean_reward": re.compile(r"Mean reward: ([\d.\-]+)"),
  "mean_episode_length": re.compile(r"Mean episode length: ([\d.\-]+)"),
  "elapsed": re.compile(r"Time elapsed: (\d+:\d+:\d+)"),
}


def main() -> None:
  if len(sys.argv) != 3:
    raise SystemExit("usage: parse_logs.py <log> <out.csv>")
  rows: list[dict[str, str]] = []
  cur: dict[str, str] = {}
  for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    for key, rx in FIELDS.items():
      m = rx.search(line)
      if m:
        cur[key] = m.group(1)
        if key == "iteration":
          cur["total_iterations"] = m.group(2)
    if "Time elapsed:" in line and cur:
      rows.append(cur)
      cur = {}
  keys = ["iteration", "total_iterations", "mean_reward", "mean_episode_length",
          "steps_per_second", "elapsed"]
  with open(sys.argv[2], "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  print(f"wrote {len(rows)} rows -> {sys.argv[2]}")


if __name__ == "__main__":
  main()
