# Run artifacts

Small, reviewable files copied from the training run on the lab node
(`/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0`). Large checkpoints and the
TensorBoard event files stay on the data disk.

| File | Content |
|-|-|
| `config.json` | full config snapshot (reward weights, PPO hyper-parameters) |
| `code_info.json` | git sha, argv, torch version |
| `metrics.jsonl` | one line per PPO update (losses, entropy, KL, throughput, reward components) |
| `episodes.jsonl` | one line per finished episode (return, score, milestones) |
| `curves/curves.png`, `curves.pdf` | summary plots |

Regenerate the plots with `python -m mcppo.plot_curves --run <run_dir>`.
