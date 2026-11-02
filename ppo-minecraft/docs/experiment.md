# Experiment: PPO on Minecraft diamond (2026-11)

## Setup

| Item | Value |
|-|-|
| Environment | real Minecraft 1.11.2 via patched MineRL 0.4.4 / Malmo |
| Task | DreamerV3 `minecraft_diamond`: 64x64 RGB, 391-dim inventory, 25 composite actions, break_speed=100 |
| Reward | hand-written dense reward (`kb/03`), undiscounted potential shaping |
| Learner | recurrent PPO, CNN+MLP+LSTM(256), 6 envs, rollout 1024/env, chunk 32, 4 epochs, lr 3e-4->1e-4 |
| Slurm | `cpu` partition, 6 envs, 10 CPUs, 26 GB, 12 h |
| Run dir | `/mnt/storage1/syk/dreamer/runs/mcppo/diamond_s0` |

## Measurements (actual)

- First reset of 6 envs: ~2-3 min (0.7 GB game-tree copy per env + world generation).
- Tick rate per env: 7-12/s; end-to-end 41-43 samples/s.
- Per update (6,144 samples): collect ~120 s, learn ~28 s.
- Reward walkthrough (ideal tech tree): 72.1 total; milestones 70.9, shaping 4.8, time -3.6.
- Random policy first two updates: 3-4 milestone events (log, planks, sticks, crafting table).

## Status

- Diamond is not reached in this run (see `kb/05` for the compute estimate).
- The run is an early-stage validation of the environment, reward, and PPO pipeline.

## Artifacts

`metrics.jsonl`, `episodes.jsonl`, `curves/`, `ckpt/` under the run directory; a
small copy is under `results/`.
