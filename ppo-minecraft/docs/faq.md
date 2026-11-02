# FAQ / troubleshooting

**Why is the first reset so slow (2-3 min for 6 envs)?**
Each Minecraft instance copies the 0.7 GB game tree into `TMPDIR` and generates a world.
Stage the tree on NVMe (`scripts/prepare_mc.sh`) and keep `TMPDIR` on NVMe; the first
reset drops to ~40 s per env.

**Why is the tick rate only 7-12/s instead of 20/s?**
The Malmo server renders a 64x64 frame for every step and the node is shared. The
bottleneck is Java/CPU, not PPO (learn time is ~20% of an update).

**Why `np.float` errors?**
MineRL 0.4.4 predates NumPy 1.24. `mcppo/envs/minecraft.py` sets the aliases before
importing the handlers, like DreamerV3 does.

**Why does the reward not include a per-step item reward?**
The potential `log(1+max inventory)` is a monotone, diminishing-return quantity reward;
using max inventory prevents crafting from subtracting reward and prevents re-collecting
the same item from being farmed.

**Why undiscounted potential shaping?**
With gamma=0.999 the term `-(1-gamma)*Phi` accumulates to about -24 on a full tech-tree
episode; it made the shaped reward dominate the task reward. `gamma_shape=1.0` fixes it
(see `kb/03`).

**How do I continue a run?**
`sbatch scripts/train_diamond.sbatch` with `--resume` (default) loads `ckpt/latest.pt`.
Change `RUN=...` to continue a different directory.
