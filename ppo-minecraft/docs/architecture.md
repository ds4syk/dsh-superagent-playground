# Architecture

```
train.py
  ├── SubprocVecEnv (6 workers, one Malmo/Minecraft each)
  │     └── MinecraftDiamondEnv: 64x64 obs, 24 composite actions, RewardEngine
  ├── FrameStack (4 frames) + observation featurisation
  ├── ActorCritic (CNN + inventory MLP + LSTM 256)
  ├── Rollout storage (T=1024, B=6, chunk=32, hidden states per chunk)
  └── ppo_update (GAE, clip, value loss, entropy, chunked BPTT)
```

## Data flow per policy step

1. Parent sends action index (0-23) to all workers over a pipe.
2. Worker composes the action dict (camera/craft/equip + movement defaults),
   calls the MineRL gym env `repeat` times (repeat=1 by default), flattens the
   nested observation dict and runs the per-env `RewardEngine`.
3. Worker returns `(obs, reward, done, info)`; on `done` it records the episode,
   resets the Minecraft world and returns the new first observation.

## Reward engine

Stateful per environment, reset on episode start. On every step it updates the
monotone max-inventory vector, visited 8x8 cells and the minimum y, computes the
undiscounted potential difference, then adds one-time milestones (speed-scaled),
health/death, hunger/breath, depth bonuses and the step penalty. All components
are returned and logged. See `kb/03` for the full design.

## PPO details

- γ=0.999, GAE λ=0.95, clip 0.2, 4 epochs, minibatch 64 chunks of 32 steps.
- Reward is divided by 5 for scale; advantages are normalised across the batch.
- The LSTM state at the start of every 32-step chunk is stored during rollout and
  used as the initial state when replaying that chunk during the update, so the
  gradient only flows through 32 steps.
- Episode end resets the corresponding LSTM state; the frame stack is refilled
  with the new first frame.

## Failure modes seen during development

| Symptom | Cause | Fix |
|-|-|-|
| `KeyError: equipped_items/mainhand/type` | MineRL returns nested Dict obs, not `/`-flattened | flatten spaces/obs like `embodied.FromGym` |
| `Input type (unsigned char)` | uint8 frames passed to conv | cast/scale before the model |
| `np.float` AttributeError inside MineRL | NumPy >= 1.24 removed aliases | set `np.float = float` before importing handlers |
| First reset appears to hang | 6 JVMs copy 0.7 GB game tree each | stage the tree on NVMe, TMPDIR on NVMe |
| `-24` shaped reward on a constant state | discounted potential (`γ<1`) drift | undiscounted potential (`gamma_shape=1.0`) |
