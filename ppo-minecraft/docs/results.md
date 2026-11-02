# Results (running experiment)

## Setup

See `docs/experiment.md`. Six Minecraft instances, 10 CPUs, repeat=1, 25 composite actions,
hand-written dense reward (undiscounted potential shaping).

## Observed so far

| Quantity | Value |
|-|-|
| samples per update | 6,144 |
| collection per update | 120-150 s |
| PPO update per update | 28 s |
| end-to-end | 35-43 samples/s |
| first episodes | early deaths (drowning) after ~2.6k steps; return -7.7 |
| random-policy milestone events | 3-4 per 6,144 samples (log / planks / sticks / table) |

## Reading the curves

`curves/curves.png` shows episode return, tech-tree score, losses, entropy/KL/clip,
throughput, reward components, episode length and a text summary. The orange curve on
the score panel is the number of milestone events per update.

If the milestone-event curve stays flat at 3-4 while the score stays 0, the policy is
still exploring; if it grows, the reward shaping is producing learning.

## v0 ablation (49,152 samples, stopped)

With `death_penalty=-5` the policy became conservative: five consecutive updates had zero
milestone events after the initial random exploration. Episodes show frequent early deaths
(drowning/falling) at 1-6k steps, so -5 dominated the return and "move less" won.
v1 lowers death to -1 (episode termination already costs future reward), doubles the
exploration potential cap to +4, and raises the step penalty to -0.0002.

## v1 first update

40.9 samples/s, 3 milestone events, exploration component 1.11 (vs 0.37 in v0),
value loss 0.079, entropy 3.218. The run continues.

## Honest status

Diamond is not reached. At ~40 samples/s, 71M steps (DreamerV3 median) is ~20 days;
100M steps is ~29 days. A pure-PPO diamond run is therefore not a single-session
deliverable. The practical next step is BC pretraining on MineRL demonstrations
followed by PPO fine-tuning, or hierarchical PPO.

## Comparison

| Run | death penalty | exploration cap | stop reason |
|-|-|-|-|
| v0 | -5 | +2 | milestone events -> 0 (conservative) |
| v1 | -1 | +4 | running |

## v1 episodes (early)

The first episodes after the reward change: one reached log/planks/crafting_table
(score 4, return -0.9) and a short death cost only -1 instead of -5. The policy is
still close to random, but the reward is no longer dominated by death.

## v1 milestone events

Per-update milestone events: 3, 3, 0, 3 (updates 0-3). Unlike v0, the events do not
disappear after the initial exploration, and one episode reached score 4. The run is
still early (~30k samples).
