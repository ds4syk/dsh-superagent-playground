# Results: JEV as a world model and action source

> 2026-11-03. All model numbers here are **our measurements** with open JEV-style
> proxies (Tacit-4B/1.7B, Qwen3-4B typed readout), not the hosted JEV API.
> Literature numbers (KIIS 2026 etc.) are cited separately in `kb/jev-wm/`.

## Setup

- **World**: deterministic two-room text gridworld. State = player room, key /
  apple / mushroom location, door (locked/closed/open), box (closed/open),
  inventory; 9 commands. Fully observable, no randomness.
- **Data**: 40 train / 4 val / 24 test worlds, vocabulary split; 2,727 / 289 /
  1,683 engine-labelled transitions. Each state is paired with 3 valid commands,
  `look`, and 4 invalid commands.
- **Queries**: 8 typed questions per transition (executes + 7 state variables);
  every option set enumerates all type-valid values.
- **Models**:
  - Tacit-4B (`morriszjm/Tacit-4B`), frozen.
  - Tacit-1.7B, frozen (CPU probe only).
  - Qwen3-4B-Instruct-2507, typed readout via AnyJev `Decider(level="raw")`, zero-shot.
  - Qwen3-4B-Instruct-2507, generative JSON, zero-shot.
  - persistence (state unchanged).
- **Hardware**: one RTX 5090 32GB via Slurm; model files under
  `/mnt/storage1/syk/jev-wm/models/`.

## Exp1: world-model prediction

500 test transitions (unseen vocabulary), 8 typed questions per transition:

| model | exact | executes | macro var | observe | simple | derive | 4-step rollout |
|-|-|-|-|-|-|-|-|
| persistence | 0.745 | 0.620 | 0.950 | – | – | – | 0.20 |
| Tacit-4B (typed) | **0.754** | 0.696 | **0.952** | **0.981** | 0.781 | 0.709 | 0.18 |
| Qwen3-4B typed (400 transitions) | **0.790** | 0.703 | **0.966** | 1.000 | 0.671 | 0.791 | 0.15 |
| Qwen3-4B generative | 0.540 | 0.740 | 0.881 | 0.459 | 0.620 | 0.526 | 0.17 |
| Tacit-1.7B (CPU probe, n=6) | 0.500 | 0.667 | 0.905 | 1.000 | 0.500 | 0.333 | – |

Key reading:

- Tacit-4B is only **+0.009** over persistence; Qwen3-4B typed is **+0.045** over persistence on exact state match. Persistence
  is a strong baseline here because 318/500 sampled actions are invalid and leave
  the state unchanged. On transitions where the command executes, Tacit-4B gets
  **0.566** exact (persistence cannot update the state at all).
- Accuracy is high for read-off (`look`: 0.981/1.000) and lower for derived
  effects (`take`/`unlock`/`go`: 0.709/0.791); simple effects are the weak spot
  for Qwen typed (0.671). Errors concentrate in whether a command executes and in
  applying its effect. On executed transitions, Tacit-4B (0.566) beats Qwen typed
  (0.481), so the two proxies fail differently: Qwen is better at validity and
  overall consistency, Tacit at post-action effects.
- Four-step open-loop exact match falls to **0.18**: errors compound. Use
  replanning every step.
- Measured 0.106 s per transition (75.6 decisions/s) on one 5090; API-equivalent
  JEV cost ~ **$0.036 / 1,000 transitions** (8 questions x ~200 tokens).

## Exp2: action generation and distillation

### Distillation upper bound (scripted oracle labels, CPU)

48 `testsame` worlds, initial state only, 6-step budget, invalid actions masked:

| policy | success | mean steps |
|-|-|-|
| random admissible | 0.625 | 5.44 |
| oracle | 1.000 | 3.67 |
| 1M-param student from oracle labels | 1.000 | 3.67 |

This validates the student/ training pipeline: a ~1M-parameter transformer can
imitate the scripted solver perfectly on in-vocabulary worlds. The remaining
question is whether JEV-generated labels are good enough.

### CPU probe: Tacit-1.7B action selection (8 worlds, 6 steps)

| policy | success | mean steps | decisions/episode | wall |
|-|-|-|-|-|
| random | 0.625 | 5.44 | 0 | – |
| choice | 0.750 | 5.38 | 4.6 | 24 s |
| score | 0.875 | 5.38 | 10.6 | 60 s |
| oracle | 1.000 | 3.67 | 0 | – |

Even a 1.7B JEV-style proxy picks useful actions zero-shot; per-action scoring is
better than a single 9-way choice. This is a preliminary signal; the 4B run and
the distillation comparison are the main experiment.

### Tacit-4B teachers (48 `testsame` worlds, 6 steps)

| teacher | success | mean steps | decisions/episode |
|-|-|-|-|
| random | 0.625 | 5.44 | 0 |
| choice (1 query/step) | 0.833 | 5.33 | 4.5 |
| score (per-action progress) | **1.000** | 4.79 | 9.8 |
| world-model lookahead (|A|×8 queries/step) | 0.833 | 5.04 | 55.8 |
| oracle | 1.000 | 3.67 | 0 |

Per-action progress scoring beats both the single choice query and the much more
expensive world-model lookahead. The lookahead does not pay for itself here:
prediction error plus the hand-written utility cancels its information advantage.

### JEV teacher → distilled student (main Exp2)

| policy | success | mean steps | JEV decisions/episode | student params |
|-|-|-|-|-|
| random admissible | 0.625 | 5.44 | 0 | – |
| oracle | 1.000 | 3.67 | 0 | – |
| teacher: choice | 0.833 | 5.33 | 4.5 | – |
| teacher: per-action score | **1.000** | 4.79 | 9.8 | – |
| teacher: world-model lookahead | 0.833 | 5.04 | 55.8 | – |
| student from oracle labels | **1.000** | 3.67 | 0 | 1M |
| student from score labels | **0.958** | 4.69 | 0 | 1M |
| student from WM-lookahead labels | 0.833 | 4.96 | 0 | 1M |

The score teacher is perfect on 48 test worlds; a 1M-parameter student distils
it to 95.8% success with **zero JEV calls at deployment**. The world-model
lookahead teacher costs 5.7x more queries and is no better (83.3%): prediction
error plus a hand-written utility cancels its information advantage. The student
trained on WM labels exactly matches its teacher (83.3%).

## Cost accounting

- JEV API list price: $0.042/MTok input, output free. At ~200 tokens/decision:
  - choice: 4.5 decisions/episode -> **$0.00004/episode**
  - score: 9.8 -> **$0.00008/episode**
  - world-model lookahead: 55.8 -> **$0.00047/episode**
  - 8 questions/transition: ~$0.036 per 1,000 transitions.
- Local proxy: Tacit-4B runs at ~75.6 decisions/s on one RTX 5090; wall time per
  500-transition Exp1 run is 52.9 s.
- Student: ~1M parameters, one forward per step, 0 JEV calls at deployment; the
  measured student latency is sub-millisecond on CPU for the gridworld.
- These are API-equivalent estimates, not a TCO (no retries/network/engineering).

## Limitations

- Open proxies, not hosted JEV; single seed; one world family; deterministic and
  fully observable; Exp2 uses `testsame` (same vocabulary) so lexical
  generalization is isolated in Exp1.
- Invalid actions are masked in Exp2 (an action interface is assumed); the
  underlying transitions still contain invalid commands for Exp1.
- See `kb/jev-wm/08.md` for the full list.
