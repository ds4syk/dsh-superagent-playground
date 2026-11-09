# Design notes and threats to validity

## Why a custom world instead of TextWorld?

The KIIS 2026 harness is the closest prior work, but it pins `textworld==1.7.0`,
a specific planner, Qwen3-4B policy and a JEV API key. We need (a) a harness we
can run end-to-end with open weights, (b) explicit state variables to score
prediction, and (c) cheap action-label generation. A 9-action, 7-variable
deterministic gridworld gives all three. It is not a TextWorld replica; results
are not directly comparable to KIIS's percentages.

## Why typed questions?

Free-form generation confounds *knowledge* with *formatting*: a model can know the
next state but emit an unparseable string. Typed questions remove that failure
mode and make probabilities comparable across models. The cost is option
enumeration; we enumerate all type-valid values per variable.

## Why separate `executes`?

KIIS measured that bundling validity with state questions drifts the answer
(0.285 -> 0.576). We ask it as its own noul question and override the predicted
values with the current state when it says "No" (the environment convention).

## Why `testsame` in Exp2?

Exp1 tests lexical generalization (unseen object names). Exp2 isolates
*distillation*: training and test worlds share the vocabulary, so a ~1M-parameter
student is not asked to ground new nouns. Lexical generalization of the student is
a separate question and is reported honestly as out of scope.

## Why mask invalid actions in Exp2?

An agent normally receives a set of currently admissible commands. Masking makes
the policy comparison about *which* valid action to choose, not about whether the
model can reject impossible commands. Exp1 keeps invalid commands in the
transition set, so validity prediction is still measured.

## Student size

A 4-layer, d=128 transformer (~1M parameters) is deliberately tiny: the point is
to show the cost ceiling of distillation, not to maximize student accuracy.
`student_oracle` (labels from the scripted solver) is the distillation upper
bound; `student_wm` (labels from the JEV teacher) is the real pipeline.

## Threats to validity

- Single seed and one world family; no confidence intervals.
- Fully observable, deterministic transitions; no stochasticity or partial
  observability.
- Open proxies, not hosted JEV; the hosted model may behave differently.
- Teacher policies use the same candidate set and heuristic utility; the
  lookahead teacher's utility is hand-designed, so a better utility could change
  the ranking.
- Cost is measured on one GPU and one API price list; not a TCO.
