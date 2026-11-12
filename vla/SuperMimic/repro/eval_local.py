"""Evaluate a local SuperMimic checkpoint on a motion library (no W&B).

The official ``play.py`` loads both the runner config and the checkpoints from
a W&B run. This reproduction trains offline, so this script evaluates the
locally saved ``world_model.pt`` + ``actor.pt`` directly.

Protocol (paper Appendix F, as used by independent reproductions):
  * every clip is played once from its first frame;
  * the deterministic actor acts on ground-truth simulator observations;
  * no observation noise, no domain randomization, no pushes;
  * normalized return = sum(step reward) / (0.1 * steps), in [0, 1];
  * a clip *fails* when a training termination fires before the clip ends.

Example:
  uv run python repro/eval_local.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/<run> \
    --motion-glob 'motions/lafan1_npz/*.npz' \
    --output results/eval_supermimic.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict

import torch

from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.task_spec import build_actor_obs_td


def advance_history(history, sensor, history_length):
  if history_length == 1:
    return sensor.unsqueeze(1)
  if history is None:
    return sensor.unsqueeze(1).repeat(1, history_length, 1)
  return torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task-id", default="SuperMimic-GenCoord-G1")
  parser.add_argument("--checkpoint-dir", required=True)
  parser.add_argument("--motion-glob", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--device", default=None)
  parser.add_argument("--history-length", type=int, default=1)
  parser.add_argument("--actor-obs", default="rep_state", choices=["rep_state", "proprio"])
  parser.add_argument("--max-steps", type=int, default=0,
                      help="optional cap on steps per clip (0 = whole clip)")
  args = parser.parse_args()

  files = sorted(glob.glob(os.path.expanduser(args.motion_glob)))
  if not files:
    raise SystemExit(f"no motion matched {args.motion_glob}")
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[eval] {len(files)} clips on {device}", flush=True)

  runner_cfg = SuperMimicRunnerCfg(history_length=args.history_length,
                                   actor_obs=args.actor_obs)
  env_cfg = load_env_cfg(args.task_id, play=True)
  # Paper F.1: deterministic evaluation, no domain randomization.
  # Keep the DR event definitions (SimContext derives its spec layout and
  # normalization ranges from them) but replace their functions with no-ops, so
  # the model keeps its nominal friction / CoM / encoder bias.
  def _noop(env, env_ids, **kwargs):
    return None
  for _name in ("foot_friction", "base_com", "encoder_bias"):
    if _name in env_cfg.events:
      env_cfg.events[_name].func = _noop
  env_cfg.commands["motion"].motion_file = files
  env = create_env(env_cfg, len(files), device)
  cmd = env.unwrapped.command_manager.get_term("motion")

  wm = build_world_model(runner_cfg, env, device)
  wm.load_bundle(torch.load(os.path.join(args.checkpoint_dir, "world_model.pt"),
                            map_location=device, weights_only=False))
  wm.dynamics.eval()
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "actor.pt"),
                                   map_location=device, weights_only=False))
  actor.eval()
  task = next(ts for ts in actor.task_specs if ts.name == "motion")

  env.reset()
  ends = cmd.motion.motion_ends.to(device)
  starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), ends[:-1]])
  # Deterministically place every env at its clip's first frame (no pose or
  # velocity randomization), then refresh the reference-body cache.
  for i in range(len(files)):
    cmd.reset_to_frame(torch.tensor([i], device=device), int(starts[i]))
  env.unwrapped.sim.forward()
  cmd.update_relative_body_poses()

  n = len(files)
  sums = torch.zeros(n, device=device)
  steps = torch.zeros(n, device=device, dtype=torch.long)
  failed = torch.zeros(n, device=device, dtype=torch.bool)
  finished = torch.zeros(n, device=device, dtype=torch.bool)
  history = None

  for t in range(args.max_steps or 10**9):
    obs = env.get_observations()
    full = obs["full_state"].to(device)
    aux = wm.representation.extract_aux_state(full)
    rep_state = wm.representation.represent(full)
    motion_times = cmd.time_steps.clone()
    goal = task.extract_training_goal(cmd.motion, motion_times, aux)
    proprio = wm.representation.actor_obs_mode == "proprio"
    sensor = wm.representation.compute_actor_sensor(
      rep_state,
      wm.sim_context.read(env.unwrapped),
      sim_actor_obs=obs["actor"].to(device) if proprio else None,
    )
    history = advance_history(history, sensor, runner_cfg.history_length)
    actor_obs = build_actor_obs_td({"motion": goal}, history)
    with torch.no_grad():
      actions = actor(actor_obs)

    _, rewards, dones, _ = env.step(actions)
    active = ~finished
    sums += rewards.squeeze(-1) * active
    steps += active.long()

    term = env.unwrapped.termination_manager.terminated
    time_outs = env.unwrapped.termination_manager.time_outs
    motion_end = env.unwrapped.termination_manager.get_term("motion_end")
    # A failure is a termination that is not the normal clip end.
    failed |= (term | dones.squeeze(-1).bool()) & ~motion_end & active
    reached_end = motion_end | (motion_times + 1 >= ends)
    finished |= (dones.squeeze(-1).bool() | reached_end) & active
    finished |= steps >= (ends - starts)
    if bool(finished.all()) or bool(dones.all()):
      break

  # Paper F.1: normalize by the full clip length, so early termination scores low.
  clip_steps = (ends - starts).clamp_min(1)
  norm = (sums / (0.1 * clip_steps)).clamp(0.0, 1.0)
  rows = []
  for i, f in enumerate(files):
    rows.append({
      "clip": os.path.splitext(os.path.basename(f))[0],
      "steps": int(steps[i]),
    "clip_steps": int((ends[i] - starts[i]).item() if hasattr(ends[i] - starts[i], "item") else (ends[i] - starts[i])),
      "sum_reward": float(sums[i]),
      "normalized_return": float(norm[i]),
      "failed": bool(failed[i]),
    })
  summary = {
    "checkpoint": os.path.abspath(args.checkpoint_dir),
    "task_id": args.task_id,
    "runner_cfg": asdict(runner_cfg),
    "clips": len(files),
    "mean_normalized_return": float(norm.mean()),
    "min_normalized_return": float(norm.min()),
    "failed": int(failed.sum()),
    "rows": rows,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
  print(f"[eval] mean normalized return {float(norm.mean()):.4f}, "
        f"failed {int(failed.sum())}/{len(files)} -> {args.output}", flush=True)


if __name__ == "__main__":
  main()
