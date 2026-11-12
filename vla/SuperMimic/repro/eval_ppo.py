"""Evaluate a local PPO-Sim checkpoint (mjlab / rsl_rl) with the Appendix-F protocol.

Mirrors ``eval_local.py`` for the SuperMimic checkpoint, but loads an mjlab
rsl_rl ``model_*.pt`` and uses the deterministic inference policy.

Example:
  uv run python repro/eval_ppo.py \
    --checkpoint /.../logs/rsl_rl/g1_tracking/<run>/model_2000.pt \
    --motion-glob 'motions/lafan1_npz/*.npz' \
    --output results/eval_ppo.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task-id", default="Mjlab-Tracking-Flat-Unitree-G1")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--motion-glob", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--device", default=None)
  parser.add_argument("--max-steps", type=int, default=0)
  args = parser.parse_args()

  files = sorted(glob.glob(os.path.expanduser(args.motion_glob)))
  if not files:
    raise SystemExit(f"no motion matched {args.motion_glob}")
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

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
  env_cfg.scene.num_envs = len(files)
  agent_cfg = load_rl_cfg(args.task_id)

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device),
                           clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)
  cmd = env.unwrapped.command_manager.get_term("motion")

  env.reset()
  ends = cmd.motion.motion_ends.to(device)
  starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), ends[:-1]])
  for i in range(len(files)):
    cmd.reset_to_frame(torch.tensor([i], device=device), int(starts[i]))
  env.unwrapped.sim.forward()
  cmd.update_relative_body_poses()

  n = len(files)
  sums = torch.zeros(n, device=device)
  steps = torch.zeros(n, device=device, dtype=torch.long)
  failed = torch.zeros(n, device=device, dtype=torch.bool)
  finished = torch.zeros(n, device=device, dtype=torch.bool)

  for _ in range(args.max_steps or 10**9):
    obs = env.get_observations()
    motion_times = cmd.time_steps.clone()
    with torch.no_grad():
      actions = policy(obs)
    _, rewards, dones, _ = env.step(actions)
    active = ~finished
    sums += rewards.squeeze(-1) * active
    steps += active.long()

    term = env.unwrapped.termination_manager.terminated
    motion_end = env.unwrapped.termination_manager.get_term("motion_end")
    reached = motion_end | (motion_times + 1 >= ends)
    done = dones.squeeze(-1).bool()
    failed |= done & ~motion_end & active
    finished |= (done | reached) & active
    finished |= steps >= (ends - starts)
    if bool(finished.all()):
      break

  # Paper F.1: normalize by the full clip length, so early termination scores low.
  clip_steps = (ends - starts).clamp_min(1)
  norm = (sums / (0.1 * clip_steps)).clamp(0.0, 1.0)
  rows = [{
    "clip": os.path.splitext(os.path.basename(f))[0],
    "steps": int(steps[i]),
    "clip_steps": int((ends[i] - starts[i]).item() if hasattr(ends[i] - starts[i], "item") else (ends[i] - starts[i])),
    "sum_reward": float(sums[i]),
    "normalized_return": float(norm[i]),
    "failed": bool(failed[i]),
  } for i, f in enumerate(files)]
  summary = {
    "checkpoint": os.path.abspath(args.checkpoint),
    "task_id": args.task_id,
    "clips": n,
    "mean_normalized_return": float(norm.mean()),
    "min_normalized_return": float(norm.min()),
    "failed": int(failed.sum()),
    "rows": rows,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
  print(f"[eval] mean normalized return {float(norm.mean()):.4f}, "
        f"failed {int(failed.sum())}/{n} -> {args.output}", flush=True)


if __name__ == "__main__":
  main()
