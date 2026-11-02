"""PPO training on Minecraft (CPU or GPU)."""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from .config import load_config, save_config
from .envs.vec import SubprocVecEnv
from .ppo.model import ActorCritic
from .ppo.ppo import Rollout, compute_gae, ppo_update


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class FrameStack:
    def __init__(self, num_envs: int, k: int):
        self.k = k
        self.buffers = [collections.deque(maxlen=k) for _ in range(num_envs)]
        self.ready = [False] * num_envs

    def reset(self, i: int, frame: np.ndarray) -> None:
        buf = self.buffers[i]
        buf.clear()
        for _ in range(self.k):
            buf.append(frame)
        self.ready[i] = True

    def push(self, i: int, frame: np.ndarray) -> None:
        if not self.ready[i]:
            self.reset(i, frame)
        else:
            self.buffers[i].append(frame)

    def stack(self, i: int) -> np.ndarray:
        return np.concatenate([f.transpose(2, 0, 1) for f in self.buffers[i]], axis=0)


def obs_to_tensors(obs_list: List[Dict[str, np.ndarray]], frames: FrameStack,
                   device: torch.device):
    images = np.stack([frames.stack(i) for i in range(len(obs_list))], axis=0)
    inv = np.stack([
        np.log1p(np.clip(o["inventory"], 0.0, 64.0)) / np.log(65.0) for o in obs_list
    ])
    inv = inv.astype(np.float32)
    scal = np.stack([
        [o["health_raw"] / 20.0, o["hunger_raw"] / 20.0, o["breath_raw"] / 300.0]
        for o in obs_list
    ]).astype(np.float32)
    equ = np.stack([o["equipped"] for o in obs_list]).astype(np.float32)
    return {
        "image": torch.from_numpy(images).to(device),
        "inventory": torch.from_numpy(inv).to(device),
        "scalars": torch.from_numpy(scal).to(device),
        "equipped": torch.from_numpy(equ).to(device),
    }, images


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--set", action="append", default=[],
                        help="dotted key=value override, e.g. ppo.lr=1e-4")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    for item in args.set:
        key, value = item.split("=", 1)
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            node[parts[-1]] = value
    if args.steps is not None:
        overrides.setdefault("ppo", {})["total_steps"] = args.steps
    if args.num_envs is not None:
        overrides.setdefault("run", {})["num_envs"] = args.num_envs
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if args.device is not None:
        overrides.setdefault("run", {})["device"] = args.device
    if args.out_dir is not None:
        overrides.setdefault("run", {})["out_dir"] = args.out_dir

    cfg = load_config(args.config, overrides)
    torch.manual_seed(cfg["run"]["seed"])
    np.random.seed(cfg["run"]["seed"])
    run_dir = cfg["run"]["out_dir"]
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.json"))
    with open(os.path.join(run_dir, "code_info.json"), "w") as fh:
        json.dump({"git": _git_sha(), "argv": sys.argv, "torch": torch.__version__,
                   "time": time.strftime("%F %T")}, fh, indent=2)

    device = torch.device(cfg["run"]["device"])
    threads = int(cfg["run"].get("torch_threads", 8))
    torch.set_num_threads(max(1, threads))
    n_envs = int(cfg["run"]["num_envs"])
    rollout_steps = int(cfg["ppo"]["rollout_steps"])
    chunk_len = int(cfg["ppo"]["chunk_len"])
    assert rollout_steps % chunk_len == 0, "rollout_steps must divide chunk_len"
    size = int(cfg["env"]["obs_size"])
    frame_channels = 3 * int(cfg["env"]["frame_stack"])
    reward_scale = float(cfg["ppo"].get("reward_scale", 5.0))

    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(run_dir, "tb"))
        except Exception as exc:  # pragma: no cover
            print("tensorboard disabled:", exc)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    episodes_path = os.path.join(run_dir, "episodes.jsonl")
    metrics_fh = open(metrics_path, "a", buffering=1)
    episodes_fh = open(episodes_path, "a", buffering=1)

    print(f"[mcppo] creating {n_envs} Minecraft envs ...", flush=True)
    t0 = time.time()
    vec = SubprocVecEnv(cfg, n_envs, seed=int(cfg["run"]["seed"]))
    print(f"[mcppo] envs ready in {time.time() - t0:.1f}s", flush=True)
    obs = vec.reset()
    frames = FrameStack(n_envs, int(cfg["env"]["frame_stack"]))
    for i, o in enumerate(obs):
        frames.reset(i, o["image"])

    model = ActorCritic(vec.n_actions, vec.obs_dim, vec.equipped_dim,
                        frame_channels=frame_channels,
                        hidden_size=int(cfg["ppo"]["hidden_size"]),
                        feature_size=int(cfg["ppo"]["feature_size"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["ppo"]["lr"]),
                                 eps=1e-5)
    hidden = model.initial_hidden(n_envs, device)
    start_update = 0
    global_step = 0
    if args.resume:
        ckpt_path = os.path.join(run_dir, "ckpt", "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_update = int(ckpt["update"]) + 1
            global_step = int(ckpt["step"])
            print(f"[mcppo] resumed from {ckpt_path} at step {global_step}", flush=True)

    samples_per_update = rollout_steps * n_envs
    total_updates = int(cfg["ppo"]["total_steps"]) // samples_per_update
    total_updates = max(total_updates, start_update)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[mcppo] samples={total_updates * samples_per_update} "
          f"updates={total_updates} samples_per_update={samples_per_update} "
          f"params={total_params/1e6:.2f}M device={device} run_dir={run_dir}", flush=True)

    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    def lr_at(u: int) -> float:
        lo, hi = float(cfg["ppo"]["lr_final"]), float(cfg["ppo"]["lr"])
        frac = min(1.0, max(0.0, u / max(1, total_updates)))
        return hi + (lo - hi) * frac

    def entropy_at(u: int) -> float:
        hi, lo = float(cfg["ppo"]["entropy_coef"]), float(cfg["ppo"]["entropy_final"])
        frac = min(1.0, max(0.0, u / max(1, total_updates)))
        return hi + (lo - hi) * frac

    def save(update: int, step: int) -> None:
        path = os.path.join(run_dir, "ckpt", f"step_{step:09d}.pt")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "update": update, "step": step, "cfg": cfg}
        torch.save(payload, path)
        torch.save(payload, os.path.join(run_dir, "ckpt", "latest.pt"))
        ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "ckpt"))
                       if f.startswith("step_"))
        for old in ckpts[:-int(cfg["logging"]["keep_checkpoints"])]:
            try:
                os.remove(os.path.join(run_dir, "ckpt", old))
            except OSError:
                pass

    last_log = time.time()
    episode_count = 0
    for update in range(start_update, total_updates):
        lr = lr_at(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rollout = Rollout(rollout_steps, n_envs, frame_channels, size,
                          vec.obs_dim, vec.equipped_dim, chunk_len,
                          hidden_size=int(cfg["ppo"]["hidden_size"]))
        t_start = time.time()
        ep_returns: List[float] = []
        ep_scores: List[int] = []
        comp_sums = collections.defaultdict(float)
        milestone_events = 0
        inv_max = collections.defaultdict(int)
        for t in range(rollout_steps):
            feats, images = obs_to_tensors(obs, frames, device)
            if t % chunk_len == 0:
                rollout.store_hidden(t, hidden[0], hidden[1])
            image = feats["image"].float().div_(255.0)
            with torch.no_grad():
                logits, values, hidden = model(
                    image.unsqueeze(0), feats["inventory"].unsqueeze(0),
                    feats["scalars"].unsqueeze(0), feats["equipped"].unsqueeze(0),
                    hidden)
                logits = logits[0]
                values = values[0]
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)
            next_obs, rewards, dones, infos = vec.step(actions.cpu().numpy())
            rollout.add(t, torch.from_numpy(images), feats["inventory"].cpu(),
                        feats["scalars"].cpu(), feats["equipped"].cpu(),
                        actions.cpu(), logp.cpu(), values.cpu(),
                        torch.from_numpy(rewards / reward_scale),
                        torch.from_numpy(dones))
            for i in range(n_envs):
                comp = next_obs[i].get("_reward_components", {})
                for name, value in comp.items():
                    comp_sums[name] += float(value)
                if comp.get("milestone", 0.0) > 0:
                    milestone_events += 1
                for name in ("log", "planks", "stick", "crafting_table",
                             "wooden_pickaxe", "cobblestone", "stone_pickaxe",
                             "iron_ore", "iron_ingot", "diamond"):
                    inv_max[name] = max(inv_max[name],
                                        int(next_obs[i].get("_inventory", {}).get(name, 0)))
                frames.push(i, next_obs[i]["image"])
                if dones[i]:
                    ep = infos[i].get("episode")
                    if ep is not None:
                        ep_returns.append(ep["return"])
                        ep_scores.append(ep["score"])
                        episode_count += 1
                        record = {"step": global_step + (t + 1) * n_envs, "update": update,
                                  "wall": time.time(), **ep}
                        episodes_fh.write(json.dumps(record) + "\n")
                        if writer is not None:
                            writer.add_scalar("episode/return", ep["return"],
                                              global_step + (t + 1) * n_envs)
                            writer.add_scalar("episode/score", ep["score"],
                                              global_step + (t + 1) * n_envs)
                        for name, value in ep["components"].items():
                            if writer is not None:
                                writer.add_scalar(f"episode/reward_{name}", value,
                                                  global_step + (t + 1) * n_envs)
                    frames.reset(i, next_obs[i]["image"])
                    hidden[0][:, i] = 0.0
                    hidden[1][:, i] = 0.0
            obs = next_obs
        collect_time = time.time() - t_start

        with torch.no_grad():
            last_feats, _ = obs_to_tensors(obs, frames, device)
            last_image = last_feats["image"].float().div_(255.0)
            _, last_values, _ = model(
                last_image.unsqueeze(0), last_feats["inventory"].unsqueeze(0),
                last_feats["scalars"].unsqueeze(0), last_feats["equipped"].unsqueeze(0),
                hidden)
            last_values = last_values[0]

        advantages, returns = compute_gae(rollout, last_values.cpu(),
                                          float(cfg["ppo"]["gamma"]),
                                          float(cfg["ppo"]["gae_lambda"]))
        t_learn = time.time()
        stats = ppo_update(model, optimizer, rollout, advantages, returns, cfg["ppo"],
                           device, entropy_at(update))
        learn_time = time.time() - t_learn
        global_step += samples_per_update
        elapsed = time.time() - t_start
        rec = {
            "step": global_step,
            "update": update,
            "lr": lr,
            "entropy_coef": entropy_at(update),
            "sps": samples_per_update / max(1e-6, elapsed),
            "ticks_per_s": samples_per_update * int(cfg["env"]["repeat"]) / max(1e-6, elapsed),
            "collect_s": collect_time,
            "learn_s": learn_time,
            "episode_return_mean": float(np.mean(ep_returns)) if ep_returns else None,
            "episode_score_max": int(np.max(ep_scores)) if ep_scores else None,
            "episodes": episode_count,
            "milestone_events": milestone_events,
            "reward_components": {k: round(v, 3) for k, v in comp_sums.items()},
            "max_inventory": dict(inv_max),
            **{f"loss_{k}": v for k, v in stats.items()},
        }
        metrics_fh.write(json.dumps(rec) + "\n")
        if writer is not None:
            for key, value in rec.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(f"train/{key}", value, global_step)
        print(json.dumps(rec), flush=True)

        if (update + 1) % int(cfg["logging"]["save_every_updates"]) == 0:
            save(update, global_step)
            print(f"[mcppo] checkpoint saved at step {global_step}", flush=True)

    if total_updates > start_update:
        save(total_updates - 1, global_step)
    vec.close()
    metrics_fh.close()
    episodes_fh.close()
    if writer is not None:
        writer.close()
    print(f"[mcppo] done at step {global_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
