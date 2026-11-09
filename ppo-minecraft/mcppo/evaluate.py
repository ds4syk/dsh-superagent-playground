"""Evaluate a checkpoint on real Minecraft and optionally record a video."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from .config import load_config
from .envs.minecraft import MinecraftDiamondEnv
from .ppo.model import ActorCritic


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--record", default=None, help="output .mp4 path")
    parser.add_argument("--out", default=None, help="episode jsonl path")
    parser.add_argument("--sample", action="store_true", help="sample instead of argmax")
    args = parser.parse_args(argv)

    cfg = load_config(os.path.join(args.run, "config.json"))
    device = torch.device(cfg["run"]["device"])
    ckpt_path = args.ckpt or os.path.join(args.run, "ckpt", "latest.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    env = MinecraftDiamondEnv(cfg, seed=int(cfg["run"]["seed"]) + 1000)
    model = ActorCritic(env.n_actions, env.obs_dim, env.equipped_dim,
                        frame_channels=3 * int(cfg["env"]["frame_stack"]),
                        hidden_size=int(cfg["ppo"]["hidden_size"]),
                        feature_size=int(cfg["ppo"]["feature_size"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = args.out or os.path.join(args.run, "eval.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fh = open(out_path, "a", buffering=1)
    frames = []
    max_steps = args.max_steps or int(cfg["env"]["episode_steps"])
    k = int(cfg["env"]["frame_stack"])
    for episode in range(args.episodes):
        obs = env.reset()
        buf = collections.deque([obs["image"]] * k, maxlen=k)
        hidden = model.initial_hidden(1, device)
        done = False
        step = 0
        t0 = time.time()
        with torch.no_grad():
            while not done and step < max_steps:
                image = np.concatenate([f.transpose(2, 0, 1) for f in buf], axis=0)[None]
                inv = (np.log1p(np.clip(obs["inventory"], 0, 64)) / np.log(65))[None]
                scal = np.asarray([[obs["health_raw"] / 20, obs["hunger_raw"] / 20,
                                    obs["breath_raw"] / 300]], np.float32)
                feats = (
                    torch.from_numpy(image).to(device).float().div_(255.0),
                    torch.from_numpy(inv.astype(np.float32)).to(device),
                    torch.from_numpy(scal).to(device),
                    torch.from_numpy(obs["equipped"][None]).to(device),
                )
                logits, _, hidden = model(*feats, hidden)
                if args.sample:
                    action = torch.distributions.Categorical(logits=logits[0]).sample().item()
                else:
                    action = int(logits[0].argmax().item())
                obs, reward, done, info = env.step(action)
                buf.append(obs["image"])
                step += 1
                if args.record and episode == 0:
                    frames.append(obs["image"])
        record = {
            "step": step,
            "episode": episode,
            "return": env.reward.episode_return,
            "score": env.reward.score(),
            "milestones": sorted(env.reward.milestones),
            "components": {k: float(v) for k, v in env.reward.components.items()},
            "wall_s": time.time() - t0,
        }
        fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    fh.close()
    env.close()
    if args.record and frames:
        import imageio
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        imageio.mimsave(args.record, frames, fps=10, macro_block_size=1)
        print("saved video", args.record, len(frames), "frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
