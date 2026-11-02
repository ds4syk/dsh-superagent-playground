"""Experiment configuration: plain dicts so runs are reproducible from JSON."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "run": {
        "name": "diamond_ppo_s0",
        "seed": 0,
        "out_dir": "runs",
        "device": "cpu",
        "torch_threads": 8,
        "num_envs": 8,
    },
    "env": {
        # DreamerV3's Minecraft task settings (see docs/01_env.md).
        "obs_size": 64,
        "break_speed": 100.0,
        "repeat": 1,              # env ticks per policy step (DreamerV3 uses 1)
        "episode_steps": 18000,   # 15 min game time; DreamerV3 uses 36000
        "sticky_attack": 0,       # Dreamer sets this to 0 when break_speed != 1
        "sticky_jump": 10,
        "pitch_limit": 60,
        "frame_stack": 4,
        "render": True,
    },
    "reward": {
        # All weights live here so ablations are one config edit away.
        "gamma_shape": 1.0,         # undiscounted potential shaping (see 03_reward)
        "beta_shape": 1.0,          # weight on Phi-based item/exploration reward
        "step_penalty": 0.0002,
        "speed_gain": 0.5,          # milestone multiplier at t=0 (1.5x)
        "speed_floor": 0.75,
        "health_heal": 0.05,
        "health_damage": 0.10,
        "death_penalty": 1.0,
        "hunger_threshold": 6.0,
        "hunger_penalty": 0.25,
        "breath_threshold": 0.30,
        "breath_penalty": 0.01,
        "explore_cell": 8,          # cell size in blocks for novelty potential
        "explore_value": 0.01,
        "explore_cap": 4.0,
        "depth_value": 0.02,
        "depth_ref_y": 64.0,
        "depth_bonus": [              # one-time, at first reaching y <= threshold
            {"y": 32, "bonus": 0.5},
            {"y": 16, "bonus": 1.0},
            {"y": 11, "bonus": 1.5},
        ],
        "clip": [-10.0, 50.0],
        "item_weights": {
            "log": 0.02,
            "planks": 0.02,
            "stick": 0.025,
            "crafting_table": 0.05,
            "wooden_pickaxe": 0.10,
            "cobblestone": 0.01,
            "stone_pickaxe": 0.15,
            "iron_ore": 0.30,
            "furnace": 0.15,
            "iron_ingot": 0.50,
            "iron_pickaxe": 0.70,
            "diamond": 4.00,
        },
        "milestone_bonus": {
            "log": 0.5,
            "planks": 0.5,
            "stick": 0.6,
            "crafting_table": 1.0,
            "wooden_pickaxe": 1.5,
            "cobblestone": 1.0,
            "stone_pickaxe": 2.0,
            "iron_ore": 3.0,
            "furnace": 2.5,
            "iron_ingot": 4.0,
            "iron_pickaxe": 5.0,
            "diamond": 40.0,
        },
    },
    "ppo": {
        "total_steps": 2_000_000,
        "rollout_steps": 1024,
        "chunk_len": 32,
        "epochs": 4,
        "minibatch_chunks": 64,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "entropy_final": 0.003,
        "lr": 3e-4,
        "lr_final": 1e-4,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "normalize_adv": True,
        "reward_norm": True,
        "hidden_size": 256,
        "feature_size": 256,
        "frame_stack_channels": 12,
    },
    "logging": {
        "log_every_updates": 5,
        "save_every_updates": 200,
        "keep_checkpoints": 3,
    },
}


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_update(cfg, json.load(fh))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
