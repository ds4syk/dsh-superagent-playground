"""Real Minecraft environment for PPO.

This is a small, self-contained adaptation of DreamerV3's Minecraft task
(``embodied/envs/minecraft_flat.py``, MIT licence, (c) Danijar Hafner), which
itself builds on MineRL's herobraine ``EnvSpec``.  We keep the task identical
-- Minecraft 1.11.2, 64x64 RGB, 24 abstract actions, break_speed=100 -- and only
replace Dreamer's wrapper with a plain gym adapter plus our reward engine.

Action set (24): noop, attack, 4x camera turn (15 deg), forward/back/left/
right, jump-forward, place dirt, and the 12 crafting/smelting/equipping steps
of the diamond tech tree.  Like Dreamer, camera/craft actions are applied on
the first tick of a repeat window, movement keys are held for the rest.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# MineRL 0.4.4 predates NumPy 1.24 and uses the removed aliases (DreamerV3 does
# the same shim in embodied/envs/minecraft_flat.py).
np.float = float  # type: ignore[attr-defined]
np.int = int  # type: ignore[attr-defined]
np.bool = bool  # type: ignore[attr-defined]

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP
from minerl.env.malmo import InstanceManager

from ..rewards.reward import RewardEngine

log = logging.getLogger(__name__)

# Each Minecraft instance copies the 0.7 GB game tree into TMPDIR on first
# reset.  On this node /mnt/storage1 is a busy HDD, so jobs can point the
# source at an NVMe staging copy (scripts/prepare_mc.sh) and TMPDIR at NVMe.
_FAST_MC = os.environ.get("MCPPO_MINECRAFT_DIR")
if _FAST_MC and os.path.exists(os.path.join(_FAST_MC, "launchClient.sh")):
    InstanceManager.MINECRAFT_DIR = _FAST_MC
    log.info("using staged Minecraft tree at %s", _FAST_MC)

BASIC_ACTIONS: Dict[str, Dict[str, Any]] = {
    "noop": dict(),
    "attack": dict(attack=1),
    "turn_up": dict(camera=(-15, 0)),
    "turn_down": dict(camera=(15, 0)),
    "turn_left": dict(camera=(0, -15)),
    "turn_right": dict(camera=(0, 15)),
    "forward": dict(forward=1),
    "back": dict(back=1),
    "left": dict(left=1),
    "right": dict(right=1),
    "jump": dict(jump=1, forward=1),
    "place_dirt": dict(place="dirt"),
}

DIAMOND_ACTIONS: Dict[str, Dict[str, Any]] = {
    **BASIC_ACTIONS,
    "craft_planks": dict(craft="planks"),
    "craft_stick": dict(craft="stick"),
    "craft_crafting_table": dict(craft="crafting_table"),
    "place_crafting_table": dict(place="crafting_table"),
    "craft_wooden_pickaxe": dict(nearbyCraft="wooden_pickaxe"),
    "craft_stone_pickaxe": dict(nearbyCraft="stone_pickaxe"),
    "craft_iron_pickaxe": dict(nearbyCraft="iron_pickaxe"),
    "equip_stone_pickaxe": dict(equip="stone_pickaxe"),
    "equip_wooden_pickaxe": dict(equip="wooden_pickaxe"),
    "equip_iron_pickaxe": dict(equip="iron_pickaxe"),
    "craft_furnace": dict(nearbyCraft="furnace"),
    "place_furnace": dict(place="furnace"),
    "smelt_iron_ingot": dict(nearbySmelt="iron_ingot"),
}

def _flatten_nested(node, prefix: str = "") -> Dict[str, Any]:
    """Flatten gym Dict spaces / obs dicts into "a/b/c" keys (like embodied.FromGym)."""
    out: Dict[str, Any] = {}
    for key, value in node.items():
        full = f"{prefix}/{key}" if prefix else key
        if hasattr(value, "spaces"):
            out.update(_flatten_nested(value.spaces, full))
        elif isinstance(value, dict):
            out.update(_flatten_nested(value, full))
        else:
            out[full] = value
    return out


NOOP = dict(
    camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
    jump=0, sneak=0, craft="none", nearbyCraft="none", nearbySmelt="none",
    place="none", equip="none",
)


class MinecraftDiamondEnv:
    """Gym-ish environment: step(int action) -> (obs dict, reward, done, info)."""

    LOCK = threading.Lock()

    def __init__(self, cfg: Dict[str, Any], seed: int = 0):
        self.cfg = cfg
        env_cfg = cfg["env"]
        self.seed = seed
        self.size = (int(env_cfg["obs_size"]), int(env_cfg["obs_size"]))
        self.repeat = int(env_cfg.get("repeat", 1))
        self.episode_steps = int(env_cfg["episode_steps"])
        self.sticky_attack = int(env_cfg.get("sticky_attack", 0))
        self.sticky_jump = int(env_cfg.get("sticky_jump", 0))
        self.pitch_limit = float(env_cfg.get("pitch_limit", 60))
        self.actions = DIAMOND_ACTIONS
        self.action_names = list(self.actions.keys())
        self.action_values = list(self.actions.values())
        self.reward = RewardEngine(cfg["reward"], self.episode_steps)
        self._rng = np.random.RandomState(seed)

        with self.LOCK:
            spec = MinecraftSpec(self.size, break_speed=float(env_cfg["break_speed"]))
            self.gym = spec.make()

        flat_space = _flatten_nested(self.gym.observation_space.spaces)
        self._inv_keys = [
            k for k in flat_space
            if k.startswith("inventory/") and k != "inventory/log2"
        ]
        self._inv_log_keys = [
            f"inventory/{k}" for k in
            ("log", "cobblestone", "iron_ingot", "diamond")
        ]
        eq = self.gym.observation_space["equipped_items"]["mainhand"]["type"]
        self._equip_enum = list(eq.values)
        self.obs_dim = len(self._inv_keys)
        self.equipped_dim = len(self._equip_enum)
        self.n_actions = len(self.action_names)
        self._step = 0
        self._pitch = 0.0
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._done = True
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        log.info("Minecraft env ready: actions=%d obs=%d equipped=%d",
                 self.n_actions, self.obs_dim, self.equipped_dim)

    # -- lifecycle -----------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        with self.LOCK:
            obs = self.gym.reset()
        self._step = 0
        self._pitch = 0.0
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._done = False
        self.reward.reset()
        out = self._obs(obs)
        r, comp = self.reward.step(
            out["_inventory"], out["health_raw"], out["hunger_raw"],
            out["breath_raw"], out["player_pos"], is_first=True)
        out["reward"] = np.float32(0.0)
        out["_reward_components"] = comp
        out["is_first"] = True
        self._last_obs = out
        return out

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        act = self._compose(int(action))
        obs = None
        done = False
        info: Dict[str, Any] = {}
        for r in range(self.repeat):
            a = act if r == 0 else self._following(act)
            with self.LOCK:
                obs, _, done, info = self.gym.step(a)
            if done or (info and "error" in info):
                done = True
                break
        self._step += 1
        truncated = self._step >= self.episode_steps
        done = bool(done or truncated)
        if obs is None:
            raise RuntimeError("env step produced no observation")
        out = self._obs(obs)
        reward, comp = self.reward.step(
            out["_inventory"], out["health_raw"], out["hunger_raw"],
            out["breath_raw"], out["player_pos"], done=done)
        out["reward"] = np.float32(reward)
        out["_reward_components"] = comp
        info = dict(info or {})
        info.update(
            score=self.reward.score(),
            milestones=sorted(self.reward.milestones),
            truncated=bool(truncated and not info.get("is_terminal", False)),
            step=self._step,
        )
        self._done = done
        self._last_obs = out
        return out, float(reward), done, info

    def close(self) -> None:
        try:
            self.gym.close()
        except Exception:
            pass

    # -- action composition -------------------------------------------
    def _compose(self, index: int) -> Dict[str, Any]:
        act = dict(NOOP)
        act.update(self.action_values[index])
        if self.sticky_attack:
            if act["attack"]:
                self._sticky_attack_counter = self.sticky_attack
            if self._sticky_attack_counter > 0:
                act["attack"] = 1
                act["jump"] = 0
                self._sticky_attack_counter -= 1
        if self.sticky_jump:
            if act["jump"]:
                self._sticky_jump_counter = self.sticky_jump
            if self._sticky_jump_counter > 0:
                act["jump"] = 1
                act["forward"] = 1
                self._sticky_jump_counter -= 1
        if self.pitch_limit and act["camera"][0]:
            lo, hi = -self.pitch_limit, self.pitch_limit
            if not (lo <= self._pitch + act["camera"][0] <= hi):
                act["camera"] = (0, act["camera"][1])
            self._pitch += act["camera"][0]
        return act

    @staticmethod
    def _following(act: Dict[str, Any]) -> Dict[str, Any]:
        """Movement keys are held across the repeat window (DreamerV3 parity)."""
        follow = dict(NOOP)
        for key in ("attack", "forward", "back", "left", "right"):
            follow[key] = act[key]
        return follow

    # -- observation ---------------------------------------------------
    def _obs(self, raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
        raw = _flatten_nested(raw)
        if "inventory/log2" in raw:
            raw["inventory/log"] = raw["inventory/log"] + raw.pop("inventory/log2")
        inv = {k.split("/", 1)[1]: int(v) for k, v in raw.items()
               if k.startswith("inventory/")}
        inv_vec = np.asarray([raw[k] for k in self._inv_keys], dtype=np.float32)
        equip_index = self._equip_enum.index(raw["equipped_items/mainhand/type"])
        equipped = np.zeros(self.equipped_dim, dtype=np.float32)
        equipped[equip_index] = 1.0
        pov = raw["pov"]
        if pov.shape[:2] != self.size:
            pov = np.transpose(pov, (1, 0, 2))
        pos = (
            float(raw.get("location_stats/xpos", 0.0)),
            float(raw.get("location_stats/ypos", 64.0)),
            float(raw.get("location_stats/zpos", 0.0)),
        )
        return {
            "image": np.ascontiguousarray(pov, dtype=np.uint8),
            "inventory": inv_vec,
            "equipped": equipped,
            "health_raw": np.float32(raw["life_stats/life"]),
            "hunger_raw": np.float32(raw["life_stats/food"]),
            "breath_raw": np.float32(raw["life_stats/air"]),
            "player_pos": np.asarray(pos, dtype=np.float32),
            "is_first": bool(self._step == 0),
            "is_last": False,
            "is_terminal": False,
            "_inventory": inv,  # dict; consumed by the worker, not the learner
        }


class MinecraftSpec(EnvSpec):
    """The DreamerV3 Minecraft task spec (flat inventory + life stats)."""

    def __init__(self, resolution=(64, 64), break_speed=50):
        self.resolution = resolution
        self.break_speed = break_speed
        super().__init__(name="MCPPODiamond-v0")

    def create_agent_start(self):
        return [BreakSpeedMultiplier(self.break_speed)]

    def create_agent_handlers(self):
        return []

    def create_server_world_generators(self):
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self):
        return [handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_initial_conditions(self):
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=0),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def create_observables(self):
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(mc.ALL_ITEMS),
            handlers.EquippedItemObservation(mc.ALL_ITEMS, _default="air", _other="other"),
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats(),
        ]

    def create_actionables(self):
        kw = dict(_other="none", _default="none")
        return [
            handlers.KeybasedCommandAction("forward", INVERSE_KEYMAP["forward"]),
            handlers.KeybasedCommandAction("back", INVERSE_KEYMAP["back"]),
            handlers.KeybasedCommandAction("left", INVERSE_KEYMAP["left"]),
            handlers.KeybasedCommandAction("right", INVERSE_KEYMAP["right"]),
            handlers.KeybasedCommandAction("jump", INVERSE_KEYMAP["jump"]),
            handlers.KeybasedCommandAction("sneak", INVERSE_KEYMAP["sneak"]),
            handlers.KeybasedCommandAction("attack", INVERSE_KEYMAP["attack"]),
            handlers.CameraAction(),
            handlers.PlaceBlock(["none"] + mc.ALL_ITEMS, **kw),
            handlers.EquipAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.CraftAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.CraftNearbyAction(["none"] + mc.ALL_ITEMS, **kw),
            handlers.SmeltItemNearby(["none"] + mc.ALL_ITEMS, **kw),
        ]

    def is_from_folder(self, folder):
        return folder == "none"

    def get_docstring(self):
        return ""

    def determine_success_from_rewards(self, rewards):
        return True

    def create_rewardables(self):
        return []

    def create_server_decorators(self):
        return []

    def create_mission_handlers(self):
        return []

    def create_monitors(self):
        return []


class BreakSpeedMultiplier(handler.Handler):
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def to_string(self):
        return f"break_speed({self.multiplier})"

    def xml_template(self):
        return "<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>"
