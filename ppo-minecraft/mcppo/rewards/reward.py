"""Hand-written dense reward for the Minecraft diamond task.

Design (see docs/03_reward.md for the long version):

1. **Potential-based item / exploration / depth shaping**

       Phi(s) = sum_i  v_i * log(1 + M_i(s))          # item quantity, diminishing
              + w_exp * min(cap, #new 8x8 cells)       # exploration
              + w_depth * max(0, 64 - min_y)           # depth progress
       r_shape = gamma * Phi(s') - Phi(s)

   M_i is the *maximum* inventory count seen so far, so crafting (which
   consumes items) never produces a negative item reward and re-collecting the
   same log cannot be farmed.  The logarithmic term makes the marginal value of
   the k-th copy of a resource decreasing, i.e. it rewards the *variety* of
   progress rather than grinding.

2. **One-time tech-tree milestones** with a speed multiplier

       r_ms = B_m * clip(1 + kappa * (1 - t/T), floor, 1+kappa)

   This is the "获取效率" term: the same achievement is worth more if it is
   reached early in the episode.

3. **Survival**: health delta (healing rewarded, damage penalised), death
   penalty, one-time hunger/breath warnings.

4. **Small step penalty** to encourage fast progress, and a hard per-step clip.

Everything is logged per component so reward hacking is visible in the curves.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .items import ITEM_TABLE, MILESTONE_ORDER, SCORE_OF


class RewardEngine:
    """Stateful per-environment reward. Call reset() on episode start."""

    def __init__(self, cfg: Dict[str, Any], episode_steps: int):
        self.cfg = cfg
        self.item_weights = dict(cfg.get("item_weights") or {})
        self.milestone_bonus = dict(cfg.get("milestone_bonus") or {})
        # Only reward keys that exist in the table; extra keys are ignored.
        self.items = [name for name in MILESTONE_ORDER if name in self.item_weights]
        self.gamma = float(cfg.get("gamma", 0.999))
        self.beta = float(cfg.get("beta_shape", 1.0))
        self.episode_steps = max(1, int(episode_steps))
        self.explore_value = float(cfg.get("explore_value", 0.0))
        self.explore_cap = float(cfg.get("explore_cap", 0.0))
        self.explore_cell = max(1, int(cfg.get("explore_cell", 8)))
        self.depth_value = float(cfg.get("depth_value", 0.0))
        self.depth_ref_y = float(cfg.get("depth_ref_y", 64.0))
        self.depth_bonuses = list(cfg.get("depth_bonus") or [])
        self.clip = cfg.get("clip") or [-10.0, 50.0]
        self.reset()

    # -- episode state -------------------------------------------------
    def reset(self) -> None:
        self.t = 0
        self.max_inv: Dict[str, int] = {}
        self.milestones: set[str] = set()
        self.cells: set[tuple[int, int]] = set()
        self.min_y: Optional[float] = None
        self.depth_levels: set[float] = set()
        self.hunger_levels: set[str] = set()
        self.breath_warned = False
        self.prev_health: Optional[float] = None
        self.last_new_milestones: list[str] = []
        self.phi = self._phi()
        self.episode_return = 0.0
        self.components: Dict[str, float] = {}

    # -- potential -----------------------------------------------------
    def _phi(self) -> float:
        phi = 0.0
        for name in self.items:
            phi += self.item_weights.get(name, 0.0) * math.log1p(self.max_inv.get(name, 0))
        if self.explore_value:
            cells = len(self.cells) * self.explore_value
            phi += min(self.explore_cap, cells)
        if self.depth_value and self.min_y is not None:
            phi += self.depth_value * max(0.0, self.depth_ref_y - self.min_y)
        return phi

    # -- helpers -------------------------------------------------------
    def _speed(self) -> float:
        gain = float(self.cfg.get("speed_gain", 0.0))
        floor = float(self.cfg.get("speed_floor", 0.75))
        if gain <= 0:
            return 1.0
        frac = 1.0 - min(1.0, self.t / self.episode_steps)
        return min(1.0 + gain, max(floor, 1.0 + gain * frac))

    def score(self) -> int:
        """Deepest milestone reached (MineRL ObtainDiamond-compatible score)."""
        return max((SCORE_OF[m] for m in self.milestones), default=0)

    @property
    def phi_value(self) -> float:
        return self.phi

    # -- main step -----------------------------------------------------
    def step(
        self,
        inventory: Dict[str, int],
        health: float,
        hunger: float,
        breath: float,
        pos: Optional[tuple],
        is_first: bool = False,
        done: bool = False,
    ) -> tuple[float, Dict[str, float]]:
        if is_first:
            self.reset()
            if pos is not None:
                self.min_y = float(pos[1])
            self.phi = self._phi()
            self.prev_health = float(health)
            return 0.0, {"shape": 0.0, "milestone": 0.0, "health": 0.0,
                         "death": 0.0, "hunger": 0.0, "breath": 0.0,
                         "depth_bonus": 0.0, "time": 0.0, "total": 0.0}

        self.t += 1
        inv = {k: int(v) for k, v in inventory.items()}

        # 1) update monotone maxima and the novelty/depth state
        for name in self.items:
            self.max_inv[name] = max(self.max_inv.get(name, 0), inv.get(name, 0))
        if pos is not None:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            self.cells.add((int(math.floor(x / self.explore_cell)),
                            int(math.floor(z / self.explore_cell))))
            self.min_y = y if self.min_y is None else min(self.min_y, y)

        # 2) potential-based shaping (item quantity + exploration + depth)
        phi_new = self._phi()
        shape = self.beta * (self.gamma * phi_new - self.phi)
        self.phi = phi_new

        # 3) one-time tech-tree milestones with speed multiplier
        milestone_r = 0.0
        new_ms = []
        speed = self._speed()
        for name in self.items:
            if name not in self.milestones and self.max_inv.get(name, 0) > 0:
                self.milestones.add(name)
                milestone_r += self.milestone_bonus.get(name, 0.0) * speed
                new_ms.append(name)

        # 4) survival
        health_r = 0.0
        h = float(health)
        if self.prev_health is not None:
            dh = h - self.prev_health
            if dh > 0:
                health_r += float(self.cfg.get("health_heal", 0.0)) * dh
            elif dh < 0:
                health_r += float(self.cfg.get("health_damage", 0.0)) * dh  # negative
        self.prev_health = h

        death_r = 0.0
        if done and h <= 0.0:
            death_r -= float(self.cfg.get("death_penalty", 0.0))
        hunger_r = 0.0
        breath_r = 0.0
        # Those are called once when a threshold is first crossed.
        for level in (6.0, 2.0):
            key = f"hunger_{level:g}"
            if hunger <= level and key not in self.hunger_levels:
                self.hunger_levels.add(key)
                hunger_r -= float(self.cfg.get("hunger_penalty", 0.0)) * (7.0 - level)
        if breath <= float(self.cfg.get("breath_threshold", 0.3)) and not self.breath_warned:
            self.breath_warned = True
            breath_r -= float(self.cfg.get("breath_penalty", 0.0)) * 20.0

        depth_r = 0.0
        if self.min_y is not None:
            for spec in self.depth_bonuses:
                key = float(spec["y"])
                if key not in self.depth_levels and self.min_y <= key:
                    self.depth_levels.add(key)
                    depth_r += float(spec["bonus"])

        time_r = -float(self.cfg.get("step_penalty", 0.0))

        total = shape + milestone_r + health_r + death_r + hunger_r + breath_r + depth_r + time_r
        total = max(float(self.clip[0]), min(float(self.clip[1]), total))
        self.episode_return += total
        comp = {
            "shape": shape,
            "milestone": milestone_r,
            "health": health_r,
            "death": death_r,
            "hunger": hunger_r,
            "breath": breath_r,
            "depth_bonus": depth_r,
            "time": time_r,
            "total": total,
        }
        for key, value in comp.items():
            self.components[key] = self.components.get(key, 0.0) + value
        self.last_new_milestones = new_ms
        return total, comp
