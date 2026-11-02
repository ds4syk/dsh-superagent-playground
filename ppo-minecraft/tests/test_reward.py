import copy

import numpy as np
import pytest

from mcppo.config import DEFAULTS, deep_update
from mcppo.rewards.reward import RewardEngine


def make_engine(**overrides):
    cfg = copy.deepcopy(DEFAULTS["reward"])
    cfg.update(overrides)
    return RewardEngine(cfg, episode_steps=1000)


def step(eng, inv=None, health=20.0, hunger=20.0, breath=300.0, pos=(0, 64, 0),
         done=False):
    return eng.step(inv or {}, health, hunger, breath, pos, done=done)


def test_first_step_is_zero():
    eng = make_engine()
    r, comp = eng.step({}, 20, 20, 300, (0, 64, 0), is_first=True)
    assert r == 0.0
    assert comp["total"] == 0.0


def test_log_milestone_and_score():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    r, comp = step(eng, {"log": 1}, pos=(0, 64, 0))
    assert comp["milestone"] > 0
    assert eng.score() == 1
    assert "log" in eng.milestones


def test_diminishing_returns_for_repeated_item():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    _, c1 = step(eng, {"log": 1}, pos=(0, 64, 0))
    _, c2 = step(eng, {"log": 2}, pos=(0, 64, 0))
    _, c3 = step(eng, {"log": 3}, pos=(0, 64, 0))
    assert c2["milestone"] == 0 and c3["milestone"] == 0
    assert 0 < c2["shape"] < c1["shape"]
    assert 0 < c3["shape"] < c2["shape"]


def test_milestone_rewarded_once_even_if_item_disappears():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    step(eng, {"planks": 4}, pos=(0, 64, 0))
    _, comp = step(eng, {"planks": 0}, pos=(0, 64, 0))
    assert comp["milestone"] == 0
    assert "planks" in eng.milestones


def test_crafting_does_not_penalise_consumed_items():
    """Log -> planks: max-inventory potential keeps the log value."""
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    step(eng, {"log": 1}, pos=(0, 64, 0))
    _, comp = step(eng, {"log": 0, "planks": 4}, pos=(0, 64, 0))
    assert comp["shape"] > 0


def test_death_penalty():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    _, comp = step(eng, {}, health=0.0, pos=(0, 64, 0), done=True)
    assert comp["death"] == -5.0


def test_health_damage_is_negative_heal_is_positive():
    eng = make_engine()
    step(eng, {}, health=20.0)
    _, comp = step(eng, {}, health=18.0)
    assert comp["health"] < 0
    _, comp = step(eng, {}, health=20.0)
    assert comp["health"] > 0


def test_speed_multiplier_favours_early_milestones():
    early = make_engine()
    step(early, {}, pos=(0, 64, 0))
    _, c_early = step(early, {"log": 1}, pos=(0, 64, 0))
    late = make_engine()
    step(late, {}, pos=(0, 64, 0))
    for _ in range(900):
        step(late, {}, pos=(0, 64, 0))
    _, c_late = step(late, {"log": 1}, pos=(0, 64, 0))
    assert c_early["milestone"] > c_late["milestone"] > 0


def test_exploration_rewards_new_cells_only():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    _, c1 = step(eng, {}, pos=(8, 64, 0))
    _, c2 = step(eng, {}, pos=(8, 64, 0))
    assert c1["shape"] > c2["shape"]


def test_reward_clip():
    eng = make_engine()
    step(eng, {}, pos=(0, 64, 0))
    _, comp = step(eng, {"diamond": 1}, pos=(0, 64, 0))
    assert comp["total"] <= eng.clip[1]


def test_config_merge():
    cfg = deep_update(DEFAULTS, {"ppo": {"lr": 1e-4}})
    assert cfg["ppo"]["lr"] == 1e-4
    assert cfg["ppo"]["gamma"] == DEFAULTS["ppo"]["gamma"]
