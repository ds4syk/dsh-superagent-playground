import random

import pytest

from jevwm.questions import transition_questions
from jevwm.world import (
    ACTIONS,
    State,
    admissible_actions,
    goal_reached,
    initial_state,
    make_specs,
    random_action,
    render_state,
    run_policy,
    solution_path,
    state_from_values,
    step,
    variable_options,
    variable_values,
)


def test_oracle_solves_every_door_mode():
    specs = make_specs("train", 12)
    for spec in specs:
        states, actions, ok = run_policy(spec, "oracle", max_steps=20)
        assert ok, (spec.world_id, spec.door_mode, actions, states[-1])


def test_oracle_path_matches_goal():
    for spec in make_specs("train", 6):
        s = initial_state(spec)
        for a in solution_path(spec):
            s, executed = step(s, a, spec)
            assert executed, (spec.world_id, spec.door_mode, a, s)
        assert goal_reached(s)


def test_state_values_roundtrip():
    rng = random.Random(0)
    for spec in make_specs("test", 8):
        s = initial_state(spec)
        for _ in range(40):
            values = variable_values(s, spec)
            assert state_from_values(values, spec) == s
            s, _ = step(s, random_action(s, rng), spec)


def test_invalid_action_is_noop():
    spec = make_specs("train", 1)[0]
    s = initial_state(spec)
    for a in ("go_east", "take_key", "take_apple"):
        if a in admissible_actions(s):
            continue
        nxt, executed = step(s, a, spec)
        assert not executed and nxt == s


def test_question_ground_truth_is_in_options():
    spec = make_specs("test", 1)[0]
    s = initial_state(spec)
    s, _ = step(s, "open_box", spec)
    s, _ = step(s, "take_key", spec)
    opts = variable_options(spec)
    gt = variable_values(s, spec)
    qs = transition_questions(s, "unlock_door", spec)
    by_id = {q["id"]: q for q in qs}
    for var in ("player_room", "key_location", "apple_location", "mushroom_location", "door_mode", "box_mode", "holding"):
        assert gt[var] in by_id[var]["options"], var
        assert gt[var] in opts[var]


def test_render_mentions_goal_objects():
    spec = make_specs("train", 1)[0]
    text = render_state(initial_state(spec), spec)
    for name in (spec.box, spec.key, spec.table, spec.apple, spec.door, spec.room_a):
        assert name in text
