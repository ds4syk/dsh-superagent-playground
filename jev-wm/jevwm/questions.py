"""Prompt construction for the JEV-style decision model and baselines."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .world import (
    ACTIONS,
    HOLDING_COMBOS,
    ITEMS,
    LOC_CODES,
    State,
    WorldSpec,
    action_text,
    admissible_actions,
    holding,
    render_goal,
    render_state,
    variable_options,
    variable_values,
)


def state_prompt(state: State, spec: WorldSpec, include_goal: bool = False) -> str:
    text = render_state(state, spec)
    if include_goal:
        text += " Goal: " + render_goal(spec)
    return text


def transition_questions(state: State, action: str, spec: WorldSpec) -> List[dict]:
    """One item per typed question.  `kind` follows AnyJev's API."""
    opts = variable_options(spec)
    act = action_text(action, spec)
    prefix = f"After the command `{act}` is attempted, "
    items: List[dict] = [
        {
            "id": "executes",
            "kind": "noul",
            "question": f"Does the command `{act}` execute successfully in this state?",
            "options": ["Yes", "No"],
        },
        {
            "id": "player_room",
            "kind": "choice",
            "question": prefix + "which room is the player in?",
            "options": opts["player_room"],
        },
        {
            "id": "key_location",
            "kind": "choice",
            "question": prefix + f"where is the {spec.key}?",
            "options": opts["key_location"],
        },
        {
            "id": "apple_location",
            "kind": "choice",
            "question": prefix + f"where is the {spec.apple}?",
            "options": opts["apple_location"],
        },
        {
            "id": "mushroom_location",
            "kind": "choice",
            "question": prefix + f"where is the {spec.mushroom}?",
            "options": opts["mushroom_location"],
        },
        {
            "id": "door_mode",
            "kind": "choice",
            "question": prefix + f"what is the state of the {spec.door}?",
            "options": opts["door_mode"],
        },
        {
            "id": "box_mode",
            "kind": "choice",
            "question": prefix + f"what is the state of the {spec.box}?",
            "options": opts["box_mode"],
        },
        {
            "id": "holding",
            "kind": "choice",
            "question": prefix + "what is the player holding?",
            "options": opts["holding"],
        },
    ]
    return items


def policy_question(state: State, spec: WorldSpec, candidate_actions: Sequence[str]) -> dict:
    """Ask the decision model to pick the next command (a policy query)."""
    return {
        "id": "next_action",
        "kind": "choice",
        "question": "Which command should the agent execute next to achieve the goal?",
        "options": [action_text(a, spec) for a in candidate_actions],
    }


def progress_question(state: State, spec: WorldSpec, action: str) -> dict:
    act = action_text(action, spec)
    return {
        "id": "progress",
        "kind": "noul",
        "question": f"Would executing `{act}` next make progress toward the goal?",
        "options": ["Yes", "No"],
    }


def valid_action_options(state: State, spec: WorldSpec, include_invalid: bool = False) -> List[str]:
    if include_invalid:
        return list(ACTIONS)
    return admissible_actions(state)


def action_from_text(text: str, spec: WorldSpec) -> str:
    for a in ACTIONS:
        if action_text(a, spec) == text:
            return a
    raise KeyError(text)
