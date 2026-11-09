"""A small deterministic text adventure used for JEV pre-experiments.

The world is deliberately close to the TextWorld "trap world" used by the
KIIS 2026 paper (two rooms, a locked door, a key in a box, a goal food and a
decoy).  It is *not* TextWorld itself: we re-implement the state machine so the
ground-truth transition function and the query schema are explicit and cheap to
run at the scale needed for action-label distillation.

State is a small factored object.  Every quantity a world model is asked about
corresponds to one field, which lets us query a decision model with typed
multiple-choice questions instead of free-form generation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Actions

ACTIONS: Tuple[str, ...] = (
    "go_east",
    "go_west",
    "open_box",
    "open_door",
    "unlock_door",
    "take_key",
    "take_apple",
    "take_mushroom",
    "look",
)

ACTION_LABELS: Dict[str, str] = {
    "go_east": "go east",
    "go_west": "go west",
    "open_box": "open the box",
    "open_door": "open the door",
    "unlock_door": "unlock the door with the key",
    "take_key": "take the key from the box",
    "take_apple": "take the apple from the table",
    "take_mushroom": "take the mushroom from the table",
    "look": "look",
}

ITEMS: Tuple[str, ...] = ("key", "apple", "mushroom")

# How much of the next state has to be *derived* rather than read off the text.
ACTION_KIND: Dict[str, str] = {
    "look": "observe",
    "open_box": "simple",
    "open_door": "simple",
    "unlock_door": "derive",
    "take_key": "derive",
    "take_apple": "derive",
    "take_mushroom": "derive",
    "go_east": "derive",
    "go_west": "derive",
}

# location codes: b=box, t=table, a=room A floor, e=room B (east) floor, i=inventory
LOC_CODES: Tuple[str, ...] = ("b", "t", "a", "e", "i")
LOC_LABELS: Dict[str, str] = {
    "b": "inside the container",
    "t": "on the table",
    "a": "on the floor of the first room",
    "e": "on the floor of the second room",
    "i": "in the player's inventory",
}

# 3 items, capacity unlimited => 8 subsets including empty
HOLDING_COMBOS: Tuple[Tuple[str, ...], ...] = (
    (),
    ("key",),
    ("apple",),
    ("mushroom",),
    ("key", "apple"),
    ("key", "mushroom"),
    ("apple", "mushroom"),
    ("key", "apple", "mushroom"),
)


@dataclass(frozen=True)
class WorldSpec:
    """Names and initial configuration of one world."""

    world_id: str
    box: str
    key: str
    table: str
    apple: str
    mushroom: str
    door: str
    room_a: str
    room_b: str
    door_mode: str = "locked"  # locked | closed | open
    mushroom_on_table: bool = True
    split: str = "train"

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "box": self.box,
            "key": self.key,
            "table": self.table,
            "apple": self.apple,
            "mushroom": self.mushroom,
            "door": self.door,
            "room_a": self.room_a,
            "room_b": self.room_b,
            "door_mode": self.door_mode,
            "mushroom_on_table": self.mushroom_on_table,
            "split": self.split,
        }

    @staticmethod
    def from_dict(d: dict) -> "WorldSpec":
        return WorldSpec(**d)


@dataclass(frozen=True)
class State:
    """Factored state.  Locations use the LOC_CODES vocabulary."""

    player: str  # "a" | "b"
    key: str
    apple: str
    mushroom: str
    door: str  # locked | closed | open
    box: str  # closed | open

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "key": self.key,
            "apple": self.apple,
            "mushroom": self.mushroom,
            "door": self.door,
            "box": self.box,
        }

    @staticmethod
    def from_dict(d: dict) -> "State":
        return State(**{k: d[k] for k in ("player", "key", "apple", "mushroom", "door", "box")})


def initial_state(spec: WorldSpec) -> State:
    return State(
        player="a",
        key="b",
        apple="t",
        mushroom="t" if spec.mushroom_on_table else "a",
        door=spec.door_mode,
        box="closed",
    )


def holding(state: State) -> Tuple[str, ...]:
    return tuple(sorted(i for i in ITEMS if getattr(state, i) == "i"))


def holding_index(held: Tuple[str, ...]) -> int:
    key = set(held)
    for i, combo in enumerate(HOLDING_COMBOS):
        if set(combo) == key:
            return i
    raise ValueError(f"unknown holding combination: {held}")


def goal_reached(state: State) -> bool:
    return state.player == "b" and state.apple == "i" and state.door == "open"


# ---------------------------------------------------------------------------
# Rendering


def action_text(action: str, spec: WorldSpec) -> str:
    mapping = {
        "go_east": "go east",
        "go_west": "go west",
        "open_box": f"open the {spec.box}",
        "open_door": f"open the {spec.door}",
        "unlock_door": f"unlock the {spec.door} with the {spec.key}",
        "take_key": f"take the {spec.key} from the {spec.box}",
        "take_apple": f"take the {spec.apple} from the {spec.table}",
        "take_mushroom": f"take the {spec.mushroom} from the {spec.table}",
        "look": "look",
    }
    return mapping[action]


def render_state(state: State, spec: WorldSpec) -> str:
    """Natural-language state.  It is fully observed (the agent knows where
    things are) so that prediction is about dynamics, not hidden-state
    inference.  This is stated as a limitation in the report."""
    room = spec.room_a if state.player == "a" else spec.room_b
    lines = [f"You are in the {room}."]
    if state.player == "a":
        lines.append(f"The {spec.room_a} contains a {spec.box} and a {spec.table}.")
    lines.append(f"The {spec.box} is {state.box}.")
    lines.append(f"The {spec.door} is {state.door}.")
    for item, label in (("key", spec.key), ("apple", spec.apple), ("mushroom", spec.mushroom)):
        loc = getattr(state, item)
        if loc == "i":
            continue
        if loc == "b":
            lines.append(f"The {label} is inside the {spec.box}.")
        elif loc == "t":
            lines.append(f"The {label} is on the {spec.table}.")
        elif loc == "a":
            lines.append(f"The {label} is on the floor of the {spec.room_a}.")
        elif loc == "e":
            lines.append(f"The {label} is on the floor of the {spec.room_b}.")
    held = holding(state)
    if not held:
        lines.append("You are holding nothing.")
    else:
        names = [{"key": spec.key, "apple": spec.apple, "mushroom": spec.mushroom}[i] for i in held]
        lines.append("You are holding the " + " and the ".join(names) + ".")
    return " ".join(lines)


def render_goal(spec: WorldSpec) -> str:
    return (
        f"Get to the {spec.room_b} while holding the {spec.apple}, "
        f"with the {spec.door} open."
    )


# ---------------------------------------------------------------------------
# Transition function


def admissible_actions(state: State) -> List[str]:
    acts: List[str] = ["look"]
    if state.player == "a" and state.door == "open":
        acts.append("go_east")
    if state.player == "b":
        acts.append("go_west")
    if state.player == "a":
        if state.box == "closed":
            acts.append("open_box")
        if state.door == "closed":
            acts.append("open_door")
        if state.door == "locked" and state.key == "i":
            acts.append("unlock_door")
        if state.key == "b" and state.box == "open":
            acts.append("take_key")
        if state.apple == "t":
            acts.append("take_apple")
        if state.mushroom == "t":
            acts.append("take_mushroom")
    elif state.player == "b" and state.door == "closed":
        acts.append("open_door")
    # de-duplicate, preserve order
    seen = set()
    return [a for a in acts if not (a in seen or seen.add(a))]


def step(state: State, action: str, spec: WorldSpec) -> Tuple[State, bool]:
    """Return (next_state, executed).  Invalid actions leave the state intact."""
    if action == "look":
        return state, True

    if action == "go_east":
        if state.player == "a" and state.door == "open":
            return replace(state, player="b"), True
        return state, False

    if action == "go_west":
        if state.player == "b":
            return replace(state, player="a"), True
        return state, False

    if action == "open_box":
        if state.player == "a" and state.box == "closed":
            return replace(state, box="open"), True
        return state, False

    if action == "open_door":
        if state.player == "a" and state.door == "closed":
            return replace(state, door="open"), True
        if state.player == "b" and state.door == "closed":
            return replace(state, door="open"), True
        return state, False

    if action == "unlock_door":
        if state.player == "a" and state.door == "locked" and state.key == "i":
            return replace(state, door="closed"), True
        return state, False

    if action == "take_key":
        if state.player == "a" and state.key == "b" and state.box == "open":
            return replace(state, key="i"), True
        return state, False

    if action == "take_apple":
        if state.player == "a" and state.apple == "t":
            return replace(state, apple="i"), True
        return state, False

    if action == "take_mushroom":
        if state.player == "a" and state.mushroom == "t":
            return replace(state, mushroom="i"), True
        return state, False

    raise ValueError(f"unknown action: {action}")


# ---------------------------------------------------------------------------
# Typed state variables used by the decision-model queries


VAR_NAMES: Tuple[str, ...] = (
    "player_room",
    "key_location",
    "apple_location",
    "mushroom_location",
    "door_mode",
    "box_mode",
    "holding",
)


def _loc_option(spec: WorldSpec, loc: str) -> str:
    return {
        "b": f"inside the {spec.box}",
        "t": f"on the {spec.table}",
        "a": f"on the floor of the {spec.room_a}",
        "e": f"on the floor of the {spec.room_b}",
        "i": "in the player's inventory",
    }[loc]


def variable_options(spec: WorldSpec) -> Dict[str, List[str]]:
    """All options for every variable.  The support is fixed per world and
    includes physically impossible values, as in the KIIS query compiler."""
    locs = [_loc_option(spec, c) for c in LOC_CODES]
    holding_opts = []
    for combo in HOLDING_COMBOS:
        if not combo:
            holding_opts.append("nothing")
        elif len(combo) == 1:
            holding_opts.append("the " + {"key": spec.key, "apple": spec.apple, "mushroom": spec.mushroom}[combo[0]])
        else:
            names = [{"key": spec.key, "apple": spec.apple, "mushroom": spec.mushroom}[i] for i in combo]
            holding_opts.append("the " + " and the ".join(names))
    return {
        "player_room": [spec.room_a, spec.room_b],
        "key_location": locs,
        "apple_location": locs,
        "mushroom_location": locs,
        "door_mode": ["locked", "closed", "open"],
        "box_mode": ["closed", "open"],
        "holding": holding_opts,
    }


def variable_values(state: State, spec: WorldSpec) -> Dict[str, str]:
    return {
        "player_room": spec.room_a if state.player == "a" else spec.room_b,
        "key_location": _loc_option(spec, state.key),
        "apple_location": _loc_option(spec, state.apple),
        "mushroom_location": _loc_option(spec, state.mushroom),
        "door_mode": state.door,
        "box_mode": state.box,
        "holding": variable_options(spec)["holding"][holding_index(holding(state))],
    }


def state_from_values(values: Dict[str, str], spec: WorldSpec) -> State:
    """Reconstruct a State from one option per variable (used for rollouts)."""
    room_opts = {spec.room_a: "a", spec.room_b: "b"}
    loc_opts = {_loc_option(spec, c): c for c in LOC_CODES}

    def loc(var: str) -> str:
        return loc_opts.get(values.get(var, ""), "a")

    key = loc("key_location")
    apple = loc("apple_location")
    mushroom = loc("mushroom_location")
    holding_opt = values.get("holding", "nothing")
    combo = HOLDING_COMBOS[
        variable_options(spec)["holding"].index(holding_opt)
        if holding_opt in variable_options(spec)["holding"]
        else 0
    ]
    for item in combo:
        if item == "key":
            key = "i"
        elif item == "apple":
            apple = "i"
        elif item == "mushroom":
            mushroom = "i"
    return State(
        player=room_opts.get(values.get("player_room", ""), "a"),
        key=key,
        apple=apple,
        mushroom=mushroom,
        door=values.get("door_mode", "locked") if values.get("door_mode") in ("locked", "closed", "open") else "locked",
        box="open" if values.get("box_mode") == "open" else "closed",
    )


# ---------------------------------------------------------------------------
# Policies used for data collection / upper bounds


def oracle_action(state: State, spec: WorldSpec) -> str:
    """A deterministic shortest-path-ish script.  It is a reference, not a
    claim of optimality."""
    if goal_reached(state):
        return "look"
    if state.apple != "i":
        if state.player == "b":
            return "go_west"
        return "take_apple"
    # apple held
    if state.player == "b":
        return "look"
    if state.door == "locked":
        if state.key != "i":
            if state.box == "open" and state.key == "b":
                return "take_key"
            return "open_box"
        return "unlock_door"
    if state.door == "closed":
        return "open_door"
    return "go_east"


def random_action(state: State, rng) -> str:
    acts = [a for a in admissible_actions(state) if a != "look"]
    return rng.choice(acts) if acts else "look"


# ---------------------------------------------------------------------------
# World construction


NAME_SETS_TRAIN = [
    ("wooden box", "brass key", "oak table", "red apple", "blue mushroom", "steel door", "workroom", "cellar"),
    ("iron chest", "silver key", "pine shelf", "green pear", "grey toadstool", "birch door", "study", "pantry"),
    ("tin crate", "copper key", "stone bench", "yellow plum", "brown fungus", "elm door", "atelier", "vault"),
    ("glass case", "bronze key", "marble stand", "white peach", "pink cap", "cedar door", "gallery", "annex"),
    ("leather trunk", "iron key", "walnut desk", "purple grape", "white truffle", "oak door", "office", "storage"),
    ("plastic bin", "steel key", "metal rack", "orange apricot", "black morel", "iron door", "lab", "closet"),
    ("cardboard carton", "zinc key", "bamboo table", "red cherry", "green moss", "glass door", "studio", "shed"),
    ("ceramic jar", "gold key", "maple counter", "green lime", "yellow chanterelle", "wooden door", "kitchen", "larder"),
    ("wicker basket", "nickel key", "teak cabinet", "blue berry", "purple cort", "painted door", "parlor", "den"),
    ("canvas bag", "chrome key", "granite slab", "red currant", "white puffball", "sliding door", "workshop", "garage"),
    ("steel safe", "titanium key", "concrete block", "green apple", "brown button", "heavy door", "depot", "archive"),
    ("rattan hamper", "brass fob", "cedar chest", "yellow lemon", "red amanita", "screen door", "lounge", "porch"),
]

NAME_SETS_TEST = [
    ("obsidian casket", "jade token", "ebony console", "crimson fig", "violet shroom", "onyx gate", "rotunda", "crypt"),
    ("porcelain urn", "coral fob", "rosewood sideboard", "amber date", "azure fungus", "lattice gate", "solarium", "nook"),
    ("brass reliquary", "opal sigil", "driftwood credenza", "scarlet quince", "ivory cap", "copper gate", "belvedere", "alcove"),
    ("granite coffer", "quartz fob", "mahogany dresser", "golden plum", "emerald mold", "basalt gate", "atrium", "grotto"),
    ("lacquered chest", "meteor key", "slate buffet", "silver nectarine", "ochre fungus", "lacquer gate", "pavilion", "bower"),
    ("woven pannier", "obsidian key", "birch credenza", "pink guava", "grey puffball", "reed gate", "veranda", "closet"),
    ("crystal coffer", "amber key", "ironwood desk", "blue damson", "red waxcap", "crystal gate", "loggia", "cell"),
    ("terracotta pot", "cobalt key", "sesame counter", "green jujube", "brown earthball", "clay gate", "orangery", "shed"),
]


def make_specs(split: str, count: int) -> List[WorldSpec]:
    pool = NAME_SETS_TRAIN if split in ("train", "testsame") else NAME_SETS_TEST
    specs: List[WorldSpec] = []
    for i in range(count):
        names = pool[i % len(pool)]
        suffix = f" {i // len(pool) + 1}" if i >= len(pool) else ""
        box, key, table, apple, mushroom, door, ra, rb = names
        specs.append(
            WorldSpec(
                world_id=f"{split}-{i:03d}",
                box=box + suffix,
                key=key + suffix,
                table=table + suffix,
                apple=apple + suffix,
                mushroom=mushroom + suffix,
                door=door + suffix,
                room_a=ra + suffix,
                room_b=rb + suffix,
                door_mode=("locked", "closed", "open")[i % 3],
                mushroom_on_table=(i % 2 == 0),
                split=split,
            )
        )
    return specs


def solution_path(spec: WorldSpec) -> List[str]:
    """A valid action sequence from the initial state to the goal."""
    acts = ["open_box", "take_key", "unlock_door", "open_door", "take_apple", "go_east"]
    if spec.door_mode == "open":
        acts = ["take_apple", "go_east"]
    elif spec.door_mode == "closed":
        acts = ["take_apple", "open_door", "go_east"]
    if spec.door_mode == "locked":
        # apple can be picked before or after unlocking; keep a fixed valid order
        acts = ["take_apple", "open_box", "take_key", "unlock_door", "open_door", "go_east"]
    return acts


def run_policy(spec: WorldSpec, policy, max_steps: int = 20, rng=None) -> Tuple[List[State], List[str], bool]:
    state = initial_state(spec)
    states = [state]
    actions: List[str] = []
    for _ in range(max_steps):
        if goal_reached(state):
            return states, actions, True
        if policy == "oracle":
            a = oracle_action(state, spec)
        elif policy == "random":
            a = random_action(state, rng)
        else:
            a = policy(state, spec, rng)
        state, _ = step(state, a, spec)
        states.append(state)
        actions.append(a)
    return states, actions, goal_reached(state)
