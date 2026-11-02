"""Item table and tech-tree knowledge for the hand-written reward.

Values are hand-written in "diamond-equivalent" units from the Minecraft tech
tree.  They encode *what each item unlocks* (its use in later recipes), not its
market price:

  log  -> planks (4) -> sticks (4) -> crafting table (4 planks)
  wooden pickaxe  (3 planks + 2 sticks)   mines stone
  cobblestone -> stone pickaxe (3 cobble + 2 sticks)  mines iron ore
  furnace (8 cobble) + fuel -> smelt iron ore -> iron ingot
  iron pickaxe (3 ingots + 2 sticks)  mines diamond ore (y <= 15 in MC 1.11)
  diamond -> the episode goal

References (Minecraft Wiki, Java Edition 1.11 recipes / mechanics):
  https://minecraft.wiki/w/Crafting
  https://minecraft.wiki/w/Log          https://minecraft.wiki/w/Planks
  https://minecraft.wiki/w/Stick        https://minecraft.wiki/w/Crafting_Table
  https://minecraft.wiki/w/Wooden_Pickaxe
  https://minecraft.wiki/w/Cobblestone  https://minecraft.wiki/w/Stone_Pickaxe
  https://minecraft.wiki/w/Iron_Ore     https://minecraft.wiki/w/Furnace
  https://minecraft.wiki/w/Iron_Ingot   https://minecraft.wiki/w/Iron_Pickaxe
  https://minecraft.wiki/w/Diamond_Ore  https://minecraft.wiki/w/Diamond
  https://minecraft.wiki/w/Health       https://minecraft.wiki/w/Hunger
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# MineRL/`ObtainDiamond`-style score: index of the deepest milestone reached.
MILESTONE_ORDER: List[str] = [
    "log",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "stone_pickaxe",
    "iron_ore",
    "furnace",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
]

SCORE_OF = {name: i + 1 for i, name in enumerate(MILESTONE_ORDER)}


@dataclass(frozen=True)
class ItemSpec:
    name: str
    value: float          # weight in the potential Phi = sum value * log1p(count)
    bonus: float          # one-time milestone reward (before speed multiplier)
    wiki: str             # short MC-wiki justification
    requires: str = ""    # tool/station needed to obtain the item (documentation)


ITEM_TABLE: Dict[str, ItemSpec] = {
    "log": ItemSpec(
        "log", 0.02, 0.5,
        "1 log -> 4 planks; punch trees by hand (no tool needed)", requires="hand"),
    "planks": ItemSpec(
        "planks", 0.02, 0.5,
        "building block and base material for sticks / table / wooden pickaxe",
        requires="log"),
    "stick": ItemSpec(
        "stick", 0.025, 0.6,
        "handle for every pickaxe (2 sticks per pickaxe)", requires="planks"),
    "crafting_table": ItemSpec(
        "crafting_table", 0.05, 1.0,
        "unlocks 3x3 recipes (pickaxes, furnace); 4 planks", requires="planks"),
    "wooden_pickaxe": ItemSpec(
        "wooden_pickaxe", 0.10, 1.5,
        "first pickaxe; mines stone/cobblestone; 3 planks + 2 sticks",
        requires="crafting_table"),
    "cobblestone": ItemSpec(
        "cobblestone", 0.01, 1.0,
        "stone tools and furnace; mined with a wooden pickaxe",
        requires="wooden_pickaxe"),
    "stone_pickaxe": ItemSpec(
        "stone_pickaxe", 0.15, 2.0,
        "mines iron ore (and all stone-tier blocks); 3 cobble + 2 sticks",
        requires="crafting_table"),
    "iron_ore": ItemSpec(
        "iron_ore", 0.30, 3.0,
        "smelts into iron ingots; needs a stone pickaxe", requires="stone_pickaxe"),
    "furnace": ItemSpec(
        "furnace", 0.15, 2.5,
        "smelting station; 8 cobblestone", requires="crafting_table"),
    "iron_ingot": ItemSpec(
        "iron_ingot", 0.50, 4.0,
        "smelt iron ore + fuel; ingredient of the iron pickaxe", requires="furnace"),
    "iron_pickaxe": ItemSpec(
        "iron_pickaxe", 0.70, 5.0,
        "only tool that can mine diamond ore; 3 ingots + 2 sticks",
        requires="crafting_table"),
    "diamond": ItemSpec(
        "diamond", 4.00, 40.0,
        "episode goal; diamond ore at y<=15 in MC 1.11, mined with iron pickaxe",
        requires="iron_pickaxe"),
}


def default_item_weights() -> Dict[str, float]:
    return {name: spec.value for name, spec in ITEM_TABLE.items()}


def default_milestone_bonus() -> Dict[str, float]:
    return {name: spec.bonus for name, spec in ITEM_TABLE.items()}
