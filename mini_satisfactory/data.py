"""Game data for the mini-Satisfactory simulator: Tier 0 (the six HUB upgrades) of Satisfactory 1.0.

Every number lives here so it can be checked against the game and fixed in one
place. HUB upgrade costs and unlocks, machine recipes, building costs, the
Miner Mk.1 rate and machine power use come from the game. Values tagged APPROX
are estimates (mostly how fast the player does things by hand); change them
freely, the rest of the code adapts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    label: str  # French in-game name, shown to the user
    stack: int  # stack size; also the size of a machine's internal buffer


ITEMS: dict[str, Item] = {
    "iron_ore": Item("Minerai de fer", 100),
    "copper_ore": Item("Minerai de cuivre", 100),
    "limestone": Item("Calcaire", 100),
    "leaves": Item("Feuilles", 500),
    "iron_ingot": Item("Lingot de fer", 100),
    "copper_ingot": Item("Lingot de cuivre", 100),
    "iron_plate": Item("Plaque de fer", 200),
    "iron_rod": Item("Barre de fer", 200),
    "screw": Item("Vis", 500),
    "wire": Item("Fil électrique", 500),
    "cable": Item("Câble", 200),
    "concrete": Item("Béton", 500),
    "reinforced_iron_plate": Item("Plaque de fer renforcée", 100),
    "portable_miner": Item("Foreuse portative", 50),
}

# Resources the player can mine, and the unlock needed before mining one is useful
# (in the game, the HUB upgrade that adds its scanner and its first recipe).
RAW_RESOURCES: dict[str, str | None] = {
    "iron_ore": None,
    "copper_ore": "copper_ingot",
    "limestone": "concrete",
}


@dataclass(frozen=True)
class Recipe:
    inputs: dict[str, float]
    outputs: dict[str, float]
    seconds: float  # one cycle in a machine
    machine: str | None  # building that can run it automatically (None: by hand only)
    handcraft_seconds: float  # one craft at the Craft Bench / Equipment Workshop (APPROX)


RECIPES: dict[str, Recipe] = {
    "iron_ingot": Recipe({"iron_ore": 1}, {"iron_ingot": 1}, 2, "smelter", 2),
    "copper_ingot": Recipe({"copper_ore": 1}, {"copper_ingot": 1}, 2, "smelter", 2),
    "iron_plate": Recipe({"iron_ingot": 3}, {"iron_plate": 2}, 6, "constructor", 6),
    "iron_rod": Recipe({"iron_ingot": 1}, {"iron_rod": 1}, 4, "constructor", 4),
    "wire": Recipe({"copper_ingot": 1}, {"wire": 2}, 4, "constructor", 4),
    "cable": Recipe({"wire": 2}, {"cable": 1}, 2, "constructor", 2),
    "screw": Recipe({"iron_rod": 1}, {"screw": 4}, 6, "constructor", 6),
    "concrete": Recipe({"limestone": 3}, {"concrete": 1}, 4, "constructor", 4),
    # Made by the Assembler, which only arrives in Tier 2: by hand in Tier 0.
    "reinforced_iron_plate": Recipe(
        {"iron_plate": 6, "screw": 12}, {"reinforced_iron_plate": 1}, 12, None, 12
    ),
    # Crafted at the Equipment Workshop.
    "portable_miner": Recipe({"iron_plate": 2, "iron_rod": 4}, {"portable_miner": 1}, 8, None, 8),
}


@dataclass(frozen=True)
class Building:
    label: str
    cost: dict[str, float]
    power_mw: float
    extracts_per_min: float = 0.0  # > 0 for miners, on a normal-purity node
    belt_ports: bool = True  # False: its output must always be collected by hand
    replaces: str | None = None  # placed building it can be built over (and recycles)


BUILDINGS: dict[str, Building] = {
    # APPROX: extraction rate of the Portable Miner.
    "portable_miner": Building(
        "Foreuse portative", {"portable_miner": 1}, 0, extracts_per_min=20, belt_ports=False
    ),
    "miner_mk1": Building(
        "Foreuse Mk.1",
        {"portable_miner": 1, "iron_plate": 10, "concrete": 10},
        5,
        extracts_per_min=60,
        replaces="portable_miner",
    ),
    "smelter": Building("Fonderie", {"iron_rod": 5, "wire": 8}, 4),
    "constructor": Building("Façonneuse", {"reinforced_iron_plate": 2, "cable": 8}, 4),
}

PURITY: dict[str, float] = {"impure": 0.5, "normal": 1.0, "pure": 2.0}

# APPROX: resource nodes close to the start (a plausible start area, not a real map).
START_NODES: dict[str, tuple[str, ...]] = {
    "iron_ore": ("normal", "normal", "normal", "impure"),
    "copper_ore": ("normal", "impure"),
    "limestone": ("normal", "normal"),
}


@dataclass(frozen=True)
class HubUpgrade:
    name: str
    cost: dict[str, float]
    unlocks: tuple[str, ...]  # recipe names, building names or features


STARTING_UNLOCKS: tuple[str, ...] = ("iron_ingot", "iron_plate", "iron_rod")

HUB_UPGRADES: tuple[HubUpgrade, ...] = (
    HubUpgrade("HUB Upgrade 1", {"iron_rod": 10}, ("portable_miner",)),
    # APPROX: we assume this upgrade also adds the HUB's Biomass Burner, since
    # the Smelter it unlocks needs power and the standalone burner comes later.
    HubUpgrade(
        "HUB Upgrade 2",
        {"iron_rod": 20, "iron_plate": 10},
        ("smelter", "copper_ingot", "wire", "cable", "hub_burner"),
    ),
    HubUpgrade(
        "HUB Upgrade 3",
        {"iron_plate": 20, "iron_rod": 20, "wire": 20},
        ("constructor", "screw", "concrete", "reinforced_iron_plate"),
    ),
    HubUpgrade(
        "HUB Upgrade 4", {"iron_plate": 75, "cable": 20, "concrete": 10}, ("conveyor_belt",)
    ),
    HubUpgrade(
        "HUB Upgrade 5", {"iron_rod": 75, "cable": 50, "concrete": 20}, ("miner_mk1",)
    ),
    HubUpgrade(
        "HUB Upgrade 6",
        {"iron_rod": 100, "iron_plate": 100, "wire": 100, "concrete": 50},
        ("space_elevator", "biomass_burner"),
    ),
)

# Power: the HUB's burner, fed with leaves the player gathers.
HUB_BURNER_MW = 20.0
LEAF_MJ = 15.0

# Time the player spends on each kind of action, in seconds of game time.
HAND_MINE_SECONDS = 10.0
HAND_MINE_AMOUNT = 10  # APPROX: about one ore per second by hand
GATHER_SECONDS = 10.0
GATHER_LEAVES = 40  # APPROX
CRAFT_BATCH_SECONDS = 10.0  # one craft action chains crafts for about this long
BUILD_SECONDS = 5.0  # APPROX
COLLECT_SECONDS = 15.0  # APPROX: walking round the machines that are not on belts
WAIT_SECONDS = 30.0
SUBMIT_SECONDS = 2.0


def label(key: str) -> str:
    """French name of an item, falling back to the key itself."""
    item = ITEMS.get(key)
    return item.label if item else key
