"""Read the real game through the Ficsit Remote Monitoring mod, in the mini-game's terms."""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # to import mini_satisfactory

from config import FRM_URL  # noqa: E402
from mini_satisfactory import data  # noqa: E402
from mini_satisfactory.sim import Simulation  # noqa: E402

# Class names the game uses for the items the mini-game knows.
ITEM_CLASSES = {
    "Desc_OreIron_C": "iron_ore",
    "Desc_OreCopper_C": "copper_ore",
    "Desc_Stone_C": "limestone",
    "Desc_Leaves_C": "leaves",
    "Desc_IronIngot_C": "iron_ingot",
    "Desc_CopperIngot_C": "copper_ingot",
    "Desc_IronPlate_C": "iron_plate",
    "Desc_IronRod_C": "iron_rod",
    "Desc_IronScrew_C": "screw",
    "Desc_Wire_C": "wire",
    "Desc_Cable_C": "cable",
    "Desc_Cement_C": "concrete",
    "Desc_IronPlateReinforced_C": "reinforced_iron_plate",
    "BP_ItemDescriptorPortableMiner_C": "portable_miner",
}
# The six HUB upgrades, in order.
HUB_SCHEMATICS = (
    "Schematic_Tutorial1_C",
    "Schematic_Tutorial1_5_C",
    "Schematic_Tutorial2_C",
    "Schematic_Tutorial3_C",
    "Schematic_Tutorial4_C",
    "Schematic_Tutorial5_C",
)
# Buildings the mini-game counts as built once (class name in the game -> mini-game name).
STRUCTURE_CLASSES = {"Build_Workshop_C": "workshop"}
CM_PER_M = 100


@dataclass(frozen=True)
class Node:
    resource: str  # mini-game item, e.g. "iron_ore"
    purity: str  # impure, normal or pure
    x: float  # metres
    y: float
    z: float
    exploited: bool  # a miner is already on it


@dataclass(frozen=True)
class GameState:
    position: tuple[float, float, float]  # metres
    yaw: float  # degrees: where the player looks
    inventory: dict[str, float]  # mini-game items carried
    other_items: dict[str, float]  # items the mini-game ignores, by in-game name
    upgrades_done: int
    deposited: dict[str, float]  # already given to the HUB for the next upgrade
    nodes: tuple[Node, ...]
    structures: frozenset[str] = frozenset()  # mini-game structures found in the factory

    def distance_to(self, node: Node) -> float:
        x, y, z = self.position
        return math.dist((x, y, z), (node.x, node.y, node.z))

    def turn_towards(self, node: Node) -> float:
        """Degrees to turn to face the node: positive to the right, negative to the left."""
        x, y, _ = self.position
        heading = math.degrees(math.atan2(node.y - y, node.x - x))
        return (heading - self.yaw + 180) % 360 - 180

    def nearest(self, resource: str, free_only: bool = True) -> Node | None:
        nodes = [n for n in self.nodes if n.resource == resource and not (free_only and n.exploited)]
        return min(nodes, key=self.distance_to, default=None)


def fetch(endpoint: str):
    with urllib.request.urlopen(f"{FRM_URL}/{endpoint}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _location(entry: dict) -> tuple[float, float, float] | None:
    loc = entry.get("location")
    if not isinstance(loc, dict):
        loc = (((entry.get("features") or {}).get("geometry") or {}).get("coordinates"))
    if not isinstance(loc, dict) or not {"x", "y", "z"} <= loc.keys():
        return None
    return loc["x"] / CM_PER_M, loc["y"] / CM_PER_M, loc["z"] / CM_PER_M


def _purity(entry: dict) -> str:
    text = f"{entry.get('EnumPurity', '')} {entry.get('Purity', '')}".lower()
    if "inpure" in text or "impur" in text:  # the game's own enum spells it "RP_Inpure"
        return "impure"
    if "pure" in text or "pur" in text:
        return "pure"
    return "normal"


def _node_resource(entry: dict) -> str | None:
    labels = {item.label.lower(): key for key, item in data.ITEMS.items()}
    for field in ("ClassName", "ResourceClass", "Resource"):
        value = entry.get(field)
        if isinstance(value, str) and value in ITEM_CLASSES:
            return ITEM_CLASSES[value]
    for field in ("Name", "ResourceName"):
        value = entry.get(field)
        if isinstance(value, str) and value.lower() in labels:
            return labels[value.lower()]
    return None


def parse_nodes(entries: list) -> tuple[Node, ...]:
    nodes = []
    for entry in entries or []:
        resource = _node_resource(entry)
        where = _location(entry)
        if resource in data.RAW_RESOURCES and where is not None:
            nodes.append(Node(resource, _purity(entry), *where, bool(entry.get("Exploited"))))
    return tuple(nodes)


def read_state(get: Callable[[str], object] = fetch) -> GameState:
    """Ask the mod for the player, the HUB upgrades and the resource nodes."""
    player = get("getPlayer")[0]
    inventory: dict[str, float] = {}
    other: dict[str, float] = {}
    for stack in player.get("Inventory", []):
        key = ITEM_CLASSES.get(stack.get("ClassName", ""))
        if key is None:
            other[stack.get("Name", "?")] = other.get(stack.get("Name", "?"), 0) + stack.get("Amount", 0)
        else:
            inventory[key] = inventory.get(key, 0) + stack.get("Amount", 0)

    schematics = {s["ClassName"]: s for s in get("getSchematics")}
    done = 0
    while done < len(HUB_SCHEMATICS) and schematics.get(HUB_SCHEMATICS[done], {}).get("Purchased"):
        done += 1
    deposited: dict[str, float] = {}
    if done < len(HUB_SCHEMATICS):
        for cost in schematics.get(HUB_SCHEMATICS[done], {}).get("Cost", []):
            key = ITEM_CLASSES.get(cost.get("ClassName", ""))
            given = cost.get("TotalCost", 0) - cost.get("RemainingCost", 0)
            if key is not None and given > 0:
                deposited[key] = given

    try:
        nodes = parse_nodes(get("getResourceNode"))
    except Exception:  # the copilot still works without node positions
        nodes = ()
    try:
        factory = {b.get("ClassName") for b in get("getFactory")}
    except Exception:
        factory = set()
    structures = frozenset(name for cls, name in STRUCTURE_CLASSES.items() if cls in factory)
    location = _location(player) or (0.0, 0.0, 0.0)
    yaw = float((player.get("location") or {}).get("rotation", 0.0))
    return GameState(location, yaw, inventory, other, done, deposited, nodes, structures)


def to_simulation(state: GameState) -> Simulation:
    """A mini-game in the same situation, for the AI to look at."""
    sim = Simulation()
    for upgrade in data.HUB_UPGRADES[: state.upgrades_done]:
        sim.unlocked.update(upgrade.unlocks)
    sim.upgrades_done = state.upgrades_done
    sim.structures.update(state.structures)
    for item, qty in (*state.inventory.items(), *state.deposited.items()):
        sim.stock[item] += qty
    return sim
