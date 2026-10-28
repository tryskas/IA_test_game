"""Reference players to compare the trained AI against."""
from __future__ import annotations

import random

from mini_satisfactory import data
from mini_satisfactory.sim import ACTION_INDEX, Action, Simulation

# Recipe that makes each item (Tier 0 has exactly one per item).
PRODUCER: dict[str, str] = {
    out: name for name, recipe in data.RECIPES.items() for out in recipe.outputs
}

# Machines to build, in order, as (building, recipe or resource).
SIMPLE_PLAN: tuple[tuple[str, str], ...] = (
    *[("portable_miner", "iron_ore")] * 3,
    ("portable_miner", "copper_ore"),
    ("portable_miner", "limestone"),
)
FACTORY_PLAN: tuple[tuple[str, str], ...] = (
    *[("portable_miner", "iron_ore")] * 3,
    ("smelter", "iron_ingot"),
    ("portable_miner", "copper_ore"),
    ("constructor", "iron_rod"),
    ("constructor", "iron_plate"),
    *[("portable_miner", "limestone")] * 2,
    ("smelter", "copper_ingot"),
    ("constructor", "wire"),
    ("constructor", "concrete"),
)

FUEL_RESERVE_S = 90  # gather leaves when the burner has less fuel than this
STARVING_BELOW = 5  # a hand-fed machine with less input than this needs a refill...
REFILL_FROM = 20  # ...if the stock has at least this much of it


class RandomBot:
    """Picks any possible action at random."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, sim: Simulation) -> int:
        mask = sim.action_mask()
        return self.rng.choice([i for i, ok in enumerate(mask) if ok])


class ScriptedBot:
    """A hand-written strategy, like a player following a guide.

    It builds the machines of its plan in order as soon as they are unlocked,
    keeps them fed (fuel, refills, emptying) and otherwise crafts what the next
    HUB upgrade needs.
    """

    def __init__(self, plan: tuple[tuple[str, str], ...] = SIMPLE_PLAN):
        self.plan = plan

    def act(self, sim: Simulation) -> int:
        if sim.is_valid(Action("submit")):
            return ACTION_INDEX[Action("submit")]
        demand = sim.power_demand()
        if demand and sim.stock["leaves"] * data.LEAF_MJ / demand < FUEL_RESERVE_S:
            return ACTION_INDEX[Action("gather")]
        if self._should_collect(sim):
            return ACTION_INDEX[Action("collect")]
        action = self._next_build(sim)
        if action is not None:
            return action
        for item, qty in sim.next_upgrade().cost.items():
            if sim.stock[item] < qty:
                action = self._obtain(sim, item)
                if action is not None:
                    return action
        return ACTION_INDEX[Action("wait")]

    def _next_build(self, sim: Simulation) -> int | None:
        """Build the first missing machine of the plan, or gather what it costs."""
        planned: dict[tuple[str, str], int] = {}
        for building, target in self.plan:
            planned[(building, target)] = planned.get((building, target), 0) + 1
            if sim.count(building, target) >= planned[(building, target)]:
                continue
            if not self._available(sim, building, target):
                continue
            build = Action("build", target, building)
            if sim.is_valid(build):
                return ACTION_INDEX[build]
            for item, qty in data.BUILDINGS[building].cost.items():
                if sim.stock[item] < qty:
                    return self._obtain(sim, item)
            return None
        return None

    @staticmethod
    def _available(sim: Simulation, building: str, target: str) -> bool:
        if building not in sim.unlocked:
            return False
        if target in data.RAW_RESOURCES:
            needed = data.RAW_RESOURCES[target]
            has_node = sim.free_nodes(target) or data.BUILDINGS[building].replaces
            return bool(has_node) and (needed is None or needed in sim.unlocked)
        return target in sim.unlocked

    @staticmethod
    def _should_collect(sim: Simulation) -> bool:
        for machine in sim.machines:
            if sim.connected(machine):
                continue
            for item, qty in machine.outputs.items():
                if qty >= data.ITEMS[item].stack / 2:
                    return True
            if machine.is_miner:
                continue
            for item in data.RECIPES[machine.target].inputs:
                if machine.inputs.get(item, 0.0) < STARVING_BELOW and sim.stock[item] >= REFILL_FROM:
                    return True
        return False

    def _obtain(self, sim: Simulation, item: str) -> int | None:
        """Next action that brings us closer to having one more `item`."""
        if item in data.RAW_RESOURCES:
            waiting = sum(
                m.outputs.get(item, 0.0) for m in sim.machines if not sim.connected(m)
            )
            if waiting >= data.HAND_MINE_AMOUNT:
                return ACTION_INDEX[Action("collect")]
            mine = Action("mine", item)
            return ACTION_INDEX[mine] if sim.is_valid(mine) else None
        recipe_name = PRODUCER[item]
        if recipe_name not in sim.unlocked:
            return None
        needs = data.RECIPES[recipe_name].needs
        if needs is not None and needs not in sim.structures:
            build = Action("build", needs, needs)
            if sim.is_valid(build):
                return ACTION_INDEX[build]
            for ingredient, qty in data.BUILDINGS[needs].cost.items():
                if sim.stock[ingredient] < qty:
                    return self._obtain(sim, ingredient)
            return None
        for ingredient, qty in data.RECIPES[recipe_name].inputs.items():
            if sim.stock[ingredient] + 1e-9 < qty:
                return self._obtain(sim, ingredient)
        return ACTION_INDEX[Action("craft", recipe_name)]


def play(bot, time_limit_s: float = 3 * 3600, record: bool = False) -> Simulation:
    """Let a bot play until Tier 0 is done or the time runs out."""
    sim = Simulation(record=record)
    while not sim.finished and sim.time < time_limit_s:
        sim.step(bot.act(sim))
    return sim
