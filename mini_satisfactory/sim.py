"""Mini-Satisfactory simulator: the early game as an economy, without 3D.

The player does one action at a time (mine by hand, craft, build, ...). Each
action takes some game time, during which the machines already built keep
working. Pure Python: no Gym or machine-learning dependency.

Simplifications compared with the real game:
- no walking or map: everything is next to the HUB, and the player's inventory
  and storage are one shared stock;
- before conveyor belts are unlocked, machines have internal buffers that the
  player fills and empties with the "collect" action; once belts are unlocked,
  machines (except Portable Miners) take from and give to the shared stock by
  themselves;
- when power runs short, every machine slows down instead of tripping a fuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from mini_satisfactory import data

SIM_TICK = 2.0  # machines are updated in steps of this many seconds
EPS = 1e-9


@dataclass(frozen=True)
class Action:
    kind: str  # wait | mine | gather | craft | build | collect | submit
    target: str | None = None  # resource or recipe
    building: str | None = None

    def label(self) -> str:
        if self.kind == "wait":
            return f"Attendre {data.WAIT_SECONDS:.0f} s"
        if self.kind == "mine":
            return f"Miner à la main : {data.label(self.target)}"
        if self.kind == "gather":
            return "Ramasser des feuilles (carburant)"
        if self.kind == "craft":
            outputs = data.RECIPES[self.target].outputs
            return "Fabriquer à la main : " + ", ".join(data.label(i) for i in outputs)
        if self.kind == "build":
            building = data.BUILDINGS[self.building].label
            return f"Construire : {building} ({data.label(self.target)})"
        if self.kind == "collect":
            return "Tournée des machines (vider / remplir)"
        return "Livrer le HUB Upgrade"


def _build_actions() -> tuple[Action, ...]:
    actions = [Action("wait")]
    actions += [Action("mine", res) for res in data.RAW_RESOURCES]
    actions.append(Action("gather"))
    actions += [Action("craft", name) for name in data.RECIPES]
    for name, building in data.BUILDINGS.items():
        if building.extracts_per_min:
            actions += [Action("build", res, name) for res in data.RAW_RESOURCES]
        else:
            actions += [
                Action("build", rec, name)
                for rec, recipe in data.RECIPES.items()
                if recipe.machine == name
            ]
    actions += [Action("collect"), Action("submit")]
    return tuple(actions)


ACTIONS: tuple[Action, ...] = _build_actions()
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}


def _item_depth() -> dict[str, int]:
    """Steps from raw resources: ore 0, ingot 1, plate 2, ... (orders machine updates)."""
    depth = {item: 0 for item in data.ITEMS}
    for _ in range(len(data.RECIPES)):
        for recipe in data.RECIPES.values():
            level = 1 + max(depth[i] for i in recipe.inputs)
            for out in recipe.outputs:
                depth[out] = max(depth[out], level)
    return depth


ITEM_DEPTH = _item_depth()


@dataclass
class Machine:
    building: str
    target: str  # recipe it runs, or resource it extracts
    rate: float  # cycles per second (recipes) or items per second (miners)
    node: int | None = None  # resource node index, for miners
    inputs: dict[str, float] = field(default_factory=dict)  # buffer while not on belts
    outputs: dict[str, float] = field(default_factory=dict)
    uid: int = 0  # stable id, to follow a machine from one moment to the next
    activity: float = 0.0  # share of full speed it ran at during the last update

    @property
    def is_miner(self) -> bool:
        return self.node is not None

    @property
    def stage(self) -> int:
        if self.is_miner:
            return 0
        return max(ITEM_DEPTH[i] for i in data.RECIPES[self.target].outputs)


class Simulation:
    def __init__(self, record: bool = False):
        self.time = 0.0
        self.stock: dict[str, float] = {item: 0.0 for item in data.ITEMS}
        self.unlocked: set[str] = set(data.STARTING_UNLOCKS)
        self.upgrades_done = 0
        self.upgrade_times: list[float] = []
        self.machines: list[Machine] = []
        self.nodes: dict[str, list[Machine | None]] = {
            res: [None] * len(purities) for res, purities in data.START_NODES.items()
        }
        # (time the action started, action) for every action, when recording.
        self.history: list[tuple[float, Action]] | None = [] if record else None
        # Called after every machine update, e.g. to film the run.
        self.on_tick: Callable[[Simulation], None] | None = None
        self.power_used_mw = 0.0  # during the last machine update
        self._next_uid = 1

    # ------------------------------------------------------------------ queries

    @property
    def finished(self) -> bool:
        return self.upgrades_done >= len(data.HUB_UPGRADES)

    def next_upgrade(self) -> data.HubUpgrade | None:
        return None if self.finished else data.HUB_UPGRADES[self.upgrades_done]

    def has(self, items: dict[str, float], times: float = 1) -> bool:
        return all(self.stock[i] + EPS >= q * times for i, q in items.items())

    def connected(self, machine: Machine) -> bool:
        return data.BUILDINGS[machine.building].belt_ports and "conveyor_belt" in self.unlocked

    def power_capacity(self) -> float:
        return data.HUB_BURNER_MW if "hub_burner" in self.unlocked else 0.0

    def power_demand(self) -> float:
        """Power the machines would draw if all of them ran (MW)."""
        return sum(data.BUILDINGS[m.building].power_mw for m in self.machines)

    def count(self, building: str, target: str) -> int:
        return sum(1 for m in self.machines if m.building == building and m.target == target)

    def free_nodes(self, resource: str) -> int:
        return sum(1 for occupant in self.nodes[resource] if occupant is None)

    def craftable(self, recipe_name: str) -> int:
        """How many times the recipe can be crafted by hand from the stock."""
        recipe = data.RECIPES[recipe_name]
        return int(min(math.floor((self.stock[i] + EPS) / q) for i, q in recipe.inputs.items()))

    def action_mask(self) -> list[bool]:
        return [self.is_valid(a) for a in ACTIONS]

    def is_valid(self, action: Action) -> bool:
        if self.finished:
            return False
        kind = action.kind
        if kind == "wait":
            return True
        if kind == "mine":
            needed = data.RAW_RESOURCES[action.target]
            return needed is None or needed in self.unlocked
        if kind == "gather":
            return "hub_burner" in self.unlocked
        if kind == "craft":
            return action.target in self.unlocked and self.craftable(action.target) >= 1
        if kind == "build":
            return self._build_plan(action) is not None
        if kind == "collect":
            return any(not self.connected(m) for m in self.machines)
        if kind == "submit":
            return self.has(self.next_upgrade().cost)
        raise ValueError(f"unknown action {action}")

    def _build_plan(self, action: Action) -> tuple[int | None, Machine | None] | None:
        """Where an action would build: (node, machine it replaces), or None if it can't."""
        building = data.BUILDINGS[action.building]
        if action.building not in self.unlocked:
            return None
        if not building.extracts_per_min:
            if action.target not in self.unlocked:
                return None
            return (None, None) if self.has(building.cost) else None

        if data.RAW_RESOURCES[action.target] not in (None, *self.unlocked):
            return None
        purities = data.START_NODES[action.target]
        candidates: list[tuple[float, bool, int, Machine | None]] = []
        for node, occupant in enumerate(self.nodes[action.target]):
            if occupant is None:
                refund: dict[str, float] = {}
            elif occupant.building == building.replaces:
                # Building over it gives back what the replaced building cost.
                refund = data.BUILDINGS[occupant.building].cost
            else:
                continue
            if all(self.stock[i] + refund.get(i, 0) + EPS >= q for i, q in building.cost.items()):
                # Best purity first; on a tie, prefer a free node to keep the old miner.
                candidates.append(
                    (data.PURITY[purities[node]], occupant is None, -node, occupant)
                )
        if not candidates:
            return None
        _, _, neg_node, occupant = max(candidates, key=lambda c: c[:3])
        return -neg_node, occupant

    # ------------------------------------------------------------------ actions

    def step(self, index: int) -> float:
        """Do action number `index` and return the game time it took (seconds)."""
        action = ACTIONS[index]
        if not self.is_valid(action):
            raise ValueError(f"action not possible now: {action.label()}")
        if self.history is not None:
            self.history.append((self.time, action))
        start = self.time
        getattr(self, f"_do_{action.kind}")(action)
        return self.time - start

    def _do_wait(self, action: Action) -> None:
        self.advance(data.WAIT_SECONDS)

    def _do_mine(self, action: Action) -> None:
        self.advance(data.HAND_MINE_SECONDS)
        self.stock[action.target] += data.HAND_MINE_AMOUNT

    def _do_gather(self, action: Action) -> None:
        self.advance(data.GATHER_SECONDS)
        self.stock["leaves"] += data.GATHER_LEAVES

    def _do_craft(self, action: Action) -> None:
        recipe = data.RECIPES[action.target]
        per_batch = max(1, int(data.CRAFT_BATCH_SECONDS // recipe.handcraft_seconds))
        crafts = min(per_batch, self.craftable(action.target))
        for item, qty in recipe.inputs.items():
            self.stock[item] -= qty * crafts
        self.advance(recipe.handcraft_seconds * crafts)
        for item, qty in recipe.outputs.items():
            self.stock[item] += qty * crafts

    def _do_build(self, action: Action) -> None:
        node, replaced = self._build_plan(action)
        building = data.BUILDINGS[action.building]
        if replaced is not None:
            self._remove(replaced)
            for item, qty in data.BUILDINGS[replaced.building].cost.items():
                self.stock[item] += qty
        for item, qty in building.cost.items():
            self.stock[item] -= qty
        if building.extracts_per_min:
            purity = data.PURITY[data.START_NODES[action.target][node]]
            rate = building.extracts_per_min / 60 * purity
        else:
            rate = 1 / data.RECIPES[action.target].seconds
        machine = Machine(action.building, action.target, rate, node, uid=self._next_uid)
        self._next_uid += 1
        if node is not None:
            self.nodes[action.target][node] = machine  # the node is taken from now on
        self.advance(data.BUILD_SECONDS)
        self.machines.append(machine)
        self.machines.sort(key=lambda m: m.stage)

    def _remove(self, machine: Machine) -> None:
        for item, qty in (*machine.inputs.items(), *machine.outputs.items()):
            self.stock[item] += qty
        self.machines.remove(machine)
        if machine.node is not None:
            self.nodes[machine.target][machine.node] = None

    def _do_collect(self, action: Action) -> None:
        self.advance(data.COLLECT_SECONDS)
        manual = [m for m in self.machines if not self.connected(m)]
        for machine in manual:
            for item, qty in machine.outputs.items():
                self.stock[item] += qty
            machine.outputs.clear()
        # Top up input buffers, sharing the stock fairly when there isn't enough.
        wants: dict[str, list[tuple[Machine, float]]] = {}
        for machine in manual:
            if machine.is_miner:
                continue
            for item in data.RECIPES[machine.target].inputs:
                room = data.ITEMS[item].stack - machine.inputs.get(item, 0.0)
                if room > EPS:
                    wants.setdefault(item, []).append((machine, room))
        for item, requests in wants.items():
            total = sum(room for _, room in requests)
            share = min(1.0, self.stock[item] / total)
            for machine, room in requests:
                given = room * share
                machine.inputs[item] = machine.inputs.get(item, 0.0) + given
                self.stock[item] -= given

    def _do_submit(self, action: Action) -> None:
        upgrade = self.next_upgrade()
        for item, qty in upgrade.cost.items():
            self.stock[item] -= qty
        self.advance(data.SUBMIT_SECONDS)
        self.unlocked.update(upgrade.unlocks)
        self.upgrades_done += 1
        self.upgrade_times.append(self.time)
        if "conveyor_belt" in upgrade.unlocks:
            # Machines now on belts hand their buffers over to the shared stock.
            for machine in self.machines:
                if self.connected(machine):
                    for item, qty in (*machine.inputs.items(), *machine.outputs.items()):
                        self.stock[item] += qty
                    machine.inputs.clear()
                    machine.outputs.clear()

    # ------------------------------------------------------------------ machines

    def advance(self, seconds: float) -> None:
        """Let the machines work for `seconds` of game time."""
        end = self.time + seconds
        while end - self.time > EPS:
            dt = min(SIM_TICK, end - self.time)
            self._tick(dt)
            self.time += dt
            if self.on_tick is not None:
                self.on_tick(self)
        self.time = end

    def _tick(self, dt: float) -> None:
        powered = [m for m in self.machines if data.BUILDINGS[m.building].power_mw > 0]
        demand = sum(
            data.BUILDINGS[m.building].power_mw for m in powered if self._can_run(m)
        )
        if demand > 0:
            fuel_mw = self.stock["leaves"] * data.LEAF_MJ / dt
            supply = min(self.power_capacity(), fuel_mw)
            power_speed = min(1.0, supply / demand)
        else:
            power_speed = 0.0
        energy_mj = 0.0
        for machine in self.machines:
            power = data.BUILDINGS[machine.building].power_mw
            speed = power_speed if power > 0 else 1.0
            ran = self._run(machine, dt * speed) if speed > 0 else 0.0  # share of full speed
            machine.activity = speed * ran
            energy_mj += power * dt * speed * ran
        self.power_used_mw = energy_mj / dt
        if energy_mj:
            self.stock["leaves"] = max(0.0, self.stock["leaves"] - energy_mj / data.LEAF_MJ)

    def _can_run(self, machine: Machine) -> bool:
        connected = self.connected(machine)
        if machine.is_miner:
            return connected or self._room(machine, machine.target) > EPS
        recipe = data.RECIPES[machine.target]
        source = self.stock if connected else machine.inputs
        has_inputs = all(source.get(i, 0.0) > EPS for i in recipe.inputs)
        has_room = connected or all(self._room(machine, o) > EPS for o in recipe.outputs)
        return has_inputs and has_room

    def _room(self, machine: Machine, item: str) -> float:
        return data.ITEMS[item].stack - machine.outputs.get(item, 0.0)

    def _run(self, machine: Machine, seconds: float) -> float:
        """Run a machine for `seconds` at full speed at most; return the fraction done."""
        connected = self.connected(machine)
        target = self.stock if connected else machine.outputs
        if machine.is_miner:
            full = machine.rate * seconds
            amount = full if connected else min(full, self._room(machine, machine.target))
            target[machine.target] = target.get(machine.target, 0.0) + amount
            return amount / full

        recipe = data.RECIPES[machine.target]
        source = self.stock if connected else machine.inputs
        full = machine.rate * seconds
        cycles = full
        for item, qty in recipe.inputs.items():
            cycles = min(cycles, source.get(item, 0.0) / qty)
        if not connected:
            for item, qty in recipe.outputs.items():
                cycles = min(cycles, self._room(machine, item) / qty)
        if cycles <= EPS:
            return 0.0
        for item, qty in recipe.inputs.items():
            source[item] = source.get(item, 0.0) - qty * cycles
        for item, qty in recipe.outputs.items():
            target[item] = target.get(item, 0.0) + qty * cycles
        return cycles / full
