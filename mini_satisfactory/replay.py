"""Film a run: a snapshot of the game every few seconds, to show it on screen afterwards."""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from mini_satisfactory import data
from mini_satisfactory.sim import EPS, Action, Machine, Simulation, upgrade_progress

# Machine states shown on screen.
RUNNING, NO_INPUT, OUTPUT_FULL, NO_POWER = "running", "no_input", "output_full", "no_power"


@dataclass(frozen=True)
class MachineView:
    uid: int
    building: str
    target: str
    node: int | None
    status: str
    activity: float  # share of full speed (0 to 1)
    connected: bool  # on conveyor belts
    input_fill: float | None  # internal buffers (0 to 1), None when on belts
    output_fill: float | None


@dataclass(frozen=True)
class Frame:
    time: float
    stock: dict[str, float]
    machines: tuple[MachineView, ...]
    upgrades_done: int
    unlocked: frozenset[str]
    structures: frozenset[str]
    power_capacity_mw: float
    power_used_mw: float
    power_demand_mw: float
    fuel_seconds: float  # how long the leaves in stock last at full demand


@dataclass(frozen=True)
class Step:
    start: float
    duration: float
    action: Action


@dataclass
class Run:
    name: str
    frames: list[Frame]
    steps: list[Step]
    upgrade_times: list[float]
    finished: bool

    @property
    def end_time(self) -> float:
        return self.frames[-1].time

    def frame_at(self, time: float) -> Frame:
        index = bisect.bisect_right(self._times, time) - 1
        return self.frames[max(0, index)]

    def step_at(self, time: float) -> int:
        """Index of the action being done at `time`."""
        index = bisect.bisect_right(self._starts, time) - 1
        return max(0, min(index, len(self.steps) - 1))

    def progress_at(self, time: float) -> float:
        """Upgrades done plus the share of the next one in stock (0 to 6)."""
        frame = self.frame_at(time)
        return frame.upgrades_done + upgrade_progress(frame.stock, frame.upgrades_done)

    def __post_init__(self) -> None:
        self._times = [f.time for f in self.frames]
        self._starts = [s.start for s in self.steps]


def machine_view(sim: Simulation, machine: Machine) -> MachineView:
    connected = sim.connected(machine)
    if machine.is_miner:
        inputs: dict[str, float] = {}
        outputs = {machine.target: 1.0}
    else:
        recipe = data.RECIPES[machine.target]
        inputs, outputs = recipe.inputs, recipe.outputs
    input_fill = output_fill = None
    if not connected:
        if inputs:
            input_fill = min(machine.inputs.get(i, 0.0) / data.ITEMS[i].stack for i in inputs)
        output_fill = max(machine.outputs.get(o, 0.0) / data.ITEMS[o].stack for o in outputs)

    if machine.activity > 0.01:
        status = RUNNING
    elif inputs and any((sim.stock if connected else machine.inputs).get(i, 0.0) <= EPS for i in inputs):
        status = NO_INPUT
    elif output_fill is not None and output_fill >= 1 - 1e-6:
        status = OUTPUT_FULL
    else:
        status = NO_POWER
    return MachineView(
        machine.uid, machine.building, machine.target, machine.node, status,
        machine.activity, connected, input_fill, output_fill,
    )


def snapshot(sim: Simulation) -> Frame:
    demand = sim.power_demand()
    return Frame(
        time=sim.time,
        stock={item: max(0.0, qty) for item, qty in sim.stock.items()},
        machines=tuple(machine_view(sim, m) for m in sim.machines),
        upgrades_done=sim.upgrades_done,
        unlocked=frozenset(sim.unlocked),
        structures=frozenset(sim.structures),
        power_capacity_mw=sim.power_capacity(),
        power_used_mw=sim.power_used_mw,
        power_demand_mw=demand,
        fuel_seconds=sim.stock["leaves"] * data.LEAF_MJ / demand if demand else 0.0,
    )


def record(player, name: str, time_limit_s: float = 3 * 3600) -> Run:
    """Let a player (anything with an `act(sim)` method) play, filming every machine update."""
    sim = Simulation(record=True)
    frames = [snapshot(sim)]
    sim.on_tick = lambda s: frames.append(snapshot(s))
    while not sim.finished and sim.time < time_limit_s:
        sim.step(player.act(sim))
        # The action's own effects (ore mined, upgrade unlocked...) land at its end.
        frame = snapshot(sim)
        if abs(frames[-1].time - frame.time) < EPS:
            frames[-1] = frame
        else:
            frames.append(frame)
    starts = [start for start, _ in sim.history]
    steps = [
        Step(start, end - start, action)
        for (start, action), end in zip(sim.history, [*starts[1:], sim.time])
    ]
    return Run(name, frames, steps, list(sim.upgrade_times), sim.finished)
