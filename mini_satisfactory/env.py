"""Gymnasium environment around the simulator, for reinforcement learning.

Reward: -1 per minute of game time. On top of that, each HUB upgrade is worth
15 points: 10 earned bit by bit while crafting the items it asks for (even
ahead of time, for a later upgrade), and 5 when it is delivered. Spending those
items on something else takes the points back. Over a finished run these 90
points are always the same, so the only thing that separates two finished
runs is how long they took.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from mini_satisfactory import data
from mini_satisfactory.sim import ACTION_INDEX, ACTIONS, Action, Simulation
from mini_satisfactory.sim import missing_for_tier, tier_progress
from mini_satisfactory.sim import upgrade_progress as sim_progress

POINTS_PER_UPGRADE = 10.0  # earned while crafting what the upgrade asks for
DELIVERY_POINTS = 5.0  # earned when it is delivered
INVALID_ACTION_PENALTY = 1.0
OBS_MAX = 10.0

ITEM_KEYS = tuple(data.ITEMS)
BUILD_ACTIONS = tuple(a for a in ACTIONS if a.kind == "build")
UNLOCK_KEYS = tuple(
    sorted({key for u in data.HUB_UPGRADES for key in u.unlocks} | set(data.STARTING_UNLOCKS))
)
RESOURCES = tuple(data.RAW_RESOURCES)
WAIT = ACTION_INDEX[Action("wait")]


def upgrade_progress(sim: Simulation) -> float:
    """Share (0 to 1) of the next HUB upgrade already in stock."""
    return sim_progress(sim.stock, sim.upgrades_done)


def potential(sim: Simulation) -> float:
    return (
        POINTS_PER_UPGRADE * tier_progress(sim.stock, sim.upgrades_done)
        + DELIVERY_POINTS * sim.upgrades_done
    )


def observe(sim: Simulation, time_limit_s: float) -> np.ndarray:
    """Everything the AI sees, as numbers roughly between 0 and 1."""
    upgrade = sim.next_upgrade()
    features: list[float] = []
    features += [math.log1p(max(0.0, sim.stock[i])) / math.log1p(200) for i in ITEM_KEYS]
    features += [sim.count(a.building, a.target) / 5 for a in BUILD_ACTIONS]
    features += [float(key in sim.unlocked) for key in UNLOCK_KEYS]
    features += [float(sim.upgrades_done == k) for k in range(len(data.HUB_UPGRADES) + 1)]
    for item in ITEM_KEYS:
        need = upgrade.cost.get(item, 0.0) if upgrade else 0.0
        features.append(need / 100)
        features.append(min(1.0, max(0.0, sim.stock[item]) / need) if need else 0.0)
    missing = missing_for_tier(sim.stock, sim.upgrades_done)
    features += [missing.get(item, 0.0) / 100 for item in ITEM_KEYS]
    demand = sim.power_demand()
    fuel_seconds = sim.stock["leaves"] * data.LEAF_MJ / demand if demand else 0.0
    features += [sim.power_capacity() / 20, demand / 20, min(2.0, fuel_seconds / 300)]
    features += [sim.free_nodes(r) / len(data.START_NODES[r]) for r in RESOURCES]
    manual = [m for m in sim.machines if not sim.connected(m)]
    for item in ITEM_KEYS:
        waiting = sum(m.outputs.get(item, 0.0) for m in manual)
        features.append(min(2.0, waiting / data.ITEMS[item].stack))
    inputs_fill = [
        m.inputs.get(i, 0.0) / data.ITEMS[i].stack
        for m in manual
        if not m.is_miner
        for i in data.RECIPES[m.target].inputs
    ]
    features.append(min(inputs_fill) if inputs_fill else 1.0)
    features.append(upgrade_progress(sim))
    features.append(tier_progress(sim.stock, sim.upgrades_done) / len(data.HUB_UPGRADES))
    features.append(min(1.0, sim.time / time_limit_s))
    return np.clip(np.asarray(features, dtype=np.float32), 0.0, OBS_MAX)


class MiniSatisfactoryEnv(gym.Env):
    """Finish Tier 0 (the six HUB upgrades) as fast as possible."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, time_limit_min: float = 180, render_mode: str | None = None):
        self.time_limit_s = time_limit_min * 60
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        size = len(observe(Simulation(), self.time_limit_s))
        self.observation_space = gym.spaces.Box(0.0, OBS_MAX, shape=(size,), dtype=np.float32)
        self.sim = Simulation()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim = Simulation(record=bool(options and options.get("record")))
        return observe(self.sim, self.time_limit_s), self._info()

    def step(self, action):
        action = int(action)
        before = potential(self.sim)
        penalty = 0.0
        if not self.sim.is_valid(ACTIONS[action]):
            action, penalty = WAIT, INVALID_ACTION_PENALTY
        seconds = self.sim.step(action)
        reward = potential(self.sim) - before - seconds / 60 - penalty
        terminated = self.sim.finished
        truncated = not terminated and self.sim.time >= self.time_limit_s
        return observe(self.sim, self.time_limit_s), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Which actions are possible right now (used by MaskablePPO)."""
        mask = np.asarray(self.sim.action_mask(), dtype=bool)
        if not mask.any():  # only once the run is over
            mask[WAIT] = True
        return mask

    def render(self):
        if self.render_mode == "ansi":
            return describe(self.sim)
        return None

    def _info(self) -> dict:
        return {
            "game_time_s": self.sim.time,
            "upgrades_done": self.sim.upgrades_done,
            "upgrade_times_s": list(self.sim.upgrade_times),
        }


def describe(sim: Simulation) -> str:
    """Short text summary of a game state."""
    minutes, seconds = divmod(int(sim.time), 60)
    stock = ", ".join(
        f"{data.label(i)} {q:.0f}" for i, q in sim.stock.items() if q >= 1
    )
    return (
        f"[{minutes:02d}:{seconds:02d}] upgrades {sim.upgrades_done}/{len(data.HUB_UPGRADES)}"
        f" | machines {len(sim.machines)} | stock: {stock or 'vide'}"
    )
