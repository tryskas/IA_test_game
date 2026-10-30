"""The walk learnt in the simulated landscape (train_walk.py), used in the real game.

Every 0.2 s it gets the same senses as in the simulation, built here from the 3D map
of the ground (navigation.HeightMap, scanned exactly as the simulation scans its
landscape) or, where that map knows nothing yet, from what the eyes see right now
(perception.View); plus what the map remembers, the next waypoint of the path,
health and being stuck. It picks one of terrain_sim.ACTIONS.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from perception import drop_distances, obstacle_distances
from terrain_sim import (BEARINGS, COLUMNS, FEATURES, RAY_STEPS, TRUST, edge_nearness, features, nearness,
                         recalled, report, scan)

KNOWN_AHEAD = 16  # samples (8 m) along a column...
KNOWN_SHARE = 0.5  # ...this share of which the 3D map must know, to be trusted over the view

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "marche.zip"


def height_senses(heights, x: float, y: float, z: float, facing: float):
    """What the 3D map shows ahead, column by column, scanned as the simulation does.
    Returns (columns the map knows well enough, distances, drops)."""
    angles = np.radians(facing + BEARINGS)
    ground = heights.sample(x + np.cos(angles)[:, None] * RAY_STEPS, y + np.sin(angles)[:, None] * RAY_STEPS)
    known = np.isfinite(ground[:, :KNOWN_AHEAD]).mean(axis=1) >= KNOWN_SHARE
    # Where unknown, the last height known along the ray carries on (nothing to report there)
    valid = np.isfinite(ground)
    last = np.maximum.accumulate(np.where(valid, np.arange(ground.shape[1])[None, :], -1), axis=1)
    filled = np.where(last >= 0, ground[np.arange(ground.shape[0])[:, None], np.maximum(last, 0)], z)
    distances, drops = report(*scan(filled.astype(np.float32), z))
    return known, distances, drops


def observe(view, pitch: float, terrain, x: float, y: float, facing: float, target: tuple[float, float],
            hp: float, stuck: bool, previous: int, z: float | None = None) -> np.ndarray:
    """The walking policy's senses in the real game. `view` may be None (nothing seen); `z`, the
    player's height, lets the 3D map of the ground be used."""
    distances: list[float | None] = [None] * COLUMNS
    drops: list[float | None] = [None] * COLUMNS
    if view is not None and len(view.obstacle_rows) == COLUMNS:
        distances = obstacle_distances(view, pitch, TRUST)
        drops = (drop_distances(view, pitch) + [None] * COLUMNS)[:COLUMNS]
    heights = getattr(terrain, "heights", None)
    if z is not None and heights is not None and len(heights):
        known, mapped, mapped_drops = height_senses(heights, x, y, z, facing)
        for i in np.flatnonzero(known):
            distances[i], drops[i] = mapped[i], mapped_drops[i]
    bearing = math.degrees(math.atan2(target[1] - y, target[0] - x)) - facing
    return features([d is not None for d in distances], [nearness(d) for d in distances],
                    [edge_nearness(d) for d in drops],
                    recalled(terrain, x, y, facing), (bearing + 180) % 360 - 180,
                    math.hypot(target[0] - x, target[1] - y), hp, stuck, previous)


class WalkPolicy:
    """Picks how to walk, from the senses. Load it with WalkPolicy.load()."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> "WalkPolicy | None":
        """The trained walker, or None if `python train_walk.py` hasn't been run yet."""
        if not path.exists():
            return None
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        if model.observation_space.shape != (FEATURES,):
            raise SystemExit("models/marche.zip date d'une autre version : relance « python train_walk.py »")
        return cls(model)

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
