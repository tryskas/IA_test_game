"""The walk learnt in the simulated landscape (train_walk.py), used in the real game.

Every 0.2 s it gets the same senses as in the simulation, built here from what the
eyes see (perception.View), what the map remembers (navigation.TerrainMap) and
the next waypoint of the path, and picks one of terrain_sim.ACTIONS.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from perception import drop_distances, obstacle_distances
from terrain_sim import COLUMNS, FEATURES, TRUST, edge_nearness, features, nearness, recalled

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "marche.zip"


def observe(view, pitch: float, terrain, x: float, y: float, facing: float, target: tuple[float, float],
            hp: float, stuck: bool, previous: int) -> np.ndarray:
    """The walking policy's senses in the real game. `view` may be None (nothing seen)."""
    distances: list[float | None] = [None] * COLUMNS
    drops: list[float | None] = [None] * COLUMNS
    if view is not None and len(view.obstacle_rows) == COLUMNS:
        distances = obstacle_distances(view, pitch, TRUST)
        drops = (drop_distances(view, pitch) + [None] * COLUMNS)[:COLUMNS]
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
