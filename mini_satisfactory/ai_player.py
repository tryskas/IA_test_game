"""The trained AI, wrapped so it plays like the reference bots."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mini_satisfactory.env import observe
from mini_satisfactory.sim import Simulation

DEFAULT_MODEL = Path("models/ia_tier0.zip")


class AIPlayer:
    def __init__(self, model_path: Path = DEFAULT_MODEL, time_limit_s: float = 3 * 3600, model=None):
        if model is None:
            from sb3_contrib import MaskablePPO  # heavy import, only when the AI is used

            model = MaskablePPO.load(model_path, device="cpu")
        self.model = model
        self.time_limit_s = time_limit_s

    def act(self, sim: Simulation) -> int:
        obs = observe(sim, self.time_limit_s)
        mask = np.asarray(sim.action_mask(), dtype=bool)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
