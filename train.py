"""Train the AI to finish Tier 0 as fast as possible (MaskablePPO).

    python train.py                  # about 1 million steps
    python train.py --steps 200000   # quick test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from mini_satisfactory.env import MiniSatisfactoryEnv

DEFAULT_MODEL = Path("models/ia_tier0.zip")


def run_episode(model, env: MiniSatisfactoryEnv) -> dict:
    """One full run with the AI always picking its preferred action."""
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def format_run(info: dict, total_upgrades: int = 6) -> str:
    minutes = info["game_time_s"] / 60
    if info["upgrades_done"] >= total_upgrades:
        return f"Tier 0 fini en {minutes:5.1f} min"
    return f"{info['upgrades_done']}/{total_upgrades} upgrades en {minutes:5.1f} min"


class ProgressCallback(BaseCallback):
    """Every `every` steps, plays one run, prints it and keeps the best model."""

    def __init__(self, every: int, save_path: Path, time_limit_min: float):
        super().__init__()
        self.every = every
        self.save_path = save_path
        self.env = MiniSatisfactoryEnv(time_limit_min=time_limit_min)
        self.best: tuple[int, float] | None = None  # (upgrades, -minutes): higher is better
        self.started = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps // self.every == (self.num_timesteps - self.training_env.num_envs) // self.every:
            return True
        info = run_episode(self.model, self.env)
        score = (info["upgrades_done"], -info["game_time_s"])
        mark = ""
        if self.best is None or score > self.best:
            self.best = score
            self.model.save(self.save_path)
            mark = "  <- nouveau record, sauvegardé"
        elapsed = time.time() - self.started
        print(
            f"{self.num_timesteps:>9,} pas ({elapsed / 60:4.1f} min réelles) | "
            f"{format_run(info)}{mark}".replace(",", " "),
            flush=True,
        )
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1_000_000, help="nombre d'actions jouées pendant l'entraînement")
    parser.add_argument("--envs", type=int, default=8, help="parties jouées en parallèle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=180, help="durée max d'une partie (minutes de jeu)")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="fichier du modèle sauvegardé")
    parser.add_argument("--eval-every", type=int, default=50_000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(
        MiniSatisfactoryEnv, n_envs=args.envs, seed=args.seed,
        env_kwargs={"time_limit_min": args.time_limit},
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    np.set_printoptions(precision=2)
    print(f"Entraînement sur {args.steps:,} pas, {args.envs} parties en parallèle...".replace(",", " "))
    callback = ProgressCallback(args.eval_every, args.out, args.time_limit)
    model.learn(total_timesteps=args.steps, callback=callback)
    print(f"Terminé. Meilleur modèle : {args.out}")


if __name__ == "__main__":
    main()
