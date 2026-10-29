"""Teach the AI to walk across rough ground: go round cliffs, stop before drops, get unstuck.

    python train_walk.py                  # about 1.5 million steps (20 to 40 min on a PC)
    python train_walk.py --steps 300000   # quicker, less good
    python train_walk.py --evaluer        # only compare the saved walker with simple ones

Trial and error (PPO) in a simulated landscape (real_game/terrain_sim.py): hills,
plateaus with cliffs, ravines, rocks, with the same senses as in the real game.
The best walker is saved to models/marche.zip; real_game/agent.py then uses it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "real_game"))

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from terrain_sim import DT, WalkEnv, greedy_action  # noqa: E402
from walk_policy import MODEL_FILE  # noqa: E402

EVAL_SEED = 10_000  # evaluation landscapes, never seen while practising


def evaluate(policy, episodes: int = 100) -> dict:
    """Walk `episodes` landscapes; share reached, share with a fall, share dead, mean time."""
    env = WalkEnv()
    reached = fell = dead = 0
    times = []
    for k in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + k)
        done, had_fall = False, False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            had_fall |= info["fell"]
            done = terminated or truncated
        reached += info["reached"]
        dead += info["dead"]
        fell += had_fall
        if info["reached"]:
            times.append(env.steps * DT)
    return {"reached": reached / episodes, "fell": fell / episodes, "dead": dead / episodes,
            "time": float(np.mean(times)) if times else float("nan")}


def describe(name: str, result: dict) -> str:
    return (f"{name:<14} arrivé {100 * result['reached']:5.1f} %   chute {100 * result['fell']:5.1f} %   "
            f"mort {100 * result['dead']:5.1f} %   temps moyen {result['time']:5.1f} s")


def score(result: dict) -> float:
    return result["reached"] - result["fell"] - 2 * result["dead"]


def model_policy(model):
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


class KeepBest(BaseCallback):
    """Every so often, test the walker on the evaluation landscapes and keep the best one."""

    def __init__(self, every: int, episodes: int):
        super().__init__()
        self.every, self.episodes = every, episodes
        self.next_at = every
        self.best = -np.inf
        self.started = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            self.check()
        return True

    def check(self) -> None:
        result = evaluate(model_policy(self.model), self.episodes)
        minutes = (time.time() - self.started) / 60
        mark = ""
        if score(result) > self.best:
            self.best = score(result)
            MODEL_FILE.parent.mkdir(exist_ok=True)
            self.model.save(MODEL_FILE)
            mark = "  <- meilleur, enregistré"
        print(describe(f"{self.num_timesteps // 1000:>5} k pas", result) + f"   ({minutes:.0f} min){mark}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1_500_000, help="pas d'entraînement (0,2 s de marche chacun)")
    parser.add_argument("--evaluer", action="store_true", help="seulement évaluer models/marche.zip")
    parser.add_argument("--envs", type=int, default=8, help="paysages joués en parallèle")
    args = parser.parse_args()
    th.set_num_threads(1)  # small network: more threads only slow it down

    rng = np.random.default_rng(0)
    print("Pour comparer, sur 100 paysages :")
    print(describe("au hasard", evaluate(lambda obs: int(rng.integers(6)))))
    print(describe("règle simple", evaluate(greedy_action)))
    if args.evaluer:
        if not MODEL_FILE.exists():
            raise SystemExit("Pas encore de models/marche.zip : lance d'abord « python train_walk.py ».")
        print(describe("IA", evaluate(model_policy(PPO.load(MODEL_FILE, device="cpu")))))
        return

    print(f"\nEntraînement : {args.steps:,} pas sur {args.envs} paysages à la fois".replace(",", " "))
    env = make_vec_env(WalkEnv, n_envs=args.envs, seed=0)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
                policy_kwargs={"net_arch": [128, 128]}, device="cpu", seed=0, verbose=0)
    keep = KeepBest(every=max(args.steps // 10, 10_000), episodes=100)
    model.learn(total_timesteps=args.steps, callback=keep)
    keep.check()
    print(f"\nTerminé. Meilleur marcheur : {MODEL_FILE.relative_to(ROOT)}")
    print("Il sera utilisé par « python real_game/agent.py jouer » (avec la vision).")


if __name__ == "__main__":
    main()
