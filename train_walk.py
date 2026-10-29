"""Teach the AI to walk across rough ground: go round cliffs, stop before drops, get unstuck.

    python train_walk.py                  # about 3 million steps (around an hour on a PC)
    python train_walk.py --steps 1000000  # quicker, less good
    python train_walk.py --evaluer        # only compare the saved walker with simple ones
    python train_walk.py --continuer      # go on training the saved walker instead of starting over

Trial and error (PPO) in simulated landscapes (real_game/landscapes.py): general ones
(hills, plateaus, ravines, rocks, trees) and ones with a trap (a wall with a gap, a
dead end, a cliff with one way down, a canyon with a bridge, a forest, terraces),
with the same senses as in the real game. The best walker is saved to
models/marche.zip; real_game/agent.py then uses it.
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

from landscapes import KINDS  # noqa: E402
from terrain_sim import ACTIONS, DT, WalkEnv, greedy_action  # noqa: E402
from walk_policy import MODEL_FILE  # noqa: E402

EVAL_SEED = 10_000  # evaluation landscapes, never seen while practising


def evaluate(policy, episodes: int = 160) -> dict:
    """Walk `episodes` landscapes (each kind in turn); share reached, with a fall, dead, mean time,
    and per kind of landscape: share reached and share dead."""
    env = WalkEnv()
    reached = fell = dead = 0
    times = []
    by_kind = {kind: [0, 0, 0] for kind in KINDS}  # kind: [walks, reached, dead]
    for k in range(episodes):
        kind = list(KINDS)[k % len(KINDS)]
        obs, _ = env.reset(seed=EVAL_SEED + k, options={"kind": kind})
        done, had_fall = False, False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            had_fall |= info["fell"]
            done = terminated or truncated
        reached += info["reached"]
        dead += info["dead"]
        fell += had_fall
        by_kind[env.kind][0] += 1
        by_kind[env.kind][1] += info["reached"]
        by_kind[env.kind][2] += info["dead"]
        if info["reached"]:
            times.append(env.steps * DT)
    return {"reached": reached / episodes, "fell": fell / episodes, "dead": dead / episodes,
            "time": float(np.mean(times)) if times else float("nan"),
            "kinds": {kind: hits / max(walks, 1) for kind, (walks, hits, _) in by_kind.items()},
            "deaths": {kind: lost / max(walks, 1) for kind, (walks, _, lost) in by_kind.items()}}


def describe(name: str, result: dict) -> str:
    return (f"{name:<14} arrivé {100 * result['reached']:5.1f} %   chute {100 * result['fell']:5.1f} %   "
            f"mort {100 * result['dead']:5.1f} %   temps moyen {result['time']:5.1f} s")


def describe_kinds(results: dict[str, dict]) -> str:
    """A table: one row per kind of landscape, one column per walker (share reached), and how
    often the last walker died there."""
    last = list(results)[-1]
    lines = [f"{'Arrivées par paysage':<26}" + "".join(f"{name:>14}" for name in results) + f"{'morts ' + last:>16}"]
    for kind, (label, _) in KINDS.items():
        lines.append(f"  {label:<24}" + "".join(f"{100 * r['kinds'][kind]:>13.0f}%" for r in results.values())
                     + f"{100 * results[last]['deaths'][kind]:>15.0f}%")
    return "\n".join(lines)


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
    parser.add_argument("--steps", type=int, default=3_000_000, help="pas d'entraînement (0,2 s de marche chacun)")
    parser.add_argument("--evaluer", action="store_true", help="seulement évaluer models/marche.zip")
    parser.add_argument("--envs", type=int, default=8, help="paysages joués en parallèle")
    parser.add_argument("--continuer", action="store_true",
                        help="reprendre l'entraînement de models/marche.zip au lieu de repartir de zéro")
    args = parser.parse_args()
    th.set_num_threads(1)  # small network: more threads only slow it down

    rng = np.random.default_rng(0)
    print("Pour comparer, sur 160 paysages (20 de chaque sorte) :")
    simple = {"au hasard": evaluate(lambda obs: int(rng.integers(len(ACTIONS)))), "règle simple": evaluate(greedy_action)}
    for name, result in simple.items():
        print(describe(name, result))
    if args.evaluer:
        if not MODEL_FILE.exists():
            raise SystemExit("Pas encore de models/marche.zip : lance d'abord « python train_walk.py ».")
        simple["IA"] = evaluate(model_policy(PPO.load(MODEL_FILE, device="cpu")))
        print(describe("IA", simple["IA"]))
        print("\n" + describe_kinds(simple))
        return

    print(f"\nEntraînement : {args.steps:,} pas sur {args.envs} paysages à la fois".replace(",", " "))
    env = make_vec_env(WalkEnv, n_envs=args.envs, seed=0)
    keep = KeepBest(every=max(args.steps // 12, 10_000), episodes=160)
    if args.continuer and MODEL_FILE.exists():
        model = PPO.load(MODEL_FILE, env=env, device="cpu")
        if model.observation_space.shape != env.observation_space.shape:
            raise SystemExit("models/marche.zip date d'une autre version : relance sans --continuer.")
        start = evaluate(model_policy(model))
        keep.best = score(start)  # only keep what does better
        print(describe("départ", start))
    else:
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
                    policy_kwargs={"net_arch": [128, 128]}, device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=args.steps, callback=keep)
    keep.check()
    simple["IA"] = evaluate(model_policy(PPO.load(MODEL_FILE, device="cpu")))
    print("\n" + describe_kinds(simple))
    print(f"\nTerminé. Meilleur marcheur : {MODEL_FILE.relative_to(ROOT)}")
    print("Il sera utilisé par « python real_game/agent.py jouer » (avec la vision).")


if __name__ == "__main__":
    main()
