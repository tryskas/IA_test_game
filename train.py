"""Train the AI to finish Tier 0 as fast as possible.

    python train.py                      # imitate the factory bot, then practise alone
    python train.py --sans-imitation     # learn from scratch, by trial and error only
    python train.py --steps 200000       # quick test

1. Imitation: the AI watches the scripted factory bot play (with some random
   detours so it also sees how to recover) and learns to pick the same actions.
2. Practice: reinforcement learning (MaskablePPO). The AI plays thousands of
   runs and keeps what makes it faster, to beat the bot it copied.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from mini_satisfactory.ai_player import DEFAULT_MODEL
from mini_satisfactory.bots import FACTORY_PLAN, ScriptedBot
from mini_satisfactory.env import MiniSatisfactoryEnv

GAMMA = 0.995


def spaced(number: int) -> str:
    return f"{number:,}".replace(",", " ")


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


# ---------------------------------------------------------------------- imitation


@dataclass
class Demonstrations:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    returns: np.ndarray  # discounted points from each moment on, to pre-train the critic


def watch_the_bot(episodes: int, detours: float, seed: int, time_limit_min: float) -> Demonstrations:
    """Record what the factory bot does in every situation.

    In all runs but the first, a share `detours` of the actions are random, so
    the AI also sees situations the bot never gets into by itself, and the bot's
    answer to them.
    """
    env = MiniSatisfactoryEnv(time_limit_min=time_limit_min)
    bot = ScriptedBot(FACTORY_PLAN)
    rng = np.random.default_rng(seed)
    obs_list, masks, actions, returns = [], [], [], []
    for episode in range(episodes):
        noise = 0.0 if episode == 0 else detours
        obs, _ = env.reset()
        rewards, done = [], False
        while not done:
            mask = env.action_masks()
            wanted = bot.act(env.sim)
            played = wanted if rng.random() >= noise else int(rng.choice(np.flatnonzero(mask)))
            obs_list.append(obs)
            masks.append(mask)
            actions.append(wanted)
            obs, reward, terminated, truncated, _ = env.step(played)
            rewards.append(reward)
            done = terminated or truncated
        future = 0.0
        episode_returns = []
        for reward in reversed(rewards):
            future = reward + GAMMA * future
            episode_returns.append(future)
        returns.extend(reversed(episode_returns))
    return Demonstrations(
        np.asarray(obs_list, dtype=np.float32), np.asarray(masks, dtype=bool),
        np.asarray(actions, dtype=np.int64), np.asarray(returns, dtype=np.float32),
    )


def imitate(model: MaskablePPO, demos: Demonstrations, epochs: int = 40, batch_size: int = 256) -> float:
    """Train the policy to copy the demonstrations; return the share of actions it copies."""
    policy = model.policy
    device = policy.device
    obs = th.as_tensor(demos.obs, device=device)
    masks = th.as_tensor(demos.masks, device=device)
    actions = th.as_tensor(demos.actions, device=device)
    returns = th.as_tensor(demos.returns, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = th.randperm(len(obs), device=device)
        for start in range(0, len(obs), batch_size):
            batch = order[start:start + batch_size]
            dist = policy.get_distribution(obs[batch], action_masks=masks[batch])
            values = policy.predict_values(obs[batch]).flatten()
            loss = -dist.log_prob(actions[batch]).mean() + th.nn.functional.mse_loss(values, returns[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    policy.set_training_mode(False)
    with th.no_grad():
        dist = policy.get_distribution(obs, action_masks=masks)
        copied = (dist.distribution.probs.argmax(dim=1) == actions).float().mean().item()
    return copied


# ---------------------------------------------------------------------- practice


class ProgressCallback(BaseCallback):
    """Every `every` steps, plays one run, prints it and keeps the best model."""

    def __init__(self, every: int, save_path: Path, time_limit_min: float):
        super().__init__()
        self.every = every
        self.save_path = save_path
        self.env = MiniSatisfactoryEnv(time_limit_min=time_limit_min)
        self.best: tuple[int, float] | None = None  # (upgrades, -seconds): higher is better
        self.started = time.time()

    def evaluate(self, model: MaskablePPO, label: str) -> None:
        info = run_episode(model, self.env)
        score = (info["upgrades_done"], -info["game_time_s"])
        mark = ""
        if self.best is None or score > self.best:
            self.best = score
            model.save(self.save_path)
            mark = "  <- nouveau record, sauvegardé"
        elapsed = (time.time() - self.started) / 60
        print(f"{label:>22} ({elapsed:4.1f} min réelles) | {format_run(info)}{mark}", flush=True)

    def _on_step(self) -> bool:
        before = self.num_timesteps - self.training_env.num_envs
        if self.num_timesteps // self.every != before // self.every:
            self.evaluate(self.model, f"{spaced(self.num_timesteps)} pas")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1_000_000, help="nombre d'actions jouées pendant l'entraînement")
    parser.add_argument("--sans-imitation", action="store_true", help="apprendre de zéro, sans regarder le bot")
    parser.add_argument("--envs", type=int, default=8, help="parties jouées en parallèle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=180, help="durée max d'une partie (minutes de jeu)")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="fichier du modèle sauvegardé")
    parser.add_argument("--eval-every", type=int, default=50_000)
    args = parser.parse_args()

    imitation = not args.sans_imitation
    th.set_num_threads(1)  # the network is tiny: one thread is faster than several
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
        # After imitation, smaller and more careful steps so practice refines
        # what was copied instead of wiping it out.
        learning_rate=1e-4 if imitation else 3e-4,
        clip_range=0.1 if imitation else 0.2,
        ent_coef=0.003 if imitation else 0.01,
        gamma=GAMMA,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    callback = ProgressCallback(args.eval_every, args.out, args.time_limit)
    if imitation:
        print("Étape 1 : l'IA regarde le script usine jouer...")
        demos = watch_the_bot(episodes=60, detours=0.15, seed=args.seed, time_limit_min=args.time_limit)
        copied = imitate(model, demos)
        print(f"  {spaced(len(demos.actions))} situations vues, l'IA choisit comme le script dans {copied:.0%} des cas")
        callback.evaluate(model, "après imitation")
    print(f"Étape 2 : l'IA s'entraîne seule sur {spaced(args.steps)} actions ({args.envs} parties en parallèle)...")
    model.learn(total_timesteps=args.steps, callback=callback)
    print(f"Terminé. Meilleur modèle : {args.out}")


if __name__ == "__main__":
    main()
