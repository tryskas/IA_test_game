"""Watch a player do a run, or compare the players.

    python play.py                    # compare: random, the two scripted bots, trained AI
    python play.py --joueur ia        # replay the AI's run step by step
    python play.py --joueur usine     # same for the scripted bot that builds a factory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mini_satisfactory import data
from mini_satisfactory.bots import FACTORY_PLAN, SIMPLE_PLAN, RandomBot, ScriptedBot, play
from mini_satisfactory.env import observe
from mini_satisfactory.sim import Simulation

DEFAULT_MODEL = Path("models/ia_tier0.zip")


class AIPlayer:
    """Wraps a trained model so it plays like the bots."""

    def __init__(self, model_path: Path, time_limit_s: float):
        from sb3_contrib import MaskablePPO  # only needed to replay the AI

        self.model = MaskablePPO.load(model_path, device="cpu")
        self.time_limit_s = time_limit_s

    def act(self, sim: Simulation) -> int:
        obs = observe(sim, self.time_limit_s)
        mask = np.asarray(sim.action_mask(), dtype=bool)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def clock(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{sec:02d}"


def summary(sim: Simulation) -> str:
    total = len(data.HUB_UPGRADES)
    if sim.finished:
        result = f"Tier 0 fini en {sim.time / 60:.1f} min"
    else:
        result = f"{sim.upgrades_done}/{total} upgrades après {sim.time / 60:.0f} min"
    splits = " ".join(f"{t / 60:.1f}" for t in sim.upgrade_times)
    return f"{result:<30} | upgrades à (min) : {splits or '-'}"


def timeline(sim: Simulation) -> None:
    """Print the run, grouping repeated actions on one line."""
    groups: list[list] = []
    for start, action in sim.history:
        if groups and groups[-1][1] == action and action.kind != "submit":
            groups[-1][2] += 1
        else:
            groups.append([start, action, 1])
    submitted = 0
    for start, action, count in groups:
        if action.kind == "submit":
            name = data.HUB_UPGRADES[submitted].name
            print(f"[{clock(sim.upgrade_times[submitted])}] *** {name} livré ***")
            submitted += 1
            continue
        times = f" ×{count}" if count > 1 else ""
        print(f"[{clock(start)}] {action.label()}{times}")
    print()
    print("Machines à la fin :")
    counts: dict[str, int] = {}
    for m in sim.machines:
        key = f"{data.BUILDINGS[m.building].label} ({data.label(m.target)})"
        counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items()):
        print(f"  {n} × {key}")
    if not counts:
        print("  aucune")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--joueur", choices=["hasard", "script", "usine", "ia"], help="rejouer la partie d'un seul joueur"
    )
    parser.add_argument("--modele", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--time-limit", type=float, default=180, help="durée max d'une partie (minutes de jeu)")
    args = parser.parse_args()
    limit_s = args.time_limit * 60

    players = {
        "hasard": lambda: RandomBot(seed=0),
        "script": lambda: ScriptedBot(SIMPLE_PLAN),
        "usine": lambda: ScriptedBot(FACTORY_PLAN),
    }
    if args.modele.exists():
        players["ia"] = lambda: AIPlayer(args.modele, limit_s)
    elif args.joueur == "ia":
        parser.error(f"pas de modèle entraîné dans {args.modele} : lance d'abord train.py")

    if args.joueur:
        sim = play(players[args.joueur](), limit_s, record=True)
        timeline(sim)
        print()
        print(summary(sim))
        return

    for name, make in players.items():
        print(f"{name:<7} | {summary(play(make(), limit_s))}")
    if "ia" not in players:
        print("(pas encore d'IA entraînée : lance train.py)")


if __name__ == "__main__":
    main()
