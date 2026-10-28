"""Watch the AI play in a window, racing the reference bots.

    python watch.py                     # the AI (if trained) and the bots
    python watch.py --vitesse 30        # 30 seconds of game per real second
    python watch.py --suivre usine      # start on the scripted factory bot

Each run is computed first (well under a second), then played on screen at the
chosen speed. In the window: Space pause, left/right arrows jump 30 s, up/down
arrows change speed, keys 1 to 4 pick the player to follow, Esc quits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mini_satisfactory.ai_player import DEFAULT_MODEL, AIPlayer
from mini_satisfactory.bots import FACTORY_PLAN, SIMPLE_PLAN, RandomBot, ScriptedBot
from mini_satisfactory.replay import record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modele", type=Path, default=DEFAULT_MODEL, help="modèle de l'IA entraînée")
    parser.add_argument("--vitesse", type=int, default=60, help="secondes de jeu par seconde réelle")
    parser.add_argument("--suivre", default=None, help="joueur affiché au départ : IA, usine, script ou hasard")
    parser.add_argument("--time-limit", type=float, default=180, help="durée max d'une partie (minutes de jeu)")
    parser.add_argument("--capture", metavar="IMAGE.png", help="enregistrer une image au lieu d'ouvrir la fenêtre")
    parser.add_argument("--a", type=float, default=10, metavar="MINUTES", help="moment de la capture (minutes de jeu)")
    args = parser.parse_args()
    limit_s = args.time_limit * 60

    players = []
    if args.modele.exists():
        players.append(("IA", AIPlayer(args.modele, limit_s)))
    else:
        print(f"Pas d'IA entraînée dans {args.modele} (lance train.py) : seuls les bots jouent.")
    players += [
        ("usine", ScriptedBot(FACTORY_PLAN)),
        ("script", ScriptedBot(SIMPLE_PLAN)),
        ("hasard", RandomBot(seed=0)),
    ]
    print("Calcul des parties...")
    runs = [record(player, name, limit_s) for name, player in players]
    names = [r.name.lower() for r in runs]
    start = names.index(args.suivre.lower()) if args.suivre and args.suivre.lower() in names else 0

    from mini_satisfactory import viewer  # pygame is only needed here

    if args.capture:
        viewer.capture(runs, args.capture, args.a * 60, start)
        print(f"Image enregistrée : {args.capture}")
        return
    viewer.show(runs, start, args.vitesse)


if __name__ == "__main__":
    main()
