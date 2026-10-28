"""The AI plays the real game.

    python real_game/agent.py calibrer        # once: measure the mouse and the walk (needs ~15 m of flat ground)
    python real_game/agent.py aller fer       # walk to the nearest free iron node
    python real_game/agent.py miner fer 10    # walk there and mine 10 iron ore by hand
    python real_game/agent.py jouer           # the AI plays Tier 0

Resources: fer, cuivre, calcaire. Each command waits 5 seconds so you can click
in the game. F12 stops at once; so does leaving the game window. When the AI
wants something it cannot do by itself yet, it asks you to do it and waits.
"""
from __future__ import annotations

import argparse
import time
import urllib.error

from game_io import RealGame, Stopped
from game_state import ROOT, read_player, read_state, to_simulation
from moves import Calibration, Mover, SkillError, calibrate

RESOURCES = {"fer": "iron_ore", "cuivre": "copper_ore", "calcaire": "limestone"}
COUNTDOWN = 5


def countdown(seconds: int = COUNTDOWN) -> None:
    print(f"Clique dans le jeu. Début dans {seconds} s (F12 pour tout arrêter).")
    for left in range(seconds, 0, -1):
        print(f"  {left}...")
        time.sleep(1)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def ask_human(game: RealGame, what: str) -> bool:
    """Let the player do something by hand; returns False to stop."""
    game.release_all()
    print(f"[{stamp()}] Je ne sais pas encore faire : « {what} ».")
    answer = input("  Fais-le dans le jeu, puis reviens ici et appuie sur Entrée (q puis Entrée pour arrêter) : ")
    if answer.strip().lower() == "q":
        return False
    countdown(3)
    return True


def play(game: RealGame, mover: Mover, model: str) -> None:
    from mini_satisfactory import data
    from mini_satisfactory.ai_player import AIPlayer
    from mini_satisfactory.sim import ACTIONS

    ai = AIPlayer(model)
    while True:
        state = read_state()
        if state.upgrades_done >= len(data.HUB_UPGRADES):
            print(f"[{stamp()}] Tier 0 terminé !")
            return
        action = ACTIONS[ai.act(to_simulation(state))]
        print(f"[{stamp()}] L'IA : {action.label()}")
        try:
            if action.kind == "mine":
                mined = mover.mine_by_hand(action.target, data.HAND_MINE_AMOUNT)
                print(f"  {mined} minés")
            elif action.kind == "wait":
                game.wait(data.WAIT_SECONDS)
            elif not ask_human(game, action.label()):
                return
        except SkillError as problem:
            if not ask_human(game, f"{action.label()} ({problem})"):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commande", choices=["calibrer", "aller", "miner", "jouer"])
    parser.add_argument("ressource", nargs="?", choices=sorted(RESOURCES))
    parser.add_argument("quantite", nargs="?", type=int, default=10)
    parser.add_argument("--modele", default=str(ROOT / "models" / "ia_tier0.zip"))
    args = parser.parse_args()
    if args.commande in ("aller", "miner") and args.ressource is None:
        parser.error("indique la ressource : fer, cuivre ou calcaire")

    try:
        read_player()
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        raise SystemExit(
            "Le mod ne répond pas. Lance le jeu depuis le Mod Manager, charge ta partie "
            "et tape dans le chat du jeu : /frm http start"
        )
    game = RealGame()
    countdown()
    try:
        if args.commande == "calibrer":
            cal = calibrate(game)
            cal.save()
            print(f"Calibrage enregistré : {cal}")
            return
        mover = Mover(game, Calibration.load())
        if args.commande == "aller":
            node = mover.go_to_node(RESOURCES[args.ressource])
            print(f"Arrivé au gisement ({node.purity}).")
        elif args.commande == "miner":
            mined = mover.mine_by_hand(RESOURCES[args.ressource], args.quantite)
            print(f"Terminé : {mined} minés.")
        else:
            play(game, mover, args.modele)
    except Stopped as reason:
        print(f"Arrêt : {reason}.")
    except SkillError as problem:
        print(f"Échec : {problem}.")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        print("Arrêt : le mod ne répond plus (jeu fermé ou serveur arrêté).")
    finally:
        game.release_all()


if __name__ == "__main__":
    main()
