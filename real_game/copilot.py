"""The AI as a copilot: it watches your game and says what to do next.

    python real_game/copilot.py              # follows the game, prints each new piece of advice
    python real_game/copilot.py --une-fois   # one reading, then stops

You play. Every 2 seconds the copilot reads your inventory and your HUB
progress through the Ficsit Remote Monitoring mod, shows them to the AI
trained in the mini-game, and prints the next action of its speedrun.
"""
from __future__ import annotations

import argparse
import time
import urllib.error

from game_state import ROOT, GameState, read_state, to_simulation
from mini_satisfactory import data
from mini_satisfactory.ai_player import DEFAULT_MODEL, AIPlayer
from mini_satisfactory.sim import ACTIONS
from moves import CALIBRATION_FILE, Calibration, heading, wrap

PURITY_LABELS = {"impure": "impur", "normal": "normal", "pure": "pur"}


def progress_lines(state: GameState) -> list[str]:
    if state.upgrades_done >= len(data.HUB_UPGRADES):
        return ["Tier 0 terminé !"]
    upgrade = data.HUB_UPGRADES[state.upgrades_done]
    parts = []
    for item, qty in upgrade.cost.items():
        have = state.inventory.get(item, 0) + state.deposited.get(item, 0)
        parts.append(f"{data.label(item)} {min(have, qty):.0f}/{qty:.0f}")
    return [f"{upgrade.name} ({state.upgrades_done}/{len(data.HUB_UPGRADES)} livrés) : " + ", ".join(parts)]


def where_to_go(state: GameState, resource: str) -> str:
    node = state.nearest(resource)
    if node is None:
        return ""
    # The mod's yaw is not the map direction: the calibration knows the difference.
    cal = Calibration.load() if CALIBRATION_FILE.exists() else Calibration()
    turn = wrap(heading(state.position[0], state.position[1], node.x, node.y) - cal.facing(state.yaw))
    side = "droite" if turn > 0 else "gauche"
    direction = "droit devant" if abs(turn) < 10 else f"tourne de {abs(turn):.0f}° vers la {side}"
    return f" | gisement {PURITY_LABELS[node.purity]} le plus proche à {state.distance_to(node):.0f} m, {direction}"


def advice(state: GameState, ai: AIPlayer) -> str:
    action = ACTIONS[ai.act(to_simulation(state))]
    text = action.label()
    if action.kind == "mine" or (action.kind == "build" and action.target in data.RAW_RESOURCES):
        text += where_to_go(state, action.target)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--une-fois", action="store_true", help="une seule lecture, puis arrêt")
    parser.add_argument("--modele", default=str(ROOT / DEFAULT_MODEL))
    args = parser.parse_args()

    ai = AIPlayer(args.modele)
    print("Copilote prêt. Joue normalement ; Ctrl+C pour arrêter.\n")
    last = None
    while True:
        try:
            state = read_state()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            raise SystemExit("Le mod ne répond pas : charge ta partie et tape /frm http start dans le chat du jeu.")
        lines = progress_lines(state)
        tip = advice(state, ai)
        if (lines, tip) != last:
            stamp = time.strftime("%H:%M:%S")
            for line in lines:
                print(f"[{stamp}] {line}")
            print(f"[{stamp}] >>> Conseil de l'IA : {tip}\n")
            last = (lines, tip)
        if args.une_fois:
            return
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Copilote arrêté.")
