"""The AI plays the real game.

    python real_game/agent.py calibrer                  # once: measure the mouse and the walk (~15 m of flat ground)
    python real_game/agent.py jouer                     # the AI plays, from the very start of a game

Single skills, to test them one by one:
    python real_game/agent.py equiper                   # equip the Xeno-Zapper (start of the tutorial)
    python real_game/agent.py construire hub            # also: etabli, atelier
    python real_game/agent.py aller fer                 # walk to the nearest free iron node
    python real_game/agent.py miner fer 10              # mine 10 iron ore by hand (also: cuivre, calcaire)
    python real_game/agent.py fabriquer "Barre de fer" 5

Each command waits 5 seconds so you can click in the game. F12 stops at once;
so does leaving the game window. While some skills are still missing, the AI
asks you to do those actions by hand and waits.
"""
from __future__ import annotations

import argparse
import time
import urllib.error

from common import find_game_window
from game_io import RealGame, Stopped
from game_state import ROOT, fetch, read_player, read_state, to_simulation
from menus import Memory, Menus
from moves import Calibration, Mover, SkillError, calibrate
from screen import Screen

RESOURCES = {"fer": "iron_ore", "cuivre": "copper_ore", "calcaire": "limestone"}
# Buildings of the start: in-game name, cost, name of the spot remembered to use it.
BUILDINGS = {
    "hub": ("Le H.U.B.", None, None),
    "etabli": ("Établi", {"iron_plate": 3, "iron_rod": 3}, "workbench"),
    "atelier": ("Atelier d'équipement", {"iron_plate": 6, "iron_rod": 4}, "workshop"),
}
COUNTDOWN = 5


def countdown(seconds: int = COUNTDOWN) -> None:
    print(f"Clique dans le jeu. Début dans {seconds} s (F12 pour tout arrêter).")
    for left in range(seconds, 0, -1):
        print(f"  {left}...")
        time.sleep(1)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def hub_id() -> str | None:
    terminals = fetch("getHUBTerminal")
    return terminals[0].get("ID") if terminals else None


def build(menus: Menus, key: str) -> None:
    name, cost, spot = BUILDINGS[key]
    done = (lambda: hub_id() is not None) if key == "hub" else menus.spend_check(cost)
    print(f"[{stamp()}] Construction : {name}")
    menus.build(name, done, spot)


def start_of_game(menus: Menus, memory: Memory) -> None:
    """The tutorial's first steps, done only if needed."""
    if menus.equip_xeno_zapper():
        print(f"[{stamp()}] Repousse-xénos équipé")
    if hub_id() is None:
        build(menus, "hub")
    memory.check_save(hub_id())
    if "workbench" not in memory.spots:
        build(menus, "etabli")


def ask_human(game: RealGame, what: str) -> bool:
    """Until every skill exists: let the player do it by hand. Returns False to stop."""
    game.release_all()
    print(f"[{stamp()}] Pas encore automatisé : « {what} ».")
    answer = input("  Fais-le dans le jeu, puis reviens ici et appuie sur Entrée (q puis Entrée pour arrêter) : ")
    if answer.strip().lower() == "q":
        return False
    countdown(3)
    return True


def play(game: RealGame, mover: Mover, menus: Menus, memory: Memory, model: str) -> None:
    from mini_satisfactory import data
    from mini_satisfactory.ai_player import AIPlayer
    from mini_satisfactory.sim import ACTIONS

    ai = AIPlayer(model)
    start_of_game(menus, memory)
    while True:
        state = read_state()
        if state.upgrades_done >= len(data.HUB_UPGRADES):
            print(f"[{stamp()}] Tier 0 terminé !")
            return
        sim = to_simulation(state)
        if "workshop" in memory.spots:
            sim.structures.add("workshop")
        action = ACTIONS[ai.act(sim)]
        print(f"[{stamp()}] L'IA : {action.label()}")
        try:
            if action.kind == "mine":
                print(f"  {mover.mine_by_hand(action.target, data.HAND_MINE_AMOUNT)} minés")
            elif action.kind == "craft" and data.RECIPES[action.target].needs is None:
                recipe = data.RECIPES[action.target]
                product, per_craft = next(iter(recipe.outputs.items()))
                batch = max(1, int(data.CRAFT_BATCH_SECONDS // recipe.handcraft_seconds))
                count = min(batch, sim.craftable(action.target)) * per_craft
                made = menus.craft(data.label(product), product, int(count))
                print(f"  {made} fabriqués")
            elif action.kind == "build" and action.building == "workshop":
                build(menus, "atelier")
            elif action.kind == "wait":
                game.wait(data.WAIT_SECONDS)
            elif not ask_human(game, action.label()):
                return
        except SkillError as problem:
            print(f"  échec : {problem}")
            if not ask_human(game, action.label()):
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commande", choices=["calibrer", "jouer", "equiper", "construire", "aller", "miner", "fabriquer"])
    parser.add_argument("quoi", nargs="?", help="ressource, bâtiment ou recette")
    parser.add_argument("quantite", nargs="?", type=int, default=10)
    parser.add_argument("--modele", default=str(ROOT / "models" / "ia_tier0.zip"))
    args = parser.parse_args()
    if args.commande in ("aller", "miner") and args.quoi not in RESOURCES:
        parser.error("indique la ressource : fer, cuivre ou calcaire")
    if args.commande == "construire" and args.quoi not in BUILDINGS:
        parser.error("indique le bâtiment : hub, etabli ou atelier")
    if args.commande == "fabriquer" and not args.quoi:
        parser.error('indique la recette, par exemple : fabriquer "Barre de fer" 5')

    try:
        read_player()
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        raise SystemExit(
            "Le mod ne répond pas. Lance le jeu depuis le Mod Manager, charge ta partie "
            "et tape dans le chat du jeu : /frm http start"
        )
    window = find_game_window()
    if window is None:
        raise SystemExit("Je ne trouve pas la fenêtre du jeu.")
    game = RealGame()
    countdown()
    try:
        if args.commande == "calibrer":
            cal = calibrate(game)
            cal.save()
            print(f"Calibrage enregistré : {cal}")
            return
        mover = Mover(game, Calibration.load())
        memory = Memory.load()
        menus = Menus(game, mover, Screen(window.left, window.top, window.width, window.height), memory)
        if args.commande == "jouer":
            play(game, mover, menus, memory, args.modele)
        elif args.commande == "equiper":
            print("Équipé." if menus.equip_xeno_zapper() else "Déjà équipé.")
        elif args.commande == "construire":
            build(menus, args.quoi)
            print("Construit.")
        elif args.commande == "aller":
            node = mover.go_to_node(RESOURCES[args.quoi])
            print(f"Arrivé au gisement ({node.purity}).")
        elif args.commande == "miner":
            print(f"Terminé : {mover.mine_by_hand(RESOURCES[args.quoi], args.quantite)} minés.")
        elif args.commande == "fabriquer":
            from mini_satisfactory import data

            labels = {data.label(item).lower(): item for item in data.ITEMS}
            product = labels.get(args.quoi.lower())
            if product is None:
                raise SkillError(f"recette inconnue : {args.quoi}")
            print(f"Terminé : {menus.craft(args.quoi, product, args.quantite)} fabriqués.")
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
