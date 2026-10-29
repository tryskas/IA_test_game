"""The AI plays the real game.

    python real_game/agent.py calibrer                  # once: measure the mouse and the walk (~15 m of flat ground)
    python real_game/agent.py jouer                     # the AI plays: HUB milestones one after the other

Single skills, to test them one by one:
    python real_game/agent.py equiper                   # equip the Xeno-Zapper (start of the tutorial)
    python real_game/agent.py construire hub            # also: etabli, atelier
    python real_game/agent.py aller fer                 # walk to the nearest free iron node
    python real_game/agent.py miner fer 10              # mine 10 iron ore by hand (also: cuivre, calcaire)
    python real_game/agent.py fabriquer "Barre de fer" 5
    python real_game/agent.py jalon                     # choose the next milestone at the HUB terminal
    python real_game/agent.py deposer                   # give the HUB terminal what it needs
    python real_game/agent.py voir                      # 20 s of what the AI sees, saved as images

The AI watches the screen (creatures, obstacles) unless --sans-vision is given.

Each command waits 5 seconds so you can click in the game. F12 stops at once;
so does leaving the game window.
"""
from __future__ import annotations

import argparse
import time
import urllib.error

from common import find_game_window
from config import FOV
from defense import defend, respawn
from game_io import Died, RealGame, Stopped, UnderAttack
from game_state import ROOT, fetch, read_player
from menus import Memory, Menus
from moves import Calibration, Mover, SkillError, calibrate
from planner import RAW, Plan, Step, next_step, parse_schematics
from screen import Screen

RESOURCES = {"fer": "iron_ore", "cuivre": "copper_ore", "calcaire": "limestone"}
PLATE, ROD = "Desc_IronPlate_C", "Desc_IronRod_C"
# Buildings of the start: in-game name, cost, name of the spot remembered to use it.
BUILDINGS = {
    "hub": ("Le H.U.B.", None, None),
    "etabli": ("Établi", {PLATE: 3, ROD: 3}, "workbench"),
    "atelier": ("Atelier d'équipement", {PLATE: 6, ROD: 4}, "workshop"),
}
COUNTDOWN = 5
WAIT_S = 10


def countdown(seconds: int = COUNTDOWN) -> None:
    print(f"Clique dans le jeu. Début dans {seconds} s (F12 pour tout arrêter).")
    for left in range(seconds, 0, -1):
        print(f"  {left}...")
        time.sleep(1)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def terminal() -> dict | None:
    terminals = fetch("getHUBTerminal")
    return terminals[0] if terminals else None


def hub_id() -> str | None:
    found = terminal()
    return found.get("ID") if found else None


def active_milestone() -> str | None:
    found = terminal()
    if not found or not found.get("HasActiveMilestone"):
        return None
    return (found.get("ActiveMilestone") or {}).get("ClassName")


def read_plan() -> Plan:
    return parse_schematics(fetch("getSchematics"))


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


def describe(step: Step, plan: Plan) -> str:
    if step.kind == "select":
        return f"choisir le jalon « {step.milestone.name} » (palier {step.milestone.tier})"
    if step.kind == "deposit":
        return f"déposer au terminal pour « {step.milestone.name} »"
    if step.kind == "mine":
        return f"miner {step.count} {plan.name(step.item)}"
    if step.kind == "craft":
        return f"fabriquer {step.count} × {step.recipe.name}"
    return step.why


def choose_milestone_at_terminal(menus: Menus, step: Step) -> None:
    menus.open_terminal(terminal())
    try:
        menus.select_milestone(step.milestone.name, step.milestone.tier,
                               lambda: active_milestone() == step.milestone.cls)
    finally:
        menus.close()


def deposit_at_terminal(menus: Menus, milestone_cls: str) -> int:
    menus.open_terminal(terminal())
    try:
        return menus.deposit(lambda: read_plan().milestones[milestone_cls].purchased)
    finally:
        menus.close()


def do(step: Step, game: RealGame, mover: Mover, menus: Menus) -> None:
    if step.kind == "select":
        choose_milestone_at_terminal(menus, step)
    elif step.kind == "deposit":
        print(f"  {deposit_at_terminal(menus, step.milestone.cls)} piles déposées")
    elif step.kind == "mine":
        print(f"  {mover.mine_by_hand(RAW[step.item], step.count)} minés")
    elif step.kind == "craft":
        wanted = step.count * step.recipe.outputs[step.item]
        print(f"  {menus.craft(step.recipe.name, step.item, int(wanted))} fabriqués")
    elif step.kind == "wait":
        game.wait(WAIT_S)


def remember_danger(game: RealGame, memory: Memory) -> None:
    """Note where the player got hurt, to mine elsewhere next time."""
    spot = game.last_hurt
    if spot is None:
        player = read_player()
        spot = (player.x, player.y)
    memory.data.setdefault("dangers", []).append([round(spot[0], 1), round(spot[1], 1)])
    memory.save()


def survive(alarm: Exception, game: RealGame, mover: Mover, menus: Menus, memory: Memory) -> None:
    """React to an attack or a death, then let the plan carry on."""
    remember_danger(game, memory)
    if isinstance(alarm, Died):
        print(f"[{stamp()}] Mort : je réapparais (l'inventaire est resté sur place)")
        respawn(game, menus.screen)
        return
    print(f"[{stamp()}] Danger : {alarm}")
    hub = terminal()
    home = (hub["location"]["x"] / 100, hub["location"]["y"] / 100) if hub else None
    outcome = defend(game, mover, game.eyes, home, hurt=getattr(alarm, "hurt", True))
    print({"safe": "  hors de danger", "harmless": "  créature inoffensive, je reprends",
           "fled": "  à l'abri"}[outcome])


def play(game: RealGame, mover: Mover, menus: Menus, memory: Memory) -> None:
    start_of_game(menus, memory)
    failures = 0
    while True:
        plan = read_plan()
        hub = terminal() or {}
        active = active_milestone()
        if active is not None and active not in plan.milestones:
            raise SkillError("le jalon actif au terminal n'est pas un jalon que je connais")
        step = next_step(plan, read_player().by_class, active, bool(hub.get("ShipDock", True)))
        print(f"[{stamp()}] {describe(step, plan)}")
        if step.kind in ("done", "stuck"):
            return
        try:
            do(step, game, mover, menus)
            failures = 0
        except (UnderAttack, Died) as alarm:
            try:
                survive(alarm, game, mover, menus, memory)
            except Died as again:
                survive(again, game, mover, menus, memory)
        except SkillError as problem:
            failures += 1
            print(f"  échec : {problem}")
            if failures >= 3:
                raise SkillError(f"trois échecs de suite, j'arrête ({problem})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sans-vision", action="store_true", help="ne pas analyser l'image (créatures, obstacles)")
    parser.add_argument("commande", choices=["calibrer", "jouer", "equiper", "construire", "aller", "miner", "voir",
                                             "fabriquer", "jalon", "deposer"])
    parser.add_argument("quoi", nargs="?", help="ressource, bâtiment ou recette")
    parser.add_argument("quantite", nargs="?", type=int, default=10)
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
    eyes = None
    if not args.sans_vision and args.commande in ("jouer", "aller", "miner", "voir"):
        from perception import Eyes

        print("Chargement de la vision (la première fois, téléchargement des modèles)...")
        eyes = Eyes((window.left, window.top, window.width, window.height), FOV).start()
        game.eyes = eyes
    if args.commande == "voir":
        watch(eyes)
        return
    countdown()
    try:
        if args.commande == "calibrer":
            cal = calibrate(game)
            cal.save()
            print(f"Calibrage enregistré : {cal}")
            return
        memory = Memory.load()
        mover = Mover(game, Calibration.load(), dangers=memory.data.setdefault("dangers", []), eyes=eyes)
        menus = Menus(game, mover, Screen(window.left, window.top, window.width, window.height), memory)
        if args.commande == "jouer":
            play(game, mover, menus, memory)
        elif args.commande == "jalon":
            plan = read_plan()
            step = next_step(plan, read_player().by_class, active_milestone(), bool((terminal() or {}).get("ShipDock", True)))
            if step.kind != "select":
                print(f"Pas de jalon à choisir maintenant : {describe(step, plan)}")
            else:
                choose_milestone_at_terminal(menus, step)
                print(f"Jalon choisi : {step.milestone.name}")
        elif args.commande == "deposer":
            active = active_milestone()
            if active is None:
                raise SkillError("aucun jalon n'est choisi au terminal")
            print(f"{deposit_at_terminal(menus, active)} piles déposées.")
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
            recipes = {r.name.lower(): (item, r) for item, r in read_plan().recipes.items()}
            if args.quoi.lower() not in recipes:
                raise SkillError(f"recette inconnue ou pas encore débloquée : {args.quoi}")
            product, recipe = recipes[args.quoi.lower()]
            print(f"Terminé : {menus.craft(recipe.name, product, args.quantite)} fabriqués.")
    except Stopped as reason:
        print(f"Arrêt : {reason}.")
    except SkillError as problem:
        print(f"Échec : {problem}.")
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        print("Arrêt : le mod ne répond plus (jeu fermé ou serveur arrêté).")
    finally:
        game.release_all()
        if eyes is not None:
            eyes.stop()


def watch(eyes, seconds: float = 20, every: float = 2.0) -> None:
    """Save what the AI sees: creatures framed, open (green) or blocked (red) directions."""
    import cv2

    from perception import annotate

    folder = ROOT / "captures" / "vision"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Je regarde pendant {seconds:.0f} s : bouge dans le jeu, approche des créatures et des obstacles.")
    end = time.perf_counter() + seconds
    shot = 0
    while time.perf_counter() < end:
        view = eyes.fresh(time.perf_counter(), timeout=5)
        image = eyes.latest_image
        if view is None or image is None:
            continue
        shot += 1
        path = folder / f"vision_{shot:02d}.jpg"
        cv2.imwrite(str(path), annotate(image, view))
        if eyes.last_depth is not None:  # small copy of the depth, to fine-tune obstacle spotting
            import numpy as np

            small = cv2.resize(eyes.last_depth, (192, 108), interpolation=cv2.INTER_AREA)
            np.savez_compressed(path.with_suffix(".npz"), depth=small.astype(np.float16))
        blocked = sum(1 for c in view.clearance if c < 0.5)
        names = ", ".join(f"{t.label} {t.bearing:+.0f}°" for t in view.threats) or "aucune"
        print(f"  {path.name} : créatures : {names} | directions bloquées : {blocked}/{len(view.clearance)}")
        time.sleep(every)
    eyes.stop()
    print(f"Images enregistrées dans {folder}")


if __name__ == "__main__":
    main()
