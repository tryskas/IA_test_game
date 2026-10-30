"""Étape 1 : vérifier que le mod HKAIBridge envoie bien l'état du jeu.

    python hollow_knight/check_bridge.py              # 15 s d'état affiché
    python hollow_knight/check_bridge.py --vitesse    # puis le jeu accéléré ×2 pendant 5 s

Lance d'abord Hollow Knight (avec le mod) et charge ta partie.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hk_bridge import Bridge, State  # noqa: E402


def describe(state: State) -> str:
    text = f"scène {state.scene:<22}"
    if state.hero:
        h = state.hero
        text += (f" chevalier ({h.x:7.2f}, {h.y:6.2f}) vie {h.hp}/{h.max_hp} âme {h.soul:3d}"
                 f"{' au sol' if h.on_ground else ' en l air'}{'' if h.control else ' (pas de contrôle)'}")
    else:
        text += " (pas de chevalier : menu ?)"
    if state.boss:
        b = state.boss
        text += f" | ennemi principal {b.name} ({b.x:6.1f}, {b.y:5.1f}) vie {b.hp}/{b.max_hp}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vitesse", action="store_true", help="essayer aussi d'accélérer le jeu")
    args = parser.parse_args()
    bridge = Bridge()
    print("J'attends des nouvelles du mod...")
    first = bridge.wait(timeout=10)
    if first is None:
        raise SystemExit(
            "Rien reçu en 10 s. Vérifie que le jeu tourne, que « HKAIBridge » apparaît en haut à gauche "
            "du menu principal, et que le pare-feu Windows n'a pas bloqué Python (réseau local)."
        )
    count, start = 0, time.perf_counter()
    last_print = 0.0
    while time.perf_counter() - start < 15:
        state = bridge.wait(timeout=2)
        if state is None:
            print("  ... plus rien reçu (jeu en pause ou fermé ?)")
            continue
        count += 1
        if time.perf_counter() - last_print >= 0.5:
            print(describe(state))
            last_print = time.perf_counter()
    print(f"\nReçu {count / 15:.0f} états par seconde : le mod fonctionne.")
    if args.vitesse:
        print("Jeu accéléré ×2 pendant 5 s...")
        bridge.set_speed(2.0)
        time.sleep(1)
        state = bridge.wait()
        print(f"  vitesse annoncée par le mod : ×{state.speed:g}" if state else "  pas de réponse")
        time.sleep(4)
        bridge.set_speed(1.0)
        print("Vitesse normale rétablie.")
    bridge.close()


if __name__ == "__main__":
    main()
