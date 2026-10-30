"""Étape 2 : vérifier que les touches envoyées par l'IA font bien bouger le chevalier.

    python hollow_knight/check_keys.py

Place le chevalier sur un sol plat avec de la place des deux côtés (le Panthéon des dieux
est parfait), lance le script, puis clique dans le jeu pendant le compte à rebours.
L'IA essaie : droite, gauche, saut, ruée (dash), attaque. Le mod dit si le chevalier a bougé.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hk_inputs as inputs  # noqa: E402
from hk_bridge import Bridge  # noqa: E402
from hk_config import KEYS  # noqa: E402


def hold(key: str, seconds: float, bridge: Bridge, watch: float = 0.0) -> list:
    """Hold a key, and note the knight's states meanwhile (and `watch` seconds after)."""
    seen = []
    inputs.key_down(KEYS[key])
    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            state = bridge.wait(timeout=1)
            if state and state.hero:
                seen.append(state.hero)
    finally:
        inputs.key_up(KEYS[key])
    end = time.perf_counter() + watch
    while time.perf_counter() < end:
        state = bridge.wait(timeout=1)
        if state and state.hero:
            seen.append(state.hero)
    return seen


def settle(bridge: Bridge) -> None:
    time.sleep(0.6)
    bridge.wait()


def main() -> None:
    bridge = Bridge()
    if bridge.wait(timeout=10) is None:
        raise SystemExit("Rien reçu du mod : lance d'abord « python hollow_knight/check_bridge.py ».")
    print("Clique dans le jeu. Début dans 5 s.")
    for k in range(5, 0, -1):
        print(f"  {k}...")
        time.sleep(1)
    start = bridge.wait().hero
    if start is None or not start.control:
        raise SystemExit("Le chevalier n'est pas jouable maintenant (menu, cinématique ?).")
    results = []

    seen = hold("right", 0.5, bridge)
    results.append(("droite", KEYS["right"], bool(seen) and seen[-1].x > start.x + 1))
    settle(bridge)
    here = bridge.latest.hero
    seen = hold("left", 0.5, bridge)
    results.append(("gauche", KEYS["left"], bool(seen) and seen[-1].x < here.x - 1))
    settle(bridge)
    here = bridge.latest.hero
    seen = hold("jump", 0.35, bridge, watch=0.4)
    results.append(("saut", KEYS["jump"], bool(seen) and max(h.y for h in seen) > here.y + 1))
    settle(bridge)
    here = bridge.latest.hero
    seen = hold("dash", 0.1, bridge, watch=0.35)
    results.append(("ruée (dash)", KEYS["dash"], bool(seen) and max(abs(h.x - here.x) for h in seen) > 2))
    settle(bridge)
    hold("attack", 0.1, bridge, watch=0.3)

    print()
    for name, key, ok in results:
        print(f"  {'OK ' if ok else 'NON'}  {name:<12} touche « {key} »")
    print("  ?    attaque      regarde dans le jeu si le chevalier a donné un coup de dard")
    if not all(ok for _, _, ok in results):
        print("\nUne touche ne marche pas : vérifie dans le jeu (Options > Clavier) quelle touche fait cette"
              " action, et écris-la dans hollow_knight/hk_config.py.")
    bridge.close()


if __name__ == "__main__":
    main()
