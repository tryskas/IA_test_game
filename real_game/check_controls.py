"""Step 3, the hands: check that Python can play.

Turns the camera, walks forward and back, jumps, opens and closes the
inventory. Stand somewhere safe first (on flat ground, nothing in hand).

    python real_game/check_controls.py

You then have 5 seconds to click in the game. Everything stops at once if
you press F12 or if the game is no longer the active window.
"""
from __future__ import annotations

import threading
import time

import inputs
from common import find_game_window, game_is_focused
from config import KEYS, STOP_KEY

COUNTDOWN = 5


class Stopped(Exception):
    pass


stop = threading.Event()


def watch_stop_key() -> None:
    from pynput import keyboard

    target = getattr(keyboard.Key, STOP_KEY)

    def on_press(key):
        if key == target:
            stop.set()
            return False  # stop listening

    keyboard.Listener(on_press=on_press, daemon=True).start()


def check() -> None:
    if stop.is_set():
        raise Stopped(f"arrêt demandé ({STOP_KEY.upper()})")
    if not game_is_focused():
        raise Stopped("le jeu n'est plus la fenêtre active")


def wait(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        check()
        time.sleep(0.02)


def hold(key: str, seconds: float) -> None:
    check()
    inputs.key_down(key)
    try:
        wait(seconds)
    finally:
        inputs.key_up(key)


def press(key: str) -> None:
    check()
    inputs.press(key)


def turn(dx: int, steps: int = 30) -> None:
    """Turn the camera smoothly: games read relative mouse moves."""
    for _ in range(steps):
        check()
        inputs.move(dx // steps, 0)
        time.sleep(0.01)


TESTS = [
    ("Tourner la caméra vers la droite", lambda: turn(600)),
    ("Tourner la caméra vers la gauche", lambda: turn(-600)),
    (f"Avancer (touche {KEYS['forward'].upper()})", lambda: hold(KEYS["forward"], 0.7)),
    (f"Reculer (touche {KEYS['back'].upper()})", lambda: hold(KEYS["back"], 0.7)),
    ("Sauter", lambda: press(KEYS["jump"])),
    ("Ouvrir l'inventaire", lambda: (press(KEYS["inventory"]), wait(1.5))),
    ("Fermer l'inventaire", lambda: (press(KEYS["inventory"]), wait(0.5))),
]


def main() -> None:
    if find_game_window() is None:
        raise SystemExit("Je ne trouve pas la fenêtre du jeu. Lance Satisfactory et charge une partie.")
    print("Touches utilisées (lettre affichée par le jeu, puis touche physique) :")
    for action, key in KEYS.items():
        scan, _ = inputs.scan_code(key)
        print(f"  {action:<10} {key.upper():<6} code {scan}")
    watch_stop_key()
    print(f"Clique dans le jeu maintenant. Début dans {COUNTDOWN} s. {STOP_KEY.upper()} pour tout arrêter.")
    for left in range(COUNTDOWN, 0, -1):
        print(f"  {left}...")
        time.sleep(1)
    try:
        for name, action in TESTS:
            print(f"- {name}")
            action()
            wait(0.6)
    except Stopped as reason:
        print(f"Arrêt : {reason}.")
        return
    print("Terminé. Le personnage a-t-il bien fait chaque mouvement ? Note ceux qui n'ont pas marché.")


if __name__ == "__main__":
    main()
