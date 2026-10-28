"""Step 1, the eyes: check that Python can see the game.

Finds the Satisfactory window, films it for 5 seconds, prints how many images
per second it gets, and saves 3 screenshots in the captures folder.

    python real_game/check_capture.py
"""
from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np

from common import GameWindow, find_game_window, screen_size
from config import CAPTURE_DIR

SECONDS = 5
SCREENSHOTS = 3


def film_with_bettercam(window: GameWindow, on_frame) -> str | None:
    """Fast capture through DirectX. Returns None if it can't be used here."""
    try:
        import bettercam
    except ImportError:
        return None
    try:
        camera = bettercam.create(output_color="BGR")
    except Exception as error:  # some graphics setups refuse DirectX capture
        print(f"  bettercam ne marche pas ici ({error}) : capture avec mss.")
        return None
    if camera is None:
        return None
    screen_w, screen_h = camera.width, camera.height
    left, top, right, bottom = window.region
    if left < 0 or top < 0 or right > screen_w or bottom > screen_h:
        print("  Le jeu n'est pas entièrement sur l'écran principal : capture plus lente avec mss.")
        camera.release()
        return None
    start = time.perf_counter()
    while time.perf_counter() - start < SECONDS:
        frame = camera.grab(region=window.region)  # None when the screen has not changed
        if frame is not None:
            on_frame(frame)
    camera.release()
    return "bettercam (DirectX, rapide)"


def film_with_mss(window: GameWindow, on_frame) -> str:
    import mss

    area = {"left": window.left, "top": window.top, "width": window.width, "height": window.height}
    with mss.mss() as screen:
        start = time.perf_counter()
        while time.perf_counter() - start < SECONDS:
            on_frame(np.asarray(screen.grab(area))[:, :, :3])  # drop the alpha channel
    return "mss (plus lent, marche partout)"


def main() -> None:
    window = find_game_window()
    if window is None:
        raise SystemExit(
            "Je ne trouve pas la fenêtre du jeu. Lance Satisfactory, charge une partie, "
            "puis relance ce script."
        )
    print(f"Fenêtre trouvée : « {window.title} », {window.width} x {window.height} pixels "
          f"à la position ({window.left}, {window.top})")
    screen_w, screen_h = screen_size()
    if (window.width, window.height) != (screen_w, screen_h):
        print(f"  Conseil : le jeu ne remplit pas l'écran ({screen_w} x {screen_h}). Passe-le en mode "
              "« Fenêtré sans bordure » pour la suite du guide.")

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    count, saved, next_save = 0, [], 0.0
    started = time.perf_counter()

    def on_frame(frame: np.ndarray) -> None:
        nonlocal count, next_save
        count += 1
        elapsed = time.perf_counter() - started
        if len(saved) < SCREENSHOTS and elapsed >= next_save:
            path = CAPTURE_DIR / f"capture_{stamp}_{len(saved) + 1}.png"
            cv2.imwrite(str(path), frame)
            saved.append(path)
            next_save = elapsed + SECONDS / SCREENSHOTS

    print(f"Je filme le jeu pendant {SECONDS} s : bouge un peu la caméra dans le jeu...")
    method = film_with_bettercam(window, on_frame) or film_with_mss(window, on_frame)
    elapsed = time.perf_counter() - started
    print(f"Méthode : {method}")
    print(f"Images reçues : {count} en {elapsed:.1f} s, soit {count / elapsed:.0f} images par seconde")
    if count / elapsed < 10:
        print("  C'est peu : vérifie que le jeu est bien au premier plan et pas réduit.")
    for path in saved:
        print(f"Capture enregistrée : {path}")
    print("Ouvre une capture : elle doit montrer le jeu, sans bande noire ni autre fenêtre.")


if __name__ == "__main__":
    main()
