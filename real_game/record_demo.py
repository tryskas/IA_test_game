"""Record one game played by hand, so the AI can learn the menus from it.

    python real_game/record_demo.py

Play normally. While the game is the active window, the script records:
- a screenshot at each mouse click (and where you clicked);
- the keys you press and how long you hold them, the mouse buttons too;
- a small screenshot every 5 seconds;
- the game's data from the mod every second (position, inventory, buildings).
Nothing is recorded while another window is active. Press F12 to stop.
Everything goes to demos/demo_<date>/.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime

import cv2
import mss
import numpy as np

from common import find_game_window, game_is_focused, make_dpi_aware
from config import STOP_KEY
from game_state import ROOT, fetch

PERIODIC_S = 5
PLAYER_S = 1.0
FACTORY_S = 5.0
CLICK_SIZE = (1280, 720)
PERIODIC_SIZE = (960, 540)


def main() -> None:
    make_dpi_aware()
    window = find_game_window()
    if window is None:
        raise SystemExit("Je ne trouve pas la fenêtre du jeu. Lance Satisfactory, puis relance ce script.")
    try:
        fetch("getPlayer")
    except Exception:
        raise SystemExit("Le mod ne répond pas : charge ta partie et tape /frm http start dans le chat du jeu.")

    from pynput import keyboard, mouse

    folder = ROOT / "demos" / f"demo_{datetime.now():%Y%m%d_%H%M%S}"
    (folder / "shots").mkdir(parents=True)
    log = (folder / "events.jsonl").open("a", encoding="utf-8")
    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    started = time.perf_counter()
    stop_key = getattr(keyboard.Key, STOP_KEY)

    def now() -> float:
        return round(time.perf_counter() - started, 3)

    def write(kind: str, **fields) -> None:
        log.write(json.dumps({"t": now(), "kind": kind, **fields}, ensure_ascii=False) + "\n")
        log.flush()

    def key_name(key) -> str:
        return getattr(key, "char", None) or str(key).replace("Key.", "")

    def on_press(key):
        if key == stop_key:
            stop.set()
            return False
        if game_is_focused():
            events.put(("key_down", {"key": key_name(key)}))

    def on_release(key):
        if game_is_focused():
            events.put(("key_up", {"key": key_name(key)}))

    def on_click(x, y, button, pressed):
        if game_is_focused():
            kind = "mouse_down" if pressed else "mouse_up"
            events.put((kind, {"button": str(button).replace("Button.", ""), "x": x, "y": y}))

    keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True).start()
    mouse.Listener(on_click=on_click, daemon=True).start()

    area = {"left": window.left, "top": window.top, "width": window.width, "height": window.height}
    write("start", window=area)
    print(f"Enregistrement dans {folder}")
    print("Joue normalement. F12 pour arrêter.")
    shots = 0
    next_periodic = next_player = next_factory = 0.0
    last_factory = None
    with mss.mss() as screen:

        def shoot(name: str, size: tuple[int, int]) -> str:
            image = np.asarray(screen.grab(area))[:, :, :3]
            path = folder / "shots" / name
            cv2.imwrite(str(path), cv2.resize(image, size, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            return f"shots/{name}"

        while not stop.is_set():
            try:
                kind, fields = events.get(timeout=0.02)
                if kind == "mouse_down":
                    shots += 1
                    fields["shot"] = shoot(f"click_{shots:04d}.jpg", CLICK_SIZE)
                write(kind, **fields)
            except queue.Empty:
                pass
            t = time.perf_counter() - started
            focused = game_is_focused()
            if focused and t >= next_periodic:
                write("view", shot=shoot(f"view_{int(t):05d}.jpg", PERIODIC_SIZE))
                next_periodic = t + PERIODIC_S
            try:
                if t >= next_player:
                    write("player", data=fetch("getPlayer"))
                    next_player = t + PLAYER_S
                if t >= next_factory:
                    factory = {"factory": fetch("getFactory"), "hub": fetch("getHUBTerminal")}
                    if factory != last_factory:
                        write("buildings", data=factory)
                        last_factory = factory
                    next_factory = t + FACTORY_S
            except Exception as error:  # the mod paused: keep recording the rest
                write("mod_error", error=str(error))
                next_player = next_factory = t + 5
    try:
        write("schematics", data=fetch("getSchematics"))
    except Exception:
        pass
    write("stop")
    log.close()
    minutes = (time.perf_counter() - started) / 60
    print(f"Arrêté après {minutes:.1f} min, {shots} clics enregistrés dans {folder}")


if __name__ == "__main__":
    main()
