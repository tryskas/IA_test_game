"""The real game as a "raw game" for moves.py: the mod to read, SendInput to act.

Safety: F12 stops at once, and so does the game losing the focus. Every key
held is released when a skill stops, whatever the reason.
"""
from __future__ import annotations

import threading
import time

import inputs
from common import game_is_focused
from config import STOP_KEY
from game_state import Node, PlayerView, fetch, parse_nodes, read_player

NODES_REFRESH_S = 30


class Stopped(Exception):
    pass


class RealGame:
    def __init__(self):
        self.held: set[str] = set()
        self._stop = threading.Event()
        self._nodes: tuple[Node, ...] = ()
        self._nodes_at = -NODES_REFRESH_S
        self._watch_stop_key()

    def _watch_stop_key(self) -> None:
        from pynput import keyboard

        target = getattr(keyboard.Key, STOP_KEY)

        def on_press(key):
            if key == target:
                self._stop.set()

        keyboard.Listener(on_press=on_press, daemon=True).start()

    def check(self) -> None:
        if self._stop.is_set():
            raise Stopped(f"arrêt demandé ({STOP_KEY.upper()})")
        if not game_is_focused():
            raise Stopped("le jeu n'est plus la fenêtre active")

    # What moves.py uses -------------------------------------------------------

    def player(self) -> PlayerView:
        return read_player()

    def nodes(self) -> tuple[Node, ...]:
        if time.perf_counter() - self._nodes_at > NODES_REFRESH_S:
            self._nodes = parse_nodes(fetch("getResourceNode"))
            self._nodes_at = time.perf_counter()
        return self._nodes

    def mouse(self, dx: int, dy: int) -> None:
        self.check()
        inputs.move(dx, dy)

    def hold(self, key: str) -> None:
        self.check()
        inputs.key_down(key)
        self.held.add(key)

    def release(self, key: str) -> None:
        inputs.key_up(key)
        self.held.discard(key)

    def now(self) -> float:
        return time.perf_counter()

    def wait(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while True:
            self.check()
            left = end - time.perf_counter()
            if left <= 0:
                return
            time.sleep(min(0.02, left))

    def release_all(self) -> None:
        for key in list(self.held):
            self.release(key)
