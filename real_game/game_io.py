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


HEALTH_CHECK_S = 0.5
CLOSE_THREAT = 0.18  # a creature this big on screen (share of its height) is too close to ignore


class Stopped(Exception):
    pass


class UnderAttack(Exception):
    """The player is losing health, or a creature is close: whatever was being done must stop."""

    def __init__(self, reason: str, hurt: bool):
        super().__init__(reason)
        self.hurt = hurt


class Died(Exception):
    pass


class RealGame:
    def __init__(self):
        self.held: set[str] = set()
        self._stop = threading.Event()
        self._nodes: tuple[Node, ...] = ()
        self._nodes_at = -NODES_REFRESH_S
        self._health_at = 0.0
        self.hp: float | None = None
        self.alarms = True  # off while defending, so the defence itself is not interrupted
        self.last_hurt: tuple[float, float] | None = None  # where the player was last hurt
        self.eyes = None  # perception.Eyes, when the AI can see
        self.ignore_sightings_until = 0.0  # after meeting a harmless creature, don't stop for it again
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
        if self.alarms and time.perf_counter() - self._health_at > HEALTH_CHECK_S:
            self.player()  # updates the health and raises if hurt
        view = self.eyes.latest if self.eyes is not None else None
        if (self.alarms and view is not None and time.perf_counter() - view.taken_at < 1.0
                and time.perf_counter() > self.ignore_sightings_until):
            threat = view.nearest_threat()
            if threat is not None and threat.size >= CLOSE_THREAT:
                self.release_all()
                raise UnderAttack(f"créature en vue ({threat.label})", hurt=False)


    # What moves.py uses -------------------------------------------------------

    def player(self) -> PlayerView:
        player = read_player()
        self._health_at = time.perf_counter()
        hurt = self.hp is not None and player.hp < self.hp - 0.5
        self.hp = player.hp
        if player.dead or player.hp <= 0:
            self.release_all()
            raise Died()
        if hurt:
            self.last_hurt = (player.x, player.y)
            if self.alarms:
                self.release_all()
                raise UnderAttack("je perds de la vie", hurt=True)
        return player

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

    def click(self, button: str = "left") -> None:
        self.check()
        inputs.click(button)

    def click_at(self, x: float, y: float) -> None:
        self.check()
        inputs.move_to(x, y)
        time.sleep(0.08)
        inputs.click()

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
