"""The link with the game: what the HKAIBridge mod sends (the state, many times a second)
and the commands sent back to it (the game's speed).
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

from hk_config import COMMAND_PORT, STATE_PORT


@dataclass(frozen=True)
class Hero:
    x: float
    y: float
    vx: float
    vy: float
    hp: int
    max_hp: int
    soul: int
    facing_right: bool
    on_ground: bool
    dead: bool
    control: bool  # the player can move it (not in a cutscene or a menu)


@dataclass(frozen=True)
class Boss:
    name: str
    x: float
    y: float
    hp: int
    max_hp: int


@dataclass(frozen=True)
class State:
    time: float  # the game's clock (seconds, not slowed down or sped up)
    speed: float  # how fast the game runs (1 = normal)
    scene: str
    hero: Hero | None  # None in the menus
    boss: Boss | None  # the enemy with the most health in the scene, if any
    received_at: float  # time.perf_counter() when it arrived


def parse_state(text: str, received_at: float = 0.0) -> State:
    data = json.loads(text)
    hero = data.get("hero")
    boss = data.get("boss")
    return State(
        float(data.get("t", 0.0)), float(data.get("speed", 1.0)), data.get("scene", ""),
        Hero(hero["x"], hero["y"], hero["vx"], hero["vy"], hero["hp"], hero["maxhp"], hero["soul"],
             hero["right"], hero["ground"], hero["dead"], hero["control"]) if hero else None,
        Boss(boss["name"], boss["x"], boss["y"], boss["hp"], boss["maxhp"]) if boss else None,
        received_at,
    )


class Bridge:
    """Listens to the mod in the background; `latest` is always the newest state."""

    def __init__(self):
        self.latest: State | None = None
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._receiver.bind(("127.0.0.1", STATE_PORT))
        self._receiver.settimeout(0.5)
        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while self._running:
            try:
                data, _ = self._receiver.recvfrom(65536)
            except (socket.timeout, OSError):
                continue
            try:
                self.latest = parse_state(data.decode("utf-8"), time.perf_counter())
            except (ValueError, KeyError):
                pass  # a broken message: the next one comes a moment later

    def wait(self, timeout: float = 5.0, after: float | None = None) -> State | None:
        """The first state received after `after` (default: now), or None after `timeout` seconds."""
        after = time.perf_counter() if after is None else after
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            state = self.latest
            if state is not None and state.received_at > after:
                return state
            time.sleep(0.005)
        return None

    def set_speed(self, factor: float) -> None:
        """Run the game `factor` times faster (1 = normal)."""
        self._sender.sendto(f"speed {factor}".encode("utf-8"), ("127.0.0.1", COMMAND_PORT))

    def close(self) -> None:
        self._running = False
        self._receiver.close()
        self._sender.close()
