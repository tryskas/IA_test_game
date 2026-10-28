"""Shared helpers: find the game window, keep coordinates right on scaled screens."""
from __future__ import annotations

import sys
from dataclasses import dataclass

from config import GAME_TITLE

# Windows whose title mentions the game but are not the game.
NOT_THE_GAME = ("visual studio code", "mini_satisfactory", "mini-satisfactory", "ia_test_game",
                "mod manager", "steam", "epic games", "discord", "chrome", "firefox", "edge")


@dataclass(frozen=True)
class GameWindow:
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


def pick_game_window(windows: list[GameWindow]) -> GameWindow | None:
    """The game among all open windows: exact title first, then the biggest match."""
    wanted = GAME_TITLE.lower()
    matches = [
        w for w in windows
        if wanted in w.title.lower()
        and not any(word in w.title.lower() for word in NOT_THE_GAME)
        and w.width >= 640 and w.height >= 360
    ]
    exact = [w for w in matches if w.title.strip().lower() == wanted]
    candidates = exact or matches
    return max(candidates, key=lambda w: w.width * w.height) if candidates else None


def make_dpi_aware() -> None:
    """With Windows display scaling (125 %, 150 %...), get real pixel coordinates."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def open_windows() -> list[GameWindow]:
    import pygetwindow

    return [
        GameWindow(w.title, w.left, w.top, w.width, w.height)
        for w in pygetwindow.getAllWindows()
        if w.title and getattr(w, "visible", True)
    ]


def find_game_window() -> GameWindow | None:
    if sys.platform != "win32":
        raise SystemExit("Ces scripts sont faits pour Windows, sur le PC où tourne le jeu.")
    make_dpi_aware()
    return pick_game_window(open_windows())


def game_is_focused() -> bool:
    """True when the game window is the one receiving the keyboard."""
    import pygetwindow

    active = pygetwindow.getActiveWindow()
    if active is None:
        return False
    window = GameWindow(active.title, active.left, active.top, active.width, active.height)
    return pick_game_window([window]) is not None
