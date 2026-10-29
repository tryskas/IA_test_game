"""Settings for the real game. Change them to match your game and keyboard."""
from pathlib import Path

# Title of the game window.
GAME_TITLE = "Satisfactory"

# Where screenshots and game data are saved.
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "captures"

# Address of the Ficsit Remote Monitoring mod's web server (step 2).
FRM_URL = "http://localhost:8080"

# Keys, as the game shows them (Options > Commandes). The keyboard layout
# decides which physical key that is, so write the letters you see in game:
# "z", "q", "s", "d" on an AZERTY keyboard, "w", "a", "s", "d" on QWERTY.
KEYS = {
    "forward": "z",
    "back": "s",
    "left": "q",
    "right": "d",
    "jump": "space",
    "inventory": "tab",
    "interact": "e",
    "build": "a",  # build menu (Q on a QWERTY keyboard)
    "craft": "space",  # hold to craft at a workbench
    "close": "escape",
}

# Key that stops any script driving the game, at any moment.
STOP_KEY = "f12"
