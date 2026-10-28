"""Settings for the real game. Change them to match your game and keyboard."""
from pathlib import Path

# Title of the game window.
GAME_TITLE = "Satisfactory"

# Where screenshots and game data are saved.
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "captures"

# Address of the Ficsit Remote Monitoring mod's web server (step 2).
FRM_URL = "http://localhost:8080"

# Keys, as shown in the game's Options > Controls. On an AZERTY keyboard with
# the French layout, forward is often "z" and left "q": if the character does
# not move during step 3, try the other letter.
KEYS = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
    "jump": "space",
    "inventory": "tab",
}

# Key that stops any script driving the game, at any moment.
STOP_KEY = "f12"
