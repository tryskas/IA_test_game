import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from common import GameWindow, pick_game_window  # noqa: E402


def window(title: str, width: int = 1920, height: int = 1080) -> GameWindow:
    return GameWindow(title, 0, 0, width, height)


class PickGameWindowTest(unittest.TestCase):
    def test_finds_the_game_among_other_windows(self):
        windows = [window("Discord"), window("Satisfactory"), window("Steam")]
        self.assertEqual(pick_game_window(windows).title, "Satisfactory")

    def test_ignores_windows_that_only_mention_the_game(self):
        windows = [
            window("viewer.py - mini_satisfactory - Visual Studio Code"),
            window("Satisfactory Mod Manager"),
            window("Mini-Satisfactory : l'IA joue"),
        ]
        self.assertIsNone(pick_game_window(windows))

    def test_prefers_the_exact_title_and_skips_tiny_windows(self):
        windows = [window("Satisfactory", 200, 100), window("Satisfactory - Early Access", 1280, 720)]
        self.assertEqual(pick_game_window(windows).title, "Satisfactory - Early Access")
        windows.append(window("Satisfactory"))
        self.assertEqual(pick_game_window(windows).width, 1920)


if __name__ == "__main__":
    unittest.main()
