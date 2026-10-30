import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from miners import PortableMiners  # noqa: E402
from moves import Mover, calibrate  # noqa: E402
from tests.test_moves import FakeGame  # noqa: E402


class Memory:
    def __init__(self):
        self.data = {}

    def save(self):
        pass


class PortableMinersTest(unittest.TestCase):
    def test_without_portable_miners_mining_is_by_hand(self):
        game = FakeGame()
        mover = Mover(game, calibrate(game), log=lambda *_: None)
        miners = PortableMiners(game, mover, None, Memory(), log=lambda *_: None)
        self.assertIsNone(miners.mine("iron_ore", 10))
        self.assertEqual(miners.placed, [])


if __name__ == "__main__":
    unittest.main()
