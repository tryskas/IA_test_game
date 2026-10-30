import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hollow_knight"))

from hk_bridge import parse_state  # noqa: E402

# As the HKAIBridge mod writes it
DURING_A_FIGHT = (
    '{"t":123.5,"speed":2,"scene":"GG_Hornet_1",'
    '"hero":{"x":45.2,"y":28.41,"vx":-8.3,"vy":0,"hp":4,"maxhp":5,"soul":33,'
    '"right":false,"ground":true,"dead":false,"control":true},'
    '"boss":{"name":"Hornet Boss 1","x":52.1,"y":29,"hp":620,"maxhp":900}}'
)


class ParseStateTest(unittest.TestCase):
    def test_reads_a_fight(self):
        state = parse_state(DURING_A_FIGHT, received_at=1.0)
        self.assertEqual(state.scene, "GG_Hornet_1")
        self.assertEqual(state.speed, 2.0)
        self.assertEqual((state.hero.x, state.hero.hp, state.hero.max_hp, state.hero.soul), (45.2, 4, 5, 33))
        self.assertFalse(state.hero.facing_right)
        self.assertTrue(state.hero.control)
        self.assertEqual((state.boss.name, state.boss.hp, state.boss.max_hp), ("Hornet Boss 1", 620, 900))

    def test_reads_a_menu(self):
        state = parse_state('{"t":1,"speed":1,"scene":"Menu_Title"}')
        self.assertIsNone(state.hero)
        self.assertIsNone(state.boss)


if __name__ == "__main__":
    unittest.main()
