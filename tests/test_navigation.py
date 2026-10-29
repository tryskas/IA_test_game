import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from navigation import TerrainMap  # noqa: E402


class PlanTest(unittest.TestCase):
    def test_straight_line_when_nothing_is_known(self):
        self.assertEqual(TerrainMap().plan((0, 0), (50, 0)), [(50, 0)])

    def test_goes_round_a_known_wall(self):
        terrain = TerrainMap()
        for y in range(-20, 21):
            terrain.mark_blocked(20, y, strength=2)
        path = terrain.plan((0, 0), (40, 0))
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (40, 0))
        self.assertTrue(any(abs(y) > 20 for _, y in path))  # round the end of the wall

    def test_keeps_away_from_danger_when_it_can(self):
        terrain = TerrainMap()
        terrain.mark_danger(25, 0)
        path = terrain.plan((0, 0), (50, 0))
        self.assertTrue(all(abs(x - 25) > 10 or abs(y) > 10 for x, y in path))

    def test_where_the_player_stood_is_not_blocked(self):
        terrain = TerrainMap()
        terrain.mark_blocked(3, 3, strength=2)
        terrain.mark_free(3, 3)
        self.assertFalse(terrain.is_blocked(terrain.cell(3, 3)))

    def test_the_map_is_saved_and_read_back(self):
        terrain = TerrainMap()
        terrain.mark_blocked(10, 10, strength=2)
        terrain.mark_danger(5, 5)
        again = TerrainMap(terrain.to_data())
        self.assertTrue(again.is_blocked(again.cell(10, 10)))
        self.assertEqual(again.danger_near(5, 5), 1)


if __name__ == "__main__":
    unittest.main()
