import math
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

    def test_a_wall_across_the_straight_line_is_noticed(self):
        terrain = TerrainMap()
        for y in range(-10, 11, 2):
            terrain.mark_blocked(20, y, strength=2)
        self.assertTrue(terrain.path_is_clear([(40, 0)]))  # the waypoint itself is clear...
        self.assertFalse(terrain.route_is_clear((0, 0), [(40, 0)]))  # ...the way to it is not
        self.assertTrue(terrain.route_is_clear((0, 0), [(0, 30), (40, 30)]))
        terrain.mark_blocked(40, 0, strength=2)  # a node to mine, seen as in the way
        self.assertTrue(terrain.route_is_clear((30, 0), [(40, 0)], goal=(40, 0)))

    def test_what_is_seen_goes_on_the_map_and_is_recalled(self):
        terrain = TerrainMap()
        for _ in range(2):
            terrain.note_sight(0, 0, 0, [-20, 0, 20], [None, 3.0, None], [False, False, True], drop_ahead=4.0)
        self.assertTrue(terrain.is_blocked(terrain.cell(3, 0)))
        drop = (4 * math.cos(math.radians(20)), 4 * math.sin(math.radians(20)))
        self.assertTrue(terrain.is_blocked(terrain.cell(*drop)))
        self.assertEqual(terrain.blocked_ahead(0, 0, 0, [-20, 0]), [None, 3.0])

    def test_a_mistake_of_the_eyes_is_undone_by_clear_looks(self):
        terrain = TerrainMap()
        for _ in range(2):
            terrain.note_sight(0, 0, 0, [0], [8.0], [False])  # wrongly seen in the way, twice
        self.assertTrue(terrain.is_blocked(terrain.cell(8, 0)))
        for _ in range(3):
            terrain.note_sight(0, 0, 0, [0], [None], [False])  # then seen clear
        self.assertFalse(terrain.is_blocked(terrain.cell(8, 0)))

    def test_what_was_bumped_into_stays_until_walked_on(self):
        terrain = TerrainMap()
        terrain.mark_blocked(8, 0, strength=2, solid=True)
        for _ in range(5):
            terrain.note_sight(0, 0, 0, [0], [None], [False])
        self.assertTrue(terrain.is_blocked(terrain.cell(8, 0)))
        again = TerrainMap(terrain.to_data())
        self.assertTrue(again.is_blocked(again.cell(8, 0)))
        again.mark_free(8, 0)
        self.assertFalse(again.is_blocked(again.cell(8, 0)))

    def test_an_edge_is_kept_while_it_cannot_be_seen(self):
        terrain = TerrainMap()
        terrain.note_sight(0, 0, 0, [0], [None], [6.0])  # the ground falls away 6 m ahead
        self.assertTrue(terrain.is_blocked(terrain.cell(6, 0)))
        terrain.note_sight(-6, 0, 0, [0], [None], [None])  # 12 m away: too far to see an edge
        terrain.note_sight(5, 0, 0, [0], [None], [None])  # 1 m away: hidden under the HUD
        self.assertTrue(terrain.is_blocked(terrain.cell(6, 0)))
        terrain.note_sight(0, 0, 0, [0], [None], [None])  # where it should show: seen safe after all
        self.assertFalse(terrain.is_blocked(terrain.cell(6, 0)))

    def test_a_ledge_walked_down_is_not_climbed_back(self):
        terrain = TerrainMap()
        for x in range(0, 22):
            terrain.mark_free(x + 0.5, 0.5, 10.0)  # on top
        for x in range(22, 41):
            terrain.mark_free(x + 0.5, 0.5, 6.0)  # 4 m lower, after the ledge
        top, bottom = terrain.cell(19, 0.5), terrain.cell(23, 0.5)
        self.assertTrue(terrain.can_climb(top, bottom))  # going down: fine
        self.assertFalse(terrain.can_climb(bottom, top))  # going back up: no
        path = terrain.plan((40, 0.5), (0, 0.5))
        self.assertTrue(any(abs(y - 0.5) > 1.0 for _, y in path[:-1]))  # round the ledge, not straight back up
        again = TerrainMap(terrain.to_data())
        self.assertFalse(again.can_climb(bottom, top))


if __name__ == "__main__":
    unittest.main()
