import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from perception import (  # noqa: E402
    BLOCKED, View, bearing_of, clearance_from_depth, threats_from_boxes,
)


def scene(wall_columns=slice(0, 0), wall_at=0.85, near_is_big=True, width=150, height=100,
          horizon=0.40, offset=0.0) -> np.ndarray:
    """A made-up depth picture with real perspective: on flat ground, nearness grows
    with the distance below the horizon line, and the sky is far. A wall stands as
    near as the ground seen at row `wall_at`. `offset` shifts every value, as depth
    models give relative numbers."""
    rows = np.clip(np.linspace(0, 1, height) - horizon, 0, None)[:, None] + 1e-3
    nearness = np.repeat(rows, width, axis=1)
    nearness[: int(height * wall_at), wall_columns] = wall_at - horizon
    nearness = nearness + offset
    return nearness if near_is_big else 1.0 / nearness


class GeometryTest(unittest.TestCase):
    def test_bearing_is_zero_in_the_middle_and_half_the_fov_at_the_edges(self):
        self.assertAlmostEqual(bearing_of(480, 960, 90), 0)
        self.assertAlmostEqual(bearing_of(960, 960, 90), 45)
        self.assertAlmostEqual(bearing_of(0, 960, 90), -45)


class ClearanceTest(unittest.TestCase):
    def open_columns(self, depth):
        return [c >= 0.5 for c in clearance_from_depth(depth)]

    def test_open_ground_is_open_whatever_the_camera_tilt(self):
        for horizon in (0.25, 0.40, 0.55):
            self.assertTrue(all(self.open_columns(scene(horizon=horizon))), horizon)

    def test_a_near_wall_on_the_left_blocks_the_left_columns_only(self):
        for horizon in (0.30, 0.45):
            columns = self.open_columns(scene(slice(0, 50), horizon=horizon))
            self.assertFalse(any(columns[:4]), horizon)
            self.assertTrue(all(columns[6:]), horizon)

    def test_a_far_wall_is_not_in_the_way(self):
        self.assertTrue(all(self.open_columns(scene(slice(0, 150), wall_at=0.50, horizon=0.40))))

    def test_the_players_hand_is_not_an_obstacle(self):
        depth = scene()
        depth[55:100, 85:150] = depth.max() * 3  # the hand and the tool, bottom right, very near
        self.assertTrue(all(self.open_columns(depth)))

    def test_a_rock_on_the_right_still_blocks_the_right(self):
        columns = self.open_columns(scene(slice(110, 150)))
        self.assertFalse(any(columns[-3:]))
        self.assertTrue(all(columns[:9]))

    def test_works_whichever_way_and_whatever_scale_the_model_uses(self):
        expected = self.open_columns(scene(slice(0, 50)))
        self.assertEqual(self.open_columns(scene(slice(0, 50), near_is_big=False)), expected)
        self.assertEqual(self.open_columns(scene(slice(0, 50), offset=3.0)), expected)
        self.assertGreater(BLOCKED, 0)


class ViewTest(unittest.TestCase):
    def view(self, clearance):
        return View(0.0, (), tuple(clearance), 90.0)

    def test_steers_to_the_nearest_open_direction(self):
        view = self.view([0.9] * 5 + [0.1] * 5 + [0.9] * 5)  # blocked straight ahead
        self.assertFalse(view.is_open(0))
        best = view.best_direction(5)
        self.assertGreater(best, 10)  # goes round on the right, the side of the target
        self.assertTrue(view.is_open(best))

    def test_nothing_open_gives_no_direction(self):
        self.assertIsNone(self.view([0.1] * 15).best_direction(0))

    def test_ignores_the_players_own_hand_and_hud(self):
        boxes = [
            (100, 100, 200, 300, 0.8, "creature"),  # in the world, on the left
            (700, 400, 900, 540, 0.9, "creature"),  # bottom right: the player's hand and tool
        ]
        threats = threats_from_boxes(boxes, 960, 540, 90)
        self.assertEqual(len(threats), 1)
        self.assertLess(threats[0].bearing, -20)
        self.assertAlmostEqual(threats[0].size, 200 / 540)

    def test_a_creature_right_next_to_the_player_is_seen(self):
        boxes = [(500, 250, 900, 540, 0.4, "animal")]  # big, low on the right: over the hand, but larger
        threats = threats_from_boxes(boxes, 960, 540, 90)
        self.assertEqual(len(threats), 1)
        self.assertGreater(threats[0].bearing, 10)


if __name__ == "__main__":
    unittest.main()
