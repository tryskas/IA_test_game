import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from perception import (  # noqa: E402
    BLOCKED, View, bearing_of, clearance_from_depth, threats_from_boxes,
)


def scene(wall_columns=slice(0, 0), near_is_big=True, width=150, height=100, horizon=0.45) -> np.ndarray:
    """A made-up depth picture with real perspective: on flat ground, nearness grows
    with the distance below the horizon line; the sky is far. Maybe a wall, as near
    as the ground at the player's feet."""
    rows = np.clip(np.linspace(0, 1, height) - horizon, 0, None)[:, None] + 1e-3
    nearness = np.repeat(rows, width, axis=1)
    nearness[:, wall_columns] = rows.max()
    return nearness if near_is_big else 1.0 / nearness


class GeometryTest(unittest.TestCase):
    def test_bearing_is_zero_in_the_middle_and_half_the_fov_at_the_edges(self):
        self.assertAlmostEqual(bearing_of(480, 960, 90), 0)
        self.assertAlmostEqual(bearing_of(960, 960, 90), 45)
        self.assertAlmostEqual(bearing_of(0, 960, 90), -45)


class ClearanceTest(unittest.TestCase):
    def test_open_ground_is_open_everywhere(self):
        self.assertTrue(all(c >= 0.5 for c in clearance_from_depth(scene())))

    def test_a_wall_on_the_left_blocks_the_left_columns_only(self):
        clearance = clearance_from_depth(scene(slice(0, 50)))  # left third
        self.assertTrue(all(c < 0.5 for c in clearance[:4]))
        self.assertTrue(all(c >= 0.5 for c in clearance[6:]))

    def test_works_whichever_way_the_model_counts(self):
        a = clearance_from_depth(scene(slice(0, 50), near_is_big=True))
        b = clearance_from_depth(scene(slice(0, 50), near_is_big=False))
        self.assertEqual([c >= 0.5 for c in a], [c >= 0.5 for c in b])
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


if __name__ == "__main__":
    unittest.main()
