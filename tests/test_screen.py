import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_game"))

from menus import BUILD_TILES_BELOW, ZAPPER_OFFSET, building_position  # noqa: E402
from screen import TextBox, best_match, hologram_state, normalize  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

DEMO = HERE.parent / "demos" / "demo_20261029_094720" / "shots"


def recorded(name: str) -> list[TextBox]:
    """What the OCR read on a screenshot of the demonstration (2560 x 1440 screen)."""
    boxes = json.loads((HERE / "fixtures" / f"ocr_{name}.json").read_text(encoding="utf-8"))
    return [TextBox(**b) for b in boxes]


def near(box: TextBox, x: float, y: float, tolerance: float = 60) -> bool:
    return abs(box.x - x) <= tolerance and abs(box.y - y) <= tolerance


class MatchingTest(unittest.TestCase):
    def test_normalize_drops_accents_case_and_punctuation(self):
        self.assertEqual(normalize("Atelier d'équipement"), "atelierdequipement")
        self.assertEqual(normalize("Le H.U.B."), "lehub")

    def test_finds_the_xeno_zapper_slot_where_it_was_clicked(self):
        label = best_match(recorded("click_0001"), "Raccourcis d'équipement")
        x, y = label.left + ZAPPER_OFFSET[0], label.y + ZAPPER_OFFSET[1]
        self.assertTrue(abs(x - 787) <= 8 and abs(y - 309) <= 8, (x, y))  # the demo's double click

    def test_finds_buildings_in_the_build_menu(self):
        boxes = recorded("click_0003")
        below = BUILD_TILES_BELOW * 1440
        self.assertTrue(near(best_match(boxes, "Le H.U.B.", below=below), 700, 440))
        self.assertTrue(near(best_match(boxes, "Établi", below=below), 686, 668))
        workshop = best_match(boxes, "Atelier d'équipement", below=below)  # written on two lines
        self.assertTrue(near(workshop, 856, 682), workshop)
        self.assertIsNotNone(best_match(boxes, "Constructeur"))

    def test_finds_recipes_at_the_workbench(self):
        boxes = recorded("click_0009")
        self.assertTrue(near(best_match(boxes, "Barre de fer"), 597, 545, 20))  # clicked in the demo
        self.assertTrue(near(best_match(boxes, "Câble"), 575, 830, 20))
        self.assertTrue(near(best_match(boxes, "Plaque de fer"), 616, 490, 10))  # not the reinforced one
        self.assertTrue(near(best_match(boxes, "Plaque de fer renforcée"), 616, 675, 20))
        self.assertIsNotNone(best_match(boxes, "Chercher une formule"))
        self.assertIsNone(best_match(boxes, "Lingot de fer"))  # not on screen (needs the search)

    def test_building_stands_where_the_camera_looked_when_placed(self):
        # Looking 45° down from 1.7 m up, the ground is 1.7 m ahead.
        x, y = building_position({"x": 10.0, "y": 5.0, "facing": 90.0, "pitch": 45.0})
        self.assertAlmostEqual(x, 10.0, places=3)
        self.assertAlmostEqual(y, 6.7, places=3)
        self.assertEqual(building_position({"building": [1.0, 2.0]}), (1.0, 2.0))

    def test_short_words_tolerate_one_misread_letter(self):
        rows = [TextBox("Vls", 538, 392, 50, 30, 0.6), TextBox("[ 228 ]", 810, 392, 60, 24, 0.9)]
        self.assertEqual(best_match(rows, "Vis").text, "Vls")
        self.assertIsNone(best_match(rows, "Fil"))


@unittest.skipIf(cv2 is None, "opencv is not installed")
class HologramTest(unittest.TestCase):
    def state(self, shot: str) -> str:
        image = cv2.imread(str(DEMO / shot))
        h, w = image.shape[:2]
        return hologram_state(image[int(h * 0.25):int(h * 0.95), int(w * 0.2):int(w * 0.8)])

    def test_sees_the_blue_hologram_of_a_building_being_placed(self):
        for shot in ("click_0004.jpg", "click_0006.jpg", "click_0008.jpg"):  # HUB, workbench, workshop
            self.assertEqual(self.state(shot), "valid", shot)

    def test_sees_no_hologram_in_menus_or_while_walking(self):
        for shot in ("click_0003.jpg", "click_0009.jpg", "view_00002.jpg", "view_00052.jpg"):
            self.assertEqual(self.state(shot), "none", shot)


if __name__ == "__main__":
    unittest.main()
