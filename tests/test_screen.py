import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_game"))

from menus import BUILD_TILES_BELOW, ZAPPER_OFFSET  # noqa: E402
from screen import TextBox, best_match, normalize  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
