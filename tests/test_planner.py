import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_game"))

from planner import choose_milestone, next_step, parse_schematics, raw_needs  # noqa: E402

# The game's own milestones and recipes (Tier 0 to 2), read with the mod.
RECORDED = json.loads((HERE / "fixtures" / "frm_milestones_t0_t2.json").read_text(encoding="utf-8"))
CONCRETE, ROD, PLATE, LIMESTONE = "Desc_Cement_C", "Desc_IronRod_C", "Desc_IronPlate_C", "Desc_Stone_C"
IRON_ORE = "Desc_OreIron_C"


def after_tier0(remaining: dict | None = None):
    """The user's save: Tier 0 done, Tier 1 open; `remaining` sets « Construction de base »."""
    schematics = copy.deepcopy(RECORDED)
    for s in schematics:
        if s["Type"] == "Tutorial":
            s["Purchased"] = True
        if s["TechTier"] == 1:
            s["Locked"] = False
        for cost in s["Cost"]:
            cost.setdefault("TotalCost", cost["Amount"])
            cost.setdefault("RemainingCost", cost["Amount"])
            if s["ClassName"] == "Schematic_1-1_C" and remaining and cost["ClassName"] in remaining:
                cost["RemainingCost"] = remaining[cost["ClassName"]]
    return parse_schematics(schematics)


class PlannerTest(unittest.TestCase):
    def test_knows_what_the_workbench_can_make_once_tier0_is_done(self):
        plan = after_tier0()
        self.assertEqual(plan.recipes[CONCRETE].name, "Béton")
        self.assertEqual(plan.recipes[CONCRETE].inputs, {LIMESTONE: 3})
        self.assertIn("Desc_Wire_C", plan.recipes)
        self.assertNotIn("Desc_Rotor_C", plan.recipes)  # unlocked in Tier 2 only

    def test_picks_the_cheapest_open_milestone(self):
        step = next_step(after_tier0(), {}, active=None)
        self.assertEqual(step.kind, "select")
        self.assertEqual(step.milestone.name, "Étude sur le terrain")  # 300 wire, 300 screws, 100 plates

    def test_waits_for_the_hub_ship(self):
        self.assertEqual(next_step(after_tier0(), {}, active=None, ship_docked=False).kind, "wait")

    def test_mines_all_the_ore_needed_then_crafts_it_all(self):
        # The demo's state: 58 concrete, 100 plates and 58 rods already given.
        plan = after_tier0({CONCRETE: 142, PLATE: 0, ROD: 42})
        active = "Schematic_1-1_C"
        step = next_step(plan, {PLATE: 29}, active=active)  # 142 concrete: 426 limestone, 300 a trip
        self.assertEqual((step.kind, step.item, step.count), ("mine", LIMESTONE, 300))
        step = next_step(plan, {LIMESTONE: 300}, active=active)
        self.assertEqual((step.kind, step.item, step.count), ("mine", LIMESTONE, 126))
        step = next_step(plan, {LIMESTONE: 426}, active=active)  # then the iron for 42 rods
        self.assertEqual((step.kind, step.item, step.count), ("mine", IRON_ORE, 42))
        step = next_step(plan, {LIMESTONE: 426, IRON_ORE: 42}, active=active)
        self.assertEqual((step.kind, step.recipe.name, step.count), ("craft", "Béton", 142))

    def test_field_research_needs_all_its_copper_at_once(self):
        step = next_step(after_tier0(), {}, active=None)
        plan = after_tier0()
        needs = raw_needs(plan, step.milestone.remaining, {})
        self.assertEqual(needs.get("Desc_OreCopper_C"), 150)  # 300 wire: 150 ingots, 150 copper ore

    def test_deposits_once_everything_is_carried(self):
        plan = after_tier0({CONCRETE: 142, PLATE: 0, ROD: 42})
        step = next_step(plan, {CONCRETE: 142, ROD: 50}, active="Schematic_1-1_C")
        self.assertEqual(step.kind, "deposit")

    def test_sets_up_the_base_between_iron_and_copper_close_together(self):
        from game_state import Node
        from planner import choose_base

        nodes = [Node("iron_ore", "normal", 100, 0, 0, False), Node("copper_ore", "normal", 400, 0, 0, False),
                 Node("iron_ore", "normal", 300, 300, 0, False), Node("copper_ore", "normal", 330, 300, 0, False),
                 Node("limestone", "normal", 315, 320, 0, False)]
        x, y, apart, trip = choose_base(nodes, 0.0, 0.0)
        self.assertEqual((x, y, apart), (315.0, 300.0, 30.0))
        self.assertIsNone(choose_base(nodes[:2], 0.0, 0.0))  # 300 m apart: too far

    def test_never_picks_a_milestone_needing_parts_it_cannot_make(self):
        plan = after_tier0()
        for s in plan.milestones.values():
            object.__setattr__(s, "locked", False) if s.tier == 2 else None
        while (target := choose_milestone(plan)) is not None:
            self.assertNotEqual(target.name, "Tremplins")  # needs rotors
            object.__setattr__(target, "purchased", True)


if __name__ == "__main__":
    unittest.main()
