import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "real_game"))

from game_state import Node, read_state, to_simulation  # noqa: E402

from mini_satisfactory import data  # noqa: E402

# Answers recorded from the real game (version 1.2.4.1) with the Ficsit Remote Monitoring mod.
RECORDED = {
    "getPlayer": json.loads((HERE / "fixtures" / "frm_getPlayer.json").read_text(encoding="utf-8")),
    "getSchematics": json.loads((HERE / "fixtures" / "frm_getSchematics.json").read_text(encoding="utf-8")),
    "getFactory": [],
    # Made up: the real format of this answer has not been recorded yet.
    "getResourceNode": [
        {"Name": "Minerai de fer", "ClassName": "Desc_OreIron_C", "Purity": "Normal", "EnumPurity": "RP_Normal",
         "Exploited": False, "location": {"x": -3627.98, "y": 257388.27 + 4000, "z": -1859.69}},
        {"Name": "Minerai de fer", "ClassName": "Desc_OreIron_C", "Purity": "Impur", "EnumPurity": "RP_Inpure",
         "Exploited": False, "location": {"x": -3627.98 + 20000, "y": 257388.27, "z": -1859.69}},
        {"Name": "Calcaire", "ClassName": "Desc_Stone_C", "Purity": "Pur", "EnumPurity": "RP_Pure",
         "Exploited": True, "location": {"x": 0, "y": 0, "z": 0}},
        {"Name": "Geyser", "ClassName": "Desc_Geyser_C", "location": {"x": 0, "y": 0, "z": 0}},
    ],
}


def answers(overrides: dict | None = None):
    recorded = copy.deepcopy(RECORDED)
    recorded.update(overrides or {})
    return lambda endpoint: recorded[endpoint]


class ReadStateTest(unittest.TestCase):
    def test_reads_the_recorded_start_of_a_game(self):
        state = read_state(answers())
        self.assertEqual(state.upgrades_done, 0)
        self.assertEqual(state.inventory, {})
        self.assertEqual(state.other_items, {"Repousse-xénos": 1})
        self.assertAlmostEqual(state.position[0], -36.28, places=2)  # centimetres -> metres
        self.assertAlmostEqual(state.yaw, 330.98, places=2)

    def test_counts_delivered_upgrades_and_items_already_in_the_hub(self):
        schematics = copy.deepcopy(RECORDED["getSchematics"])
        for s in schematics:
            if s["ClassName"] == "Schematic_Tutorial1_C":
                s["Purchased"] = True
            if s["ClassName"] == "Schematic_Tutorial1_5_C":
                s["Cost"][0]["RemainingCost"] = 5  # 15 of the 20 iron rods already given
        player = copy.deepcopy(RECORDED["getPlayer"])
        player[0]["Inventory"].append({"Name": "Plaque de fer", "ClassName": "Desc_IronPlate_C", "Amount": 4})
        state = read_state(answers({"getSchematics": schematics, "getPlayer": player}))
        self.assertEqual(state.upgrades_done, 1)
        self.assertEqual(state.deposited, {"iron_rod": 15})
        sim = to_simulation(state)
        self.assertEqual(sim.upgrades_done, 1)
        self.assertIn("portable_miner", sim.unlocked)
        self.assertEqual(sim.stock["iron_rod"], 15)
        self.assertEqual(sim.stock["iron_plate"], 4)

    def test_finds_the_nearest_free_node_and_the_way_to_it(self):
        state = read_state(answers())
        self.assertEqual(len(state.nodes), 3)  # the geyser is not a mini-game resource
        node = state.nearest("iron_ore")
        self.assertEqual(node.purity, "normal")
        self.assertAlmostEqual(state.distance_to(node), 40, places=0)
        self.assertIsNone(state.nearest("limestone"))  # the only one is taken
        self.assertEqual(state.nearest("limestone", free_only=False).purity, "pure")

    def test_turn_is_positive_to_the_right(self):
        state = read_state(answers())
        state = type(state)((0, 0, 0), 0.0, {}, {}, 0, {}, ())
        self.assertAlmostEqual(state.turn_towards(Node("iron_ore", "normal", 0, 10, 0, False)), 90)
        self.assertAlmostEqual(state.turn_towards(Node("iron_ore", "normal", 0, -10, 0, False)), -90)
        self.assertAlmostEqual(state.turn_towards(Node("iron_ore", "normal", 10, 0, 0, False)), 0)

    def test_workshop_found_in_the_factory(self):
        state = read_state(answers({"getFactory": [{"ClassName": "Build_Workshop_C"}]}))
        self.assertIn("workshop", to_simulation(state).structures)

    def test_hub_costs_match_the_mini_game(self):
        """The game's own HUB upgrade costs must be the ones the AI was trained with."""
        from game_state import HUB_SCHEMATICS, ITEM_CLASSES

        schematics = {s["ClassName"]: s for s in RECORDED["getSchematics"]}
        for name, upgrade in zip(HUB_SCHEMATICS, data.HUB_UPGRADES):
            cost = {ITEM_CLASSES[c["ClassName"]]: c["TotalCost"] for c in schematics[name]["Cost"]}
            self.assertEqual(cost, upgrade.cost)


if __name__ == "__main__":
    unittest.main()
