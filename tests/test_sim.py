import unittest

from mini_satisfactory import data
from mini_satisfactory.bots import FACTORY_PLAN, RandomBot, ScriptedBot, play
from mini_satisfactory.sim import ACTION_INDEX, Action, Simulation, missing_for_tier, tier_progress


def do(sim: Simulation, kind: str, target: str | None = None, building: str | None = None) -> float:
    return sim.step(ACTION_INDEX[Action(kind, target, building)])


class HandActionsTest(unittest.TestCase):
    def test_mining_by_hand_gives_ore_and_takes_time(self):
        sim = Simulation()
        seconds = do(sim, "mine", "iron_ore")
        self.assertEqual(seconds, data.HAND_MINE_SECONDS)
        self.assertEqual(sim.stock["iron_ore"], data.HAND_MINE_AMOUNT)

    def test_copper_cannot_be_mined_before_it_is_unlocked(self):
        sim = Simulation()
        self.assertFalse(sim.is_valid(Action("mine", "copper_ore")))
        sim.unlocked.add("copper_ingot")
        self.assertTrue(sim.is_valid(Action("mine", "copper_ore")))

    def test_craft_chains_as_many_crafts_as_fit_in_a_batch(self):
        sim = Simulation()
        sim.stock["iron_ingot"] = 12
        seconds = do(sim, "craft", "iron_rod")  # 2 s per rod by hand, 10 s batch -> 5 rods
        self.assertEqual(sim.stock["iron_rod"], 5)
        self.assertEqual(sim.stock["iron_ingot"], 7)
        self.assertEqual(seconds, 10)

    def test_portable_miner_needs_the_equipment_workshop(self):
        sim = Simulation()
        sim.unlocked.update(data.HUB_UPGRADES[0].unlocks)
        sim.stock.update({"iron_plate": 8, "iron_rod": 8})
        self.assertFalse(sim.is_valid(Action("craft", "portable_miner")))
        do(sim, "build", "workshop", "workshop")
        self.assertIn("workshop", sim.structures)
        self.assertFalse(sim.is_valid(Action("build", "workshop", "workshop")))  # only one
        seconds = do(sim, "craft", "portable_miner")
        self.assertEqual(sim.stock["portable_miner"], 1)
        self.assertEqual(seconds, data.RECIPES["portable_miner"].handcraft_seconds)

    def test_craft_needs_the_ingredients(self):
        sim = Simulation()
        sim.stock["iron_ingot"] = 2
        self.assertFalse(sim.is_valid(Action("craft", "iron_plate")))

    def test_hub_upgrade_takes_the_items_and_unlocks(self):
        sim = Simulation()
        self.assertFalse(sim.is_valid(Action("submit")))
        sim.stock["iron_rod"] = 12
        do(sim, "submit")
        self.assertEqual(sim.upgrades_done, 1)
        self.assertEqual(sim.stock["iron_rod"], 2)
        self.assertIn("portable_miner", sim.unlocked)
        self.assertEqual(sim.upgrade_times, [data.SUBMIT_SECONDS])


class ProgressTest(unittest.TestCase):
    def test_stock_counts_toward_later_upgrades_once(self):
        stock = {item: 0.0 for item in data.ITEMS}
        stock["iron_rod"] = 30  # 10 for upgrade 1, the other 20 for upgrade 2
        # Upgrade 2 also wants 10 plates; a plate takes 1.5 ore, a rod 1.
        self.assertAlmostEqual(tier_progress(stock, 0), 1 + 20 / (20 + 15))
        missing = missing_for_tier(stock, 0)
        self.assertEqual(missing["iron_rod"], 20 + 75 + 100)
        self.assertEqual(missing["iron_plate"], 10 + 20 + 75 + 100)

    def test_finished_tier_has_full_progress(self):
        stock = {item: 0.0 for item in data.ITEMS}
        self.assertEqual(tier_progress(stock, len(data.HUB_UPGRADES)), len(data.HUB_UPGRADES))
        self.assertEqual(missing_for_tier(stock, len(data.HUB_UPGRADES)), {})


class MachinesTest(unittest.TestCase):
    def test_portable_miner_output_waits_for_collection(self):
        sim = Simulation()
        sim.unlocked.add("portable_miner")
        sim.stock["portable_miner"] = 1
        do(sim, "build", "iron_ore", "portable_miner")
        do(sim, "wait")
        self.assertEqual(sim.stock["iron_ore"], 0)
        do(sim, "collect")
        # 20 ore per minute on a normal node, during the wait and the collection.
        expected = 20 / 60 * (data.WAIT_SECONDS + data.COLLECT_SECONDS)
        self.assertAlmostEqual(sim.stock["iron_ore"], expected)

    def test_portable_miner_stops_when_its_buffer_is_full(self):
        sim = Simulation()
        sim.unlocked.add("portable_miner")
        sim.stock["portable_miner"] = 1
        do(sim, "build", "iron_ore", "portable_miner")
        sim.advance(3600)
        self.assertAlmostEqual(sim.machines[0].outputs["iron_ore"], data.ITEMS["iron_ore"].stack)

    def test_miners_take_the_best_free_node(self):
        sim = Simulation()
        sim.unlocked.add("portable_miner")
        sim.stock["portable_miner"] = 4
        for _ in range(4):
            do(sim, "build", "iron_ore", "portable_miner")
        rates = [m.rate * 60 for m in sim.machines]
        self.assertEqual(rates, [20, 20, 20, 10])  # the impure node comes last
        self.assertFalse(sim.is_valid(Action("build", "iron_ore", "portable_miner")))

    def _smelter_sim(self) -> Simulation:
        sim = Simulation()
        sim.unlocked.update({"smelter", "hub_burner", "copper_ingot"})
        sim.stock.update({"iron_rod": 5, "wire": 8, "iron_ore": 100})
        do(sim, "build", "iron_ingot", "smelter")
        return sim

    def test_smelter_does_nothing_without_fuel(self):
        sim = self._smelter_sim()
        do(sim, "collect")  # fills the smelter with ore
        do(sim, "wait")
        self.assertEqual(sim.machines[0].outputs.get("iron_ingot", 0), 0)

    def test_smelter_runs_on_leaves(self):
        sim = self._smelter_sim()
        sim.stock["leaves"] = 100
        do(sim, "collect")
        do(sim, "wait")
        # 30 ingots per minute, burning 4 MW worth of leaves.
        self.assertAlmostEqual(sim.machines[0].outputs["iron_ingot"], 15)
        self.assertAlmostEqual(sim.stock["leaves"], 100 - 4 * 30 / data.LEAF_MJ)

    def test_conveyor_belts_connect_machines_to_the_stock(self):
        sim = self._smelter_sim()
        sim.stock["leaves"] = 100
        do(sim, "collect")
        sim.unlocked.add("conveyor_belt")
        self.assertFalse(sim.is_valid(Action("collect")))
        before = sim.stock["iron_ore"] + sim.machines[0].inputs["iron_ore"]
        sim.machines[0].inputs.clear()
        sim.stock["iron_ore"] = before
        do(sim, "wait")
        self.assertAlmostEqual(sim.stock["iron_ingot"], 15)
        self.assertAlmostEqual(sim.stock["iron_ore"], before - 15)

    def test_belts_unlock_empties_machine_buffers_into_the_stock(self):
        sim = self._smelter_sim()
        sim.stock["leaves"] = 100
        do(sim, "collect")
        do(sim, "wait")
        sim.upgrades_done = 3
        sim.stock.update({"iron_plate": 75, "cable": 20, "concrete": 10})
        do(sim, "submit")  # HUB Upgrade 4: conveyor belts
        self.assertEqual(sim.machines[0].inputs, {})
        self.assertEqual(sim.machines[0].outputs, {})
        self.assertAlmostEqual(sim.stock["iron_ore"] + sim.stock["iron_ingot"], 100, places=6)

    def test_miner_mk1_is_built_over_a_portable_miner(self):
        sim = Simulation()
        sim.unlocked.update({"portable_miner", "miner_mk1", "conveyor_belt"})
        sim.stock["portable_miner"] = 4
        for _ in range(4):
            do(sim, "build", "iron_ore", "portable_miner")
        sim.stock.update({"iron_plate": 10, "concrete": 10})
        do(sim, "build", "iron_ore", "miner_mk1")
        kinds = sorted(m.building for m in sim.machines)
        self.assertEqual(kinds, ["miner_mk1", "portable_miner", "portable_miner", "portable_miner"])
        mk1 = next(m for m in sim.machines if m.building == "miner_mk1")
        self.assertEqual(mk1.rate * 60, 60)  # took a normal node, not the impure one
        self.assertEqual(sim.stock["portable_miner"], 0)


class BotsTest(unittest.TestCase):
    def test_scripted_bot_finishes_tier0(self):
        sim = play(ScriptedBot())
        self.assertTrue(sim.finished)
        self.assertEqual(len(sim.upgrade_times), len(data.HUB_UPGRADES))

    def test_building_a_factory_beats_crafting_by_hand(self):
        simple = play(ScriptedBot())
        factory = play(ScriptedBot(FACTORY_PLAN))
        self.assertTrue(factory.finished)
        self.assertLess(factory.time, simple.time * 0.75)

    def test_random_bot_only_plays_possible_actions(self):
        sim = play(RandomBot(seed=1), time_limit_s=1800)
        self.assertGreaterEqual(sim.time, 1800)
        for value in sim.stock.values():
            self.assertGreater(value, -1e-6)


if __name__ == "__main__":
    unittest.main()
