import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from config import KEYS  # noqa: E402
from terrain_sim import (BEARINGS, COLUMNS, FEATURES, SIZE, STUCK_STEPS, WalkEnv,  # noqa: E402
                         greedy_action)

FLAT = np.zeros((SIZE, SIZE), dtype=np.float32)


def walk(env, actions):
    infos = []
    for action in actions:
        infos.append(env.step(action)[4])
    return infos


class WalkEnvTest(unittest.TestCase):
    def test_follows_the_gymnasium_rules(self):
        from gymnasium.utils.env_checker import check_env

        check_env(WalkEnv(), skip_render_check=True)

    def test_observation_has_the_shared_layout(self):
        obs, _ = WalkEnv().reset(seed=1)
        self.assertEqual(obs.shape, (FEATURES,))
        self.assertEqual(FEATURES, 4 * COLUMNS + 3 + 2 + 6)

    def test_simple_walker_crosses_flat_ground(self):
        env = WalkEnv(noise=False, terrain=FLAT)
        obs, _ = env.reset(seed=3)
        for _ in range(300):
            obs, _, terminated, truncated, info = env.step(greedy_action(obs))
            if terminated or truncated:
                break
        self.assertTrue(info["reached"])

    def test_a_cliff_stops_the_walk_and_goes_on_the_map(self):
        h = FLAT.copy()
        h[:, 70:] = 8.0  # a cliff, 6 m east of the start
        env = WalkEnv(noise=False, terrain=h)
        env.reset(seed=0, options={"start": (63, 64), "goal": (40.5, 64.5), "facing": 0.0})
        obs = env.observe()
        self.assertTrue(obs[COLUMNS // 2])  # seen in the way
        infos = walk(env, [0] * 12)
        self.assertTrue(infos[-1]["bumped"])
        self.assertLess(env.x, 70)
        self.assertTrue(env.observe()[4 * COLUMNS + 3 + 1])  # knows it is stuck
        self.assertTrue(env.map.is_blocked(env.map.cell(env.x + 1.5, env.y)))
        self.assertGreaterEqual(env.stuck, STUCK_STEPS)

    def test_a_drop_is_seen_then_hurts_or_kills(self):
        for depth, dead in ((9.0, False), (20.0, True)):
            h = FLAT.copy()
            h[:, 70:] = -depth  # the ground falls away 6 m east of the start
            env = WalkEnv(noise=False, terrain=h)
            env.reset(seed=0, options={"start": (65, 64), "goal": (40.5, 64.5), "facing": 0.0})
            self.assertTrue(env.observe()[2 * COLUMNS + COLUMNS // 2])  # the edge is seen
            infos = walk(env, [0] * 8)
            self.assertTrue(any(i["fell"] for i in infos))
            self.assertEqual(any(i["dead"] for i in infos), dead)
            self.assertLess(env.hp, 100)

    def test_jumping_gets_up_a_step(self):
        h = FLAT.copy()
        h[:, 66:] = 1.0  # a step too high to walk up
        env = WalkEnv(noise=False, terrain=h)
        env.reset(seed=0, options={"start": (64, 64), "goal": (90.5, 64.5), "facing": 0.0})
        self.assertTrue(walk(env, [0, 0, 0])[-1]["bumped"])
        walk(env, [5])
        self.assertGreaterEqual(env.x, 66)

    def test_columns_match_the_real_view(self):
        from perception import View

        view = View(0.0, (), (1.0,) * COLUMNS, 90.0)
        self.assertTrue(np.allclose(BEARINGS, [view.column_bearing(i) for i in range(COLUMNS)]))


class RealSensesTest(unittest.TestCase):
    def test_built_from_what_the_eyes_see(self):
        from navigation import TerrainMap
        from perception import View, ground_distance
        from walk_policy import observe

        rows = [None] * COLUMNS
        rows[7] = 0.66  # something in the way, straight ahead
        drops = [False] * COLUMNS
        drops[2] = True
        view = View(0.0, (), (1.0,) * COLUMNS, 90.0, tuple(rows), tuple(drops))
        terrain = TerrainMap()
        terrain.mark_blocked(13.0, 1.0, strength=2)  # remembered 3 m ahead
        obs = observe(view, 5.0, terrain, 10.0, 1.0, 0.0, (10.0, 31.0), 80.0, True, 4)
        self.assertEqual(obs.shape, (FEATURES,))
        self.assertEqual(list(obs[:COLUMNS]), [1.0 if i == 7 else 0.0 for i in range(COLUMNS)])
        self.assertAlmostEqual(obs[COLUMNS + 7], 1 - max(ground_distance(0.66, 5.0, 90.0, 9 / 16), 7.6) / 15, 5)
        self.assertEqual(obs[2 * COLUMNS + 2], 1.0)
        self.assertGreater(obs[3 * COLUMNS + 7], 0)  # the map's memory, ahead
        goal = obs[4 * COLUMNS:4 * COLUMNS + 3]
        self.assertAlmostEqual(math.degrees(math.atan2(goal[0], goal[1])), 90.0, 3)  # to the right
        self.assertEqual(obs[4 * COLUMNS + 3], 0.8)
        self.assertEqual(obs[4 * COLUMNS + 4], 1.0)
        self.assertEqual(obs[-6 + 4], 1.0)

    def test_blind_without_a_view(self):
        from navigation import TerrainMap
        from walk_policy import observe

        obs = observe(None, 5.0, TerrainMap(), 0.0, 0.0, 0.0, (20.0, 0.0), 100.0, False, 0)
        self.assertFalse(obs[:4 * COLUMNS].any())


class LearntWalkTest(unittest.TestCase):
    def test_mover_walks_round_a_wall_with_a_walker(self):
        from moves import Mover, calibrate
        from tests.test_moves import RayEyes, WallWorld

        class Walker:
            def __init__(self):
                self.seen = []

            def act(self, obs):
                self.seen.append(obs)
                return greedy_action(obs)

        world = WallWorld(walls=[(18, -15, 20, 15)])  # a wall across the straight line
        mover = Mover(world, calibrate(world), log=lambda *_: None)
        world.x, world.y = 0.0, 0.0
        mover.eyes = RayEyes(world)
        mover.walker = Walker()
        self.assertTrue(mover.walk_to(40, 0, radius=2))
        self.assertGreater(len(mover.walker.seen), 20)
        self.assertTrue(any(obs[:COLUMNS].any() for obs in mover.walker.seen))  # it saw the wall
        self.assertNotIn(KEYS["forward"], world.held)  # keys let go at the end


if __name__ == "__main__":
    unittest.main()
