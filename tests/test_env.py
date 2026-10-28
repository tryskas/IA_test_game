import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from mini_satisfactory import data
from mini_satisfactory.bots import FACTORY_PLAN, ScriptedBot
from mini_satisfactory.env import DELIVERY_POINTS, POINTS_PER_UPGRADE, MiniSatisfactoryEnv


class EnvTest(unittest.TestCase):
    def test_follows_the_gymnasium_api(self):
        check_env(MiniSatisfactoryEnv(), skip_render_check=True)

    def test_reward_is_upgrade_points_minus_minutes(self):
        env = MiniSatisfactoryEnv()
        env.reset()
        bot = ScriptedBot(FACTORY_PLAN)
        total, done, info = 0.0, False, {}
        while not done:
            _, reward, terminated, truncated, info = env.step(bot.act(env.sim))
            total += reward
            done = terminated or truncated
        self.assertTrue(terminated)
        points = (POINTS_PER_UPGRADE + DELIVERY_POINTS) * len(data.HUB_UPGRADES)
        expected = points - info["game_time_s"] / 60
        self.assertAlmostEqual(total, expected, places=6)

    def test_random_masked_run_stops_at_the_time_limit(self):
        env = MiniSatisfactoryEnv(time_limit_min=20)
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        done = False
        while not done:
            self.assertTrue(env.observation_space.contains(obs))
            mask = env.action_masks()
            obs, _, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(mask)))
            done = terminated or truncated
        self.assertTrue(truncated)
        self.assertGreaterEqual(info["game_time_s"], 20 * 60)

    def test_impossible_action_waits_and_costs_a_penalty(self):
        env = MiniSatisfactoryEnv()
        env.reset()
        submit = len(env.action_masks()) - 1
        self.assertFalse(env.action_masks()[submit])
        _, reward, *_ = env.step(submit)
        self.assertLess(reward, -data.WAIT_SECONDS / 60)
        self.assertEqual(env.sim.time, data.WAIT_SECONDS)


if __name__ == "__main__":
    unittest.main()
