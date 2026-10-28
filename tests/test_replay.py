import os
import tempfile
import unittest

from mini_satisfactory import data
from mini_satisfactory.bots import FACTORY_PLAN, RandomBot, ScriptedBot
from mini_satisfactory.replay import RUNNING, record
from mini_satisfactory.sim import SIM_TICK

try:
    import pygame
except ImportError:  # the window is optional for the rest of the project
    pygame = None


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.film = record(ScriptedBot(FACTORY_PLAN), "usine")

    def test_films_the_whole_run_every_tick(self):
        times = [f.time for f in self.film.frames]
        self.assertEqual(times[0], 0)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertLessEqual(max(b - a for a, b in zip(times, times[1:])), SIM_TICK + 1e-9)
        self.assertTrue(self.film.finished)
        self.assertAlmostEqual(self.film.end_time, self.film.upgrade_times[-1])

    def test_steps_follow_each_other(self):
        for before, after in zip(self.film.steps, self.film.steps[1:]):
            self.assertAlmostEqual(before.start + before.duration, after.start)
        self.assertEqual(self.film.step_at(0), 0)
        self.assertEqual(self.film.step_at(self.film.end_time + 100), len(self.film.steps) - 1)

    def test_progress_goes_from_zero_to_six(self):
        self.assertEqual(self.film.progress_at(0), 0)
        self.assertEqual(self.film.progress_at(self.film.end_time), len(data.HUB_UPGRADES))
        middle = self.film.upgrade_times[2] + 1
        self.assertGreaterEqual(self.film.progress_at(middle), 3)

    def test_machines_are_seen_working(self):
        statuses = {m.status for f in self.film.frames for m in f.machines}
        self.assertIn(RUNNING, statuses)


@unittest.skipIf(pygame is None, "pygame is not installed")
class ViewerTest(unittest.TestCase):
    def test_draws_every_moment_of_every_run(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from mini_satisfactory import viewer

        runs = [
            record(ScriptedBot(FACTORY_PLAN), "usine"),
            record(ScriptedBot(), "script"),
            record(RandomBot(seed=0), "hasard", time_limit_s=1200),
        ]
        with tempfile.TemporaryDirectory() as folder:
            for current in range(len(runs)):
                for minutes in (0, 5, 20, 40, 200):
                    path = os.path.join(folder, f"{current}_{minutes}.png")
                    viewer.capture(runs, path, minutes * 60, current)
                    self.assertGreater(os.path.getsize(path), 0)

    def test_keys_change_speed_time_and_player(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from mini_satisfactory import viewer

        pygame.font.init()
        runs = [record(ScriptedBot(FACTORY_PLAN), "usine"), record(ScriptedBot(), "script")]
        view = viewer.Viewer(runs, speed=60)
        key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k)
        view.handle(key(pygame.K_UP), None)
        self.assertEqual(viewer.SPEEDS[view.speed_index], 120)
        view.handle(key(pygame.K_RIGHT), None)
        self.assertEqual(view.time, viewer.SEEK_SECONDS)
        view.handle(key(pygame.K_2), None)
        self.assertEqual(view.current, 1)
        view.handle(key(pygame.K_SPACE), None)
        self.assertTrue(view.paused)


if __name__ == "__main__":
    unittest.main()
