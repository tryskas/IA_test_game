import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "real_game"))

from config import KEYS  # noqa: E402
from game_state import Node, PlayerView  # noqa: E402
from moves import Calibration, Mover, SkillError, calibrate, wrap  # noqa: E402


class FakeGame:
    """A tiny flat world with the same basic abilities as the real game."""

    def __init__(self, yaw_sign=-1, yaw_offset=90.0, per_count=0.07, nodes=(), blocked_s=0.0, fail_after=None):
        self.yaw_sign, self.yaw_offset, self.per_count = yaw_sign, yaw_offset, per_count
        self.x = self.y = 0.0
        self.facing = 30.0  # map direction the player faces
        self.pitch = 0.0  # degrees looking down
        self.t = 0.0
        self.held: dict[str, float] = {}
        self.inventory: dict[str, float] = {}
        self._nodes = tuple(nodes)
        self.blocked_s = blocked_s  # cannot walk at first, like against a rock
        self.fail_after = fail_after  # raise after this many waits, like F12

    def player(self):
        yaw = (self.facing - self.yaw_offset) * self.yaw_sign
        return PlayerView(self.x, self.y, 0.0, yaw % 360, dict(self.inventory), {})

    def nodes(self):
        return self._nodes

    def mouse(self, dx, dy):
        self.facing = wrap(self.facing + dx * self.per_count)
        self.pitch = max(-89, min(89, self.pitch + dy * abs(self.per_count)))

    def hold(self, key):
        self.held[key] = self.t

    def release(self, key):
        pressed_at = self.held.pop(key, None)
        if key == KEYS["interact"] and pressed_at is not None:
            self._try_mining()
        if key == KEYS["jump"]:
            self.blocked_s = 0.0  # jumping gets over the rock

    def _try_mining(self):
        for node in self._nodes:
            distance = math.hypot(node.x - self.x, node.y - self.y)
            direction = math.degrees(math.atan2(node.y - self.y, node.x - self.x))
            needed_pitch = math.degrees(math.atan2(1.7, max(distance, 0.1)))
            if distance < 3.2 and abs(wrap(direction - self.facing)) < 25 and abs(self.pitch - needed_pitch) < 9:
                self.inventory[node.resource] = self.inventory.get(node.resource, 0) + 1

    def now(self):
        return self.t

    def wait(self, seconds):
        if self.fail_after is not None:
            self.fail_after -= 1
            if self.fail_after < 0:
                raise RuntimeError("stop")
        steps = max(1, round(seconds / 0.05))
        for _ in range(steps):
            dt = seconds / steps
            self.t += dt
            if KEYS["forward"] in self.held and self.t > self.blocked_s:
                self.x += 5.5 * math.cos(math.radians(self.facing)) * dt
                self.y += 5.5 * math.sin(math.radians(self.facing)) * dt


class WallWorld(FakeGame):
    """The fake world, with walls (rectangles) the player can't walk through."""

    def __init__(self, walls, **kwargs):
        super().__init__(**kwargs)
        self.walls = walls
        self.moves: list[int] = []

    def inside_wall(self, x, y):
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in self.walls)

    def mouse(self, dx, dy):
        self.moves.append(abs(dx))
        super().mouse(dx, dy)

    def wait(self, seconds):
        x, y = self.x, self.y
        super().wait(seconds)
        if self.inside_wall(self.x, self.y):
            self.x, self.y = x, y  # bumped into a wall


class RayEyes:
    """Sees the walls of a WallWorld, as the depth analysis would report them."""

    def __init__(self, world, fov=90.0, aspect=9 / 16):
        from perception import View

        self.world, self.fov, self.aspect, self.View = world, fov, aspect, View

    @property
    def latest(self):
        import math as m

        rows = []
        for i in range(15):
            bearing = m.degrees(m.atan((i + 0.5 - 7.5) / 7.5 * m.tan(m.radians(self.fov / 2))))
            angle = m.radians(self.world.facing + bearing)
            hit = None
            for k in range(1, 49):
                d = k * 0.25
                if self.world.inside_wall(self.world.x + d * m.cos(angle), self.world.y + d * m.sin(angle)):
                    hit = d
                    break
            if hit is None:
                rows.append(None)
                continue
            below = m.degrees(m.atan(1.7 / hit)) - self.world.pitch
            rows.append((0.5 * m.tan(m.radians(below)) + self.aspect / 2) / self.aspect)
        return self.View(self.world.now(), (), (0.9,) * 15, self.fov, tuple(rows), (False,) * 15, self.aspect)


class CreatureEyes:
    """Always sees one creature, 30° to the right, of a given size on screen."""

    def __init__(self, game, size):
        from perception import Threat, View

        self.game, self.size, self.View, self.Threat = game, size, View, Threat
        self.confidence = 0.18

    def fresh(self, since, timeout=2.0):
        threat = self.Threat(30.0, 0.0, self.size, 0.5, "animal", (0, 0, 1, 1))
        return self.View(self.game.now(), (threat,), (), 90.0)


class CalibrationTest(unittest.TestCase):
    def test_finds_how_yaw_and_mouse_relate_to_the_map(self):
        for sign, offset, per_count in ((-1, 90.0, 0.07), (1, -45.0, -0.05), (1, 0.0, 0.12)):
            game = FakeGame(sign, offset, per_count)
            cal = calibrate(game)
            self.assertEqual(cal.yaw_sign, sign)
            self.assertAlmostEqual(wrap(cal.yaw_offset - offset), 0, delta=1)
            self.assertAlmostEqual(cal.facing_per_count, per_count, delta=abs(per_count) * 0.05)
            self.assertAlmostEqual(cal.walk_speed, 5.5, delta=0.6)

    def test_refuses_when_the_camera_does_not_turn(self):
        game = FakeGame(per_count=0.0)
        with self.assertRaises(SkillError):
            calibrate(game)


class MoverTest(unittest.TestCase):
    def mover(self, game):
        return Mover(game, calibrate(game), log=lambda *_: None)

    def test_walks_to_a_point_in_any_direction(self):
        for target in ((30, -40), (-25, 5), (0, 60)):
            game = FakeGame()
            self.assertTrue(self.mover(game).walk_to(*target, radius=2))
            self.assertLess(math.hypot(target[0] - game.x, target[1] - game.y), 2.01)
            self.assertNotIn(KEYS["forward"], game.held)

    def test_gets_unstuck_by_jumping(self):
        game = FakeGame()
        mover = self.mover(game)
        game.blocked_s = game.t + 100  # stuck until it jumps
        self.assertTrue(mover.walk_to(game.x + 20, game.y, radius=2))

    def test_mines_the_nearest_free_node(self):
        nodes = (
            Node("iron_ore", "normal", 40.0, 10.0, 0.0, False),
            Node("iron_ore", "pure", 5.0, 5.0, 0.0, True),  # taken by a miner
        )
        game = FakeGame(nodes=nodes)
        mined = self.mover(game).mine_by_hand("iron_ore", 10)
        self.assertGreaterEqual(mined, 10)
        self.assertLess(math.hypot(40 - game.x, 10 - game.y), 3)

    def test_says_so_when_no_node_is_known(self):
        game = FakeGame()
        with self.assertRaises(SkillError):
            self.mover(game).mine_by_hand("copper_ore", 5)

    def test_walks_round_a_wall_it_discovers(self):
        world = WallWorld(walls=[(18, -15, 20, 15)])  # a wall across the straight line
        mover = Mover(world, calibrate(world), log=lambda *_: None)
        world.x, world.y = 0.0, 0.0
        mover.eyes = RayEyes(world)
        self.assertTrue(mover.walk_to(40, 0, radius=2))
        self.assertTrue(any(mover.terrain.is_blocked(mover.terrain.cell(19, y)) for y in range(-6, 7)))

    def test_turns_smoothly(self):
        world = WallWorld(walls=[])
        mover = Mover(world, calibrate(world), log=lambda *_: None)
        world.moves.clear()
        mover.turn(180)
        self.assertGreater(len(world.moves), 10)  # many small moves...
        self.assertLess(max(world.moves), 0.1 * sum(world.moves))  # ...none of them a jump

    def fight(self, size, hurt):
        from defense import defend

        game = FakeGame()
        game.alarms, game.ignore_sightings_until, game.clicks = True, 0.0, 0
        game.click = lambda button="left": setattr(game, "clicks", game.clicks + 1)
        mover = Mover(game, calibrate(game), log=lambda *_: None)
        outcome = defend(game, mover, CreatureEyes(game, size), home=(0, 0), hurt=hurt, log=lambda *_: None)
        self.assertTrue(game.alarms)  # alarms back on after the fight
        return outcome, game

    def test_leaves_a_calm_creature_alone(self):
        outcome, game = self.fight(size=0.25, hurt=False)
        self.assertEqual(outcome, "harmless")
        self.assertEqual(game.clicks, 0)
        self.assertGreater(game.ignore_sightings_until, game.now())

    def test_strikes_back_when_hurt(self):
        outcome, game = self.fight(size=0.25, hurt=True)
        self.assertGreater(game.clicks, 0)

    def test_releases_keys_when_stopped(self):
        game = FakeGame()
        mover = Mover(game, Calibration(-1, 90.0, 0.07, 0.07, 5.5), log=lambda *_: None)
        game.fail_after = 5
        with self.assertRaises(RuntimeError):
            mover.walk_to(100, 0)
        self.assertNotIn(KEYS["forward"], game.held)


if __name__ == "__main__":
    unittest.main()
