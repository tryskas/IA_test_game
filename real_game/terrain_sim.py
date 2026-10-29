"""A small simulated landscape to teach the AI to walk: hills, cliffs, rocks, ravines.

Thousands of walks per minute, with the same senses as in the real game: what the
depth analysis reports for 15 directions (in the way or not, how far, ground
falling away), what the learnt map remembers just ahead, where the next waypoint
is, health, being stuck. What is learnt here then drives the character in the
real game (real_game/walk_policy.py, moves.Mover.walk_to).
"""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np

from navigation import BLOCK_SCORE, MEMORY_REACH, TerrainMap

SIZE = 128  # metres; one height per square metre
DT = 0.2  # seconds per decision
SPEED = 5.0  # metres per second, walking (calibrated: 5.03)
STEP_UP = 0.6  # metres the character walks up in one step (1 m) without jumping
JUMP_UP = 1.4  # metres it gets up by jumping
SAFE_FALL = 4.0  # falls up to this height do no harm
DEADLY_FALL = 14.0  # falls from this height kill
COLUMNS = 15
FOV = 90.0
SIGHT = 15.0  # metres the depth analysis reaches
TRUST = 12.0  # obstacles farther than this are not reported (as moves.TRUST_SIGHT)
NEAREST_SEEN = 7.6  # looking 5° down, the depth band starts this far: nearer obstacles read as this
DROP_SEEN = (3.5, 7.6)  # metres ahead where the ground band shows a drop (5° down, above the HUD)
STEEP_UP, STEEP_DOWN = STEP_UP, -2.0  # height change over one metre that shows as a wall / an edge
STEER, TURN = 15.0, 35.0  # degrees: steering while walking, turning on the spot
ACTIONS = ("avancer", "avancer à gauche", "avancer à droite",
           "tourner à gauche", "tourner à droite", "sauter en avançant")
FORWARD_ACTIONS = (0, 1, 2, 5)
GOAL_REACHED = 2.5
MAX_STEPS = 400
STALE_VIEW = 0.3  # chance that the eyes have nothing new this step (the real depth model is slower)
JUMP_COST = 0.05  # jumping for nothing looks odd: only when it helps
STUCK_STEPS = 3  # bumping this many steps in a row (0.6 s) is noticed, as in the real game


def column_bearings() -> np.ndarray:
    """Directions (degrees, + to the right) of the 15 columns, as perception.View has them."""
    focal = (COLUMNS / 2) / math.tan(math.radians(FOV) / 2)
    return np.degrees(np.arctan((np.arange(COLUMNS) + 0.5 - COLUMNS / 2) / focal))


BEARINGS = column_bearings()


def nearness(distance: float | None) -> float:
    """0 when nothing is in the way, else bigger when nearer."""
    if distance is None:
        return 0.0
    return 1.0 - min(max(distance, NEAREST_SEEN), SIGHT) / SIGHT


def recalled(terrain: TerrainMap, x: float, y: float, facing: float) -> np.ndarray:
    """What the map remembers just ahead, per column: 0 nothing, bigger when nearer."""
    reach = MEMORY_REACH[-1] + MEMORY_REACH[0]
    return np.array([0.0 if d is None else 1.0 - d / reach
                     for d in terrain.blocked_ahead(x, y, facing, BEARINGS)], dtype=np.float32)


def features(blocked, near, drops, remembered, goal_bearing: float, goal_distance: float, hp: float,
             stuck: bool, previous_action: int) -> np.ndarray:
    """What the walking policy sees, the same in the simulation and in the real game."""
    one_hot = np.zeros(len(ACTIONS), dtype=np.float32)
    one_hot[previous_action] = 1.0
    goal = [math.sin(math.radians(goal_bearing)), math.cos(math.radians(goal_bearing)),
            min(goal_distance / 50.0, 1.0)]
    return np.concatenate([
        np.asarray(blocked, dtype=np.float32), np.asarray(near, dtype=np.float32),
        np.asarray(drops, dtype=np.float32), np.asarray(remembered, dtype=np.float32),
        np.asarray(goal, dtype=np.float32),
        np.asarray([min(max(hp, 0.0), 100.0) / 100.0, float(stuck)], dtype=np.float32), one_hot,
    ])


FEATURES = len(features([0] * COLUMNS, [0] * COLUMNS, [0] * COLUMNS, [0] * COLUMNS, 0, 0, 100, False, 0))


def make_terrain(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Gentle hills, plateaus with cliffs, ravines and rocks."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    h = np.zeros((size, size), dtype=np.float32)
    for _ in range(rng.integers(3, 7)):
        cx, cy, spread, height = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(10, 30), rng.uniform(-4, 6)
        h += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * spread ** 2))
    for _ in range(rng.integers(2, 6)):  # plateaus: walls to go round
        cx, cy, radius = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(5, 18)
        h += rng.uniform(3, 12) * (((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2)
    for _ in range(rng.integers(0, 3)):  # ravines: drops to stay away from
        cx, cy, radius = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(4, 12)
        h -= rng.uniform(6, 20) * (((x - cx) ** 2 + (y - cy) ** 2) < radius ** 2)
    for _ in range(rng.integers(8, 26)):  # rocks and trunks
        cx, cy, half = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(0.5, 2.5)
        h += rng.uniform(2, 4) * ((np.abs(x - cx) < half) & (np.abs(y - cy) < half))
    return h


def reachable(h: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Cells the character can walk to from `start` without getting hurt."""
    size = h.shape[0]
    seen = np.zeros_like(h, dtype=bool)
    seen[start[1], start[0]] = True
    todo = deque([start])
    while todo:
        cx, cy = todo.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and not seen[ny, nx]:
                rise = h[ny, nx] - h[cy, cx]
                if -SAFE_FALL <= rise <= JUMP_UP:
                    seen[ny, nx] = True
                    todo.append((nx, ny))
    return seen


class WalkEnv(gym.Env):
    """Reach a goal 25 to 70 m away across rough ground, without falling or getting stuck."""

    metadata = {"render_modes": []}

    def __init__(self, noise: bool = True, terrain: np.ndarray | None = None):
        self.noise = noise
        self.fixed_terrain = terrain  # tests: always this ground
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(FEATURES,), dtype=np.float32)
        self.h = np.zeros((SIZE, SIZE), dtype=np.float32)

    # The world -------------------------------------------------------------------------

    def height(self, x: float, y: float) -> float:
        return float(self.h[min(SIZE - 1, max(0, int(y))), min(SIZE - 1, max(0, int(x)))])

    def goal_bearing(self) -> float:
        heading = math.degrees(math.atan2(self.goal[1] - self.y, self.goal[0] - self.x))
        return (heading - self.facing + 180) % 360 - 180

    def goal_distance(self) -> float:
        return math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)

    def look(self):
        """What the depth analysis would report: per column, metres to what is in the way
        (None: nothing within TRUST) and whether the ground falls away ahead."""
        rng = self.np_random
        angles = np.radians(self.facing + BEARINGS)
        steps = np.arange(0.5, SIGHT + 0.01, 0.5)
        xs = np.clip((self.x + np.cos(angles)[:, None] * steps).astype(int), 0, SIZE - 1)
        ys = np.clip((self.y + np.sin(angles)[:, None] * steps).astype(int), 0, SIZE - 1)
        ground = np.concatenate([np.full((COLUMNS, 2), self.z, dtype=np.float32), self.h[ys, xs]], axis=1)
        change = ground[:, 2:] - ground[:, :-2]  # height change over the last metre
        in_way = change > STEEP_UP
        first = np.where(in_way.any(axis=1), in_way.argmax(axis=1), -1)
        distance = np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)
        ahead = (steps >= DROP_SEEN[0]) & (steps <= DROP_SEEN[1])
        before_wall = steps[None, :] < distance[:, None]
        drops = ((change < STEEP_DOWN) & ahead[None, :] & before_wall).any(axis=1)
        if self.noise:  # the real analysis misses things, and sometimes sees what isn't there
            flip = rng.random(COLUMNS) < 0.06
            distance = np.where(flip & np.isinf(distance), rng.uniform(NEAREST_SEEN, TRUST, COLUMNS),
                                np.where(flip, np.inf, distance))
            drops = (drops & (rng.random(COLUMNS) > 0.25)) | (rng.random(COLUMNS) < 0.02)
            distance = distance * rng.uniform(0.8, 1.2, COLUMNS)
        seen = [None if not d <= TRUST else max(float(d), NEAREST_SEEN) for d in distance]
        return seen, [bool(d) for d in drops]

    def observe(self) -> np.ndarray:
        if self.sight is None or not (self.noise and self.np_random.random() < STALE_VIEW):
            self.sight = self.look()
            self.map.note_sight(self.x, self.y, self.facing, BEARINGS, *self.sight)
        distances, drops = self.sight
        return features([d is not None for d in distances], [nearness(d) for d in distances], drops,
                        recalled(self.map, self.x, self.y, self.facing), self.goal_bearing(),
                        self.goal_distance(), self.hp, self.stuck >= STUCK_STEPS, self.previous)

    # Gym ---------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        while True:
            self.h = self.fixed_terrain if self.fixed_terrain is not None else make_terrain(rng)
            start = (int(rng.integers(10, SIZE - 10)), int(rng.integers(10, SIZE - 10)))
            ok = reachable(self.h, start)
            gy, gx = np.nonzero(ok)
            far = np.hypot(gx - start[0], gy - start[1])
            choice = np.nonzero((far >= 25) & (far <= 70))[0]
            if len(choice):
                pick = int(rng.choice(choice))
                self.goal = (gx[pick] + 0.5, gy[pick] + 0.5)
                break
        if options and "start" in options:
            start = options["start"]
        if options and "goal" in options:
            self.goal = options["goal"]
        self.x, self.y = start[0] + 0.5, start[1] + 0.5
        self.z = self.height(self.x, self.y)
        self.facing = float(options["facing"]) if options and "facing" in options else float(rng.uniform(-180, 180))
        # The real game is never exactly the same: slightly different speed and turns
        self.speed = SPEED * (rng.uniform(0.9, 1.1) if self.noise else 1.0)
        self.turn_scale = rng.uniform(0.8, 1.2) if self.noise else 1.0
        self.hp, self.steps, self.stuck, self.previous = 100.0, 0, 0, 0
        self.map = TerrainMap()
        self.sight = None
        return self.observe(), {}

    def step(self, action):
        action = int(action)
        before = self.goal_distance()
        reward, terminated = -0.02, False
        turn = {1: -STEER, 2: STEER, 3: -TURN, 4: TURN}.get(action, 0.0)
        self.facing = (self.facing + turn * self.turn_scale + 180) % 360 - 180
        bumped = fell = False
        if action in FORWARD_ACTIONS:
            nx = self.x + self.speed * DT * math.cos(math.radians(self.facing))
            ny = self.y + self.speed * DT * math.sin(math.radians(self.facing))
            rise = self.height(nx, ny) - self.z
            if rise > (JUMP_UP if action == 5 else STEP_UP) or not (1 <= nx < SIZE - 1 and 1 <= ny < SIZE - 1):
                bumped = True
            else:
                if -rise > SAFE_FALL:
                    fell = True
                    self.map.mark_blocked(self.x, self.y, strength=BLOCK_SCORE)  # the edge it fell from
                    harm = 100.0 * (-rise - SAFE_FALL) / (DEADLY_FALL - SAFE_FALL)
                    self.hp -= harm
                    reward -= 0.1 * harm
                self.x, self.y, self.z = nx, ny, self.height(nx, ny)
                self.map.mark_free(self.x, self.y)
        self.stuck = self.stuck + 1 if bumped else 0
        if self.stuck == STUCK_STEPS:  # noticed being stuck: as in the real game, the spot ahead goes on the map
            ahead = math.radians(self.facing)
            self.map.mark_blocked(self.x + 1.5 * math.cos(ahead), self.y + 1.5 * math.sin(ahead), strength=BLOCK_SCORE)
        if bumped:
            reward -= 0.1
        if action == 5:
            reward -= JUMP_COST
        after = self.goal_distance()
        reward += 0.5 * (before - after)
        if self.hp <= 0:
            reward -= 20.0
            terminated = True
        elif after <= GOAL_REACHED:
            reward += 10.0
            terminated = True
        self.steps += 1
        self.previous = action
        truncated = not terminated and self.steps >= MAX_STEPS
        info = {"reached": after <= GOAL_REACHED, "dead": self.hp <= 0, "fell": fell, "bumped": bumped}
        return self.observe(), reward, terminated, truncated, info


def greedy_action(obs: np.ndarray) -> int:
    """A simple hand-written walker, to compare with: head for the goal, go round what is in the way
    by turning one way and following it."""
    blocked = obs[:COLUMNS]
    drops = obs[2 * COLUMNS:3 * COLUMNS]
    remembered = obs[3 * COLUMNS:4 * COLUMNS]
    sin_b, cos_b = obs[4 * COLUMNS], obs[4 * COLUMNS + 1]
    previous = int(np.argmax(obs[-len(ACTIONS):]))
    bearing = math.degrees(math.atan2(sin_b, cos_b))
    danger = np.maximum(np.maximum(blocked, drops), remembered > 0)
    middle = COLUMNS // 2
    if danger[middle - 2:middle + 3].max():  # in the way: keep turning the same way
        if previous in (3, 4):
            return previous
        return 3 if danger[:middle].sum() < danger[middle + 1:].sum() else 4
    if abs(bearing) > 100:
        return 3 if bearing < 0 else 4
    goal_side = danger[:middle] if bearing < 0 else danger[middle + 1:]
    if abs(bearing) > 45:  # the goal is to the side: go back to it once that side is clear
        return (3 if bearing < 0 else 4) if not goal_side.any() else 0
    if bearing < -8:
        return 1
    if bearing > 8:
        return 2
    return 0
