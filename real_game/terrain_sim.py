"""Simulated landscapes to teach the AI to walk: hills, cliffs, walls, ravines, forests...

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

from landscapes import SIZE, Landscape, make_landscape
from navigation import BLOCK_SCORE, CELL, EDGE_SEEN, GUARD_RUN, GUARD_WALK, MEMORY_REACH, TerrainMap

DT = 0.2  # seconds per decision
SPEED = 5.0  # metres per second, walking (calibrated: 5.03)
RUN_SPEED = 9.0  # metres per second, running (shift held)
STEP_UP = 0.6  # metres the ground may rise at once for the character to walk on without jumping
JUMP_UP = 1.4  # metres it gets up by jumping
SAFE_FALL = 4.0  # falls up to this height do no harm
DEADLY_FALL = 14.0  # falls from this height kill
COLUMNS = 15
FOV = 90.0
SIGHT = 15.0  # metres the depth analysis reaches
TRUST = 12.0  # obstacles farther than this are not reported (as moves.TRUST_SIGHT)
NEAREST_SEEN = 7.6  # looking 5° down, the depth band starts this far: nearer obstacles read as this
DROP_SEEN = EDGE_SEEN  # metres ahead where the ground band shows a drop (5° down, above the HUD)
EDGE_FAR = 10.0  # an edge this far would read as 0
STEEP_UP, STEEP_DOWN = STEP_UP, -2.0  # height change over one metre that shows as a wall / an edge
STEER, TURN = 15.0, 35.0  # degrees: steering while walking, turning on the spot
ACTIONS = ("avancer", "avancer à gauche", "avancer à droite",
           "tourner à gauche", "tourner à droite", "sauter en avançant",
           "courir", "courir à gauche", "courir à droite")
FORWARD_ACTIONS = (0, 1, 2, 5, 6, 7, 8)
RUN_ACTIONS = (6, 7, 8)
TURNS = {1: -STEER, 2: STEER, 3: -TURN, 4: TURN, 7: -STEER, 8: STEER}  # degrees, + to the right
GOAL_REACHED = 2.5
WAYPOINT_REACHED = 2.5  # as moves.WAYPOINT_REACHED
MAX_STEPS = 800  # 160 s: enough to walk along a whole cliff to find the way down
FALL_COST = 0.3  # reward lost per point of health lost falling...
FALL_PENALTY = 15.0  # ...plus this for any fall that hurts: more than any shortcut could save
DEATH_COST = 50.0
STALE_VIEW = 0.3  # chance that the eyes have nothing new this step (the real depth model is slower)
JUMP_COST = 0.05  # jumping for nothing looks odd: only when it helps
STUCK_STEPS = 3  # bumping this many steps in a row (0.6 s) is noticed, as in the real game
GOAL_RANGE = {"general": (25, 70), "forest": (25, 70)}  # metres to the goal; other kinds: SPECIAL_RANGE
SPECIAL_RANGE = (25, 110)


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


def edge_nearness(distance: float | None) -> float:
    """0 when the ground doesn't fall away ahead, else bigger when the edge is nearer."""
    return 0.0 if distance is None else 1.0 - min(distance, EDGE_FAR) / EDGE_FAR


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


def place(land: Landscape, rng: np.random.Generator, tries: int = 3):
    """A start (cell) on flat ground in the start zone, and a goal (metres) the character can walk
    to from there without getting hurt. None if this landscape has none."""
    h = land.h
    padded = np.pad(h, 1, mode="edge")
    around = np.stack([padded[1 + dy:SIZE + 1 + dy, 1 + dx:SIZE + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    inside = np.zeros((SIZE, SIZE), dtype=bool)
    inside[10:-10, 10:-10] = True
    sy, sx = np.nonzero(land.start_zone & inside & (around.max(axis=0) - around.min(axis=0) < 0.7))
    low, high = GOAL_RANGE.get(land.kind, SPECIAL_RANGE)
    for _ in range(tries if len(sx) else 0):
        pick = int(rng.integers(len(sx)))
        start = (int(sx[pick]), int(sy[pick]))
        gy, gx = np.nonzero(reachable(h, start) & land.goal_zone)
        far = np.hypot(gx - start[0], gy - start[1])
        choice = np.nonzero((far >= low) & (far <= high))[0]
        if len(choice):
            pick = int(rng.choice(choice))
            return start, (gx[pick] + 0.5, gy[pick] + 0.5)
    return None


class WalkEnv(gym.Env):
    """Reach a goal 25 to 110 m away across rough ground, without falling or getting stuck.
    As in the real game, the learnt map and A* give the next waypoint; the walker decides how
    to get there, and is rewarded for shortening the way left.
    options: kind (see landscapes.KINDS), start (cell), goal (metres), facing (degrees)."""

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

    def move(self, length: float, jump: bool) -> tuple[bool, float]:
        """Go `length` metres ahead, a little at a time: stop against what is too high to step
        (or jump) up, fall where the ground drops away. Returns (bumped, height fallen)."""
        angle = math.radians(self.facing)
        pieces = max(2, math.ceil(length / 0.25))
        dx, dy = length / pieces * math.cos(angle), length / pieces * math.sin(angle)
        jumped = False
        for _ in range(pieces):
            nx, ny = self.x + dx, self.y + dy
            if not (1 <= nx < SIZE - 1 and 1 <= ny < SIZE - 1):
                return True, 0.0
            ground = self.height(nx, ny)
            if ground - self.z > STEP_UP:
                if not jump or jumped or ground - self.z > JUMP_UP:
                    return True, 0.0
                jumped = True
            drop = self.z - ground
            self.x, self.y, self.z = nx, ny, ground
            if drop > SAFE_FALL:
                return False, drop
        return False, 0.0

    # The way, as in the real game: the learnt map and A* give the next waypoint ------------

    def replan(self) -> None:
        self.route = self.map.plan((self.x, self.y), self.goal) or [self.goal]

    def follow_route(self) -> None:
        while len(self.route) > 1 and math.hypot(self.route[0][0] - self.x, self.route[0][1] - self.y) < WAYPOINT_REACHED:
            self.route.pop(0)

    def to_go(self) -> float:
        """Metres left along the route: what the walker is rewarded for shortening."""
        points = [(self.x, self.y), *self.route]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))

    def look(self):
        """What the depth analysis would report: per column, metres to what is in the way
        (None: nothing within TRUST) and to where the ground falls away (None: it doesn't)."""
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
        sudden = (change < STEEP_DOWN) & before_wall  # the ground falls away within a metre: an edge
        first = sudden.argmax(axis=1)
        top = ground[np.arange(COLUMNS), first]  # the ground just before the edge
        # Beyond the edge, the void shows, even when the edge itself is hidden under the HUD
        fallen = (sudden.any(axis=1)[:, None] & (np.arange(len(steps))[None, :] >= first[:, None])
                  & (ground[:, 2:] < top[:, None] + STEEP_DOWN) & ahead[None, :] & before_wall)
        edge_at = np.where(fallen.any(axis=1), steps[fallen.argmax(axis=1)], np.inf)
        if self.noise:  # the real analysis misses things, and sometimes sees what isn't there
            flip = rng.random(COLUMNS) < 0.06
            distance = np.where(flip & np.isinf(distance), rng.uniform(NEAREST_SEEN, TRUST, COLUMNS),
                                np.where(flip, np.inf, distance))
            distance = distance * rng.uniform(0.8, 1.2, COLUMNS)
            edge_at = np.where(rng.random(COLUMNS) < 0.25, np.inf, edge_at * rng.uniform(0.85, 1.15, COLUMNS))
            edge_at = np.where(rng.random(COLUMNS) < 0.02, rng.uniform(*DROP_SEEN, COLUMNS), edge_at)
        seen = [None if not d <= TRUST else max(float(d), NEAREST_SEEN) for d in distance]
        drops = [None if np.isinf(d) else min(max(float(d), DROP_SEEN[0]), DROP_SEEN[1]) for d in edge_at]
        return seen, drops

    def observe(self) -> np.ndarray:
        if self.sight is None or not (self.noise and self.np_random.random() < STALE_VIEW):
            self.sight = self.look()
            if self.map.note_sight(self.x, self.y, self.facing, BEARINGS, *self.sight, reach=TRUST) and \
                    not self.map.route_is_clear((self.x, self.y), self.route[:2], self.goal):
                self.replan()
        self.follow_route()
        distances, drops = self.sight
        target = self.route[0]
        bearing = math.degrees(math.atan2(target[1] - self.y, target[0] - self.x)) - self.facing
        return features([d is not None for d in distances], [nearness(d) for d in distances],
                        [edge_nearness(d) for d in drops],
                        recalled(self.map, self.x, self.y, self.facing), (bearing + 180) % 360 - 180,
                        math.hypot(target[0] - self.x, target[1] - self.y), self.hp, self.stuck >= STUCK_STEPS,
                        self.previous)

    # Gym ---------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        for attempt in range(40):
            if self.fixed_terrain is not None:
                every = np.ones((SIZE, SIZE), dtype=bool)
                land = Landscape("general", self.fixed_terrain, every, every)
            else:
                land = make_landscape(rng, options.get("kind") if attempt < 30 else "general")
            placed = place(land, rng)
            if placed:
                break
        else:
            raise RuntimeError("no landscape with a goal that can be reached")
        self.kind, self.h = land.kind, land.h
        start, self.goal = placed
        if options and "start" in options:
            start = options["start"]
        if options and "goal" in options:
            self.goal = options["goal"]
        self.x, self.y = start[0] + 0.5, start[1] + 0.5
        self.z = self.height(self.x, self.y)
        self.facing = float(options["facing"]) if options and "facing" in options else float(rng.uniform(-180, 180))
        # The real game is never exactly the same: slightly different speed and turns
        self.speed = SPEED * (rng.uniform(0.9, 1.1) if self.noise else 1.0)
        self.run_speed = RUN_SPEED * (rng.uniform(0.85, 1.1) if self.noise else 1.0)
        self.turn_scale = rng.uniform(0.8, 1.2) if self.noise else 1.0
        self.hp, self.steps, self.stuck, self.previous = 100.0, 0, 0, 0
        self.map = TerrainMap()
        cells = int(SIZE / CELL)
        for i in range(-1, cells + 1):  # the edge of the simulated world: known from the start
            for cell in ((i, -1), (i, cells), (-1, i), (cells, i)):
                self.map.solid.add(cell)
        self.sight = None
        self.route = [self.goal]
        self.replan()
        return self.observe(), {}

    def step(self, action):
        action = int(action)
        before = self.to_go()
        reward, terminated = -0.02, False
        turn = TURNS.get(action, 0.0)
        self.facing = (self.facing + turn * self.turn_scale + 180) % 360 - 180
        bumped = fell = held = False
        running = action in RUN_ACTIONS
        if action in FORWARD_ACTIONS and self.map.edge_ahead(self.x, self.y, self.facing,
                                                             GUARD_RUN if running else GUARD_WALK):
            held = True  # the guard, as in the real game: no step onto a known edge
        elif action in FORWARD_ACTIONS:
            edge = (self.x, self.y)
            bumped, drop = self.move((self.run_speed if running else self.speed) * DT, jump=action == 5)
            if drop > SAFE_FALL:
                fell = True
                self.map.mark_blocked(*edge, strength=BLOCK_SCORE, solid=True)  # the edge it fell from
                self.map.mark_edge(*edge, votes=4)
                harm = 100.0 * (drop - SAFE_FALL) / (DEADLY_FALL - SAFE_FALL)
                self.hp -= harm
                reward -= FALL_PENALTY + FALL_COST * min(harm, 100.0)
            self.map.mark_free(self.x, self.y)
        self.stuck = self.stuck + 1 if bumped else 0
        if self.stuck == STUCK_STEPS:  # noticed being stuck: as in the real game, the spot ahead goes on the map
            ahead = math.radians(self.facing)
            self.map.mark_blocked(self.x + 1.5 * math.cos(ahead), self.y + 1.5 * math.sin(ahead), strength=BLOCK_SCORE, solid=True)
        if fell or self.stuck == STUCK_STEPS:
            self.replan()
        if bumped:
            reward -= 0.1
        if action == 5:
            reward -= JUMP_COST
        observation = self.observe()  # may find a new route
        after = self.goal_distance()
        reward += 0.5 * (before - self.to_go())
        if self.hp <= 0:
            reward -= DEATH_COST
            terminated = True
        elif after <= GOAL_REACHED:
            reward += 10.0
            terminated = True
        self.steps += 1
        self.previous = action
        truncated = not terminated and self.steps >= MAX_STEPS
        info = {"reached": after <= GOAL_REACHED, "dead": self.hp <= 0, "fell": fell, "bumped": bumped, "held": held,
                "kind": self.kind}
        return observation, reward, terminated, truncated, info


def greedy_action(obs: np.ndarray) -> int:
    """A simple hand-written walker, to compare with: head for the goal, go round what is in the way
    by turning one way and following it."""
    near = obs[COLUMNS:2 * COLUMNS]
    drops = obs[2 * COLUMNS:3 * COLUMNS]
    remembered = obs[3 * COLUMNS:4 * COLUMNS]
    sin_b, cos_b = obs[4 * COLUMNS], obs[4 * COLUMNS + 1]
    previous = int(np.argmax(obs[-len(ACTIONS):]))
    bearing = math.degrees(math.atan2(sin_b, cos_b))
    danger = (near > 0.35) | (drops > 0) | (remembered > 0)  # in the way within about 10 m
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
