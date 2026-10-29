"""The landscapes the AI learns to walk on: general ones, and some with a particular trap.

Each landscape is a height map (one height per square metre) plus where the walk
may start and where the goal may be. The special ones are drawn in one orientation
(start on the left, goal on the right), then turned and mirrored at random.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SIZE = 128
KINDS = {  # kind: (name shown, share of the landscapes)
    "general": ("Terrain varié", 0.30),
    "wall": ("Mur avec une ouverture", 0.10),
    "dead_end": ("Cul-de-sac", 0.07),
    "descent": ("Descendre sans mourir", 0.15),
    "ascent": ("Monter sur un plateau", 0.06),
    "canyon": ("Canyon et pont", 0.12),
    "forest": ("Forêt dense", 0.08),
    "terraces": ("Terrasses", 0.12),
}


@dataclass
class Landscape:
    kind: str
    h: np.ndarray  # metres, indexed [y, x]
    start_zone: np.ndarray  # where the walk may start
    goal_zone: np.ndarray  # where the goal may be

    @property
    def name(self) -> str:
        return KINDS[self.kind][0]


def pick_kind(rng: np.random.Generator) -> str:
    kinds = list(KINDS)
    shares = np.array([KINDS[k][1] for k in kinds])
    return kinds[int(rng.choice(len(kinds), p=shares / shares.sum()))]


# Pieces ------------------------------------------------------------------------------


def _grid(size: int):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return x, y


def _hills(rng, x, y) -> np.ndarray:
    h = np.zeros_like(x)
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.uniform(0, x.shape[1]), rng.uniform(0, x.shape[0])
        spread, height = rng.uniform(10, 30), rng.uniform(-4, 6)
        h += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * spread ** 2))
    return h


def _boxes(rng, x, y, count, half, height) -> np.ndarray:
    """Rocks (low, wide) or trees (tall, thin): square blocks."""
    h = np.zeros_like(x)
    for _ in range(int(count)):
        cx, cy, size = rng.uniform(0, x.shape[1]), rng.uniform(0, x.shape[0]), rng.uniform(*half)
        tall = rng.uniform(*height)
        x0, x1 = max(0, math.floor(cx - size) + 1), min(x.shape[1], math.ceil(cx + size))
        y0, y1 = max(0, math.floor(cy - size) + 1), min(x.shape[0], math.ceil(cy + size))
        h[y0:y1, x0:x1] = np.maximum(h[y0:y1, x0:x1], tall)
    return h


def _rocks(rng, x, y, count) -> np.ndarray:
    return _boxes(rng, x, y, count, (0.5, 2.5), (2, 4))


def _trees(rng, x, y, count) -> np.ndarray:
    return _boxes(rng, x, y, count, (0.5, 1.0), (6, 10))


def _discs(rng, x, y, count, radius, height) -> np.ndarray:
    h = np.zeros_like(x)
    for _ in range(int(count)):
        cx, cy, r = rng.uniform(0, x.shape[1]), rng.uniform(0, x.shape[0]), rng.uniform(*radius)
        h += rng.uniform(*height) * (((x - cx) ** 2 + (y - cy) ** 2) < r ** 2)
    return h


def _wall_pieces(rng, x, y, count) -> np.ndarray:
    """Short straight walls, any direction."""
    h = np.zeros_like(x)
    for _ in range(int(count)):
        cx, cy = rng.uniform(0, x.shape[1]), rng.uniform(0, x.shape[0])
        angle, length, thick = rng.uniform(0, math.pi), rng.uniform(10, 30), rng.uniform(2, 3)
        along = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
        across = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        h = np.maximum(h, rng.uniform(4, 8) * ((np.abs(along) < length / 2) & (np.abs(across) < thick / 2)))
    return h


def _winding(rng, y, middle: float, sway: float) -> np.ndarray:
    """A line across the map, from top to bottom, gently winding around `middle`."""
    return middle + rng.uniform(0, sway) * np.sin(2 * math.pi * y / rng.uniform(50, 120) + rng.uniform(0, 2 * math.pi))


def _gaps(rng, y, count, width) -> np.ndarray:
    open_ = np.zeros_like(y, dtype=bool)
    for _ in range(int(count)):
        start, w = rng.uniform(10, y.shape[0] - 16), rng.uniform(*width)
        open_ |= (y >= start) & (y < start + w)
    return open_


def _way_down(rng, x, y, edge, height: float, stairs: bool) -> np.ndarray:
    """A ramp or (if `stairs`) sometimes a staircase going down from a cliff edge towards +x.
    Returns the height added."""
    added = np.zeros_like(x)
    start, along = rng.uniform(12, y.shape[0] - 20), x - edge
    if not stairs or rng.random() < 0.6:  # a ramp, gentle enough to walk up
        width, slope = rng.uniform(4, 7), rng.uniform(0.25, 0.4)
        length = height / slope
        strip = (y >= start) & (y < start + width) & (along >= 0) & (along < length)
        added[strip] = (height * (1 - along / length))[strip]
    else:  # steps going down, low enough that the eyes don't take them for an edge
        width = rng.uniform(5, 9)
        drop = rng.uniform(0.8, 1.5)
        tread = rng.uniform(1.5, 3)
        steps = math.ceil(height / drop)
        strip = (y >= start) & (y < start + width) & (along >= 0) & (along < steps * tread)
        level = np.maximum(height - (np.floor(along / tread) + 1) * drop, 0)
        added[strip] = level[strip]
    return added


# Landscapes --------------------------------------------------------------------------


def make_landscape(rng: np.random.Generator, kind: str | None = None, size: int = SIZE) -> Landscape:
    kind = kind or pick_kind(rng)
    x, y = _grid(size)
    h = _hills(rng, x, y)
    everywhere = np.ones_like(h, dtype=bool)
    if kind == "general":
        h += _discs(rng, x, y, rng.integers(2, 7), (5, 18), (3, 12))  # plateaus
        h -= _discs(rng, x, y, rng.integers(0, 4), (4, 12), (6, 20))  # ravines
        h += np.maximum.reduce([_rocks(rng, x, y, rng.integers(15, 45)), _trees(rng, x, y, rng.integers(10, 60)),
                                _wall_pieces(rng, x, y, rng.integers(0, 3))])
        return Landscape(kind, h, everywhere, everywhere.copy())
    if kind == "forest":
        h += np.maximum(_trees(rng, x, y, rng.integers(180, 360)), _rocks(rng, x, y, rng.integers(5, 20)))
        return Landscape(kind, h, everywhere, everywhere.copy())

    middle = size / 2
    if kind == "wall":
        line = _winding(rng, y, middle + rng.uniform(-10, 10), 10)
        wall = (np.abs(x - line) < rng.uniform(2, 3) / 2) & ~_gaps(rng, y, rng.integers(1, 3), (3, 6))
        h += rng.uniform(4, 10) * wall
        start, goal = x < line - 10, x > line + 10
    elif kind == "dead_end":
        back, centre = rng.uniform(70, 88), rng.uniform(40, size - 40)
        half, depth, thick = rng.uniform(7, 12), rng.uniform(12, 22), 2.5
        sides = (np.abs(np.abs(y - centre) - half) < thick / 2) & (x >= back - depth) & (x <= back)
        rear = (np.abs(x - back) < thick / 2) & (np.abs(y - centre) <= half + thick / 2)
        h += rng.uniform(5, 9) * (sides | rear)
        start = (x > back - depth + 3) & (x < back - 3) & (np.abs(y - centre) < half - 3)
        goal = x > back + 10
    elif kind in ("descent", "ascent"):
        edge = _winding(rng, y, middle - rng.uniform(0, 12), 6)
        height = rng.uniform(10, 22) if kind == "descent" else rng.uniform(7, 14)
        h += height * (x < edge)
        h += np.maximum.reduce([_way_down(rng, x, y, edge, height, stairs=kind == "descent")
                                for _ in range(rng.integers(1, 3))])
        start, goal = x < edge - 6, x > edge + 12
        if kind == "ascent":
            start, goal = goal, start
    elif kind == "canyon":
        line = _winding(rng, y, middle + rng.uniform(-10, 10), 10)
        trench = (np.abs(x - line) < rng.uniform(6, 12) / 2) & ~_gaps(rng, y, rng.integers(1, 3), (3, 5))
        h -= rng.uniform(12, 25) * trench
        start, goal = x < line - 10, x > line + 10
    elif kind == "terraces":
        level = np.zeros_like(x)
        for edge in np.linspace(34, size - 34, int(rng.integers(5, 10))):  # higher on the left; some drops hurt
            level += (x < edge) * (rng.uniform(5, 8) if rng.random() < 0.3 else rng.uniform(1.5, 3.0))
        smooth = level.max() * np.clip((size - 20 - x) / (size - 40), 0, 1)  # ramps: the same descent, gently
        h += np.where(_gaps(rng, y, rng.integers(1, 3), (4, 7)), smooth, level)
        start, goal = x < 30, x > size - 30
    else:
        raise ValueError(kind)
    h += _rocks(rng, x, y, rng.integers(6, 20))  # a few rocks everywhere, as in any landscape
    turns, mirror = int(rng.integers(4)), bool(rng.random() < 0.5)

    def turn(a: np.ndarray) -> np.ndarray:
        a = np.rot90(a, turns)
        return np.ascontiguousarray(a[:, ::-1] if mirror else a)

    return Landscape(kind, turn(h.astype(np.float32)), turn(start), turn(goal))
