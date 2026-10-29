"""A map the AI learns as it goes, and paths worked out on it.

The ground is cut into square cells. What the eyes see in the way, where the
player got stuck or fell, and where it was attacked are written on the map. To go
somewhere, the AI looks for the shortest path that avoids blocked cells and keeps
away from danger (A* search), and works it out again whenever it learns more.
Places never seen are assumed open: the map fills in while exploring.
"""
from __future__ import annotations

import heapq
import math

CELL = 2.0  # metres
SEARCH_MARGIN = 80.0  # metres around the start and the goal the search may use
BLOCK_SCORE = 2  # a cell is blocked once seen in the way this many times
DANGER_RADIUS = 30.0  # metres around an attack that the path avoids if it can
DANGER_COST = 6.0  # extra cost of a cell near danger, per danger
WALL_MARGIN_COST = 1.5  # extra cost of a cell next to a blocked one: keep some room
MAX_EXPANSIONS = 60000


class TerrainMap:
    def __init__(self, data: dict | None = None):
        data = data or {}
        self.blocked: dict[tuple[int, int], int] = {tuple(c): n for c, n in data.get("blocked", [])}
        self.dangers: list[tuple[float, float]] = [tuple(d) for d in data.get("dangers", [])]

    def to_data(self) -> dict:
        return {"blocked": [[list(c), n] for c, n in self.blocked.items()],
                "dangers": [list(d) for d in self.dangers]}

    @staticmethod
    def cell(x: float, y: float) -> tuple[int, int]:
        return math.floor(x / CELL), math.floor(y / CELL)

    @staticmethod
    def centre(cell: tuple[int, int]) -> tuple[float, float]:
        return (cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL

    def is_blocked(self, cell: tuple[int, int]) -> bool:
        return self.blocked.get(cell, 0) >= BLOCK_SCORE

    def mark_blocked(self, x: float, y: float, strength: int = 1) -> bool:
        """Something is in the way at (x, y). Returns True if the cell just became blocked."""
        cell = self.cell(x, y)
        before = self.is_blocked(cell)
        self.blocked[cell] = min(self.blocked.get(cell, 0) + strength, 10)
        return not before and self.is_blocked(cell)

    def mark_free(self, x: float, y: float) -> None:
        """The player stands there: it is not blocked, whatever was thought before."""
        self.blocked.pop(self.cell(x, y), None)

    def mark_danger(self, x: float, y: float) -> None:
        self.dangers.append((x, y))
        self.dangers = self.dangers[-50:]

    def danger_near(self, x: float, y: float) -> int:
        return sum(1 for dx, dy in self.dangers if math.hypot(x - dx, y - dy) < DANGER_RADIUS)

    def path_is_clear(self, points: list[tuple[float, float]]) -> bool:
        return not any(self.is_blocked(self.cell(*p)) for p in points)

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        """Waypoints (cell centres) from start to goal, the goal itself last; None if no way is known."""
        s, g = self.cell(*start), self.cell(*goal)
        if s == g:
            return [goal]
        x0 = math.floor((min(start[0], goal[0]) - SEARCH_MARGIN) / CELL)
        x1 = math.floor((max(start[0], goal[0]) + SEARCH_MARGIN) / CELL)
        y0 = math.floor((min(start[1], goal[1]) - SEARCH_MARGIN) / CELL)
        y1 = math.floor((max(start[1], goal[1]) + SEARCH_MARGIN) / CELL)

        def cost(c: tuple[int, int]) -> float:
            extra = 0.0
            if any(self.is_blocked((c[0] + dx, c[1] + dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                extra += WALL_MARGIN_COST
            if self.dangers:
                extra += DANGER_COST * self.danger_near(*self.centre(c))
            return extra

        def h(c: tuple[int, int]) -> float:
            return math.hypot(c[0] - g[0], c[1] - g[1])

        best = {s: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        todo = [(h(s), s)]
        expansions = 0
        while todo and expansions < MAX_EXPANSIONS:
            _, c = heapq.heappop(todo)
            if c == g:
                break
            expansions += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if (dx == dy == 0) or not (x0 <= n[0] <= x1 and y0 <= n[1] <= y1):
                        continue
                    if self.is_blocked(n) and n != g:
                        continue
                    if dx and dy and (self.is_blocked((c[0] + dx, c[1])) or self.is_blocked((c[0], c[1] + dy))):
                        continue  # don't cut a blocked corner
                    step = math.hypot(dx, dy) + cost(n)
                    if best.get(n, math.inf) > best[c] + step:
                        best[n] = best[c] + step
                        came[n] = c
                        heapq.heappush(todo, (best[n] + h(n), n))
        if g not in came:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        return simplify([self.centre(c) for c in cells[1:-1]] + [goal], start, self)


def simplify(points: list[tuple[float, float]], start: tuple[float, float], terrain: TerrainMap) -> list[tuple[float, float]]:
    """Drop the waypoints that can be skipped in a straight, clear line: a smoother walk."""
    out, here, i = [], start, 0
    while i < len(points):
        j = len(points) - 1
        while j > i and not _clear_line(here, points[j], terrain):
            j -= 1
        out.append(points[j])
        here, i = points[j], j + 1
    return out


def _clear_line(a: tuple[float, float], b: tuple[float, float], terrain: TerrainMap) -> bool:
    steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (CELL / 2)))
    return terrain.path_is_clear([(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
                                  for k in range(steps + 1)]) and all(
        terrain.danger_near(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        <= terrain.danger_near(*a) for k in range(0, steps + 1, 4))
