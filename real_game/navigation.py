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
DROP_AHEAD = 4.0  # metres: where a drop is put on the map when its distance is unknown
SIGHT_CAP = 4  # a cell seen in the way counts at most this much: a few clear looks undo a mistake
CLEAR_FROM = 3.5  # metres: a direction seen clear frees the cells from here to the reach of sight
# (nearer, the ground is hidden under the HUD: an edge there can't be seen, so it must not be forgotten)
EDGE_SEEN = (3.5, 7.6)  # metres ahead where the eyes can tell the ground falls away (5° down)
EDGE_VOTES = 2  # an edge is dangerous: blocked at the first sighting, until seen safe again
GUARD_REACH = (0.75, 1.5)  # metres ahead checked before each step forward: never onto a known edge
MEMORY_REACH = (1.5, 3.0, 4.5)  # metres ahead where the walker recalls what the map knows


class TerrainMap:
    def __init__(self, data: dict | None = None):
        data = data or {}
        self.blocked: dict[tuple[int, int], int] = {tuple(c): n for c, n in data.get("blocked", [])}
        self.edges: dict[tuple[int, int], int] = {tuple(c): n for c, n in data.get("edges", [])}  # ground falls away
        self.solid: set[tuple[int, int]] = {tuple(c) for c in data.get("solid", [])}  # known by bumping or falling
        self.dangers: list[tuple[float, float]] = [tuple(d) for d in data.get("dangers", [])]

    def to_data(self) -> dict:
        return {"blocked": [[list(c), n] for c, n in self.blocked.items()],
                "edges": [[list(c), n] for c, n in self.edges.items()],
                "solid": [list(c) for c in self.solid],
                "dangers": [list(d) for d in self.dangers]}

    @staticmethod
    def cell(x: float, y: float) -> tuple[int, int]:
        return math.floor(x / CELL), math.floor(y / CELL)

    @staticmethod
    def centre(cell: tuple[int, int]) -> tuple[float, float]:
        return (cell[0] + 0.5) * CELL, (cell[1] + 0.5) * CELL

    def is_blocked(self, cell: tuple[int, int]) -> bool:
        return cell in self.solid or self.blocked.get(cell, 0) >= BLOCK_SCORE or self.edges.get(cell, 0) >= BLOCK_SCORE

    def walls(self) -> set[tuple[int, int]]:
        """Every blocked cell."""
        return ({c for c, votes in self.blocked.items() if votes >= BLOCK_SCORE} | self.solid
                | {c for c, votes in self.edges.items() if votes >= BLOCK_SCORE})

    def mark_edge(self, x: float, y: float, votes: int = EDGE_VOTES) -> bool:
        """The ground falls away at (x, y). Returns True if the cell just became blocked."""
        cell = self.cell(x, y)
        before = self.is_blocked(cell)
        self.edges[cell] = min(self.edges.get(cell, 0) + votes, SIGHT_CAP)
        return not before and self.is_blocked(cell)

    def edge_ahead(self, x: float, y: float, facing: float) -> bool:
        """Would a step forward go onto an edge the map knows? (The guard against falling.)"""
        angle = math.radians(facing)
        here = self.cell(x, y)
        for d in GUARD_REACH:
            cell = self.cell(x + d * math.cos(angle), y + d * math.sin(angle))
            if cell != here and self.edges.get(cell, 0) >= BLOCK_SCORE:
                return True
        return False

    def mark_blocked(self, x: float, y: float, strength: int = 1, solid: bool = False) -> bool:
        """Something is in the way at (x, y): seen (counts as a vote), or `solid` when the player
        bumped into it or fell from there. Returns True if the cell just became blocked."""
        cell = self.cell(x, y)
        before = self.is_blocked(cell)
        if solid:
            self.solid.add(cell)
        self.blocked[cell] = min(self.blocked.get(cell, 0) + strength, 10 if solid else max(SIGHT_CAP, strength))
        return not before and self.is_blocked(cell)

    def mark_free(self, x: float, y: float) -> None:
        """The player stands there: it is not blocked, whatever was thought before."""
        cell = self.cell(x, y)
        self.blocked.pop(cell, None)
        self.edges.pop(cell, None)
        self.solid.discard(cell)

    def _seen_clear(self, votes: dict, x: float, y: float, angle: float, start: float, end: float) -> None:
        """Nothing seen from `start` to `end` metres in that direction: cells there lose a vote."""
        steps = int((end - start) / (CELL / 2)) + 1
        cos, sin = math.cos(angle), math.sin(angle)
        cells = {self.cell(x + (start + k * CELL / 2) * cos, y + (start + k * CELL / 2) * sin) for k in range(steps)}
        for cell in cells - self.solid:
            count = votes.get(cell, 0)
            if count > 1:
                votes[cell] = min(count, SIGHT_CAP) - 1
            elif count:
                del votes[cell]

    def mark_danger(self, x: float, y: float) -> None:
        self.dangers.append((x, y))
        self.dangers = self.dangers[-50:]

    def danger_near(self, x: float, y: float) -> int:
        return sum(1 for dx, dy in self.dangers if math.hypot(x - dx, y - dy) < DANGER_RADIUS)

    def note_sight(self, x: float, y: float, facing: float, bearings, distances, drops,
                   drop_ahead: float = DROP_AHEAD, reach: float = 12.0) -> bool:
        """Put on the map what is seen from (x, y): per direction (degrees from `facing`), the
        distance of what is in the way (None: nothing within `reach`) and of where the ground falls
        away (None: it doesn't; True: it does, distance unknown). What is seen clear loses a vote:
        mistakes of the eyes are undone, but an edge only where an edge could have been seen.
        True if a cell just became blocked."""
        changed = False
        for bearing, distance, drop in zip(bearings, distances, drops):
            angle = math.radians(facing + bearing)
            if drop is True:
                drop = drop_ahead
            elif drop is False:
                drop = None
            if distance is None:
                self._seen_clear(self.blocked, x, y, angle, CLEAR_FROM, reach)
            else:
                changed |= self.mark_blocked(x + distance * math.cos(angle), y + distance * math.sin(angle))
            if drop is None:
                if distance is None or distance > EDGE_SEEN[1]:
                    self._seen_clear(self.edges, x, y, angle, *EDGE_SEEN)
            else:
                changed |= self.mark_edge(x + drop * math.cos(angle), y + drop * math.sin(angle))
        return changed

    def blocked_ahead(self, x: float, y: float, facing: float, bearings,
                      reach: tuple[float, ...] = MEMORY_REACH) -> list[float | None]:
        """Per direction: how far the nearest spot known to be blocked is (None: none within reach)."""
        found = []
        for bearing in bearings:
            angle = math.radians(facing + bearing)
            found.append(next((d for d in reach if self.is_blocked(
                self.cell(x + d * math.cos(angle), y + d * math.sin(angle)))), None))
        return found

    def path_is_clear(self, points: list[tuple[float, float]]) -> bool:
        return not any(self.is_blocked(self.cell(*p)) for p in points)

    def route_is_clear(self, start: tuple[float, float], points: list[tuple[float, float]],
                       goal: tuple[float, float] | None = None) -> bool:
        """Is the straight walk from `start` through `points` free of blocked cells? The goal's
        own cell doesn't count (a node to mine is often seen as in the way)."""
        skip = {self.cell(*start)} | ({self.cell(*goal)} if goal is not None else set())
        here = start
        for point in points:
            steps = max(1, int(math.hypot(point[0] - here[0], point[1] - here[1]) / (CELL / 2)))
            for k in range(steps + 1):
                cell = self.cell(here[0] + (point[0] - here[0]) * k / steps, here[1] + (point[1] - here[1]) * k / steps)
                if cell not in skip and self.is_blocked(cell):
                    return False
            here = point
        return True

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        """Waypoints (cell centres) from start to goal, the goal itself last; None if no way is known."""
        s, g = self.cell(*start), self.cell(*goal)
        if s == g:
            return [goal]
        x0 = math.floor((min(start[0], goal[0]) - SEARCH_MARGIN) / CELL)
        x1 = math.floor((max(start[0], goal[0]) + SEARCH_MARGIN) / CELL)
        y0 = math.floor((min(start[1], goal[1]) - SEARCH_MARGIN) / CELL)
        y1 = math.floor((max(start[1], goal[1]) + SEARCH_MARGIN) / CELL)

        walls = self.walls()
        by_walls = {(c[0] + dx, c[1] + dy) for c in walls for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        danger_costs: dict[tuple[int, int], float] = {}

        def cost(c: tuple[int, int]) -> float:
            extra = WALL_MARGIN_COST if c in by_walls else 0.0
            if self.dangers:
                if c not in danger_costs:
                    danger_costs[c] = DANGER_COST * self.danger_near(*self.centre(c))
                extra += danger_costs[c]
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
                    if n in walls and n != g:
                        continue
                    if dx and dy and ((c[0] + dx, c[1]) in walls or (c[0], c[1] + dy) in walls):
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
