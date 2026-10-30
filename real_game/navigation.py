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
GUARD_WALK, GUARD_RUN = 1.5, 3.0  # metres ahead checked before each step forward, walking or running
DROP_LIMIT = 3.5  # metres: a neighbouring cell this much lower is not gone to (a fall that hurts)
UNKNOWN_COST = 0.3  # extra cost of a cell never walked: known safe ways are preferred
HEIGHT_CELL = 1.0  # metres: cells of the 3D map of the ground
HEIGHT_SEEN = 2  # a height counts once seen this many times
HEIGHT_WEIGHT = 0.3  # how much a new look changes a height already known
CLIMB = 2.2  # metres: a neighbouring cell whose ground is known to be this much higher can't be
# climbed (a ledge; a walkable slope rises less over 2 m), though it can be walked down
MEMORY_REACH = (1.5, 3.0, 4.5)  # metres ahead where the walker recalls what the map knows


class HeightMap:
    """The ground's height, cell by cell (1 m), measured by the eyes: the depth of every picture,
    placed in the world from where the player stood. Heights are in the mod's player z units:
    the height the player would have standing there."""

    def __init__(self):
        self.cells: dict[tuple[int, int], list[float]] = {}  # cell: [height, times seen]

    def __len__(self) -> int:
        return len(self.cells)

    def add_view(self, x: float, y: float, z: float, facing: float, points) -> None:
        """Points seen (ahead, right, height above the player's ground) from (x, y, z), facing `facing`."""
        import numpy as np

        if points is None or len(points) == 0:
            return
        ahead, right, above = points[:, 0], points[:, 1], points[:, 2]
        cos, sin = math.cos(math.radians(facing)), math.sin(math.radians(facing))
        ix = np.floor((x + ahead * cos - right * sin) / HEIGHT_CELL).astype(np.int64)
        iy = np.floor((y + ahead * sin + right * cos) / HEIGHT_CELL).astype(np.int64)
        heights = z + above
        keys = ix * 2_000_003 + iy
        order = np.argsort(keys, kind="stable")
        keys, ix, iy, heights = keys[order], ix[order], iy[order], heights[order]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for start, end in zip(np.r_[0, bounds], np.r_[bounds, len(keys)]):
            cell, height = (int(ix[start]), int(iy[start])), float(np.median(heights[start:end]))
            known = self.cells.get(cell)
            if known is None:
                self.cells[cell] = [height, 1]
            else:
                known[0] += HEIGHT_WEIGHT * (height - known[0])
                known[1] = min(known[1] + 1, 50)

    def height(self, x: float, y: float) -> float | None:
        known = self.cells.get((math.floor(x / HEIGHT_CELL), math.floor(y / HEIGHT_CELL)))
        return known[0] if known and known[1] >= HEIGHT_SEEN else None

    def sample(self, xs, ys):
        """Heights at many points at once (NaN where unknown)."""
        import numpy as np

        out = np.full(np.shape(xs), np.nan, dtype=np.float32)
        for index in np.ndindex(out.shape):
            h = self.height(float(xs[index]), float(ys[index]))
            if h is not None:
                out[index] = h
        return out

    def ground(self, x0: float, y0: float, size: float) -> float | None:
        """The ground's height in a square (median of the cells known there), None if unknown."""
        found = sorted(h for i in range(int(size / HEIGHT_CELL)) for j in range(int(size / HEIGHT_CELL))
                       if (h := self.height(x0 + (i + 0.5) * HEIGHT_CELL, y0 + (j + 0.5) * HEIGHT_CELL)) is not None)
        return found[len(found) // 2] if found else None

    def save(self, path) -> None:
        import numpy as np

        cells = list(self.cells.items())
        np.savez_compressed(path, cells=np.array([c for c, _ in cells], dtype=np.int64).reshape(-1, 2),
                            values=np.array([v for _, v in cells], dtype=np.float32).reshape(-1, 2))

    def load(self, path) -> None:
        import numpy as np

        try:
            data = np.load(path)
        except (OSError, ValueError):
            return
        self.cells = {(int(c[0]), int(c[1])): [float(v[0]), int(v[1])] for c, v in zip(data["cells"], data["values"])}


class TerrainMap:
    def __init__(self, data: dict | None = None):
        data = data or {}
        self.blocked: dict[tuple[int, int], int] = {tuple(c): n for c, n in data.get("blocked", [])}
        self.edges: dict[tuple[int, int], int] = {tuple(c): n for c, n in data.get("edges", [])}  # ground falls away
        self.solid: set[tuple[int, int]] = {tuple(c) for c in data.get("solid", [])}  # known by bumping or falling
        # lowest height where the player stood, per cell: the ground (in the air, it is higher)
        self.levels: dict[tuple[int, int], float] = {tuple(c): z for c, z in data.get("levels", [])}
        self.heights = HeightMap()  # what the eyes measured (kept in its own file)
        self.dangers: list[tuple[float, float]] = [tuple(d) for d in data.get("dangers", [])]

    def to_data(self) -> dict:
        return {"blocked": [[list(c), n] for c, n in self.blocked.items()],
                "edges": [[list(c), n] for c, n in self.edges.items()],
                "solid": [list(c) for c in self.solid],
                "levels": [[list(c), z] for c, z in self.levels.items()],
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

    def edge_ahead(self, x: float, y: float, facing: float, reach: float = GUARD_WALK,
                   z: float | None = None) -> bool:
        """Would a step forward go onto an edge the map knows, or (given the player's height `z`)
        down a drop the 3D map shows? (The guard against falling.)"""
        angle = math.radians(facing)
        here = self.cell(x, y)
        for k in range(1, int(reach / 0.75 + 1e-9) + 1):
            px, py = x + 0.75 * k * math.cos(angle), y + 0.75 * k * math.sin(angle)
            cell = self.cell(px, py)
            if cell != here and self.edges.get(cell, 0) >= BLOCK_SCORE:
                return True
            if z is not None and self.heights.cells:
                ground = self.heights.height(px, py)
                if ground is not None and z - ground > DROP_LIMIT:
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

    def mark_free(self, x: float, y: float, z: float | None = None) -> None:
        """The player stands there: it is not blocked, whatever was thought before. `z`: the
        player's height there."""
        cell = self.cell(x, y)
        self.blocked.pop(cell, None)
        self.edges.pop(cell, None)
        self.solid.discard(cell)
        if z is not None:
            self.levels[cell] = min(self.levels.get(cell, z), z)

    def ground(self, cell: tuple[int, int]) -> float | None:
        """The ground's height in a cell: where the player stood, else what the eyes measured."""
        if cell in self.levels:
            return self.levels[cell]
        if not self.heights.cells:
            return None
        return self.heights.ground(cell[0] * CELL, cell[1] * CELL, CELL)

    def can_climb(self, a: tuple[int, int], b: tuple[int, int], grounds: dict | None = None) -> bool:
        """Can the player go from cell a to cell b, as far as the heights known tell? Not up a
        ledge, not down a drop that hurts. `grounds`: a cache of ground heights."""
        if grounds is None:
            ga, gb = self.ground(a), self.ground(b)
        else:
            for c in (a, b):
                if c not in grounds:
                    grounds[c] = self.ground(c)
            ga, gb = grounds[a], grounds[b]
        return ga is None or gb is None or -DROP_LIMIT <= gb - ga <= CLIMB

    def _ray(self, x: float, y: float, angle: float, start: float, end: float) -> set[tuple[int, int]]:
        """The cells from `start` to `end` metres in that direction."""
        steps = int((end - start) / (CELL / 2)) + 1
        cos, sin = math.cos(angle), math.sin(angle)
        return {self.cell(x + (start + k * CELL / 2) * cos, y + (start + k * CELL / 2) * sin) for k in range(steps)}

    def _vote(self, votes: dict, seen: set, clear: set, strength: int) -> bool:
        """One look: cells seen gain `strength` votes, cells seen clear lose one, each once
        (several directions often cross the same cell). True if a cell just became blocked."""
        changed = False
        for cell in seen:
            before = self.is_blocked(cell)
            votes[cell] = min(votes.get(cell, 0) + strength, SIGHT_CAP)
            changed |= not before and self.is_blocked(cell)
        for cell in clear - seen - self.solid:
            count = votes.get(cell, 0)
            if count > 1:
                votes[cell] = min(count, SIGHT_CAP) - 1
            elif count:
                del votes[cell]
        return changed

    def mark_danger(self, x: float, y: float) -> None:
        self.dangers.append((x, y))
        self.dangers = self.dangers[-50:]

    def danger_near(self, x: float, y: float) -> int:
        return sum(1 for dx, dy in self.dangers if math.hypot(x - dx, y - dy) < DANGER_RADIUS)

    def note_sight(self, x: float, y: float, facing: float, bearings, distances, drops,
                   drop_ahead: float = DROP_AHEAD, reach: float = 12.0) -> bool:
        """Put on the map what one look shows from (x, y): per direction (degrees from `facing`),
        the distance of what is in the way (None: nothing within `reach`) and of where the ground
        falls away (None: it doesn't; True: it does, distance unknown). What is seen clear loses a
        vote: mistakes of the eyes are undone, but an edge only where an edge could have been seen.
        True if a cell just became blocked."""
        walls, open_, edges, safe = set(), set(), set(), set()
        for bearing, distance, drop in zip(bearings, distances, drops):
            angle = math.radians(facing + bearing)
            cos, sin = math.cos(angle), math.sin(angle)
            if drop is True:
                drop = drop_ahead
            elif drop is False:
                drop = None
            if distance is None:
                open_ |= self._ray(x, y, angle, CLEAR_FROM, reach)
            else:
                walls.add(self.cell(x + distance * cos, y + distance * sin))
            if drop is not None:
                edges.add(self.cell(x + drop * cos, y + drop * sin))
            elif distance is None or distance > EDGE_SEEN[1]:
                safe |= self._ray(x, y, angle, *EDGE_SEEN)
        changed = self._vote(self.blocked, walls, open_, 1)
        return self._vote(self.edges, edges, safe, EDGE_VOTES) or changed

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
        levels = self.levels
        heights = bool(self.levels or self.heights.cells)
        grounds: dict = {}
        by_walls = {(c[0] + dx, c[1] + dy) for c in walls for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        danger_costs: dict[tuple[int, int], float] = {}

        def cost(c: tuple[int, int]) -> float:
            extra = WALL_MARGIN_COST if c in by_walls else 0.0
            if levels and c not in levels:
                extra += UNKNOWN_COST
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
                    if heights and not self.can_climb(c, n, grounds):
                        continue  # a ledge: fine to go down, not up
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
    points = [(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps) for k in range(steps + 1)]
    cells = [terrain.cell(*p) for p in points]
    if (terrain.levels or terrain.heights.cells) and not all(
            terrain.can_climb(c, n) for c, n in zip(cells, cells[1:]) if c != n):
        return False
    return terrain.path_is_clear(points) and all(
        terrain.danger_near(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        <= terrain.danger_near(*a) for k in range(0, steps + 1, 4))
