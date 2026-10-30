"""Moving in the real game: calibration, turning, walking, mining by hand.

Everything here drives a "raw game" that only knows a few basic things, so it
can be tested without the game (see tests/test_moves.py):

    player() -> PlayerView      nodes() -> tuple[Node, ...]
    mouse(dx, dy)               hold(key) / release(key)
    now() -> seconds            wait(seconds)

game_io.RealGame is the real one: the mod for reading, SendInput for acting.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from config import KEYS
from game_state import Node, PlayerView

CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"
WALK_PITCH = 5  # degrees below the horizon while walking: the eyes need to see what is ahead
FRESH_VIEW_S = 1.5  # a view older than this is not used
MOUSE_TICK = 0.012  # seconds between two small mouse moves
TURN_SPEED = 300.0  # degrees per second, turning on the spot
STEER_SPEED = 150.0  # degrees per second at most, correcting the course while walking
WAYPOINT_REACHED = 2.5  # metres
STUCK_S, STUCK_MOVE = 1.2, 0.4  # moved less than this (m) in this time (s): stuck
NO_PROGRESS_S = 25.0  # no closer to the goal for this long: give up
FALL_DROP = 3.0  # metres lost in height between two looks: a fall
BLOCK_NOW = 2  # stuck or fell there: blocked at once
TRUST_SIGHT = 12.0  # metres: obstacles seen farther than this are not put on the map
DROP_AHEAD = 4.0  # metres: where a drop seen ahead is put on the map
STEP_S = 0.12  # how often the walk is corrected
DETOUR_SIDES = (90, -90)  # degrees from the goal's direction: right, then left
DETOUR_NEAR, DETOUR_FAR = 12.0, 30.0  # metres to the detour's waypoint
ESCAPE_TURN = 70.0  # degrees turned to go round what keeps blocking the way
GUARD_ESCAPE = 4.0  # metres ahead checked before stepping off while getting out
LEARNT_STUCK_S, LEARNT_STUCK_MOVE = 0.6, 0.5  # learnt walk: moved less than this (m) in this time (s), as terrain_sim.STUCK_STEPS
MINE_S_PER_ORE = 2.0  # seconds allowed per ore mined by hand
MINE_RADIUS = 2.5  # metres from the node's centre where we stop to mine
EYE_HEIGHT = 1.7  # metres
AIM_PITCHES = (15, 25, 35, 45, 55, 65)  # degrees below the horizon tried to find the node
TAP_S, LONG_PRESS_S = 0.1, 1.5


class SkillError(Exception):
    """A skill could not be done; the message says why, in French."""


def wrap(degrees: float) -> float:
    """Angle between -180 and 180."""
    return (degrees + 180) % 360 - 180


def heading(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    return math.degrees(math.atan2(to_y - from_y, to_x - from_x))


# ---------------------------------------------------------------------- calibration


@dataclass
class Calibration:
    """How the mod's yaw and the mouse relate to where the player really faces.

    Facing is the direction the player walks to, as an angle on the map:
    facing = yaw_sign * yaw + yaw_offset.
    """

    yaw_sign: int = 1
    yaw_offset: float = 0.0
    facing_per_count: float = 0.1  # degrees of facing per mouse count to the right
    pitch_per_count: float = 0.1  # degrees looked down per mouse count down
    walk_speed: float = 5.0  # metres per second

    def facing(self, yaw: float) -> float:
        return wrap(self.yaw_sign * yaw + self.yaw_offset)

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "Calibration":
        if not path.exists():
            raise SkillError("pas de calibrage : lance d'abord « python real_game/agent.py calibrer »")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _walk_straight(raw, seconds: float) -> tuple[float, float, float]:
    """Walk forward; return (map direction walked, mod yaw meanwhile, distance)."""
    start = raw.player()
    raw.hold(KEYS["forward"])
    raw.wait(seconds)
    raw.release(KEYS["forward"])
    raw.wait(0.5)
    end = raw.player()
    distance = math.hypot(end.x - start.x, end.y - start.y)
    if distance < 1.0:
        raise SkillError("le personnage n'a pas avancé : il faut un terrain plat et dégagé devant lui")
    return heading(start.x, start.y, end.x, end.y), start.yaw, distance / seconds


def calibrate(raw, counts: int = 400) -> Calibration:
    """Turn and walk a little to measure the mouse and the mod's yaw. Needs ~15 m of open ground."""
    before = raw.player().yaw
    raw.mouse(counts, 0)
    raw.wait(0.5)
    turned = wrap(raw.player().yaw - before)
    if abs(turned) < 1:
        raise SkillError("la caméra n'a pas tourné : le jeu reçoit-il la souris ?")
    yaw_per_count = turned / counts

    heading1, yaw1, speed1 = _walk_straight(raw, 1.2)
    raw.mouse(round(120 / abs(yaw_per_count)), 0)  # face another way to tell sign from offset
    raw.wait(0.5)
    heading2, yaw2, speed2 = _walk_straight(raw, 1.2)
    same = abs(wrap((heading2 - heading1) - (yaw2 - yaw1)))
    mirrored = abs(wrap((heading2 - heading1) + (yaw2 - yaw1)))
    sign = 1 if same <= mirrored else -1
    offsets = [wrap(heading1 - sign * yaw1), wrap(heading2 - sign * yaw2)]
    offset = math.degrees(math.atan2(sum(math.sin(math.radians(o)) for o in offsets),
                                     sum(math.cos(math.radians(o)) for o in offsets)))
    facing_per_count = sign * yaw_per_count
    return Calibration(sign, offset, facing_per_count, abs(facing_per_count), (speed1 + speed2) / 2)


# ---------------------------------------------------------------------- skills


class Mover:
    """The player's body: turns, walks and mines, checking the result through the mod."""

    def __init__(self, raw, calibration: Calibration, log=print, terrain=None, eyes=None):
        from navigation import TerrainMap

        self.raw = raw
        self.cal = calibration
        self.log = log
        self.eyes = eyes  # perception.Eyes: to see obstacles and drops while walking
        self.terrain = terrain if terrain is not None else TerrainMap()  # learnt map of the ground
        self.pitch: float | None = None  # degrees looked down, once known
        self.avoid: dict[tuple[int, int], float] = {}  # nodes to leave alone until a given time
        self.current_node: Node | None = None
        self.walker = None  # walk_policy.WalkPolicy: the walk learnt in the simulation, if trained
        self._seen_at = 0.0
        self._detouring = False

    def facing(self, player: PlayerView | None = None) -> float:
        return self.cal.facing((player or self.raw.player()).yaw)

    # Hand on the mouse -------------------------------------------------------------

    def _glide(self, dx_counts: float, dy_counts: float, seconds: float) -> None:
        """Move the mouse smoothly: speed up, then slow down, like a hand would."""
        steps = max(1, int(seconds / MOUSE_TICK))
        sent_x = sent_y = 0
        for k in range(1, steps + 1):
            t = k / steps
            eased = t * t * (3 - 2 * t)
            x, y = round(dx_counts * eased), round(dy_counts * eased)
            if (x, y) != (sent_x, sent_y):
                self.raw.mouse(x - sent_x, y - sent_y)
                sent_x, sent_y = x, y
            self.raw.wait(MOUSE_TICK)

    def turn(self, degrees: float, speed: float = TURN_SPEED) -> None:
        """Turn right by `degrees` (left if negative), smoothly."""
        if abs(degrees) < 0.3:
            return
        self._glide(degrees / self.cal.facing_per_count, 0, max(0.08, 1.3 * abs(degrees) / speed))

    def tilt(self, degrees_down: float, speed: float = TURN_SPEED) -> None:
        """Look further down by `degrees_down` (up if negative), smoothly."""
        if self.pitch is not None:
            degrees_down = max(-85 - self.pitch, min(85 - self.pitch, degrees_down))
        if abs(degrees_down) < 0.3:
            return
        self._glide(0, degrees_down / self.cal.pitch_per_count, max(0.08, 1.3 * abs(degrees_down) / speed))
        if self.pitch is not None:
            self.pitch += degrees_down
            if self.eyes is not None:
                self.eyes.pitch = self.pitch  # to place what they see on the 3D map

    def set_pitch(self, degrees_down: float) -> None:
        """Look `degrees_down` below the horizon. The first time, the view is levelled once
        (all the way up, the game stops it there); after that every change is a smooth move."""
        if self.pitch is None:
            self.raw.mouse(0, -round(120 / self.cal.pitch_per_count))
            self.raw.wait(0.05)
            self.pitch = -89.0
        self.tilt(degrees_down - self.pitch)

    def face(self, x: float, y: float) -> None:
        player = self.raw.player()
        self.turn(wrap(heading(player.x, player.y, x, y) - self.facing(player)))
        self.raw.wait(0.1)

    # Walking ---------------------------------------------------------------------------

    def walk_to(self, x: float, y: float, radius: float = 2.0, timeout: float | None = None) -> bool:
        """Walk to (x, y) along a path worked out on the learnt map, avoiding what is seen
        in the way and learning where the ground can't be crossed. If no way is found, try going
        round by one side, then by the other. False if it can't get there."""
        if self._walk(x, y, radius, timeout):
            return True
        if timeout is not None or self._detouring:  # a walk with a time limit (fleeing) or a detour itself
            return False
        self._detouring = True
        try:
            for side in DETOUR_SIDES:
                player = self.raw.player()
                distance = math.hypot(x - player.x, y - player.y)
                if distance <= radius:
                    return True
                angle = math.radians(heading(player.x, player.y, x, y) + side)
                reach = min(DETOUR_FAR, max(DETOUR_NEAR, distance / 2))
                self.log(f"  je tente un détour par la {'droite' if side > 0 else 'gauche'}")
                self._walk(player.x + reach * math.cos(angle), player.y + reach * math.sin(angle), 4.0, None)
                if self._walk(x, y, radius, None):
                    return True
            return False
        finally:
            self._detouring = False

    def _walk(self, x: float, y: float, radius: float, timeout: float | None) -> bool:
        player = self.raw.player()
        goal = (x, y)
        distance = math.hypot(x - player.x, y - player.y)
        timeout = timeout or 30 + 4 * distance / self.cal.walk_speed
        if self.walker is not None and self.eyes is not None:
            return self._walk_learnt(goal, radius, timeout)
        start = best_at = self.raw.now()
        best = distance
        path = self._plan(player, goal)
        forward = False
        check_at, check_pos, unstick = start + STUCK_S, (player.x, player.y), 0
        last_z = player.z
        if self.eyes is not None:
            self.set_pitch(WALK_PITCH)
        try:
            while self.raw.now() - start < timeout:
                player = self.raw.player()
                now = self.raw.now()
                distance = math.hypot(x - player.x, y - player.y)
                if distance <= radius:
                    return True
                self.terrain.mark_free(player.x, player.y, player.z)
                if last_z - player.z > FALL_DROP:
                    self.log("  chute : je retiens ce bord")
                    self.terrain.mark_blocked(check_pos[0], check_pos[1], strength=BLOCK_NOW, solid=True)
                last_z = player.z
                if self._learn_from_eyes(player) and not self.terrain.route_is_clear((player.x, player.y), path[:2], goal):
                    path = self._plan(player, goal)
                if distance < best - 1.0:
                    best, best_at = distance, now
                elif now - best_at > NO_PROGRESS_S:
                    self.log("  je n'avance plus vers le but")
                    return False
                while len(path) > 1 and math.hypot(path[0][0] - player.x, path[0][1] - player.y) < WAYPOINT_REACHED:
                    path.pop(0)
                delta = wrap(heading(player.x, player.y, *path[0]) - self.facing(player))
                if abs(delta) > 70:  # far off course: stop and turn on the spot
                    if forward:
                        self.raw.release(KEYS["forward"])
                        forward = False
                    self.turn(delta)
                    check_at, check_pos = self.raw.now() + STUCK_S, (player.x, player.y)
                    continue
                correction = max(-STEER_SPEED * STEP_S, min(STEER_SPEED * STEP_S, delta))
                if not forward:
                    self.raw.hold(KEYS["forward"])
                    forward = True
                if abs(correction) > 1:
                    self.turn(correction, speed=2 * STEER_SPEED)  # steer while walking
                else:
                    self.raw.wait(STEP_S)
                if self.raw.now() >= check_at:
                    moved = math.hypot(player.x - check_pos[0], player.y - check_pos[1])
                    if moved < STUCK_MOVE:  # stuck against something: it goes on the map
                        unstick += 1
                        ahead = math.radians(self.facing(player))
                        self.terrain.mark_blocked(player.x + 1.5 * math.cos(ahead), player.y + 1.5 * math.sin(ahead),
                                                  strength=BLOCK_NOW, solid=True)
                        self._unstick(unstick)
                        path = self._plan(self.raw.player(), goal)
                    check_at, check_pos = self.raw.now() + STUCK_S, (player.x, player.y)
            return False
        finally:
            if forward:
                self.raw.release(KEYS["forward"])

    def _walk_learnt(self, goal: tuple[float, float], radius: float, timeout: float) -> bool:
        """walk_to with the walk learnt in the simulation: the map and A* choose the way, the
        learnt policy chooses every 0.2 s how to get to the next waypoint (go round, jump, stop
        before an edge), seeing what the eyes see."""
        from navigation import GUARD_RUN, GUARD_WALK
        from terrain_sim import DT, FORWARD_ACTIONS, RUN_ACTIONS, TURNS
        from walk_policy import observe

        player = self.raw.player()
        start = best_at = self.raw.now()
        best = math.hypot(goal[0] - player.x, goal[1] - player.y)
        path = self._plan(player, goal)
        forward, running, stuck, previous, stuck_times = False, False, False, 0, 0
        last_z, last_spot = player.z, (player.x, player.y)
        moves: list[tuple[float, float, float]] = []  # (time, x, y) while walking forward
        self.set_pitch(WALK_PITCH)
        try:
            while self.raw.now() - start < timeout:
                player = self.raw.player()
                now = self.raw.now()
                distance = math.hypot(goal[0] - player.x, goal[1] - player.y)
                if distance <= radius:
                    return True
                self.terrain.mark_free(player.x, player.y, player.z)
                if last_z - player.z > FALL_DROP:
                    self.log("  chute : je retiens ce bord")
                    self.terrain.mark_blocked(*last_spot, strength=BLOCK_NOW, solid=True)
                    self.terrain.mark_edge(*last_spot, votes=4)
                    path = self._plan(player, goal)
                last_z, last_spot = player.z, (player.x, player.y)
                if self._learn_from_eyes(player) and not self.terrain.route_is_clear((player.x, player.y), path[:2], goal):
                    path = self._plan(player, goal)
                if distance < best - 1.0:
                    best, best_at = distance, now
                elif now - best_at > NO_PROGRESS_S:
                    self.log("  je n'avance plus vers le but")
                    return False
                while len(path) > 1 and math.hypot(path[0][0] - player.x, path[0][1] - player.y) < WAYPOINT_REACHED:
                    path.pop(0)
                # Stuck: walking forward for a while but hardly moving
                moves = [m for m in moves if now - m[0] <= 3 * LEARNT_STUCK_S] if forward else []
                moves.append((now, player.x, player.y))
                earlier = [m for m in moves if now - m[0] >= LEARNT_STUCK_S]
                held_long = bool(earlier)
                moved = math.hypot(player.x - earlier[-1][1], player.y - earlier[-1][2]) if earlier else 0.0
                if held_long and moved < LEARNT_STUCK_MOVE:
                    if not stuck:  # just got stuck: the spot ahead goes on the map
                        stuck_times += 1
                        ahead = math.radians(self.facing(player))
                        self.terrain.mark_blocked(player.x + 1.5 * math.cos(ahead), player.y + 1.5 * math.sin(ahead),
                                                  strength=BLOCK_NOW, solid=True)
                        path = self._plan(player, goal)
                        if stuck_times % 2 == 0:  # stuck again: back off and go round
                            for key, held in ((KEYS["forward"], forward), (KEYS["sprint"], running)):
                                if held:
                                    self.raw.release(key)
                            forward = running = False
                            moves = []
                            self._escape(stuck_times // 2)
                    stuck = True
                elif held_long or not forward:
                    stuck = False
                pitch = self.pitch if self.pitch is not None else WALK_PITCH
                obs = observe(self._fresh_view(), pitch, self.terrain, player.x, player.y, self.facing(player),
                              path[0], player.hp, stuck, previous, z=player.z)
                action = self.walker.act(obs)
                turn = TURNS.get(action, 0.0)
                facing = self.facing(player)
                reach = GUARD_RUN if action in RUN_ACTIONS else GUARD_WALK
                wants_forward = action in FORWARD_ACTIONS and not (  # the guard: never onto a known edge
                    self.terrain.edge_ahead(player.x, player.y, facing, reach, z=player.z)
                    or self.terrain.edge_ahead(player.x, player.y, facing + turn, reach, z=player.z))
                wants_run = wants_forward and action in RUN_ACTIONS
                if wants_run != running:
                    (self.raw.hold if wants_run else self.raw.release)(KEYS["sprint"])
                    running = wants_run
                if wants_forward != forward:
                    (self.raw.hold if wants_forward else self.raw.release)(KEYS["forward"])
                    forward = wants_forward
                left = DT
                if action == 5 and wants_forward:  # jump while walking
                    self.raw.hold(KEYS["jump"])
                    self.raw.wait(TAP_S)
                    self.raw.release(KEYS["jump"])
                    left -= TAP_S
                if turn:
                    self.turn(turn, speed=1.3 * abs(turn) / left)  # smooth, lasts the step
                else:
                    self.raw.wait(left)
                previous = action
            return False
        finally:
            if forward:
                self.raw.release(KEYS["forward"])
            if running:
                self.raw.release(KEYS["sprint"])

    def _plan(self, player: PlayerView, goal: tuple[float, float]) -> list[tuple[float, float]]:
        path = self.terrain.plan((player.x, player.y), goal)
        return path if path else [goal]  # no known way: head straight and learn on the way

    def _fresh_view(self):
        """The latest view of the eyes, if recent enough."""
        view = self.eyes.latest if self.eyes is not None else None
        if view is None or self.raw.now() - view.taken_at > FRESH_VIEW_S:
            return None
        return view

    def _learn_from_eyes(self, player: PlayerView) -> bool:
        """Put on the map what the eyes see in the way. True if the map changed."""
        from perception import drop_distances, obstacle_distances

        view = self._fresh_view()
        if view is None or not view.obstacle_rows or view.taken_at <= self._seen_at:
            return False
        self._seen_at = view.taken_at
        pitch = self.pitch if self.pitch is not None else WALK_PITCH
        columns = range(len(view.obstacle_rows))
        drops = drop_distances(view, pitch, DROP_AHEAD)
        drops += [None] * (len(columns) - len(drops))
        if view.points is not None and view.pose is not None:  # the 3D map of the ground
            x, y, z, yaw = view.pose
            self.terrain.heights.add_view(x, y, z, self.cal.facing(yaw), view.points)
        return self.terrain.note_sight(player.x, player.y, self.facing(player),
                                       [view.column_bearing(i) for i in columns],
                                       obstacle_distances(view, pitch, TRUST_SIGHT), drops, DROP_AHEAD,
                                       reach=TRUST_SIGHT)

    def _escape(self, attempt: int) -> None:
        """Stuck again and again: back off, go round by one side (the other side next time) with a
        jump, never towards an edge the map knows."""
        side = ESCAPE_TURN if attempt % 2 else -ESCAPE_TURN
        self.log(f"  coincée : je recule et je contourne par la {'droite' if side > 0 else 'gauche'}")
        player = self.raw.player()
        if not self.terrain.edge_ahead(player.x, player.y, self.facing(player) + 180, z=player.z):
            self.raw.hold(KEYS["back"])
            self.raw.wait(0.8)
            self.raw.release(KEYS["back"])
        self.turn(side)
        player = self.raw.player()
        if self.terrain.edge_ahead(player.x, player.y, self.facing(player), GUARD_ESCAPE, z=player.z):
            return
        self.raw.hold(KEYS["forward"])
        self.raw.wait(0.3)
        self.raw.hold(KEYS["jump"])
        self.raw.wait(TAP_S)
        self.raw.release(KEYS["jump"])
        self.raw.wait(0.9)
        self.raw.release(KEYS["forward"])

    def _unstick(self, attempt: int) -> None:
        """Jump; after a few tries, also step sideways."""
        self.log(f"  bloqué, tentative {attempt} pour passer")
        self.raw.hold(KEYS["jump"])
        self.raw.wait(TAP_S)
        self.raw.release(KEYS["jump"])
        if attempt >= 2:
            side = KEYS["right"] if attempt % 2 else KEYS["left"]
            self.raw.hold(side)
            self.raw.wait(0.8)
            self.raw.release(side)

    def avoid_node(self, node: Node | None, seconds: float = 900) -> None:
        """Leave a node alone for a while (dangerous or out of reach)."""
        if node is not None:
            self.avoid[(round(node.x), round(node.y))] = self.raw.now() + seconds

    def _press_interact(self, seconds: float) -> None:
        self.raw.hold(KEYS["interact"])
        self.raw.wait(seconds)
        self.raw.release(KEYS["interact"])
        self.raw.wait(0.35)

    def _amount(self, item: str) -> float:
        return self.raw.player().inventory.get(item, 0)

    def go_to_node(self, resource: str) -> Node:
        player = self.raw.player()
        free = [n for n in self.raw.nodes() if n.resource == resource and not n.exploited]
        if not free:
            raise SkillError("aucun gisement libre de ce type n'est connu")
        now = self.raw.now()
        usable = [n for n in free if self.avoid.get((round(n.x), round(n.y)), 0) < now] or free
        safe = [n for n in usable if self.terrain.danger_near(n.x, n.y) == 0]
        if not safe:
            self.log("  tous les gisements connus sont près d'un danger : je prends le plus proche")
        node = min(safe or usable, key=lambda n: math.hypot(n.x - player.x, n.y - player.y))
        self.current_node = node
        far = math.hypot(node.x - player.x, node.y - player.y)
        self.log(f"  gisement {node.purity} à {far:.0f} m")
        if not self.walk_to(node.x, node.y, radius=MINE_RADIUS):
            self.avoid_node(node)
            raise SkillError("je n'arrive pas à atteindre le gisement : j'en choisirai un autre")
        return node

    def mine_by_hand(self, resource: str, amount: int, timeout: float | None = None) -> int:
        """Walk to the nearest free node of `resource` and mine `amount` of it. Returns how many."""
        timeout = timeout or 60 + MINE_S_PER_ORE * amount
        start_amount = self._amount(resource)
        node = self.go_to_node(resource)
        self.face(node.x, node.y)
        press = self._find_node_with_eyes(resource)
        self.log(f"  gisement trouvé du regard, minage de {amount}")
        deadline = self.raw.now() + timeout
        while self._amount(resource) < start_amount + amount and self.raw.now() < deadline:
            before = self._amount(resource)
            for _ in range(4):
                self._press_interact(press)
            if self._amount(resource) <= before:  # moved, or the view drifted: look again
                press = self._find_node_with_eyes(resource)
        return int(self._amount(resource) - start_amount)

    def _find_node_with_eyes(self, resource: str) -> float:
        """Look lower and lower until mining works; return how long to press the key."""
        for press in (TAP_S, LONG_PRESS_S):  # tap first; some actions need a long press
            for down in AIM_PITCHES:
                self.set_pitch(down)
                before = self._amount(resource)
                self._press_interact(press)
                self._press_interact(press)
                if self._amount(resource) > before:
                    return press
        raise SkillError("je ne trouve pas le gisement du regard")
