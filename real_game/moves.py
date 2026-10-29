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
FRESH_VIEW_S = 1.5  # a view older than this is not used to steer
STEP_S = 0.15  # how often the walk is corrected
MINE_RADIUS = 2.5  # metres from the node's centre where we stop to mine
DANGER_RADIUS = 50.0  # metres around a place where the player was attacked
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

    def __init__(self, raw, calibration: Calibration, log=print, dangers: list | None = None, eyes=None):
        self.raw = raw
        self.cal = calibration
        self.log = log
        self.eyes = eyes  # perception.Eyes: to see obstacles while walking
        self.dangers = dangers if dangers is not None else []  # [x, y] where the player was attacked

    def facing(self, player: PlayerView | None = None) -> float:
        return self.cal.facing((player or self.raw.player()).yaw)

    def turn(self, degrees: float) -> None:
        """Turn right by `degrees` (left if negative), in small mouse moves like a hand would."""
        counts = round(degrees / self.cal.facing_per_count)
        while counts:
            part = max(-150, min(150, counts))
            self.raw.mouse(part, 0)
            counts -= part
            self.raw.wait(0.01)

    def set_pitch(self, degrees_down: float) -> None:
        """Look `degrees_down` below the horizon: look all the way up first, then down."""
        up = round(120 / self.cal.pitch_per_count)
        self.raw.mouse(0, -up)
        self.raw.wait(0.05)
        self.raw.mouse(0, round((89 + degrees_down) / self.cal.pitch_per_count))
        self.raw.wait(0.1)

    def face(self, x: float, y: float) -> None:
        player = self.raw.player()
        self.turn(wrap(heading(player.x, player.y, x, y) - self.facing(player)))
        self.raw.wait(0.2)

    def walk_to(self, x: float, y: float, radius: float = 2.0, timeout: float | None = None) -> bool:
        """Walk until within `radius` metres of (x, y) on the map. Returns False if it can't."""
        player = self.raw.player()
        distance = math.hypot(x - player.x, y - player.y)
        timeout = timeout or 15 + 3 * distance / self.cal.walk_speed
        start = self.raw.now()
        forward = False
        check_at, check_pos, unstick = start + 1.5, (player.x, player.y), 0
        if self.eyes is not None:
            self.set_pitch(WALK_PITCH)
        try:
            while self.raw.now() - start < timeout:
                player = self.raw.player()
                distance = math.hypot(x - player.x, y - player.y)
                if distance <= radius:
                    return True
                delta = self.steer(wrap(heading(player.x, player.y, x, y) - self.facing(player)))
                if abs(delta) > 35:  # far off course: stop and turn on the spot
                    if forward:
                        self.raw.release(KEYS["forward"])
                        forward = False
                    self.turn(delta)
                    self.raw.wait(0.2)
                    check_at, check_pos = self.raw.now() + 1.5, (player.x, player.y)
                    continue
                if abs(delta) > 3:
                    self.turn(delta)
                if not forward:
                    self.raw.hold(KEYS["forward"])
                    forward = True
                self.raw.wait(STEP_S)
                if self.raw.now() >= check_at:
                    moved = math.hypot(player.x - check_pos[0], player.y - check_pos[1])
                    if moved < 0.5:  # stuck against something
                        unstick += 1
                        self._unstick(unstick)
                    check_at, check_pos = self.raw.now() + 1.5, (player.x, player.y)
            return False
        finally:
            if forward:
                self.raw.release(KEYS["forward"])

    def steer(self, delta: float) -> float:
        """Direction to walk, given the target's: around what the eyes see in the way."""
        view = self.eyes.latest if self.eyes is not None else None
        if view is None or not view.clearance or self.raw.now() - view.taken_at > FRESH_VIEW_S:
            return delta
        if abs(delta) > view.fov / 2 - 5 or view.is_open(delta):
            return delta  # target out of sight (turn first), or the way is clear
        best = view.best_direction(delta)
        if best is not None:
            return best
        half = len(view.clearance) // 2
        left, right = sum(view.clearance[:half]), sum(view.clearance[-half:])
        return delta + (60 if right >= left else -60)  # everything blocked: turn to the more open side

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
        safe = [n for n in free if all(math.hypot(n.x - x, n.y - y) > DANGER_RADIUS for x, y in self.dangers)]
        if safe:
            free = safe
        else:
            self.log("  tous les gisements connus sont près d'un danger : je prends le plus proche")
        node = min(free, key=lambda n: math.hypot(n.x - player.x, n.y - player.y))
        far = math.hypot(node.x - player.x, node.y - player.y)
        self.log(f"  gisement {node.purity} à {far:.0f} m")
        if not self.walk_to(node.x, node.y, radius=MINE_RADIUS):
            raise SkillError("je n'arrive pas à atteindre le gisement")
        return node

    def mine_by_hand(self, resource: str, amount: int, timeout: float = 240) -> int:
        """Walk to the nearest free node of `resource` and mine `amount` of it. Returns how many."""
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
