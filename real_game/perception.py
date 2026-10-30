"""The AI's eyes: it looks at the game screen and works out what is there.

- Creatures: an open-vocabulary detector (YOLO-World) finds what we describe in
  words ("creature", "animal", "spider"...), without being trained on Satisfactory.
- Obstacles and relief: a depth model (Depth Anything V2) tells, for every pixel,
  how near it is. Each column of the view is then open or blocked.

Both models run in the background a few times per second, so the latest view is
always ready for walking and fighting.
"""
from __future__ import annotations

import math
import threading
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

CREATURE_WORDS = ("creature", "animal", "wild boar", "lizard", "dinosaur", "spider", "monster")
DETECTOR_WEIGHTS = "yolov8s-worldv2.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_PRECISE = "depth-anything/Depth-Anything-V2-Base-hf"  # finer, about three times slower
DETECT_CONFIDENCE = 0.18
DETECT_SIZE = 800  # pixels the detector works at: bigger finds far creatures better, but is slower
ANALYSIS_WIDTH = 960  # images are shrunk to this width before analysis
COLUMNS = 15  # the view is cut into this many columns, left to right
HORIZON_BAND = (0.30, 0.62)  # rows (fractions of the height) where something in the way shows
GROUND_BAND = (0.62, 0.84)  # rows that show the ground in front in nearly every view (above the HUD)
SKY_BAND = (0.0, 0.15)
GROUND_PERCENTILE = 20  # the ground is found among the farthest points of each row
DROP = 0.45  # ground this much farther than expected (share of its nearness): a drop
BLOCKED = 0.35  # sticking out of the ground by this share of the ground's depth range: in the way
# Parts of the screen where the player's own hand, tool and HUD are: never creatures.
POINT_STEP = 6  # one depth pixel in this many, each way, is placed in the 3D map
POINT_FAR = 20.0  # metres: farther, the depth is too rough to place things
GROUND_FIT_ROWS = 8  # rows of ground needed to set the depth's scale
NOT_THE_WORLD = ((0.55, 0.55, 1.0, 1.0), (0.0, 0.85, 1.0, 1.0))


def bearing_of(x: float, width: float, fov: float) -> float:
    """Horizontal angle (degrees, + to the right) of a point at column `x` of an image."""
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    return math.degrees(math.atan((x - width / 2) / focal))


def vertical_of(y: float, height: float, width: float, fov: float) -> float:
    """Vertical angle (degrees, + below the centre) of a point at row `y`."""
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    return math.degrees(math.atan((y - height / 2) / focal))


@dataclass(frozen=True)
class DepthReading:
    clearance: tuple[float, ...]  # per column: 1 = clear, 0.5 = at the limit, 0 = right in front
    obstacle_rows: tuple[float | None, ...]  # per column: where the obstacle meets the ground (row fraction)
    drops: tuple[bool, ...]  # per column: the ground falls away (cliff edge, hole, water)
    drop_rows: tuple[float | None, ...] = ()  # per column: the nearest row where it has fallen away


def nearness(depth: np.ndarray) -> np.ndarray:
    """The depth picture as « bigger is nearer », whichever way the model gives it, with the
    player's hand and the HUD left out (NaN). A copy."""
    height = depth.shape[0]
    width = depth.shape[1]
    fractions = (np.arange(height) + 0.5) / height
    ground = (fractions >= GROUND_BAND[0]) & (fractions <= GROUND_BAND[1])
    sky = (fractions >= SKY_BAND[0]) & (fractions <= SKY_BAND[1])
    if np.median(depth[ground]) < np.median(depth[sky]):
        depth = 1.0 / np.clip(depth, 1e-6, None)  # a distance: its inverse is the nearness
    depth = depth.astype(np.float64)
    for x0, y0, x1, y1 in NOT_THE_WORLD:
        depth[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = np.nan
    return depth


def depth_points(depth: np.ndarray, pitch: float, fov: float, eye_height: float = 1.7,
                 step: int = POINT_STEP, far: float = POINT_FAR) -> np.ndarray | None:
    """3D points seen, in metres: (ahead, to the right, height above the ground under the player),
    from a depth picture taken looking `pitch` degrees down.

    The depth model gives nearness up to an unknown scale and offset (it is « affine in 1 /
    distance »). Both come from the ground just ahead, whose distance is known: it lies
    `eye_height` below the eyes. None if that ground can't be made out."""
    near = nearness(depth)
    height, width = near.shape
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    down = ((np.arange(height) + 0.5) - height / 2) / focal  # per row: how far below the centre
    across = ((np.arange(width) + 0.5) - width / 2) / focal  # per column: how far to the right
    sin_p, cos_p = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sinking = sin_p + down * cos_p  # how steeply each row's rays go down
    fractions = (np.arange(height) + 0.5) / height
    band = (fractions >= GROUND_BAND[0]) & (fractions <= GROUND_BAND[1]) & (sinking > 0.05)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ground = np.nanpercentile(near[band], GROUND_PERCENTILE, axis=1)  # the farthest surface: the ground
    inverse = sinking[band] / eye_height  # 1 / depth of the ground, row by row
    known = np.isfinite(ground)
    if known.sum() < GROUND_FIT_ROWS:
        return None
    scale, offset = np.polyfit(inverse[known], ground[known], 1)
    if scale <= 0:
        return None
    rows, cols = np.arange(0, height, step), np.arange(0, width, step)
    with np.errstate(all="ignore"):
        inv = (near[np.ix_(rows, cols)] - offset) / scale
        distance = 1.0 / inv  # along the view's axis
        d, a = down[rows][:, None], across[cols][None, :]
        ahead = distance * (cos_p - d * sin_p)
        right = distance * a
        above = eye_height - distance * (sin_p + d * cos_p)
        keep = np.isfinite(distance) & (inv > 0) & (ahead > 0.5) & (ahead < far) & (np.abs(above) < 30)
    return np.stack([ahead[keep], right[keep], above[keep]], axis=1).astype(np.float32)


def read_depth(depth: np.ndarray, columns: int = COLUMNS) -> DepthReading:
    """What a depth picture says about the way ahead.

    The bottom of the view is nearly always ground: how its nearness grows from row
    to row gives a model of the ground, whatever the camera's tilt. What sticks out
    of that ground (a rock, a trunk, a wall) is in the way; ground much farther than
    the model expects is a drop. Depth models give a relative scale, sometimes
    « bigger is nearer », sometimes the opposite: both work.
    """
    height, width = depth.shape
    fractions = (np.arange(height) + 0.5) / height

    def rows(band):
        return (fractions >= band[0]) & (fractions <= band[1])

    depth = nearness(depth)
    # On one row, the ground is the farthest surface: whatever hides it is nearer.
    ground = rows(GROUND_BAND)
    slope, offset = np.polyfit(fractions[ground], np.nanpercentile(depth[ground], GROUND_PERCENTILE, axis=1), 1)
    slope = max(slope, 1e-6)  # the ground gets nearer towards the bottom
    far = float(np.nanpercentile(depth, 5))
    expected = np.maximum(slope * fractions + offset, far)  # the ground's nearness, row by row
    unit = slope * (GROUND_BAND[1] - HORIZON_BAND[0])  # the ground's nearness range in view
    band = rows(HORIZON_BAND)
    band_rows = fractions[band]
    sticking_out = (depth[band] - expected[band][:, None]) / unit
    clearance, obstacle_rows, drops, drop_rows = [], [], [], []
    ground_rows = fractions[ground]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns hidden by the hand have no data
        for i in range(columns):
            cols = slice(int(i * width / columns), int((i + 1) * width / columns))
            part = sticking_out[:, cols]
            score = float(np.nanpercentile(part, 80))
            clearance.append(min(1.0, max(0.0, 1 - 0.5 * score / BLOCKED)))
            hits = np.where(np.nanmax(part, axis=1) > BLOCKED)[0]
            obstacle_rows.append(float(band_rows[hits.max()]) if len(hits) and score > BLOCKED else None)
            seen = np.nanmedian(depth[ground][:, cols])
            wanted = float(np.median(expected[ground]))
            drop = bool(np.isfinite(seen) and seen < DROP * wanted)
            drops.append(drop)
            fallen = np.nanmedian(depth[ground][:, cols], axis=1) < DROP * expected[ground]  # row by row
            drop_rows.append(float(ground_rows[fallen].max()) if drop and fallen.any() else None)
    return DepthReading(tuple(clearance), tuple(obstacle_rows), tuple(drops), tuple(drop_rows))


def clearance_from_depth(depth: np.ndarray, columns: int = COLUMNS) -> tuple[float, ...]:
    """How open each column is: 1 = clear, 0.5 = at the limit, 0 = something right in front."""
    return read_depth(depth, columns).clearance


def ground_distance(row: float, pitch: float, fov: float, aspect: float, eye_height: float = 1.7,
                    farthest: float = 15.0) -> float:
    """Metres to the ground seen at a row (fraction of the height), looking `pitch` degrees down."""
    below = pitch + vertical_of(row * aspect, aspect, 1.0, fov)
    if below <= 1:
        return farthest
    return min(farthest, eye_height / math.tan(math.radians(below)))


def obstacle_distances(view: "View", pitch: float, trust: float = 12.0) -> list[float | None]:
    """Per column: metres to what is in the way, None if nothing (or farther than `trust`)."""
    found = []
    for row in view.obstacle_rows:
        distance = None if row is None else ground_distance(row, pitch, view.fov, view.aspect)
        found.append(distance if distance is not None and distance <= trust else None)
    return found


def drop_distances(view: "View", pitch: float, unknown: float = 4.0) -> list[float | None]:
    """Per column: metres to where the ground falls away (`unknown` if it does but the row is
    unknown), None if it doesn't."""
    found = []
    for i, drop in enumerate(view.drops):
        row = view.drop_rows[i] if i < len(view.drop_rows) else None
        found.append(None if not drop else unknown if row is None
                     else ground_distance(row, pitch, view.fov, view.aspect))
    return found


@dataclass(frozen=True)
class Threat:
    bearing: float  # degrees to turn to face it, + to the right
    below: float  # degrees below the centre of the screen
    size: float  # share of the screen height it covers: bigger means nearer
    score: float
    label: str
    box: tuple[float, float, float, float]  # in analysed-image pixels


@dataclass(frozen=True)
class View:
    taken_at: float  # time.perf_counter() when the image was captured
    threats: tuple[Threat, ...]
    clearance: tuple[float, ...]  # one value per column, left to right (empty: no depth)
    fov: float
    obstacle_rows: tuple[float | None, ...] = ()  # where each column's obstacle meets the ground
    drops: tuple[bool, ...] = ()  # columns where the ground falls away
    aspect: float = 9 / 16  # height / width of the image analysed
    drop_rows: tuple[float | None, ...] = ()  # where each column's ground has fallen away, nearest
    pose: tuple[float, float, float, float] | None = None  # player's x, y, z, yaw when the picture was taken
    pitch: float | None = None  # degrees the view looked down then
    points: np.ndarray | None = field(default=None, compare=False, repr=False)  # depth_points of the picture

    def column_bearing(self, index: int) -> float:
        width = len(self.clearance)
        return bearing_of(index + 0.5, width, self.fov)

    def is_open(self, bearing: float, half_width: float = 10.0) -> bool:
        """Is the way clear in that direction (within the view)?"""
        near = [c for i, c in enumerate(self.clearance) if abs(self.column_bearing(i) - bearing) <= half_width]
        return bool(near) and min(near) >= 0.5

    def best_direction(self, target: float) -> float | None:
        """The open direction closest to `target` (degrees), or None if everything is blocked."""
        options = [self.column_bearing(i) for i in range(len(self.clearance)) if self.is_open(self.column_bearing(i))]
        return min(options, key=lambda b: abs(b - target), default=None)

    def nearest_threat(self) -> Threat | None:
        return max(self.threats, key=lambda t: t.size, default=None)


def threats_from_boxes(boxes, width: int, height: int, fov: float) -> tuple[Threat, ...]:
    """Detections (x0, y0, x1, y1, score, label) -> threats, leaving out the player's own hand and
    HUD: what lies entirely in their corner. A creature right next to the player reaches out of it."""
    found = []
    for x0, y0, x1, y1, score, label in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(a * width <= x0 and x1 <= c * width and b * height <= y0 and y1 <= d * height
               for a, b, c, d in NOT_THE_WORLD):
            continue
        found.append(Threat(bearing_of(cx, width, fov), vertical_of(cy, height, width, fov),
                            (y1 - y0) / height, score, label, (x0, y0, x1, y1)))
    return tuple(found)


class Eyes:
    """Watches the game window in the background and keeps the latest View."""

    def __init__(self, box: tuple[int, int, int, int], fov: float, creatures: bool = True, depth: bool = True,
                 log=print):
        self.box = box  # left, top, width, height of the game window
        self.fov = fov
        self.log = log
        self.latest: View | None = None
        self.latest_image: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.confidence = DETECT_CONFIDENCE  # lowered during a fight: a doubtful shape nearby is worth a look
        self.pitch: float | None = None  # degrees the view looks down, told by the Mover
        self.where = None  # () -> (x, y, z, yaw) of the player: to place what is seen on the 3D map
        self.depth_model = DEPTH_MODEL
        self._want = (creatures, depth)
        self._detector = self._depth = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "Eyes":
        creatures, depth = self._want
        if creatures:
            from ultralytics import YOLOWorld  # downloads the weights the first time

            self._detector = YOLOWorld(DETECTOR_WEIGHTS)
            self._detector.set_classes(list(CREATURE_WORDS))
        if depth:
            from transformers import pipeline  # downloads the model the first time

            self._depth = pipeline("depth-estimation", model=self.depth_model, device=-1)
        self._running = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False

    def look(self, image: np.ndarray, taken_at: float, pose=None, pitch: float | None = None) -> View:
        """Analyse one picture of the game (BGR), taken from `pose` looking `pitch` degrees down."""
        import cv2

        scale = ANALYSIS_WIDTH / image.shape[1]
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = small.shape[:2]
        threats: tuple[Threat, ...] = ()
        if self._detector is not None:
            result = self._detector.predict(small, imgsz=DETECT_SIZE, conf=self.confidence, verbose=False)[0]
            boxes = [
                (*xyxy, score, CREATURE_WORDS[int(cls)])
                for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                            result.boxes.cls.tolist())
            ]
            threats = threats_from_boxes(boxes, width, height, self.fov)
        reading = DepthReading((), (), ())
        points = None
        if self._depth is not None:
            from PIL import Image

            depth = self._depth(Image.fromarray(small[:, :, ::-1]))["predicted_depth"]
            self.last_depth = np.asarray(depth.squeeze().cpu().numpy(), dtype=np.float32)
            reading = read_depth(self.last_depth)
            if pose is not None and pitch is not None:
                points = depth_points(self.last_depth, pitch, self.fov)
        return View(taken_at, threats, reading.clearance, self.fov, reading.obstacle_rows, reading.drops,
                    height / width, reading.drop_rows, pose, pitch, points)

    def _watch(self) -> None:
        import mss

        left, top, width, height = self.box
        with mss.mss() as screen:
            while self._running:
                taken_at = time.perf_counter()
                image = np.asarray(screen.grab({"left": left, "top": top, "width": width, "height": height}))[:, :, :3]
                pitch = self.pitch
                try:
                    pose = self.where() if self.where is not None else None
                except Exception:  # the mod didn't answer: the picture is still good for the rest
                    pose = None
                try:
                    self.latest = self.look(image, taken_at, pose, pitch)
                    self.latest_image = image
                except Exception as error:  # a bad frame must not blind the AI for good
                    self.log(f"  vision : {error}")
                    time.sleep(0.5)

    def fresh(self, since: float, timeout: float = 3.0) -> View | None:
        """The first view taken after `since`, waiting for it up to `timeout` seconds."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            view = self.latest
            if view is not None and view.taken_at >= since:
                return view
            time.sleep(0.05)
        return None


def annotate(image: np.ndarray, view: View) -> np.ndarray:
    """A copy of the image showing what the AI saw: creatures framed, open (green) / blocked (red) columns."""
    import cv2

    scale = ANALYSIS_WIDTH / image.shape[1]
    out = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).copy()
    height, width = out.shape[:2]
    for i, share in enumerate(view.clearance):
        x0, x1 = int(i * width / len(view.clearance)), int((i + 1) * width / len(view.clearance))
        colour = (0, 200, 0) if share >= 0.5 else (0, 0, 230)
        cv2.rectangle(out, (x0 + 2, int(height * HORIZON_BAND[0])), (x1 - 2, int(height * HORIZON_BAND[1])), colour, 2)
        cv2.putText(out, f"{share:.2f}", (x0 + 6, int(height * HORIZON_BAND[0]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    for t in view.threats:
        x0, y0, x1, y1 = (int(v) for v in t.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 140, 255), 3)
        cv2.putText(out, f"{t.label} {t.score:.2f} {t.bearing:+.0f} deg", (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return out
