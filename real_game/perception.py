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
from dataclasses import dataclass

import numpy as np

CREATURE_WORDS = ("creature", "animal", "spider", "monster", "alien")
DETECTOR_WEIGHTS = "yolov8s-worldv2.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DETECT_CONFIDENCE = 0.25
ANALYSIS_WIDTH = 960  # images are shrunk to this width before analysis
COLUMNS = 15  # the view is cut into this many columns, left to right
HORIZON_BAND = (0.35, 0.62)  # rows (fractions of the height) where obstacles at body height show
GROUND_BAND = (0.85, 0.98)  # rows showing the ground just in front of the player
SKY_BAND = (0.0, 0.15)
BLOCKED = 0.55  # a column is blocked when it is at least this near, compared with the ground in front
# Parts of the screen where the player's own hand, tool and HUD are: never creatures.
NOT_THE_WORLD = ((0.55, 0.55, 1.0, 1.0), (0.0, 0.85, 1.0, 1.0))


def bearing_of(x: float, width: float, fov: float) -> float:
    """Horizontal angle (degrees, + to the right) of a point at column `x` of an image."""
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    return math.degrees(math.atan((x - width / 2) / focal))


def vertical_of(y: float, height: float, width: float, fov: float) -> float:
    """Vertical angle (degrees, + below the centre) of a point at row `y`."""
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    return math.degrees(math.atan((y - height / 2) / focal))


def clearance_from_depth(depth: np.ndarray, columns: int = COLUMNS) -> tuple[float, ...]:
    """How open each column is: 1 = far away, 0.5 = at the limit, 0 = right in front.

    Depth models give a relative scale (sometimes « bigger is nearer », sometimes the
    opposite): the ground just in front of the player sets both the direction and the
    unit, so no calibration is needed.
    """
    height, width = depth.shape
    def rows(band):
        return depth[int(height * band[0]):max(int(height * band[1]), int(height * band[0]) + 1)]

    if np.median(rows(GROUND_BAND)) < np.median(rows(SKY_BAND)):
        depth = 1.0 / np.clip(depth, 1e-6, None)  # a distance: its inverse is the nearness
    reference = float(np.median(rows(GROUND_BAND))) or 1e-6
    band = rows(HORIZON_BAND)
    open_share = []
    for i in range(columns):
        part = band[:, int(i * width / columns):int((i + 1) * width / columns)]
        nearness = float(np.percentile(part, 90)) / reference
        open_share.append(min(1.0, max(0.0, 1 - 0.5 * nearness / BLOCKED)))
    return tuple(open_share)


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
    """Detections (x0, y0, x1, y1, score, label) -> threats, leaving out the player's own hand and HUD."""
    found = []
    for x0, y0, x1, y1, score, label in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(a * width <= cx <= c * width and b * height <= cy <= d * height for a, b, c, d in NOT_THE_WORLD):
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

            self._depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=-1)
        self._running = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False

    def look(self, image: np.ndarray, taken_at: float) -> View:
        """Analyse one picture of the game (BGR)."""
        import cv2

        scale = ANALYSIS_WIDTH / image.shape[1]
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = small.shape[:2]
        threats: tuple[Threat, ...] = ()
        if self._detector is not None:
            result = self._detector.predict(small, imgsz=640, conf=DETECT_CONFIDENCE, verbose=False)[0]
            boxes = [
                (*xyxy, score, CREATURE_WORDS[int(cls)])
                for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                            result.boxes.cls.tolist())
            ]
            threats = threats_from_boxes(boxes, width, height, self.fov)
        clearance: tuple[float, ...] = ()
        if self._depth is not None:
            from PIL import Image

            depth = self._depth(Image.fromarray(small[:, :, ::-1]))["predicted_depth"]
            clearance = clearance_from_depth(np.asarray(depth.squeeze().cpu().numpy(), dtype=np.float32))
        return View(taken_at, threats, clearance, self.fov)

    def _watch(self) -> None:
        import mss

        left, top, width, height = self.box
        with mss.mss() as screen:
            while self._running:
                taken_at = time.perf_counter()
                image = np.asarray(screen.grab({"left": left, "top": top, "width": width, "height": height}))[:, :, :3]
                try:
                    self.latest = self.look(image, taken_at)
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
        cv2.putText(out, f"{share:.1f}", (x0 + 6, int(height * HORIZON_BAND[0]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    for t in view.threats:
        x0, y0, x1, y1 = (int(v) for v in t.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 140, 255), 3)
        cv2.putText(out, f"{t.label} {t.score:.2f} {t.bearing:+.0f} deg", (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return out
