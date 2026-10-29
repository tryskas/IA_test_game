"""Seeing the screen: capture it, and find a text on it (OCR, French and English).

Reading the whole screen is slow (several seconds), so menus are read in the
area where they appear. Areas are fractions of the screen, so they work at any
resolution.
"""
from __future__ import annotations

import difflib
import re
import time
import unicodedata
from dataclasses import dataclass

# Where each menu shows its texts: (left, top, right, bottom) as fractions of the
# screen, and the size it is read at (smaller is faster, bigger reads short words better).
AREAS = {
    "inventory": ((0.25, 0.10, 0.55, 0.30), 0.5),
    "build_menu": ((0.15, 0.08, 0.50, 0.55), 0.5),
    "workbench": ((0.17, 0.14, 0.36, 0.80), 1.0),
    "workbench_recipe": ((0.40, 0.26, 0.54, 0.34), 1.0),  # name of the selected recipe
}
HOLOGRAM_AREA = (0.20, 0.25, 0.80, 0.95)
SHORT_WORD = 4  # words this short tolerate one misread letter


@dataclass(frozen=True)
class TextBox:
    text: str
    x: float  # centre, in screen pixels
    y: float
    w: float
    h: float
    confidence: float

    @property
    def left(self) -> float:
        return self.x - self.w / 2


def normalize(text: str) -> str:
    """Lower case, no accents, letters and digits only: « Équiper » -> « equiper »."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def similarity(found: str, wanted: str) -> float:
    a, b = normalize(found), normalize(wanted)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _two_line_labels(boxes: list[TextBox]) -> list[TextBox]:
    """Join each text with the one just below it, for labels written on two lines."""
    joined = []
    for top in boxes:
        for below in boxes:
            gap = (below.y - below.h / 2) - (top.y + top.h / 2)
            if below is not top and -4 <= gap <= top.h and abs(below.x - top.x) <= max(top.w, below.w) / 2:
                left = min(top.left, below.left)
                right = max(top.left + top.w, below.left + below.w)
                y0, y1 = top.y - top.h / 2, below.y + below.h / 2
                joined.append(TextBox(f"{top.text} {below.text}", (left + right) / 2, (y0 + y1) / 2,
                                      right - left, y1 - y0, min(top.confidence, below.confidence)))
    return joined


def best_match(boxes: list[TextBox], wanted: str, min_score: float = 0.75,
               below: float | None = None) -> TextBox | None:
    """The text that reads most like `wanted`; the highest on screen when several do.

    `below`: only look under this height on screen (to skip a search field, say).
    """
    boxes = [b for b in boxes if below is None or b.y > below]
    scored = [(similarity(b.text, wanted), b) for b in boxes + _two_line_labels(boxes)]
    if len(normalize(wanted)) <= SHORT_WORD:
        min_score = min(min_score, 0.66)
    good = [(s, b) for s, b in scored if s >= min_score]
    if not good:
        return None
    top = max(s for s, _ in good)
    candidates = [b for s, b in good if s >= top - 0.05]
    return min(candidates, key=lambda b: b.y)


def hologram_state(image) -> str:
    """Building placement, from a picture of the middle of the screen (BGR):
    "valid" (blue hologram), "invalid" (red hologram) or "none"."""
    import cv2
    import numpy as np

    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.int16)
    b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    cyan = ((b > 150) & (g > 120) & (r < 110) & (b - r > 70)).mean()
    red = ((r > 160) & (g < 90) & (b < 90)).mean()
    if cyan >= 0.03 and red < cyan / 2:
        return "valid"
    if red >= 0.03:
        return "invalid"
    return "none"


class Screen:
    def __init__(self, left: int, top: int, width: int, height: int):
        self.left, self.top, self.width, self.height = left, top, width, height
        self._reader = None
        self._grabber = None

    def _ocr(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
        return self._reader

    def area(self, name: str) -> tuple[int, int, int, int]:
        return self._box(AREAS[name][0])

    def _box(self, fractions) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = fractions
        return (round(self.left + x0 * self.width), round(self.top + y0 * self.height),
                round(self.left + x1 * self.width), round(self.top + y1 * self.height))

    def hologram(self) -> str:
        return hologram_state(self.grab(self._box(HOLOGRAM_AREA)))

    def grab(self, box: tuple[int, int, int, int]):
        import mss
        import numpy as np

        if self._grabber is None:
            self._grabber = mss.mss()
        left, top, right, bottom = box
        shot = self._grabber.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
        return np.asarray(shot)[:, :, :3]

    def read(self, area: str) -> list[TextBox]:
        import cv2

        left, top, right, bottom = self.area(area)
        scale = AREAS[area][1]
        image = self.grab((left, top, right, bottom))
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes = []
        for corners, text, confidence in self._ocr().readtext(image):
            xs = [p[0] / scale for p in corners]
            ys = [p[1] / scale for p in corners]
            boxes.append(TextBox(text, left + sum(xs) / 4, top + sum(ys) / 4,
                                 max(xs) - min(xs), max(ys) - min(ys), float(confidence)))
        return boxes

    def find(self, wanted: str, area: str, below: float | None = None) -> TextBox | None:
        return best_match(self.read(area), wanted, below=below)

    def wait_for(self, wanted: str, area: str, timeout: float = 6.0) -> TextBox | None:
        end = time.perf_counter() + timeout
        while True:
            found = self.find(wanted, area)
            if found is not None or time.perf_counter() > end:
                return found
            time.sleep(0.2)
