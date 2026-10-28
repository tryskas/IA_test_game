"""Step 4, reading the screen: find the text on a screenshot (OCR).

    python real_game/check_ocr.py captures/capture_XXXX_1.png

Prints every piece of text found, with where it is, and saves a copy of the
image with the text framed (same name, ending in _ocr.png). The first run
downloads the French and English reading models (about 100 MB).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Donne le chemin d'une capture : python real_game/check_ocr.py captures/xxx.png")
    path = Path(sys.argv[1])
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Impossible de lire l'image {path}")

    import easyocr

    print("Lecture de l'image (la première fois, téléchargement des modèles)...")
    reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
    found = reader.readtext(image)
    found.sort(key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))  # top to bottom, left to right
    for box, text, confidence in found:
        x, y = (int(v) for v in box[0])
        color = (0, 200, 0) if confidence >= 0.6 else (0, 165, 255)
        cv2.polylines(image, [np.array(box, dtype=np.int32)], True, color, 2)
        print(f"  ({x:4d}, {y:4d})  {confidence:4.0%}  {text}")
    out = path.with_name(path.stem + "_ocr.png")
    cv2.imwrite(str(out), image)
    print(f"{len(found)} textes trouvés. Image annotée : {out}")


if __name__ == "__main__":
    main()
