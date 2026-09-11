from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .core import FacePair, Frame, LandmarkSet
from .pipeline import GeometryPipeline


def load_image(path: str) -> Frame:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), source=path)


def load_landmarks(path: str | None) -> LandmarkSet | None:
    if path is None:
        return None
    obj = json.loads(Path(path).read_text())
    return LandmarkSet(
        np.asarray(obj["xy"], float), tuple(obj.get("names", [])) or None
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="facemath")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare")
    p.add_argument("image_a")
    p.add_argument("image_b")
    p.add_argument("--landmarks-a")
    p.add_argument("--landmarks-b")
    p.add_argument("--out", default="facemath_evidence.json")
    args = ap.parse_args()
    if args.cmd == "compare":
        pair = FacePair(
            load_image(args.image_a),
            load_image(args.image_b),
            load_landmarks(args.landmarks_a),
            load_landmarks(args.landmarks_b),
        )
        bundle = GeometryPipeline().run(pair)
        bundle.ledger.write_json(args.out)
        print(args.out)
