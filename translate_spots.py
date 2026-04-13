#!/usr/bin/env python3
"""
Interactively translate parking spot geometry over an aligned/test image.

Run:
    python translate_spots.py --image aligned_images/IMG_3285.jpeg --spots spots.json

Controls:
    - Left click + drag: move all spots together
    - Arrow keys or h/j/k/l: nudge by 1 px
    - H/J/K/L (uppercase): nudge by 10 px
    - r: reset translation to the initial offset
    - s: save translated spots JSON (and optional transform JSON), then quit
    - q or Esc: quit without saving

Notes:
    - This applies a global translation only (dx, dy).
    - Input spots schema is preserved; only supported geometry fields are shifted.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Spots Translation Tool"


@dataclass
class TranslationState:
    dx: int
    dy: int
    initial_dx: int
    initial_dy: int
    dragging: bool = False
    drag_start_display: Tuple[int, int] = (0, 0)
    drag_start_offset: Tuple[int, int] = (0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively translate spots geometry over an aligned image"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to aligned/test image to visualize",
    )
    parser.add_argument(
        "--spots",
        default="spots.json",
        help="Input spots JSON path",
    )
    parser.add_argument(
        "--output",
        default="spots_translated.json",
        help="Output path for translated spots JSON",
    )
    parser.add_argument(
        "--transform-json",
        default="",
        help="Optional path to save chosen translation as {'dx': ..., 'dy': ...}",
    )
    parser.add_argument(
        "--dx",
        type=int,
        default=0,
        help="Initial x translation in pixels",
    )
    parser.add_argument(
        "--dy",
        type=int,
        default=0,
        help="Initial y translation in pixels",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1600,
        help="Maximum display width for the interactive window",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=1000,
        help="Maximum display height for the interactive window",
    )
    return parser.parse_args()


def load_spots(spots_path: str) -> List[Dict]:
    try:
        with open(spots_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"Could not read spots file '{spots_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{spots_path}': {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Spots JSON must be a list of spot objects.")

    return data


def fit_scale(image: np.ndarray, max_w: int, max_h: int) -> float:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return 1.0
    return min(max_w / float(w), max_h / float(h), 1.0)


def to_display_xy(x: float, y: float, scale: float) -> Tuple[int, int]:
    return int(round(x * scale)), int(round(y * scale))


def shift_hexagon(hexagon: Sequence[Sequence[float]], dx: int, dy: int) -> List[List[int]]:
    out: List[List[int]] = []
    for pt in hexagon:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            continue
        x = int(round(float(pt[0]))) + dx
        y = int(round(float(pt[1]))) + dy
        out.append([x, y])
    return out


def shift_bbox(bbox: Sequence[float], dx: int, dy: int) -> List[int]:
    if len(bbox) != 4:
        return [int(round(float(v))) for v in bbox]
    x, y, w, h = bbox
    return [int(round(float(x))) + dx, int(round(float(y))) + dy, int(round(float(w))), int(round(float(h)))]


def translate_spots(spots: List[Dict], dx: int, dy: int) -> List[Dict]:
    translated: List[Dict] = []
    for entry in spots:
        if not isinstance(entry, dict):
            translated.append(entry)
            continue

        out = copy.deepcopy(entry)
        if "hexagon" in out and isinstance(out["hexagon"], list):
            out["hexagon"] = shift_hexagon(out["hexagon"], dx, dy)
        if "bbox" in out and isinstance(out["bbox"], list):
            out["bbox"] = shift_bbox(out["bbox"], dx, dy)
        translated.append(out)
    return translated


def draw_overlay(
    image_bgr: np.ndarray,
    spots: List[Dict],
    dx: int,
    dy: int,
    scale: float,
) -> np.ndarray:
    if scale < 1.0:
        disp = cv2.resize(
            image_bgr,
            (max(1, int(round(image_bgr.shape[1] * scale))), max(1, int(round(image_bgr.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        disp = image_bgr.copy()

    overlay = disp.copy()

    for idx, entry in enumerate(spots, start=1):
        if not isinstance(entry, dict):
            continue

        spot_id = str(entry.get("spot_id", f"spot_{idx:03d}"))
        color = (60, 210, 255)

        if "hexagon" in entry and isinstance(entry["hexagon"], list):
            shifted = shift_hexagon(entry["hexagon"], dx, dy)
            if len(shifted) >= 3:
                pts = np.asarray(shifted, dtype=np.int32).reshape(-1, 2)
                pts_disp = np.asarray([to_display_xy(p[0], p[1], scale) for p in pts], dtype=np.int32).reshape(-1, 1, 2)

                cv2.polylines(overlay, [pts_disp], True, color, 2, lineType=cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts_disp], (30, 100, 150))

                centroid = pts.mean(axis=0)
                cx, cy = to_display_xy(float(centroid[0]), float(centroid[1]), scale)
                cv2.putText(
                    overlay,
                    spot_id,
                    (cx - 18, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        elif "bbox" in entry and isinstance(entry["bbox"], list):
            sb = shift_bbox(entry["bbox"], dx, dy)
            if len(sb) == 4:
                x, y, w, h = sb
                x1, y1 = to_display_xy(x, y, scale)
                x2, y2 = to_display_xy(x + w, y + h, scale)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay,
                    spot_id,
                    (x1 + 3, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    blended = cv2.addWeighted(overlay, 0.28, disp, 0.72, 0.0)

    lines = [
        f"dx={dx}, dy={dy}",
        "Drag left mouse to move spots",
        "Arrows or h/j/k/l: 1 px | H/J/K/L: 10 px",
        "r: reset | s: save and quit | q or Esc: quit",
    ]
    y0 = 26
    for line in lines:
        cv2.putText(
            blended,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y0 += 28

    return blended


def save_outputs(spots: List[Dict], output_path: str, transform_path: str, dx: int, dy: int) -> None:
    translated = translate_spots(spots, dx, dy)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, indent=2)

    if transform_path:
        tf_dir = os.path.dirname(transform_path)
        if tf_dir:
            os.makedirs(tf_dir, exist_ok=True)
        with open(transform_path, "w", encoding="utf-8") as f:
            json.dump({"dx": int(dx), "dy": int(dy)}, f, indent=2)


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        return 1
    if not os.path.exists(args.spots):
        print(f"Error: spots file not found: {args.spots}")
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: failed to load image: {args.image}")
        return 1

    try:
        spots = load_spots(args.spots)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    scale = fit_scale(
        image,
        max(200, int(args.max_display_width)),
        max(200, int(args.max_display_height)),
    )

    state = TranslationState(
        dx=int(args.dx),
        dy=int(args.dy),
        initial_dx=int(args.dx),
        initial_dy=int(args.dy),
    )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.drag_start_display = (int(x), int(y))
            state.drag_start_offset = (state.dx, state.dy)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            ddx = int(round((x - state.drag_start_display[0]) / float(scale)))
            ddy = int(round((y - state.drag_start_display[1]) / float(scale)))
            state.dx = state.drag_start_offset[0] + ddx
            state.dy = state.drag_start_offset[1] + ddy
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Use a practical initial window size for large parking-lot frames.
    h, w = image.shape[:2]
    cv2.resizeWindow(
        WINDOW_NAME,
        max(720, int(round(w * scale))),
        max(420, int(round(h * scale))),
    )

    print("Interactive spots translation")
    print("- Drag left mouse to move all spots")
    print("- Arrows or h/j/k/l: move by 1 px")
    print("- H/J/K/L: move by 10 px")
    print("- r: reset | s: save+quit | q/Esc: quit without save")

    LEFT_KEYS = {81, 2424832}
    UP_KEYS = {82, 2490368}
    RIGHT_KEYS = {83, 2555904}
    DOWN_KEYS = {84, 2621440}

    while True:
        frame = draw_overlay(image, spots, state.dx, state.dy, scale)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKeyEx(20)
        if key < 0:
            continue

        if key in (27, ord("q"), ord("Q")):
            print("Quit without saving.")
            break

        if key in (ord("s"), ord("S")):
            save_outputs(spots, args.output, args.transform_json, state.dx, state.dy)
            print(f"Saved translated spots: {args.output}")
            if args.transform_json:
                print(f"Saved translation metadata: {args.transform_json}")
            print(f"Final translation: dx={state.dx}, dy={state.dy}")
            cv2.destroyAllWindows()
            return 0

        if key in (ord("r"), ord("R")):
            state.dx = state.initial_dx
            state.dy = state.initial_dy
            continue

        if key in LEFT_KEYS or key == ord("h"):
            state.dx -= 1
        elif key in RIGHT_KEYS or key == ord("l"):
            state.dx += 1
        elif key in UP_KEYS or key == ord("k"):
            state.dy -= 1
        elif key in DOWN_KEYS or key == ord("j"):
            state.dy += 1
        elif key == ord("H"):
            state.dx -= 10
        elif key == ord("L"):
            state.dx += 10
        elif key == ord("K"):
            state.dy -= 10
        elif key == ord("J"):
            state.dy += 10

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
