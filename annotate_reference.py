#!/usr/bin/env python3
"""
Interactive parking-spot annotation tool for a single reference image.

Run:
    python annotate_reference.py
or:
    python annotate_reference.py --image reference_images/parking_topdown_halffull.jpeg

Controls:
    - Left click: add hexagon vertex (6 clicks per spot, in clockwise order)
    - n: finalize current hexagon (requires 6 clicks)
    - u: undo last completed spot
    - r: reset all annotations
    - s: save spots.json and spots_overlay.png
    - q: quit

Notes:
    - Each parking spot is annotated as a projected hexagon, which approximates
      a rectangular-solid car footprint in perspective.
    - This is a practical geometric annotation for downstream cropping/tracking,
      not true 3D reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Parking Spot Annotator"
HEXAGON_POINTS = 6


def clamp_point(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    """Clamp a point so it stays inside image bounds."""
    cx = int(np.clip(x, 0, width - 1))
    cy = int(np.clip(y, 0, height - 1))
    return cx, cy


def fit_for_display(image: np.ndarray, max_w: int, max_h: int) -> Tuple[np.ndarray, float]:
    """Resize image for display while preserving aspect ratio."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0

    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    return image.copy(), 1.0


def make_spot_id(index_1based: int) -> str:
    """Create spot ID in sequence: A01, A02, A03, ..."""
    return f"A{index_1based:02d}"


@dataclass
class Spot:
    spot_id: str
    hexagon: List[Tuple[int, int]]


@dataclass
class AnnotationState:
    image_width: int
    image_height: int
    display_scale: float
    spots: List[Spot] = field(default_factory=list)
    current_points: List[Tuple[int, int]] = field(default_factory=list)
    mouse_pos: Optional[Tuple[int, int]] = None

    def display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        src_x = int(round(x / self.display_scale))
        src_y = int(round(y / self.display_scale))
        return clamp_point(src_x, src_y, self.image_width, self.image_height)

    def add_click(self, x: int, y: int) -> None:
        px, py = self.display_to_image(x, y)

        if len(self.current_points) >= HEXAGON_POINTS:
            print("Current hexagon already has 6 points. Press 'n' to finalize or 'u' to undo.")
            return

        self.current_points.append((px, py))
        print(
            f"Added vertex {len(self.current_points)}/{HEXAGON_POINTS}: "
            f"({px}, {py})"
        )

    def update_mouse(self, x: int, y: int) -> None:
        self.mouse_pos = self.display_to_image(x, y)

    def current_polygon(self) -> Optional[np.ndarray]:
        if not self.current_points:
            return None
        return np.asarray(self.current_points, dtype=np.int32).reshape(-1, 1, 2)

    def finalize_current_spot(self) -> bool:
        if len(self.current_points) != HEXAGON_POINTS:
            print(
                f"Need {HEXAGON_POINTS} points before finalizing. "
                f"Current points: {len(self.current_points)}"
            )
            return False

        spot_id = make_spot_id(len(self.spots) + 1)
        hexagon = [(int(x), int(y)) for x, y in self.current_points]
        self.spots.append(Spot(spot_id=spot_id, hexagon=hexagon))
        self.current_points.clear()
        print(f"Saved {spot_id}: {hexagon}")
        return True

    def undo_last_spot(self) -> None:
        if self.current_points:
            removed = self.current_points.pop()
            print(
                f"Removed current vertex ({removed[0]}, {removed[1]}). "
                f"Remaining: {len(self.current_points)}"
            )
            return

        if not self.spots:
            print("No points or completed spots to undo.")
            return

        removed = self.spots.pop()
        print(f"Removed {removed.spot_id}: {removed.hexagon}")

    def reset_all(self) -> None:
        self.spots.clear()
        self.current_points.clear()
        self.mouse_pos = None
        print("Reset all annotations.")


def draw_annotations(base_image: np.ndarray, state: AnnotationState, show_help: bool) -> np.ndarray:
    """Draw saved hexagons and current in-progress polygon on image-space canvas."""
    canvas = base_image.copy()

    for spot in state.spots:
        poly = np.asarray(spot.hexagon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, (0, 220, 0), 2, lineType=cv2.LINE_AA)

        for pt in spot.hexagon:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        centroid = np.mean(np.asarray(spot.hexagon, dtype=np.float32), axis=0)
        cv2.putText(
            canvas,
            spot.spot_id,
            (int(centroid[0]) + 4, max(16, int(centroid[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if state.current_points:
        current_poly = np.asarray(state.current_points, dtype=np.int32)
        for idx, pt in enumerate(current_poly, start=1):
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 180, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(idx),
                (int(pt[0]) + 5, int(pt[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if len(current_poly) >= 2:
            cv2.polylines(
                canvas,
                [current_poly.reshape(-1, 1, 2)],
                False,
                (0, 180, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        if state.mouse_pos is not None and len(current_poly) < HEXAGON_POINTS:
            last = (int(current_poly[-1][0]), int(current_poly[-1][1]))
            cv2.line(canvas, last, state.mouse_pos, (255, 180, 80), 1, lineType=cv2.LINE_AA)

        if len(current_poly) == HEXAGON_POINTS:
            cv2.polylines(
                canvas,
                [current_poly.reshape(-1, 1, 2)],
                True,
                (0, 120, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        label = f"Current points: {len(current_poly)}/{HEXAGON_POINTS}"
        cv2.putText(
            canvas,
            label,
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    if show_help:
        help_line = "Click 6 vertices clockwise | n: finalize | u: undo | r: reset | s: save | q: quit"
        cv2.putText(
            canvas,
            help_line,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    count_line = f"Completed spots: {len(state.spots)}"
    cv2.putText(
        canvas,
        count_line,
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    return canvas


def save_outputs(state: AnnotationState, image: np.ndarray, json_path: str, overlay_path: str) -> bool:
    """Save spot annotations JSON and overlay image to disk."""
    json_data = [
        {
            "spot_id": spot.spot_id,
            "hexagon": [[int(x), int(y)] for x, y in spot.hexagon],
        }
        for spot in state.spots
    ]

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    except OSError as exc:
        print(f"Failed to write JSON file '{json_path}': {exc}")
        return False

    overlay = draw_annotations(image, state, show_help=False)
    if not cv2.imwrite(overlay_path, overlay):
        print(f"Failed to write overlay image '{overlay_path}'.")
        return False

    print(f"Saved {len(state.spots)} spots to {json_path}")
    print(f"Saved overlay image to {overlay_path}")
    return True


def build_mouse_callback(state: AnnotationState):
    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            state.update_mouse(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.add_click(x, y)

    return on_mouse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate parking spots on one reference image")
    parser.add_argument(
        "--image",
        default="reference_images/parking_topdown_halffull.jpeg",
        help="Path to reference image",
    )
    parser.add_argument(
        "--json",
        default="spots.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--overlay",
        default="spots_overlay.png",
        help="Output overlay image path",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1600,
        help="Maximum display width",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=900,
        help="Maximum display height",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return 1

    display_image, display_scale = fit_for_display(
        image, args.max_display_width, args.max_display_height
    )

    state = AnnotationState(
        image_width=image.shape[1],
        image_height=image.shape[0],
        display_scale=display_scale,
    )

    print(
        "\nInstructions:\n"
        "  - Click 6 vertices per spot in clockwise order\n"
        "  - Press 'n' to finalize current hexagon\n"
        "  - Press 'u' to undo last point (or last saved spot)\n"
        "  - Press 'r' to reset everything\n"
        "  - Press 's' to save JSON + overlay\n"
        "  - Press 'q' to quit\n"
    )

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, build_mouse_callback(state))
    except cv2.error as exc:
        print("OpenCV GUI is not available in this environment.")
        print(f"Details: {exc}")
        return 1

    while True:
        overlay_full = draw_annotations(image, state, show_help=True)

        if display_scale < 1.0:
            frame = cv2.resize(
                overlay_full,
                (display_image.shape[1], display_image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame = overlay_full

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            state.finalize_current_spot()
        elif key == ord("u"):
            state.undo_last_spot()
        elif key == ord("r"):
            state.reset_all()
        elif key == ord("s"):
            save_outputs(state, image, args.json, args.overlay)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
