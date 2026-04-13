#!/usr/bin/env python3
"""
Interactive parking-spot annotation tool for a single reference image.

Run:
    python annotate_reference.py
or:
    python annotate_reference.py --image reference_images/IMG_3275.jpeg
or:
    python annotate_reference.py --image reference_images/ref_image.jpeg --edit spots.json

Controls:
    - Left click: add hexagon vertex (6 clicks per spot, in clockwise order)
    - Edit mode: left click + drag on a spot polygon to move that spot
    - Right click + drag (or W/A/D/X): move/pan view
    - Mouse wheel or +/-: zoom in/out
    - 0: reset zoom/pan view
    - n: finalize current hexagon (requires 6 clicks)
    - u: undo last point (or last completed spot in normal mode)
    - r: reset all annotations (or restore target in redo mode)
    - s: save spots.json and spots_overlay.png
    - q: quit

Redo mode:
    - Use --redo A01 (or any spot id) to redraw one saved spot while keeping
      all other saved spots unchanged.

Load mode:
    - Use --load to preload existing spots from --json (for example an
        incomplete spots.json) and continue annotating new spots.

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


def load_spots_from_json(json_path: str) -> List["Spot"]:
    """Load existing hexagon spot annotations from JSON."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as exc:
        raise ValueError(f"Could not read JSON file '{json_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{json_path}': {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError("Spots JSON must be a list.")

    loaded: List[Spot] = []
    for idx, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {idx} is not a JSON object.")

        spot_id = str(entry.get("spot_id", "")).strip()
        if not spot_id:
            raise ValueError(f"Entry {idx} missing spot_id.")

        hexagon = entry.get("hexagon")
        if not isinstance(hexagon, list) or len(hexagon) != HEXAGON_POINTS:
            raise ValueError(
                f"Entry {spot_id} must contain exactly {HEXAGON_POINTS} hexagon points."
            )

        points: List[Tuple[int, int]] = []
        for p_idx, point in enumerate(hexagon, start=1):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Entry {spot_id} has invalid point {p_idx}.")
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            points.append((x, y))

        loaded.append(Spot(spot_id=spot_id, hexagon=points))

    return loaded


def clone_spots(spots: List["Spot"]) -> List["Spot"]:
    """Deep-copy spot list so edits can be reset safely."""
    return [
        Spot(spot_id=s.spot_id, hexagon=[(int(x), int(y)) for x, y in s.hexagon])
        for s in spots
    ]


@dataclass
class Spot:
    spot_id: str
    hexagon: List[Tuple[int, int]]


@dataclass
class AnnotationState:
    image_width: int
    image_height: int
    view_width: int
    view_height: int
    spots: List[Spot] = field(default_factory=list)
    loaded_spot_count: int = 0
    current_points: List[Tuple[int, int]] = field(default_factory=list)
    mouse_pos: Optional[Tuple[int, int]] = None
    redo_spot_id: Optional[str] = None
    redo_index: Optional[int] = None
    redo_original_hexagon: Optional[List[Tuple[int, int]]] = None
    edit_mode: bool = False
    edit_original_spots: Optional[List["Spot"]] = None
    selected_spot_index: Optional[int] = None
    dragging_spot: bool = False
    spot_drag_start_image: Optional[Tuple[int, int]] = None
    spot_drag_start_hexagon: Optional[List[Tuple[int, int]]] = None
    zoom: float = 1.0
    min_zoom: float = 1.0
    max_zoom: float = 10.0
    view_x: float = 0.0
    view_y: float = 0.0
    is_panning: bool = False
    pan_last_xy: Optional[Tuple[int, int]] = None

    def view_roi_size(self) -> Tuple[int, int]:
        roi_w = max(1, int(round(self.image_width / self.zoom)))
        roi_h = max(1, int(round(self.image_height / self.zoom)))
        roi_w = min(roi_w, self.image_width)
        roi_h = min(roi_h, self.image_height)
        return roi_w, roi_h

    def clamp_view(self) -> None:
        roi_w, roi_h = self.view_roi_size()
        max_x = max(0.0, float(self.image_width - roi_w))
        max_y = max(0.0, float(self.image_height - roi_h))
        self.view_x = float(np.clip(self.view_x, 0.0, max_x))
        self.view_y = float(np.clip(self.view_y, 0.0, max_y))

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.clamp_view()

    def pan_by_screen_delta(self, dx: int, dy: int) -> None:
        roi_w, roi_h = self.view_roi_size()
        if self.view_width <= 0 or self.view_height <= 0:
            return

        # Dragging right moves the viewport left (hand-pan behavior).
        self.view_x -= dx * (roi_w / float(self.view_width))
        self.view_y -= dy * (roi_h / float(self.view_height))
        self.clamp_view()

    def pan_by_fraction(self, fx: float, fy: float) -> None:
        roi_w, roi_h = self.view_roi_size()
        self.view_x += fx * roi_w
        self.view_y += fy * roi_h
        self.clamp_view()

    def zoom_at(self, sx: int, sy: int, zoom_factor: float) -> None:
        if zoom_factor <= 0:
            return

        sx = int(np.clip(sx, 0, max(0, self.view_width - 1)))
        sy = int(np.clip(sy, 0, max(0, self.view_height - 1)))
        anchor_x, anchor_y = self.display_to_image(sx, sy)

        new_zoom = float(np.clip(self.zoom * zoom_factor, self.min_zoom, self.max_zoom))
        if abs(new_zoom - self.zoom) < 1e-6:
            return

        self.zoom = new_zoom
        roi_w, roi_h = self.view_roi_size()
        scale_x = roi_w / float(max(1, self.view_width))
        scale_y = roi_h / float(max(1, self.view_height))

        # Keep the same image point under the cursor after zoom.
        self.view_x = float(anchor_x) - sx * scale_x
        self.view_y = float(anchor_y) - sy * scale_y
        self.clamp_view()

    def display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        self.clamp_view()
        roi_w, roi_h = self.view_roi_size()

        scale_x = roi_w / float(max(1, self.view_width))
        scale_y = roi_h / float(max(1, self.view_height))

        src_x = int(round(self.view_x + x * scale_x))
        src_y = int(round(self.view_y + y * scale_y))
        return clamp_point(src_x, src_y, self.image_width, self.image_height)

    def add_click(self, x: int, y: int) -> None:
        if self.edit_mode:
            print("Edit mode: drag a spot to move it. Point-adding is disabled.")
            return

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
        if self.edit_mode:
            print("Edit mode: finalize is disabled. Press 's' to save edited spots.")
            return False

        if len(self.current_points) != HEXAGON_POINTS:
            print(
                f"Need {HEXAGON_POINTS} points before finalizing. "
                f"Current points: {len(self.current_points)}"
            )
            return False

        hexagon = [(int(x), int(y)) for x, y in self.current_points]

        if self.redo_spot_id is not None and self.redo_index is not None:
            if self.redo_index < 0 or self.redo_index >= len(self.spots):
                print("Redo target is invalid. Please restart with a valid --redo id.")
                return False

            self.spots[self.redo_index] = Spot(spot_id=self.redo_spot_id, hexagon=hexagon)
            self.current_points.clear()
            print(f"Updated {self.redo_spot_id}: {hexagon}")
            return True

        spot_id = make_spot_id(len(self.spots) + 1)
        self.spots.append(Spot(spot_id=spot_id, hexagon=hexagon))
        self.current_points.clear()
        print(f"Saved {spot_id}: {hexagon}")
        return True

    def undo_last_spot(self) -> None:
        if self.edit_mode:
            print("Edit mode: use drag to move spots. Press 'r' to reset edits.")
            return

        if self.current_points:
            removed = self.current_points.pop()
            print(
                f"Removed current vertex ({removed[0]}, {removed[1]}). "
                f"Remaining: {len(self.current_points)}"
            )
            return

        if self.redo_spot_id is not None:
            print("Redo mode: existing saved spots are locked. Add points for target and press 'n'.")
            return

        if not self.spots:
            print("No points or completed spots to undo.")
            return

        if self.loaded_spot_count > 0 and len(self.spots) <= self.loaded_spot_count:
            print("Load mode: cannot undo preloaded spots. Only newly added spots can be undone.")
            return

        removed = self.spots.pop()
        print(f"Removed {removed.spot_id}: {removed.hexagon}")

    def reset_all(self) -> None:
        if self.edit_mode and self.edit_original_spots is not None:
            self.spots = clone_spots(self.edit_original_spots)
            self.current_points.clear()
            self.mouse_pos = None
            self.selected_spot_index = None
            self.dragging_spot = False
            self.spot_drag_start_image = None
            self.spot_drag_start_hexagon = None
            print("Edit mode reset: restored spots to their original loaded locations.")
            return

        if self.redo_spot_id is not None and self.redo_index is not None:
            if self.redo_original_hexagon is not None and 0 <= self.redo_index < len(self.spots):
                self.spots[self.redo_index] = Spot(
                    spot_id=self.redo_spot_id,
                    hexagon=[(int(x), int(y)) for x, y in self.redo_original_hexagon],
                )
            self.current_points.clear()
            self.mouse_pos = None
            print("Redo mode reset: cleared current points and restored original target spot.")
            return

        if self.loaded_spot_count > 0:
            self.spots = self.spots[: self.loaded_spot_count]
            self.current_points.clear()
            self.mouse_pos = None
            print("Load mode reset: cleared unsaved points and removed newly added spots.")
            return

        self.spots.clear()
        self.current_points.clear()
        self.mouse_pos = None
        print("Reset all annotations.")

    def _find_spot_index_at(self, image_x: int, image_y: int) -> Optional[int]:
        """Return index of spot near click, preferring polygons that contain the point."""
        query = (float(image_x), float(image_y))
        best_idx: Optional[int] = None
        best_score = -1e9

        # Reverse iteration gives recently drawn spots a slight priority if overlap happens.
        for idx in range(len(self.spots) - 1, -1, -1):
            pts = np.asarray(self.spots[idx].hexagon, dtype=np.float32)
            if pts.shape != (HEXAGON_POINTS, 2):
                continue
            dist = cv2.pointPolygonTest(pts.reshape(-1, 1, 2), query, True)
            # Accept inside points and near-border clicks.
            if dist >= -10.0 and dist > best_score:
                best_score = dist
                best_idx = idx

        return best_idx

    def begin_spot_drag(self, x: int, y: int) -> None:
        if not self.edit_mode:
            return

        px, py = self.display_to_image(x, y)
        idx = self._find_spot_index_at(px, py)
        if idx is None:
            self.selected_spot_index = None
            self.dragging_spot = False
            self.spot_drag_start_image = None
            self.spot_drag_start_hexagon = None
            return

        self.selected_spot_index = idx
        self.dragging_spot = True
        self.spot_drag_start_image = (px, py)
        self.spot_drag_start_hexagon = [
            (int(sx), int(sy)) for sx, sy in self.spots[idx].hexagon
        ]

    def update_spot_drag(self, x: int, y: int) -> None:
        if not self.edit_mode or not self.dragging_spot:
            return
        if self.selected_spot_index is None:
            return
        if self.spot_drag_start_image is None or self.spot_drag_start_hexagon is None:
            return

        px, py = self.display_to_image(x, y)
        dx = int(px - self.spot_drag_start_image[0])
        dy = int(py - self.spot_drag_start_image[1])

        moved: List[Tuple[int, int]] = []
        for ox, oy in self.spot_drag_start_hexagon:
            mx, my = clamp_point(ox + dx, oy + dy, self.image_width, self.image_height)
            moved.append((mx, my))

        self.spots[self.selected_spot_index].hexagon = moved

    def end_spot_drag(self) -> None:
        if not self.edit_mode:
            return

        self.dragging_spot = False
        self.spot_drag_start_image = None
        self.spot_drag_start_hexagon = None


def render_view(base_image: np.ndarray, state: AnnotationState) -> np.ndarray:
    """Render current zoom/pan viewport from the full-resolution overlay image."""
    state.clamp_view()
    roi_w, roi_h = state.view_roi_size()

    x0 = int(round(state.view_x))
    y0 = int(round(state.view_y))
    x0 = int(np.clip(x0, 0, max(0, state.image_width - roi_w)))
    y0 = int(np.clip(y0, 0, max(0, state.image_height - roi_h)))

    roi = base_image[y0 : y0 + roi_h, x0 : x0 + roi_w]
    interpolation = cv2.INTER_LINEAR if state.zoom > 1.0 else cv2.INTER_AREA
    return cv2.resize(roi, (state.view_width, state.view_height), interpolation=interpolation)


def draw_hud(frame: np.ndarray, state: AnnotationState) -> np.ndarray:
    """Draw on-screen controls and status in display coordinates."""
    hud = frame.copy()

    if state.edit_mode:
        selected = "none"
        if state.selected_spot_index is not None and 0 <= state.selected_spot_index < len(state.spots):
            selected = state.spots[state.selected_spot_index].spot_id
        line1 = "EDIT mode | Left-drag: move selected spot | r: reset edits"
        line2 = "Right-drag/W/A/D/X: pan | Wheel or +/-: zoom | 0: reset view | s: save | q: quit"
        line3 = f"Loaded spots: {len(state.spots)} | Selected: {selected} | Zoom: {state.zoom:.2f}x"
    elif state.redo_spot_id is not None:
        line1 = (
            f"REDO {state.redo_spot_id} | Left click: add point | n: finalize target | "
            "u: undo point | r: restore target"
        )
        line2 = "Right-drag/W/A/D/X: pan | Wheel or +/-: zoom | 0: reset view | s: save | q: quit"
        line3 = (
            f"Completed spots: {len(state.spots)} | "
            f"Current points: {len(state.current_points)}/{HEXAGON_POINTS} | Zoom: {state.zoom:.2f}x"
        )
    else:
        line1 = "Left click: add point | n: finalize | u: undo | r: reset"
        line2 = "Right-drag/W/A/D/X: pan | Wheel or +/-: zoom | 0: reset view | s: save | q: quit"
        line3 = (
            f"Completed spots: {len(state.spots)} | "
            f"Current points: {len(state.current_points)}/{HEXAGON_POINTS} | Zoom: {state.zoom:.2f}x"
        )

    cv2.putText(hud, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, line3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 2, cv2.LINE_AA)
    return hud


def draw_annotations(base_image: np.ndarray, state: AnnotationState, show_help: bool) -> np.ndarray:
    """Draw saved hexagons and current in-progress polygon on image-space canvas."""
    canvas = base_image.copy()

    for idx, spot in enumerate(state.spots):
        poly = np.asarray(spot.hexagon, dtype=np.int32).reshape(-1, 1, 2)
        is_redo_target = state.redo_spot_id is not None and spot.spot_id == state.redo_spot_id
        is_selected = state.edit_mode and state.selected_spot_index == idx
        if is_selected:
            poly_color = (255, 0, 255)
            point_color = (255, 120, 255)
        else:
            poly_color = (255, 170, 0) if is_redo_target else (0, 220, 0)
            point_color = (255, 170, 0) if is_redo_target else (0, 255, 0)
        cv2.polylines(canvas, [poly], True, poly_color, 2, lineType=cv2.LINE_AA)

        for pt in spot.hexagon:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, point_color, -1, lineType=cv2.LINE_AA)

        centroid = np.mean(np.asarray(spot.hexagon, dtype=np.float32), axis=0)
        if is_selected:
            label_text = f"{spot.spot_id} (selected)"
        else:
            label_text = f"{spot.spot_id} (redo)" if is_redo_target else spot.spot_id
        cv2.putText(
            canvas,
            label_text,
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

    # Help text and counters are drawn in display coordinates after zoom/pan render.
    _ = show_help

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


def decode_mouse_wheel_delta(flags: int) -> int:
    """Decode mouse-wheel delta from OpenCV event flags without cv2 helper APIs."""
    # OpenCV packs wheel delta in high 16 bits as signed short.
    raw = (int(flags) >> 16) & 0xFFFF
    return int(np.int16(raw))


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
        help="Output JSON path (in --edit mode, defaults to the --edit file if omitted)",
    )
    parser.add_argument(
        "--edit",
        default=None,
        help="Edit mode: load existing spots JSON and drag-move individual spots",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing spots from --json and continue annotation from next spot id",
    )
    parser.add_argument(
        "--redo",
        default=None,
        help="Redo one existing spot id from JSON while keeping others unchanged (example: --redo A01)",
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

    display_image, _ = fit_for_display(
        image, args.max_display_width, args.max_display_height
    )

    state = AnnotationState(
        image_width=image.shape[1],
        image_height=image.shape[0],
        view_width=display_image.shape[1],
        view_height=display_image.shape[0],
    )

    if args.edit is not None:
        if args.redo is not None:
            print("Error: --edit and --redo cannot be used together.")
            return 1
        if args.load:
            print("Error: --edit and --load cannot be used together.")
            return 1

        edit_json_path = args.edit
        if not os.path.exists(edit_json_path):
            print(f"Error: --edit requested, but JSON file not found: {edit_json_path}")
            return 1

        try:
            loaded_spots = load_spots_from_json(edit_json_path)
        except ValueError as exc:
            print(f"Error loading edit spots: {exc}")
            return 1

        state.spots = loaded_spots
        state.loaded_spot_count = len(loaded_spots)
        state.edit_mode = True
        state.edit_original_spots = clone_spots(loaded_spots)

        # If user didn't override --json, save back to the edited file.
        if args.json == "spots.json":
            args.json = edit_json_path

        print(
            f"Loaded {state.loaded_spot_count} spots from {edit_json_path} for edit mode."
        )
    elif args.redo is not None:
        redo_id = args.redo.strip()
        if not redo_id:
            print("Error: --redo requires a non-empty spot id, e.g. --redo A01")
            return 1

        if not os.path.exists(args.json):
            print(f"Error: --redo requested, but JSON file not found: {args.json}")
            return 1

        try:
            loaded_spots = load_spots_from_json(args.json)
        except ValueError as exc:
            print(f"Error loading existing spots: {exc}")
            return 1

        target_index = None
        for idx, spot in enumerate(loaded_spots):
            if spot.spot_id.lower() == redo_id.lower():
                target_index = idx
                break

        if target_index is None:
            available = ", ".join(spot.spot_id for spot in loaded_spots)
            print(f"Error: spot '{redo_id}' not found in {args.json}")
            print(f"Available spots: {available}")
            return 1

        state.spots = loaded_spots
        state.redo_spot_id = loaded_spots[target_index].spot_id
        state.redo_index = target_index
        state.redo_original_hexagon = [
            (int(x), int(y)) for x, y in loaded_spots[target_index].hexagon
        ]
    elif args.load:
        if not os.path.exists(args.json):
            print(f"Error: --load requested, but JSON file not found: {args.json}")
            return 1

        try:
            loaded_spots = load_spots_from_json(args.json)
        except ValueError as exc:
            print(f"Error loading existing spots: {exc}")
            return 1

        state.spots = loaded_spots
        state.loaded_spot_count = len(loaded_spots)
        print(
            f"Loaded {state.loaded_spot_count} spots from {args.json}. "
            f"Continue annotating from {make_spot_id(len(state.spots) + 1)}."
        )

    if state.edit_mode:
        print(
            "\nInstructions (EDIT mode):\n"
            "  - Left click and drag a spot polygon to move only that spot\n"
            "  - Selected spot is highlighted\n"
            "  - Press 'r' to restore all spots to initially loaded positions\n"
            "  - Right-drag or W/A/D/X to pan, mouse wheel or +/- to zoom\n"
            "  - Press '0' to reset zoom/pan view\n"
            "  - Press 's' to save JSON + overlay\n"
            "  - Press 'q' to quit\n"
        )
    elif state.redo_spot_id is not None:
        print(
            "\nInstructions (REDO mode):\n"
            f"  - Redo target: {state.redo_spot_id}\n"
            "  - Click 6 vertices for the new hexagon\n"
            "  - Press 'n' to replace the target spot only\n"
            "  - Press 'u' to undo current points\n"
            "  - Press 'r' to restore original target and clear current points\n"
            "  - Right-drag or W/A/D/X to pan, mouse wheel or +/- to zoom\n"
            "  - Press '0' to reset zoom/pan view\n"
            "  - Press 's' to save (other spots remain unchanged)\n"
            "  - Press 'q' to quit\n"
        )
    else:
        print(
            "\nInstructions:\n"
            "  - Click 6 vertices per spot in clockwise order\n"
            "  - Press 'n' to finalize current hexagon\n"
            "  - Press 'u' to undo last point (or last new saved spot)\n"
            "  - Press 'r' to reset current work (keeps preloaded spots in --load mode)\n"
            "  - Right-drag or W/A/D/X to pan, mouse wheel or +/- to zoom\n"
            "  - Press '0' to reset zoom/pan view\n"
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

    def on_mouse(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        state.update_mouse(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if state.edit_mode:
                state.begin_spot_drag(x, y)
            else:
                state.add_click(x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if state.edit_mode:
                state.end_spot_drag()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            state.is_panning = True
            state.pan_last_xy = (x, y)
            return

        if event == cv2.EVENT_RBUTTONUP:
            state.is_panning = False
            state.pan_last_xy = None
            return

        if event == cv2.EVENT_MOUSEMOVE and state.is_panning and state.pan_last_xy is not None:
            last_x, last_y = state.pan_last_xy
            state.pan_by_screen_delta(x - last_x, y - last_y)
            state.pan_last_xy = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and state.edit_mode and state.dragging_spot:
            state.update_spot_drag(x, y)
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = decode_mouse_wheel_delta(flags)
            factor = 1.2 if delta > 0 else (1.0 / 1.2)
            state.zoom_at(x, y, factor)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        overlay_full = draw_annotations(image, state, show_help=False)
        frame = render_view(overlay_full, state)
        frame = draw_hud(frame, state)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            cx = state.view_width // 2
            cy = state.view_height // 2
            state.zoom_at(cx, cy, 1.2)
        elif key == ord("-") or key == ord("_"):
            cx = state.view_width // 2
            cy = state.view_height // 2
            state.zoom_at(cx, cy, 1.0 / 1.2)
        elif key == ord("0"):
            state.reset_view()
        elif key == ord("a"):
            state.pan_by_fraction(-0.08, 0.0)
        elif key == ord("d"):
            state.pan_by_fraction(0.08, 0.0)
        elif key == ord("w"):
            state.pan_by_fraction(0.0, -0.08)
        elif key == ord("S"):
            state.pan_by_fraction(0.0, 0.08)
        elif key == ord("s"):
            save_outputs(state, image, args.json, args.overlay)
        elif key == ord("x"):
            # Optional extra downward pan key to avoid conflict with save 's'.
            state.pan_by_fraction(0.0, 0.08)
        elif key == ord("n"):
            state.finalize_current_spot()
        elif key == ord("u"):
            state.undo_last_spot()
        elif key == ord("r"):
            state.reset_all()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
