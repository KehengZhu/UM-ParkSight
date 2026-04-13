#!/usr/bin/env python3
"""
Perspective rectification helper for parking-lot occupancy preprocessing.

Run:
    python perspective_crop.py
or:
    python perspective_crop.py --image reference_images/IMG_3275.jpeg

Controls:
    - Left click: select up to 4 source points on the parking-lot ground plane
    - u: undo last point
    - r: reset all points
    - s: save outputs (only after 4 points are selected)
    - q: quit

Notes:
    - This produces an approximate top-down view, not a true 3D reconstruction.
    - A homography works reasonably well because the parking-lot surface is treated
      as one approximately planar surface.
    - Expected failure cases include: non-planar ground (ramps/curbs), poor point
      placement, severe lens distortion, and selecting points on objects above the
      ground plane (cars, poles, window blinds, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np


WINDOW_ORIGINAL = "Original - Select 4 parking-lot points"
WINDOW_WARPED = "Warped top-down-style preview"


def order_points_tl_tr_br_bl(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Return 4 points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Exactly 4 points are required to order corners.")

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left

    # Fallback if duplicate assignments occur due to unusual geometry.
    if len({(float(p[0]), float(p[1])) for p in ordered}) != 4:
        x_sorted = pts[np.argsort(pts[:, 0])]
        left = x_sorted[:2]
        right = x_sorted[2:]
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]
        ordered = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

    return ordered


def compute_output_size(ordered: np.ndarray) -> Tuple[int, int]:
    """Compute output rectangle size from selected quadrilateral side lengths."""
    tl, tr, br, bl = ordered

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = max(int(round(max(width_top, width_bottom))), 1)
    height = max(int(round(max(height_left, height_right))), 1)
    return width, height


def crop_valid_region_after_warp(
    image_shape: Tuple[int, int],
    ordered_src: np.ndarray,
    homography: np.ndarray,
    warped: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    """Crop warped image to valid mapped region to remove empty black borders."""
    src_h, src_w = image_shape
    src_mask = np.zeros((src_h, src_w), dtype=np.uint8)
    cv2.fillConvexPoly(src_mask, np.round(ordered_src).astype(np.int32), 255)

    warped_mask = cv2.warpPerspective(
        src_mask,
        homography,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_NEAREST,
    )

    nonzero = cv2.findNonZero(warped_mask)
    if nonzero is None:
        return warped, (0, 0, warped.shape[1], warped.shape[0]), False

    x, y, w, h = cv2.boundingRect(nonzero)
    if w <= 0 or h <= 0:
        return warped, (0, 0, warped.shape[1], warped.shape[0]), False

    unchanged = x == 0 and y == 0 and w == warped.shape[1] and h == warped.shape[0]
    if unchanged:
        return warped, (x, y, w, h), False

    cropped = warped[y : y + h, x : x + w].copy()
    return cropped, (x, y, w, h), True


def fit_for_display(image: np.ndarray, max_w: int, max_h: int) -> Tuple[np.ndarray, float]:
    """Resize image for display while preserving aspect ratio."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0

    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale

    return image.copy(), 1.0


def to_display_points(points: Sequence[Tuple[float, float]], scale: float) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    arr = np.round(arr * scale).astype(np.int32)
    return arr


def draw_overlay(
    base_image: np.ndarray,
    points: Sequence[Tuple[float, float]],
    scale: float = 1.0,
    show_help_text: bool = True,
) -> np.ndarray:
    canvas = base_image.copy()
    disp_pts = to_display_points(points, scale)

    for idx, p in enumerate(disp_pts):
        px, py = int(p[0]), int(p[1])
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"P{idx + 1}",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(disp_pts) >= 2:
        poly = disp_pts.reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (255, 255, 0), 2, lineType=cv2.LINE_AA)

    if len(points) == 4:
        ordered = order_points_tl_tr_br_bl(points)
        ordered_disp = to_display_points(ordered, scale).reshape(-1, 1, 2)
        cv2.polylines(canvas, [ordered_disp], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        labels = ["TL", "TR", "BR", "BL"]
        for label, pt in zip(labels, ordered_disp.reshape(-1, 2)):
            cv2.putText(
                canvas,
                label,
                (int(pt[0]) + 8, int(pt[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    if show_help_text:
        help_text = "Left-click: add point | u: undo | r: reset | s: save | q: quit"
        cv2.putText(
            canvas,
            help_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def compute_warp(image: np.ndarray, points: Sequence[Tuple[float, float]]) -> dict:
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to compute perspective warp.")

    ordered = order_points_tl_tr_br_bl(points)
    width, height = compute_output_size(ordered)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, homography, (width, height))

    cropped, crop_rect, was_cropped = crop_valid_region_after_warp(
        image.shape[:2], ordered, homography, warped
    )

    return {
        "ordered_points": ordered,
        "homography": homography,
        "warped": warped,
        "final_warp": cropped,
        "crop_rect": crop_rect,
        "was_cropped": was_cropped,
        "initial_size": (width, height),
        "final_size": (cropped.shape[1], cropped.shape[0]),
    }


@dataclass
class PointSelector:
    image_width: int
    image_height: int
    display_scale: float
    points: List[Tuple[float, float]] = field(default_factory=list)
    version: int = 0

    def add_point_from_display(self, x: int, y: int) -> None:
        if len(self.points) >= 4:
            print("Already selected 4 points. Press 'u' to undo or 'r' to reset.")
            return

        src_x = float(np.clip(x / self.display_scale, 0, self.image_width - 1))
        src_y = float(np.clip(y / self.display_scale, 0, self.image_height - 1))

        self.points.append((src_x, src_y))
        self.version += 1
        print(f"Selected point {len(self.points)}: ({src_x:.1f}, {src_y:.1f})")

    def reset(self) -> None:
        if self.points:
            self.points.clear()
            self.version += 1
            print("Selection reset.")

    def undo(self) -> None:
        if not self.points:
            print("No points to undo.")
            return

        removed = self.points.pop()
        self.version += 1
        print(f"Removed point: ({removed[0]:.1f}, {removed[1]:.1f})")

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point_from_display(x, y)


def print_transform_summary(result: dict) -> None:
    ordered = result["ordered_points"]
    labels = ["TL", "TR", "BR", "BL"]

    print("\nSelected source points (ordered TL, TR, BR, BL):")
    for label, pt in zip(labels, ordered):
        print(f"  {label}: ({pt[0]:.2f}, {pt[1]:.2f})")

    iw, ih = result["initial_size"]
    fw, fh = result["final_size"]
    print(f"Initial warp size: {iw} x {ih}")
    if result["was_cropped"]:
        x, y, w, h = result["crop_rect"]
        print(f"Post-warp crop applied at x={x}, y={y}, w={w}, h={h}")
    else:
        print("Post-warp crop: not needed")
    print(f"Final output size: {fw} x {fh}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive parking-lot perspective rectification")
    parser.add_argument(
        "--image",
        default="reference_images/IMG_3275.jpeg",
        help="Input image path (default: reference_images/IMG_3275.jpeg)",
    )
    parser.add_argument(
        "--output",
        default="parking_topdown.png",
        help="Output warped image path (default: parking_topdown.png)",
    )
    parser.add_argument(
        "--overlay-output",
        default="parking_points_overlay.png",
        help="Output overlay image path (default: parking_points_overlay.png)",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1500,
        help="Maximum display window width in pixels",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=900,
        help="Maximum display window height in pixels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: failed to load image: {args.image}")
        return 1

    display_image, display_scale = fit_for_display(
        image, args.max_display_width, args.max_display_height
    )

    selector = PointSelector(
        image_width=image.shape[1],
        image_height=image.shape[0],
        display_scale=display_scale,
    )

    print(
        "\nInstructions:\n"
        "  1) Click 4 points around the parking-lot ground region\n"
        "  2) Use 'u' to undo, 'r' to reset\n"
        "  3) Press 's' to save once 4 points are selected\n"
        "  4) Press 'q' to quit\n"
    )

    cached_result = None
    cached_version = -1

    try:
        cv2.namedWindow(WINDOW_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_WARPED, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_ORIGINAL, selector.mouse_callback)
    except cv2.error as exc:
        print("Error: OpenCV GUI is not available in this environment.")
        print(f"Details: {exc}")
        return 1

    while True:
        overlay_display = draw_overlay(display_image, selector.points, scale=display_scale)
        cv2.imshow(WINDOW_ORIGINAL, overlay_display)

        if len(selector.points) == 4:
            if selector.version != cached_version:
                try:
                    cached_result = compute_warp(image, selector.points)
                    cached_version = selector.version
                    print_transform_summary(cached_result)
                except ValueError as exc:
                    cached_result = None
                    print(f"Could not compute warp: {exc}")

            if cached_result is not None:
                warped_preview, _ = fit_for_display(
                    cached_result["final_warp"],
                    args.max_display_width,
                    args.max_display_height,
                )
                cv2.imshow(WINDOW_WARPED, warped_preview)
        else:
            hint = np.zeros((220, 620, 3), dtype=np.uint8)
            cv2.putText(
                hint,
                "Select exactly 4 points to preview/save warp",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hint,
                "u: undo   r: reset   s: save   q: quit",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_WARPED, hint)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            selector.reset()
            cached_result = None
            cached_version = -1
        elif key == ord("u"):
            selector.undo()
            cached_result = None
            cached_version = -1
        elif key == ord("s"):
            if len(selector.points) != 4:
                print("Select 4 points before saving.")
                continue

            if cached_result is None:
                try:
                    cached_result = compute_warp(image, selector.points)
                except ValueError as exc:
                    print(f"Could not save results: {exc}")
                    continue

            overlay_full = draw_overlay(image, selector.points, scale=1.0)

            save_warp_ok = cv2.imwrite(args.output, cached_result["final_warp"])
            save_overlay_ok = cv2.imwrite(args.overlay_output, overlay_full)

            if not save_warp_ok or not save_overlay_ok:
                print("Error: failed to write one or more output files.")
                continue

            print_transform_summary(cached_result)
            print(f"Saved warped output: {args.output}")
            print(f"Saved points overlay: {args.overlay_output}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
