#!/usr/bin/env python3
"""
Crop per-spot segmentations from an annotated parking overlay image.

Run:
    python crop_spots.py
or:
    python crop_spots.py --image spots_overlay.png --spots spots.json --output-dir spot_segments

This script creates one output image per spot entry in spots.json.
If a spot has a hexagon polygon, a masked polygon crop is saved.
If a spot has a bbox, a rectangular crop is saved.
Crops are written as JPG files.
Hexagon crops can be inpainted to fill black masked areas using neighboring pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop parking spot segmentations from spots_overlay image"
    )
    parser.add_argument(
        "--image",
        default="spots_overlay.png",
        help="Input overlay image path",
    )
    parser.add_argument(
        "--spots",
        default="spots.json",
        help="Input spots JSON path",
    )
    parser.add_argument(
        "--output-dir",
        default="spot_segments",
        help="Directory to save per-spot crops",
    )
    parser.add_argument(
        "--no-fill-black",
        action="store_true",
        help="Disable filling black masked regions in hexagon crops",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=float,
        default=3.0,
        help="Inpainting radius used when --fill-black is enabled",
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


def clamp_xy(points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0, img_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img_h - 1)
    return points


def fill_black_regions_with_inpaint(
    segmented: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: float,
) -> np.ndarray:
    """Fill masked-out black regions using neighboring non-black pixels."""
    if segmented.size == 0:
        return segmented

    if mask.shape[:2] != segmented.shape[:2]:
        raise ValueError("Mask shape does not match segmented image.")

    # Inpaint where mask == 0 (outside polygon) to reduce black side artifacts.
    inpaint_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(inpaint_mask) == 0:
        return segmented

    filled = cv2.inpaint(segmented, inpaint_mask, float(max(inpaint_radius, 0.1)), cv2.INPAINT_TELEA)
    return filled


def crop_from_hexagon(
    image: np.ndarray,
    hexagon: List[List[int]],
    fill_black: bool,
    inpaint_radius: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    img_h, img_w = image.shape[:2]
    pts = np.asarray(hexagon, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("hexagon must contain at least 3 points in [x, y] format.")

    pts = clamp_xy(pts, img_w, img_h)
    poly = pts.reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(poly)
    if w <= 0 or h <= 0:
        raise ValueError("hexagon produced an empty crop.")

    roi = image[y : y + h, x : x + w].copy()
    shifted = (pts - np.array([x, y], dtype=np.int32)).reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)
    segmented = cv2.bitwise_and(roi, roi, mask=mask)

    if fill_black:
        segmented = fill_black_regions_with_inpaint(segmented, mask, inpaint_radius)

    return segmented, (x, y, w, h)


def crop_from_bbox(image: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    img_h, img_w = image.shape[:2]
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values: [x, y, w, h].")

    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        raise ValueError("bbox width and height must be > 0.")

    x1 = int(np.clip(x, 0, img_w - 1))
    y1 = int(np.clip(y, 0, img_h - 1))
    x2 = int(np.clip(x + w, 0, img_w))
    y2 = int(np.clip(y + h, 0, img_h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is outside image bounds after clamping.")

    crop = image[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)


def safe_spot_id(spot: Dict, index: int) -> str:
    raw = str(spot.get("spot_id", f"spot_{index:03d}"))
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_"))
    if not cleaned:
        cleaned = f"spot_{index:03d}"
    return cleaned


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

    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    skipped = 0

    for idx, spot in enumerate(spots, start=1):
        if not isinstance(spot, dict):
            print(f"Skipping entry {idx}: not an object")
            skipped += 1
            continue

        spot_id = safe_spot_id(spot, idx)
        out_path = os.path.join(args.output_dir, f"{spot_id}.jpg")

        try:
            if "hexagon" in spot:
                crop, rect = crop_from_hexagon(
                    image,
                    spot["hexagon"],
                    fill_black=not args.no_fill_black,
                    inpaint_radius=args.inpaint_radius,
                )
            elif "bbox" in spot:
                crop, rect = crop_from_bbox(image, spot["bbox"])
            else:
                raise ValueError("missing 'hexagon' or 'bbox'")

            if not cv2.imwrite(out_path, crop):
                raise ValueError("failed to write image")

            x, y, w, h = rect
            print(f"Saved {spot_id}: {out_path} (x={x}, y={y}, w={w}, h={h})")
            saved += 1
        except ValueError as exc:
            print(f"Skipping {spot_id}: {exc}")
            skipped += 1

    print("\nDone.")
    print(f"Spots in JSON: {len(spots)}")
    print(f"Saved crops: {saved}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")

    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
