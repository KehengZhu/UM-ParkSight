#!/usr/bin/env python3
"""
align_with_loftr.py

Align parking-lot test image(s) to a reference image using a pretrained LoFTR matcher,
RANSAC homography estimation, and perspective warping.

Run:
    python align_with_loftr.py

Single-image mode (same interface as align_test_to_reference.py):
    python align_with_loftr.py --input test_images/example.jpg
    python align_with_loftr.py --input test_images/example.jpg --output aligned_images/example.jpg

Dependencies:
    pip install torch kornia opencv-python numpy
"""

import os
import glob
import numpy as np
import cv2
import torch
import kornia


# =========================
# Paths
# =========================
REFERENCE_IMAGE_PATH = "./reference_images/ref_image.jpg"
TEST_IMAGES_DIR = "test_images"
ALIGNED_OUTPUT_DIR = "aligned_images"
DEBUG_MATCHES_DIR = "debug_matches"
DEBUG_OVERLAYS_DIR = "debug_overlays"


# =========================
# Parameters
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MATCH_CONFIDENCE_THRESHOLD = 0.20
MIN_MATCHES_REQUIRED = 25
RANSAC_REPROJ_THRESHOLD = 4.0
MIN_INLIERS_REQUIRED = 18
MAX_DEBUG_MATCHES = 120
BLEND_ALPHA = 0.55
RESIZE_MAX_SIDE = 1536

# Optional cap on homography estimation points (top confidence first).
MAX_HOMOGRAPHY_MATCHES = 2000

# Optional ROI matching on the parking-lot ground plane.
USE_ROI = False
REF_ROI = (0, 0, 0, 0)   # (x, y, w, h)
TEST_ROI = (0, 0, 0, 0)  # (x, y, w, h)


def print_usage() -> None:
    print("Usage:")
    print("  python align_with_loftr.py")
    print("  python align_with_loftr.py --input path/to/input_image.jpg")
    print("  python align_with_loftr.py --input path/to/input_image.jpg --output path/to/output_image.jpg")


def parse_args_from_argv(argv):
    """Minimal CLI parser to keep dependencies simple and compatible with --input/--output."""
    args = {
        "input": "",
        "output": "",
    }

    i = 0
    while i < len(argv):
        token = argv[i]

        if token in ("-h", "--help"):
            args["help"] = True
            return args

        if token == "--input":
            if i + 1 >= len(argv):
                raise ValueError("--input requires a path argument")
            args["input"] = argv[i + 1]
            i += 2
            continue

        if token == "--output":
            if i + 1 >= len(argv):
                raise ValueError("--output requires a path argument")
            args["output"] = argv[i + 1]
            i += 2
            continue

        raise ValueError(f"Unknown argument: {token}")

    return args


def ensure_dirs() -> None:
    os.makedirs(ALIGNED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_MATCHES_DIR, exist_ok=True)
    os.makedirs(DEBUG_OVERLAYS_DIR, exist_ok=True)


def list_test_images(folder):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(paths)


def resolve_reference_path() -> str:
    """Try a few common locations to reduce path friction across projects."""
    candidates = [
        REFERENCE_IMAGE_PATH,
        os.path.join("reference_images", "ref_image.jpg"),
        os.path.join("reference_images", "ref_image.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return REFERENCE_IMAGE_PATH


def clamp_roi(roi, image_w: int, image_h: int):
    x, y, w, h = [int(v) for v in roi]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    if x >= image_w:
        x = image_w - 1
    if y >= image_h:
        y = image_h - 1

    w = min(w, image_w - x)
    h = min(h, image_h - y)
    return x, y, w, h


def prepare_gray_for_loftr(gray_image: np.ndarray, roi, use_roi: bool):
    """
    Prepare grayscale image for LoFTR and return tensor + mapping metadata.

    If ROI is used, LoFTR runs on ROI crop.
    If resizing is used, keypoints are mapped back via inverse scaling.
    """
    h, w = gray_image.shape[:2]

    if use_roi:
        x, y, rw, rh = clamp_roi(roi, w, h)
        crop = gray_image[y : y + rh, x : x + rw]
        offset_x = float(x)
        offset_y = float(y)
    else:
        crop = gray_image
        offset_x = 0.0
        offset_y = 0.0

    crop_h, crop_w = crop.shape[:2]
    scale = 1.0

    if RESIZE_MAX_SIDE is not None and RESIZE_MAX_SIDE > 0:
        max_side = max(crop_h, crop_w)
        if max_side > RESIZE_MAX_SIDE:
            scale = float(RESIZE_MAX_SIDE) / float(max_side)
            new_w = max(1, int(round(crop_w * scale)))
            new_h = max(1, int(round(crop_h * scale)))
            proc = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            proc = crop
    else:
        proc = crop

    # LoFTR expects float tensor in [0,1], shape Bx1xHxW.
    tensor = torch.from_numpy(proc).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    meta = {
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scale": float(scale),
    }
    return tensor, meta


def map_points_to_full_image(points: np.ndarray, meta: dict) -> np.ndarray:
    """Map points from LoFTR processed coordinates back to full image coordinates."""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    out = points.astype(np.float32).copy()
    scale = float(meta["scale"])
    offset_x = float(meta["offset_x"])
    offset_y = float(meta["offset_y"])

    out[:, 0] = out[:, 0] / scale + offset_x
    out[:, 1] = out[:, 1] / scale + offset_y
    return out


def run_loftr_matches(matcher, ref_tensor, test_tensor):
    with torch.inference_mode():
        out = matcher({"image0": ref_tensor, "image1": test_tensor})

    if "keypoints0" not in out or "keypoints1" not in out or "confidence" not in out:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    kpts0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
    kpts1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
    conf = out["confidence"].detach().cpu().numpy().astype(np.float32)
    return kpts0, kpts1, conf


def draw_match_visualization(test_bgr, ref_bgr, test_pts, ref_pts, conf, inlier_mask):
    """Draw line correspondences on a side-by-side canvas for debugging."""
    h0, w0 = test_bgr.shape[:2]
    h1, w1 = ref_bgr.shape[:2]
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = test_bgr
    canvas[:h1, w0:w0 + w1] = ref_bgr

    if len(conf) == 0:
        return canvas

    order = np.argsort(-conf)
    if MAX_DEBUG_MATCHES > 0:
        order = order[: min(MAX_DEBUG_MATCHES, len(order))]

    inlier_bool = None
    if inlier_mask is not None and len(inlier_mask) == len(test_pts):
        inlier_bool = inlier_mask.reshape(-1).astype(bool)

    for idx in order:
        x0, y0 = test_pts[idx]
        x1, y1 = ref_pts[idx]

        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + w0)), int(round(y1)))

        if inlier_bool is not None:
            color = (0, 255, 0) if inlier_bool[idx] else (0, 165, 255)
        else:
            color = (255, 255, 0)

        cv2.circle(canvas, p0, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)

    return canvas


def create_overlay(reference_bgr, warped_bgr):
    return cv2.addWeighted(reference_bgr, BLEND_ALPHA, warped_bgr, 1.0 - BLEND_ALPHA, 0.0)


def align_one_image(matcher, reference_bgr, reference_gray, reference_tensor, reference_meta, test_path, aligned_output_path=None):
    name = os.path.basename(test_path)
    base, ext = os.path.splitext(name)

    test_bgr = cv2.imread(test_path)
    if test_bgr is None:
        print(f"[WARN] {name}: failed to load image, skipping")
        return

    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    test_tensor, test_meta = prepare_gray_for_loftr(test_gray, TEST_ROI, USE_ROI)

    # LoFTR is a detector-free local feature matcher with pretrained weights.
    # Kornia recommends high-level APIs such as LoFTR for image matching.
    # We use pretrained="outdoor" because this is an outdoor parking-lot scene.
    ref_pts_proc, test_pts_proc, conf_all = run_loftr_matches(matcher, reference_tensor, test_tensor)
    total_matches = len(conf_all)

    print(f"\nImage: {name}")
    print(f"  LoFTR raw matches: {total_matches}")

    if total_matches == 0:
        print("  [WARN] no LoFTR matches, skipping")
        return

    keep_mask = conf_all >= float(MATCH_CONFIDENCE_THRESHOLD)
    ref_pts_kept = ref_pts_proc[keep_mask]
    test_pts_kept = test_pts_proc[keep_mask]
    conf_kept = conf_all[keep_mask]

    print(f"  Kept after confidence filter: {len(conf_kept)} (threshold={MATCH_CONFIDENCE_THRESHOLD:.3f})")

    if len(conf_kept) < MIN_MATCHES_REQUIRED:
        print(
            f"  [WARN] not enough kept matches (need {MIN_MATCHES_REQUIRED}, got {len(conf_kept)}), skipping"
        )
        return

    order = np.argsort(-conf_kept)
    if MAX_HOMOGRAPHY_MATCHES > 0:
        order = order[: min(MAX_HOMOGRAPHY_MATCHES, len(order))]

    ref_pts_kept = ref_pts_kept[order]
    test_pts_kept = test_pts_kept[order]
    conf_kept = conf_kept[order]

    # Map LoFTR points back to full-image coordinates if ROI and/or resizing was used.
    dst_pts = map_points_to_full_image(ref_pts_kept, reference_meta)   # reference coords
    src_pts = map_points_to_full_image(test_pts_kept, test_meta)       # test coords

    # Homography is still estimated with RANSAC because the parking lot is approximately planar.
    h_mat, inlier_mask = cv2.findHomography(
        src_pts.reshape(-1, 1, 2),
        dst_pts.reshape(-1, 1, 2),
        cv2.RANSAC,
        float(RANSAC_REPROJ_THRESHOLD),
    )

    if h_mat is None or inlier_mask is None:
        print("  [WARN] homography failed, skipping")
        return

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / float(max(len(src_pts), 1))
    print(f"  Inliers: {inliers}")
    print(f"  Inlier ratio: {inlier_ratio:.3f}")

    if inliers < MIN_INLIERS_REQUIRED:
        print(
            f"  [WARN] too few inliers (need {MIN_INLIERS_REQUIRED}, got {inliers}), skipping"
        )
        return

    if inlier_ratio < 0.15:
        print("  [WARN] low inlier ratio; alignment may be unstable")

    ref_h, ref_w = reference_bgr.shape[:2]
    # The warp aligns the parking surface, not all 3D objects.
    warped = cv2.warpPerspective(test_bgr, h_mat, (ref_w, ref_h))

    if aligned_output_path:
        aligned_path = aligned_output_path
    else:
        aligned_path = os.path.join(ALIGNED_OUTPUT_DIR, base + ext)

    aligned_dir = os.path.dirname(aligned_path)
    if aligned_dir:
        os.makedirs(aligned_dir, exist_ok=True)

    match_debug_path = os.path.join(DEBUG_MATCHES_DIR, f"{base}_loftr_matches.jpg")
    overlay_debug_path = os.path.join(DEBUG_OVERLAYS_DIR, f"{base}_overlay.jpg")
    homography_path = os.path.splitext(aligned_path)[0] + "_H.txt"

    match_vis = draw_match_visualization(test_bgr, reference_bgr, src_pts, dst_pts, conf_kept, inlier_mask)
    overlay_vis = create_overlay(reference_bgr, warped)

    ok_aligned = cv2.imwrite(aligned_path, warped)
    ok_matches = cv2.imwrite(match_debug_path, match_vis)
    ok_overlay = cv2.imwrite(overlay_debug_path, overlay_vis)

    try:
        np.savetxt(homography_path, h_mat, fmt="%.8f")
        ok_h = True
    except Exception as exc:
        ok_h = False
        print(f"  [WARN] failed to save homography text: {exc}")

    if not (ok_aligned and ok_matches and ok_overlay):
        print("  [WARN] failed writing one or more image outputs")
        return

    print("  Alignment: SUCCESS")
    print(f"  Saved warped: {aligned_path}")
    print(f"  Saved matches: {match_debug_path}")
    print(f"  Saved overlay: {overlay_debug_path}")
    if ok_h:
        print(f"  Saved homography: {homography_path}")


def main() -> int:
    try:
        args = parse_args_from_argv(os.sys.argv[1:])
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print_usage()
        return 1

    if args.get("help", False):
        print_usage()
        return 0

    ensure_dirs()

    input_path = args["input"]
    output_path = args["output"]

    if output_path and not input_path:
        print("[ERROR] --output is only supported when --input is provided")
        return 1

    ref_path = resolve_reference_path()
    if not os.path.exists(ref_path):
        print(f"[ERROR] reference image not found: {ref_path}")
        return 1

    if input_path:
        if not os.path.exists(input_path):
            print(f"[ERROR] input image not found: {input_path}")
            return 1
        test_paths = [input_path]
        print("Single-image mode enabled")
    else:
        test_paths = list_test_images(TEST_IMAGES_DIR)
        if len(test_paths) == 0:
            print(f"[ERROR] no test images found in: {TEST_IMAGES_DIR}")
            return 1

    reference_bgr = cv2.imread(ref_path)
    if reference_bgr is None:
        print(f"[ERROR] failed to load reference image: {ref_path}")
        return 1

    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    reference_tensor, reference_meta = prepare_gray_for_loftr(reference_gray, REF_ROI, USE_ROI)

    # LoFTR model creation (pretrained outdoor weights).
    matcher = kornia.feature.LoFTR(pretrained="outdoor").to(DEVICE).eval()

    print("=== LoFTR Homography Alignment Start ===")
    print(f"Device: {DEVICE}")
    print(f"Reference: {ref_path}")
    print(f"Test images found: {len(test_paths)}")
    print(f"USE_ROI: {USE_ROI}")
    if USE_ROI:
        print(f"REF_ROI: {REF_ROI}")
        print(f"TEST_ROI: {TEST_ROI}")

    if input_path and output_path:
        print(f"Aligned output path: {output_path}")
    else:
        print(f"Output dirs: {ALIGNED_OUTPUT_DIR}, {DEBUG_MATCHES_DIR}, {DEBUG_OVERLAYS_DIR}")

    for test_path in test_paths:
        try:
            aligned_override = output_path if (input_path and output_path) else None
            align_one_image(
                matcher,
                reference_bgr,
                reference_gray,
                reference_tensor,
                reference_meta,
                test_path,
                aligned_output_path=aligned_override,
            )
        except Exception as exc:
            name = os.path.basename(test_path)
            print(f"[WARN] {name}: unexpected error, skipping ({exc})")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    os.sys.exit(main())
