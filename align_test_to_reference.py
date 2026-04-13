#!/usr/bin/env python3
"""
align_test_to_reference.py

Align each parking-lot test image to a reference image using a homography.

Run:
    python align_test_to_reference.py

Input:
    ref_image.jpg
    test_images/

Outputs:
    aligned_images/<name>
    debug_matches/<name>_matches.jpg
    debug_overlays/<name>_overlay.jpg

Notes:
- Homography is appropriate because the parking-lot surface is approximately planar.
- Affine is usually too weak for nearby viewpoint changes because it cannot model
  full perspective distortion.
- Cars, poles, and trees are 3D objects and may still look imperfect after warping,
  but the parking-lot ground plane should align better.
"""

import os
import glob
import argparse
import cv2
import numpy as np


# =========================
# Paths
# =========================
REFERENCE_IMAGE_PATH = "./reference_images/ref_image.jpeg"
TEST_IMAGES_DIR = "test_images"
ALIGNED_OUTPUT_DIR = "aligned_images"
DEBUG_MATCHES_DIR = "debug_matches"
DEBUG_OVERLAYS_DIR = "debug_overlays"


# =========================
# Tunable parameters
# =========================
MAX_FEATURES = 8000
KEEP_BEST_MATCHES = 350
RANSAC_REPROJ_THRESHOLD = 4.0
MIN_MATCHES_REQUIRED = 25
MIN_INLIERS_REQUIRED = 18
BLEND_ALPHA = 0.55

# ORB matching quality controls.
ORB_RATIO_TEST = 0.86

# Nice-to-have optional fallback if ORB is weak.
ENABLE_SIFT_FALLBACK = True
SIFT_KEEP_BEST_MATCHES = 900
SIFT_RATIO_TEST = 0.75

# Restrict features to parking-lot ground-plane area to reduce 3D outliers.
USE_GROUND_PLANE_MASK = True
GROUND_MASK_TOP_RATIO = 0.34
GROUND_MASK_TOP_LEFT_RATIO = 0.04
GROUND_MASK_TOP_RIGHT_RATIO = 0.96


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align parking-lot image(s) to a reference image using homography."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional single image path. If set, only this image is processed.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional aligned output image path. Only valid with --input. "
            "If omitted in single-image mode, output is saved in aligned_images/."
        ),
    )
    return parser.parse_args()


def ensure_dirs():
    os.makedirs(ALIGNED_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_MATCHES_DIR, exist_ok=True)
    os.makedirs(DEBUG_OVERLAYS_DIR, exist_ok=True)


def list_test_images(folder):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(paths)


def build_ground_plane_mask(gray_image):
    h, w = gray_image.shape[:2]
    y_top = int(h * GROUND_MASK_TOP_RATIO)
    pts = np.array(
        [
            [int(w * GROUND_MASK_TOP_LEFT_RATIO), y_top],
            [int(w * GROUND_MASK_TOP_RIGHT_RATIO), y_top],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def _ratio_map(knn_matches, ratio):
    out = {}
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            out[m.queryIdx] = (m.trainIdx, float(m.distance))
    return out


def mutual_ratio_match(desc_a, desc_b, norm_type, ratio, keep_best):
    if desc_a is None or desc_b is None:
        return []

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    fwd = matcher.knnMatch(desc_a, desc_b, k=2)
    rev = matcher.knnMatch(desc_b, desc_a, k=2)

    fwd_map = _ratio_map(fwd, ratio)
    rev_map = _ratio_map(rev, ratio)

    matches = []
    for q_idx, (t_idx, dist) in fwd_map.items():
        rev_item = rev_map.get(t_idx)
        if rev_item is None:
            continue
        rev_q_idx, _ = rev_item
        if rev_q_idx != q_idx:
            continue
        matches.append(cv2.DMatch(_queryIdx=q_idx, _trainIdx=t_idx, _distance=dist))

    matches = sorted(matches, key=lambda m: m.distance)
    return matches[: min(keep_best, len(matches))]


def detect_and_match_orb(ref_gray, test_gray, ref_mask=None, test_mask=None):
    orb = cv2.ORB_create(
        nfeatures=MAX_FEATURES,
        fastThreshold=10,
        edgeThreshold=15,
        patchSize=31,
    )

    ref_kp, ref_desc = orb.detectAndCompute(ref_gray, ref_mask)
    test_kp, test_desc = orb.detectAndCompute(test_gray, test_mask)

    if ref_desc is None or test_desc is None:
        return ref_kp, test_kp, [], "orb"

    good_matches = mutual_ratio_match(
        test_desc,
        ref_desc,
        cv2.NORM_HAMMING,
        ORB_RATIO_TEST,
        KEEP_BEST_MATCHES,
    )

    return ref_kp, test_kp, good_matches, "orb"


def detect_and_match_sift(ref_gray, test_gray, ref_mask=None, test_mask=None):
    if not hasattr(cv2, "SIFT_create"):
        return [], [], [], "sift-unavailable"

    sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    ref_kp, ref_desc = sift.detectAndCompute(ref_gray, ref_mask)
    test_kp, test_desc = sift.detectAndCompute(test_gray, test_mask)

    if ref_desc is None or test_desc is None:
        return ref_kp, test_kp, [], "sift"

    good_matches = mutual_ratio_match(
        test_desc,
        ref_desc,
        cv2.NORM_L2,
        SIFT_RATIO_TEST,
        SIFT_KEEP_BEST_MATCHES,
    )

    return ref_kp, test_kp, good_matches, "sift"


def estimate_homography(test_kp, ref_kp, matches):
    if len(matches) < MIN_MATCHES_REQUIRED:
        return None, None

    src_pts = np.float32([test_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    method = cv2.RANSAC
    if hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC

    h_mat, inlier_mask = cv2.findHomography(src_pts, dst_pts, method, RANSAC_REPROJ_THRESHOLD)

    # Optional refine with only inliers to stabilize the final warp.
    if h_mat is not None and inlier_mask is not None:
        inlier_bool = inlier_mask.ravel().astype(bool)
        if np.count_nonzero(inlier_bool) >= 8:
            src_in = src_pts[inlier_bool]
            dst_in = dst_pts[inlier_bool]
            h_refined, _ = cv2.findHomography(src_in, dst_in, 0)
            if h_refined is not None:
                h_mat = h_refined

    return h_mat, inlier_mask


def evaluate_method(ref_gray, test_gray, ref_mask, test_mask, method_name):
    if method_name == "orb":
        ref_kp, test_kp, matches, _ = detect_and_match_orb(ref_gray, test_gray, ref_mask, test_mask)
    else:
        ref_kp, test_kp, matches, _ = detect_and_match_sift(ref_gray, test_gray, ref_mask, test_mask)

    h_mat, inlier_mask = estimate_homography(test_kp, ref_kp, matches)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = (inliers / float(max(len(matches), 1))) if len(matches) > 0 else 0.0

    return {
        "method": method_name,
        "ref_kp": ref_kp,
        "test_kp": test_kp,
        "matches": matches,
        "homography": h_mat,
        "inlier_mask": inlier_mask,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
    }


def draw_match_debug(ref_img, test_img, ref_kp, test_kp, matches, inlier_mask):
    if inlier_mask is None:
        return cv2.drawMatches(
            test_img,
            test_kp,
            ref_img,
            ref_kp,
            matches[: min(60, len(matches))],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    inlier_mask_flat = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(matches, inlier_mask_flat) if keep]

    if len(inlier_matches) == 0:
        return cv2.drawMatches(
            test_img,
            test_kp,
            ref_img,
            ref_kp,
            matches[: min(60, len(matches))],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    return cv2.drawMatches(
        test_img,
        test_kp,
        ref_img,
        ref_kp,
        inlier_matches[: min(120, len(inlier_matches))],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def create_overlay(reference_bgr, warped_bgr):
    return cv2.addWeighted(reference_bgr, BLEND_ALPHA, warped_bgr, 1.0 - BLEND_ALPHA, 0.0)


def align_one_image(reference_bgr, reference_gray, test_path, aligned_output_path=None):
    name = os.path.basename(test_path)
    base, ext = os.path.splitext(name)

    test_bgr = cv2.imread(test_path)
    if test_bgr is None:
        print(f"[WARN] {name}: failed to load image, skipping")
        return

    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    if USE_GROUND_PLANE_MASK:
        ref_mask = build_ground_plane_mask(reference_gray)
        test_mask = build_ground_plane_mask(test_gray)
    else:
        ref_mask = None
        test_mask = None

    # ORB first (recommended main path).
    orb_result = evaluate_method(reference_gray, test_gray, ref_mask, test_mask, "orb")
    best_result = orb_result

    # Optional fallback to SIFT when ORB geometric quality is weak.
    sift_result = None
    if ENABLE_SIFT_FALLBACK:
        sift_result = evaluate_method(reference_gray, test_gray, ref_mask, test_mask, "sift")
        if sift_result["inliers"] > best_result["inliers"]:
            best_result = sift_result
        elif sift_result["inliers"] == best_result["inliers"] and sift_result["inlier_ratio"] > best_result["inlier_ratio"]:
            best_result = sift_result

    method = best_result["method"]
    ref_kp = best_result["ref_kp"]
    test_kp = best_result["test_kp"]
    matches = best_result["matches"]
    h_mat = best_result["homography"]
    inlier_mask = best_result["inlier_mask"]

    ref_kp_count = len(ref_kp)
    test_kp_count = len(test_kp)
    raw_match_count = len(matches)

    print(f"\nImage: {name}")
    print(f"  Method: {method}")
    print(f"  Keypoints (ref/test): {ref_kp_count}/{test_kp_count}")
    print(f"  Raw matches: {raw_match_count}")
    print(
        f"  ORB inliers/ratio: {orb_result['inliers']}/{orb_result['inlier_ratio']:.3f}"
    )
    if sift_result is not None:
        print(
            f"  SIFT inliers/ratio: {sift_result['inliers']}/{sift_result['inlier_ratio']:.3f}"
        )

    if raw_match_count < MIN_MATCHES_REQUIRED:
        print(
            f"  [WARN] not enough matches (need {MIN_MATCHES_REQUIRED}, got {raw_match_count}), skipping"
        )
        return

    if h_mat is None or inlier_mask is None:
        print("  [WARN] homography failed, skipping")
        return

    inliers = int(inlier_mask.sum())
    print(f"  Good matches used: {len(matches)}")
    print(f"  Inliers: {inliers}")

    if inliers < MIN_INLIERS_REQUIRED:
        print(
            f"  [WARN] too few inliers (need {MIN_INLIERS_REQUIRED}, got {inliers}), skipping"
        )
        return

    ref_h, ref_w = reference_bgr.shape[:2]
    warped = cv2.warpPerspective(test_bgr, h_mat, (ref_w, ref_h))

    if aligned_output_path:
        aligned_path = aligned_output_path
    else:
        aligned_path = os.path.join(ALIGNED_OUTPUT_DIR, base + ext)

    aligned_dir = os.path.dirname(aligned_path)
    if aligned_dir:
        os.makedirs(aligned_dir, exist_ok=True)

    match_debug_path = os.path.join(DEBUG_MATCHES_DIR, f"{base}_matches.jpg")
    overlay_debug_path = os.path.join(DEBUG_OVERLAYS_DIR, f"{base}_overlay.jpg")

    match_vis = draw_match_debug(reference_bgr, test_bgr, ref_kp, test_kp, matches, inlier_mask)
    overlay_vis = create_overlay(reference_bgr, warped)

    ok_aligned = cv2.imwrite(aligned_path, warped)
    ok_matches = cv2.imwrite(match_debug_path, match_vis)
    ok_overlay = cv2.imwrite(overlay_debug_path, overlay_vis)

    if not (ok_aligned and ok_matches and ok_overlay):
        print("  [WARN] failed writing one or more outputs")
        return

    print("  Alignment: SUCCESS")
    print(f"  Saved warped: {aligned_path}")
    print(f"  Saved matches: {match_debug_path}")
    print(f"  Saved overlay: {overlay_debug_path}")


def main():
    args = parse_args()
    ensure_dirs()

    if args.output and not args.input:
        print("[ERROR] --output is only supported when --input is provided")
        return

    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"[ERROR] reference image not found: {REFERENCE_IMAGE_PATH}")
        return

    if args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] input image not found: {args.input}")
            return
        test_paths = [args.input]
        print("Single-image mode enabled")
    else:
        test_paths = list_test_images(TEST_IMAGES_DIR)
        if len(test_paths) == 0:
            print(f"[ERROR] no test images found in: {TEST_IMAGES_DIR}")
            return

    reference_bgr = cv2.imread(REFERENCE_IMAGE_PATH)
    if reference_bgr is None:
        print(f"[ERROR] failed to load reference image: {REFERENCE_IMAGE_PATH}")
        return

    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    print("=== Homography Alignment Start ===")
    print(f"Reference: {REFERENCE_IMAGE_PATH}")
    print(f"Test images found: {len(test_paths)}")
    if args.input and args.output:
        print(f"Aligned output path: {args.output}")
    else:
        print(f"Output dirs: {ALIGNED_OUTPUT_DIR}, {DEBUG_MATCHES_DIR}, {DEBUG_OVERLAYS_DIR}")

    for test_path in test_paths:
        try:
            aligned_override = args.output if args.input and args.output else None
            align_one_image(reference_bgr, reference_gray, test_path, aligned_output_path=aligned_override)
        except Exception as exc:
            name = os.path.basename(test_path)
            print(f"[WARN] {name}: unexpected error, skipping ({exc})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
