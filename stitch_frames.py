#!/usr/bin/env python3
"""
Stitch consecutive CORONA/KH-4 satellite frames into a single mosaic.

Uses SIFT feature detection on downscaled images for speed,
then applies the computed transforms at full resolution.

Usage:
    python3 stitch_frames.py frame1.tif frame2.tif [frame3.tif ...] -o output.tif
"""

import argparse
import sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # These are large satellite scans


def load_image(path):
    """Load a grayscale TIF as numpy array."""
    img = Image.open(path)
    if img.mode == "RGB":
        img = img.convert("L")
    return np.array(img)


def find_homography(img_a, img_b, scale=0.25, min_matches=10):
    """
    Find homography mapping img_b coordinates into img_a coordinate space.
    Uses downscaled images for feature detection, then scales the homography back.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Use a higher scale for better matching on these detailed images
    small_a = cv2.resize(img_a, (int(w_a * scale), int(h_a * scale)))
    small_b = cv2.resize(img_b, (int(w_b * scale), int(h_b * scale)))

    # CLAHE to normalize contrast before feature detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small_a = clahe.apply(small_a)
    small_b = clahe.apply(small_b)

    # SIFT is much better than ORB for satellite imagery
    sift = cv2.SIFT_create(nfeatures=10000)
    kp_a, desc_a = sift.detectAndCompute(small_a, None)
    kp_b, desc_b = sift.detectAndCompute(small_b, None)

    print(f"  Features: img_a={len(kp_a)}, img_b={len(kp_b)}")

    if desc_a is None or desc_b is None:
        print("ERROR: Could not detect features in one of the images")
        sys.exit(1)

    # FLANN-based matcher (faster and better for SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(desc_b, desc_a, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    print(f"  Good matches after ratio test: {len(good)} (from {len(raw_matches)} raw)")

    if len(good) < min_matches:
        print(f"ERROR: Not enough matches ({len(good)} < {min_matches})")
        sys.exit(1)

    # Extract matched point coordinates (in downscaled space)
    pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography
    H_small, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0,
                                        maxIters=5000, confidence=0.999)
    inliers = mask.ravel().sum()
    print(f"  RANSAC inliers: {inliers}/{len(good)}")

    if H_small is None:
        print("ERROR: Could not compute homography")
        sys.exit(1)

    # Validate: the homography should mostly be a translation + slight rotation/scale
    # Check that the determinant is near 1 (no large scale change)
    det = np.linalg.det(H_small[:2, :2])
    print(f"  Homography determinant: {det:.4f} (should be near 1.0)")
    if abs(det - 1.0) > 0.5:
        print("WARNING: Large scale change detected — homography may be wrong")

    # Scale homography to full resolution
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[1/scale, 0, 0],
                      [0, 1/scale, 0],
                      [0, 0, 1]], dtype=np.float64)
    H_full = S_inv @ H_small @ S

    # Print where the corners of img_b map to in img_a space
    corners = np.float32([[0, 0], [w_b-1, 0], [w_b-1, h_b-1], [0, h_b-1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, H_full)
    print(f"  img_b corners map to:")
    labels = ["TL", "TR", "BR", "BL"]
    for label, pt in zip(labels, mapped.reshape(-1, 2)):
        print(f"    {label}: ({pt[0]:.0f}, {pt[1]:.0f})")

    return H_full


def stitch_pair(img_a, img_b, H):
    """
    Warp img_b into img_a's coordinate space and blend them together.
    Returns the combined image.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Find where corners of img_b end up in img_a's space
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    corners_b_warped = cv2.perspectiveTransform(corners_b, H)

    # Also include img_a's own corners
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners_a, corners_b_warped], axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel()) - 1
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel()) + 1

    # Translation to shift everything into positive coordinates
    tx, ty = -x_min, -y_min
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    out_w = x_max - x_min
    out_h = y_max - y_min
    print(f"  Output canvas: {out_w}x{out_h}")

    # Sanity check: output shouldn't be absurdly large
    max_dim = max(w_a + w_b, h_a + h_b) * 2
    if out_w > max_dim or out_h > max_dim:
        print(f"ERROR: Output size {out_w}x{out_h} seems too large, aborting")
        sys.exit(1)

    # Warp img_b into the output space
    warped_b = cv2.warpPerspective(img_b, T @ H, (out_w, out_h))

    # Place img_a into the output space (simple translation)
    canvas_a = np.zeros((out_h, out_w), dtype=img_a.dtype)
    canvas_a[ty:ty+h_a, tx:tx+w_a] = img_a

    # Create masks for blending
    mask_a = np.zeros((out_h, out_w), dtype=np.uint8)
    mask_a[ty:ty+h_a, tx:tx+w_a] = (img_a > 0).astype(np.uint8) * 255

    mask_b = cv2.warpPerspective(
        np.ones_like(img_b) * 255, T @ H, (out_w, out_h)
    ).astype(np.uint8)

    # In overlap region, use weighted average; elsewhere use whichever exists
    overlap = (mask_a > 0) & (mask_b > 0)
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_a == 0) & (mask_b > 0)

    result = np.zeros((out_h, out_w), dtype=np.uint8)
    result[only_a] = canvas_a[only_a]
    result[only_b] = warped_b[only_b]

    # Feathered blend in overlap region
    if overlap.any():
        dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5)
        dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5)
        total = dist_a + dist_b
        total[total == 0] = 1
        weight_a = dist_a / total
        weight_b = dist_b / total
        blended = (canvas_a.astype(np.float32) * weight_a +
                   warped_b.astype(np.float32) * weight_b)
        result[overlap] = blended[overlap].astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description="Stitch consecutive satellite frames")
    parser.add_argument("frames", nargs="+", help="Input TIF frames (in order)")
    parser.add_argument("-o", "--output", required=True, help="Output TIF path")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Downscale factor for feature detection (default: 0.25)")
    args = parser.parse_args()

    if len(args.frames) < 2:
        print("Need at least 2 frames to stitch")
        sys.exit(1)

    print(f"Stitching {len(args.frames)} frames...")

    # Load first frame as the base
    print(f"\nLoading {args.frames[0]}...")
    result = load_image(args.frames[0])
    print(f"  Size: {result.shape[1]}x{result.shape[0]}")

    # Iteratively stitch each subsequent frame
    for i in range(1, len(args.frames)):
        print(f"\nLoading {args.frames[i]}...")
        next_img = load_image(args.frames[i])
        print(f"  Size: {next_img.shape[1]}x{next_img.shape[0]}")

        print(f"  Finding homography (scale={args.scale})...")
        H = find_homography(result, next_img, scale=args.scale)
        print(f"  Homography:\n{H}")

        print(f"  Stitching...")
        result = stitch_pair(result, next_img, H)
        print(f"  Current mosaic: {result.shape[1]}x{result.shape[0]}")

    # Save output
    print(f"\nSaving to {args.output}...")
    out_img = Image.fromarray(result)
    out_img.save(args.output, compression="lzw")
    print(f"Done! Final size: {result.shape[1]}x{result.shape[0]}")


if __name__ == "__main__":
    main()
