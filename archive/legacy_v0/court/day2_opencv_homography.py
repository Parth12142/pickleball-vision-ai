#!/usr/bin/env python3
"""
day2_opencv_homography.py — Week 1, Plan Day 2 (Rows 5, 6, 7 of Pickleball_AI_Tracker.xlsx)
Completes Work Date 1 (2026-11-03 Tue: Plan Days 1 & 2, 8.0 Hours Total).

Covers:
  1. Row 5: cv2.findHomography with RANSAC — Compute H from point pairs and reject outliers
  2. Row 6: cv2.perspectiveTransform — Map lists of points through H
  3. Row 7: cv2.warpPerspective (Assignment 2) — Produce a top-down bird's-eye image of a real frame
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from court.day1_homography_basics import solve_homography_4pt_numpy


def demo_find_homography_ransac():
    """
    Row 5: Compute H from point pairs with cv2.findHomography(..., cv2.RANSAC)
    and demonstrate automatic outlier rejection.
    """
    # 4 true camera corners + 2 true kitchen corners + 2 corrupted outlier points
    src_pts = np.array([
        [420.0, 180.0],             # 0: Far-Left Corner (INLIER)
        [860.0, 180.0],             # 1: Far-Right Corner (INLIER)
        [1140.0, 640.0],            # 2: Near-Right Corner (INLIER)
        [140.0, 640.0],             # 3: Near-Left Corner (INLIER)
        [640.0, 2885.0 / 9.0],      # 4: Exact Net Center (INLIER: 320.5556)
        [640.0, 180.0],             # 5: Far Baseline Center (INLIER)
        [999.0, 120.0],             # 6: BAD OUTLIER #1 (false detection in crowd)
        [50.0, 200.0],              # 7: BAD OUTLIER #2 (false detection on wall)
    ], dtype=np.float64)

    dst_pts = np.array([
        [0.0, 0.0],         # 0: (0 ft, 0 ft)
        [20.0, 0.0],        # 1: (20 ft, 0 ft)
        [20.0, 44.0],       # 2: (20 ft, 44 ft)
        [0.0, 44.0],        # 3: (0 ft, 44 ft)
        [10.0, 22.0],       # 4: Net Center (10 ft, 22 ft)
        [10.0, 0.0],        # 5: Far Baseline Center (10 ft, 0 ft)
        [10.0, 37.0],       # 6: Wrong target for Outlier #1
        [15.0, 15.0],       # 7: Wrong target for Outlier #2
    ], dtype=np.float64)

    # Compute H with RANSAC (threshold = 0.5 ft in court space)
    H_ransac, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=0.5)
    inlier_flags = mask.ravel().tolist()

    # Compare against our Day 1 exact 4-point NumPy solver on the 4 true corners
    H_day1, _, _ = solve_homography_4pt_numpy(src_pts[:4], dst_pts[:4])
    max_diff = float(np.max(np.abs(H_ransac - H_day1)))

    print("\n[1] Row 5 — cv2.findHomography with RANSAC (Outlier Rejection):")
    print("    Input Points        : 8 pairs (6 true court points + 2 corrupted outliers)")
    print(f"    RANSAC Inlier Mask  : {inlier_flags}  (1 = Inlier kept, 0 = Outlier rejected)")
    print(f"    Max |H_ransac - H_day1| : {max_diff:.8e}  [PASS]")

    assert inlier_flags == [1, 1, 1, 1, 1, 1, 0, 0], f"Unexpected RANSAC mask: {inlier_flags}"
    assert max_diff < 1e-4, "RANSAC H did not match Day 1 exact H!"
    return H_ransac


def demo_perspective_transform(H: np.ndarray):
    """
    Row 6: Map a list of points through H using cv2.perspectiveTransform.
    """
    # List of 5 camera pixels: 4 corners + Net Center
    query_pixels = np.array([
        [420.0, 180.0],
        [860.0, 180.0],
        [1140.0, 640.0],
        [140.0, 640.0],
        [640.0, 2885.0 / 9.0],
    ], dtype=np.float32)

    # OpenCV expects shape (N, 1, 2)
    pts_reshaped = query_pixels.reshape(-1, 1, 2)
    mapped_court_ft = cv2.perspectiveTransform(pts_reshaped, H).reshape(-1, 2)

    print("\n[2] Row 6 — cv2.perspectiveTransform (Batch Point Mapping):")
    names = ["Far-Left ", "Far-Right", "Near-Right", "Near-Left", "Net-Center"]
    for name, px, ft in zip(names, query_pixels, mapped_court_ft):
        print(f"    {name:10s} : Pixel ({px[0]:6.1f}, {px[1]:6.2f})  -->  Court ({ft[0]:5.2f} ft, {ft[1]:5.2f} ft)")

    return mapped_court_ft


def demo_warp_perspective_still_frame():
    """
    Row 7 (Assignment 2): Warp one still frame from match1.mp4 to a top-down bird's-eye image.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clip_path = os.path.join(root_dir, "data", "clips", "match1.mp4")
    out_dir = os.path.join(root_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "day2_topdown_warp.jpg")

    cap = cv2.VideoCapture(clip_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame from {clip_path}")

    h_in, w_in = frame.shape[:2]

    # 4 exact white court corners in match1.mp4 (1302 x 740 broadcast perspective)
    src_corners_px = np.array([
        [425.0, 177.0],   # Far-Left baseline corner
        [855.0, 177.0],   # Far-Right baseline corner
        [1082.0, 602.0],  # Near-Right baseline corner
        [204.0, 602.0],   # Near-Left baseline corner
    ], dtype=np.float32)

    # Top-down bird's-eye canvas (Width = 480 px, Height = 960 px; 20x44 ft court inside margin)
    out_w, out_h = 480, 960
    margin_x, margin_y = 60, 60
    court_w_px = out_w - 2 * margin_x       # 360 px (18 px per foot)
    court_h_px = out_h - 2 * margin_y       # 840 px (~19.09 px per foot)

    dst_corners_topdown = np.array([
        [margin_x, margin_y],
        [margin_x + court_w_px, margin_y],
        [margin_x + court_w_px, margin_y + court_h_px],
        [margin_x, margin_y + court_h_px],
    ], dtype=np.float32)

    # Compute homography and warp frame to bird's-eye view
    H_warp, _ = cv2.findHomography(src_corners_px, dst_corners_topdown, method=cv2.RANSAC)
    birdseye_img = cv2.warpPerspective(frame, H_warp, (out_w, out_h))

    # Draw the 4 calibrated corner markers on a copy of the original frame
    annotated_src = frame.copy()
    pts_poly = src_corners_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated_src, [pts_poly], isClosed=True, color=(0, 255, 223), thickness=3, lineType=cv2.LINE_AA)
    for idx, (cx, cy) in enumerate(src_corners_px.astype(int), 1):
        cv2.circle(annotated_src, (cx, cy), 8, (0, 77, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            annotated_src,
            f"C{idx}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw USAP reference boundary & kitchen lines on the warped bird's-eye image
    cv2.rectangle(
        birdseye_img,
        (margin_x, margin_y),
        (margin_x + court_w_px, margin_y + court_h_px),
        (0, 255, 223),
        2,
        cv2.LINE_AA,
    )
    net_y = margin_y + court_h_px // 2
    nvz_offset = round(court_h_px * (7.0 / 44.0))
    cv2.line(birdseye_img, (margin_x, net_y), (margin_x + court_w_px, net_y), (0, 77, 255), 2, cv2.LINE_AA)
    cv2.line(
        birdseye_img,
        (margin_x, net_y - nvz_offset),
        (margin_x + court_w_px, net_y - nvz_offset),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        birdseye_img,
        (margin_x, net_y + nvz_offset),
        (margin_x + court_w_px, net_y + nvz_offset),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Combine side-by-side (Original Perspective Frame vs Top-Down Bird's-Eye Warp)
    scale_src_h = out_h
    scale_src_w = round(w_in * (scale_src_h / h_in))
    src_resized = cv2.resize(annotated_src, (scale_src_w, scale_src_h))
    combined = np.hstack([src_resized, birdseye_img])

    cv2.imwrite(out_path, combined)

    print("\n[3] Row 7 — Assignment 2: cv2.warpPerspective (Top-Down Bird's-Eye Image):")
    print(f"    Source Clip Frame   : {clip_path} ({w_in}x{h_in})")
    print(f"    Top-Down Output Size: {out_w}x{out_h} px (Bird's-Eye View)")
    print(f"    Saved Comparison To : {out_path}  [PASS]")
    return out_path


def main():
    print("=" * 78)
    print(" 🏓 WORK DATE 1 (2026-11-03) • PLAN DAY 2: OPENCV HOMOGRAPHY & BIRD'S-EYE WARP")
    print("=" * 78)
    H = demo_find_homography_ransac()
    demo_perspective_transform(H)
    demo_warp_perspective_still_frame()
    print("=" * 78)


if __name__ == "__main__":
    main()
