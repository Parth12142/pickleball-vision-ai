#!/usr/bin/env python3
"""
day1_homography_basics.py — Week 1, Day 1 (Rows 2, 3, 4 of Pickleball_AI_Tracker.xlsx)

Covers:
  1. Row 2: Homogeneous coordinates — Write 2D points as (x, y, 1) and apply 3x3 matrices
  2. Row 3: Transform types — Tell Translation, Affine, and Projective transforms apart
  3. Row 4: Assignment 1 — Hand-solve a 4-point mapping (8x8 system A*h = b) and check in NumPy
"""
import numpy as np

# ============================================================================
# 1. HOMOGENEOUS COORDINATES (Row 2)
# ============================================================================

def to_homogeneous(pts_2d: np.ndarray) -> np.ndarray:
    """
    Convert (N, 2) Cartesian 2D points [(x, y), ...] into (N, 3) homogeneous points [(x, y, 1), ...].
    """
    pts = np.asarray(pts_2d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones])


def from_homogeneous(pts_h: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) homogeneous points [(X', Y', W'), ...] back to (N, 2) 2D points
    via perspective division: (x', y') = (X' / W', Y' / W').
    """
    pts_h = np.asarray(pts_h, dtype=np.float64)
    if pts_h.ndim == 1:
        pts_h = pts_h.reshape(1, 3)
    w = pts_h[:, 2:3]
    return pts_h[:, :2] / w


def apply_homography_matrix(H: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """
    Multiply 2D points by a 3x3 matrix H using homogeneous coordinates:
      p' = H @ [x, y, 1]^T  -->  divide by W'.
    """
    pts_h = to_homogeneous(pts_2d)            # shape (N, 3)
    transformed_h = (H @ pts_h.T).T           # shape (N, 3)
    return from_homogeneous(transformed_h)    # shape (N, 2)


# ============================================================================
# 2. TRANSFORM TYPES: TRANSLATION vs AFFINE vs PROJECTIVE (Row 3)
# ============================================================================

def classify_transform_matrix(H: np.ndarray, tol: float = 1e-7) -> str:
    """
    Inspect a 3x3 homogeneous matrix and classify it as:
      - 'Translation' (2 DoF: top-left 2x2 is Identity, bottom row is [0, 0, 1])
      - 'Affine'      (6 DoF: bottom row is [0, 0, 1], parallel lines stay parallel)
      - 'Projective'  (8 DoF: h31 != 0 or h32 != 0, parallel lines converge)
    """
    M = np.asarray(H, dtype=np.float64)
    M = M / M[2, 2]  # Normalize so h33 = 1

    # Check bottom row [h31, h32, 1]
    is_bottom_zero = np.allclose(M[2, :2], [0.0, 0.0], atol=tol)
    if not is_bottom_zero:
        return "Projective (Homography — 8 DoF, parallel lines converge)"

    # Bottom row is [0, 0, 1] -> Check if top-left 2x2 is Identity
    is_top_identity = np.allclose(M[:2, :2], np.eye(2), atol=tol)
    if is_top_identity:
        return "Translation (2 DoF — shifts (x, y) by (tx, ty))"

    return "Affine (6 DoF — preserves parallel lines, W' = 1 everywhere)"


# ============================================================================
# 3. ASSIGNMENT 1: SOLVE H FOR 4 POINT PAIRS IN PURE NUMPY (Row 4)
# ============================================================================

def solve_homography_4pt_numpy(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    Hand-formulated 8x8 linear solver A * h = b for 4 point correspondences:
      (x_i, y_i)  -->  (X_i, Y_i),  for i = 0, 1, 2, 3 (with h33 = 1).

    For each point pair:
      x_i*h11 + y_i*h12 + h13 - x_i*X_i*h31 - y_i*X_i*h32 = X_i
      x_i*h21 + y_i*h22 + h23 - x_i*Y_i*h31 - y_i*Y_i*h32 = Y_i

    Returns:
      H (3x3 np.ndarray), A (8x8 np.ndarray), b (8x1 np.ndarray)
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    assert src.shape == (4, 2) and dst.shape == (4, 2), "Must provide exactly 4 (x, y) point pairs"

    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8,), dtype=np.float64)

    for i in range(4):
        x, y = src[i]
        X, Y = dst[i]

        # Row 2*i: equation for X_i
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * X, -y * X]
        b[2 * i] = X

        # Row 2*i + 1: equation for Y_i
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * Y, -y * Y]
        b[2 * i + 1] = Y

    # Solve the 8x8 linear system A * h = b
    h = np.linalg.solve(A, b)

    # Assemble the 3x3 Homography matrix with h33 = 1.0
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=np.float64)

    return H, A, b


def main():
    print("=" * 78)
    print(" 🏓 WEEK 1 • DAY 1: HOMOGRAPHY FUNDAMENTALS & WORKED 4-POINT EXAMPLE")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # PART 1: Homogeneous Coordinates (Row 2)
    # ------------------------------------------------------------------------
    sample_pts = np.array([[420.0, 180.0], [640.0, 360.0]])
    pts_h = to_homogeneous(sample_pts)
    print("\n[1] Row 2 — Homogeneous Coordinates (x, y) -> (x, y, 1):")
    for p2, ph in zip(sample_pts, pts_h):
        print(f"    2D Point {p2}  -->  Homogeneous {ph}")

    # ------------------------------------------------------------------------
    # PART 2: Tell Translation, Affine, and Projective Apart (Row 3)
    # ------------------------------------------------------------------------
    T_mat = np.array([[1.0, 0.0, 120.0],
                      [0.0, 1.0, -45.0],
                      [0.0, 0.0,   1.0]])

    A_mat = np.array([[0.866, -0.500, 50.0],
                      [0.500,  0.866, 30.0],
                      [0.000,  0.000,  1.0]])

    P_mat = np.array([[0.025, 0.015, -13.48],
                      [0.000, 0.042,  -7.56],
                      [0.000, -0.00072, 1.0]])

    print("\n[2] Row 3 — Transform Types Classification:")
    print(f"    Matrix 1 (bottom=[0, 0, 1], 2x2=I)  -> {classify_transform_matrix(T_mat)}")
    print(f"    Matrix 2 (bottom=[0, 0, 1], rot)    -> {classify_transform_matrix(A_mat)}")
    print(f"    Matrix 3 (bottom=[0, -0.00072, 1])  -> {classify_transform_matrix(P_mat)}")

    # ------------------------------------------------------------------------
    # PART 3: Assignment 1 — Worked 4-Point Mapping in NumPy (Row 4)
    # ------------------------------------------------------------------------
    # 4 Camera Pixel Corners of a Pickleball Court (Trapezoid due to perspective)
    camera_corners_px = np.array([
        [420.0, 180.0],   # Corner 1: Far-Left Baseline Corner (px)
        [860.0, 180.0],   # Corner 2: Far-Right Baseline Corner (px)
        [1140.0, 640.0],  # Corner 3: Near-Right Baseline Corner (px)
        [140.0, 640.0],   # Corner 4: Near-Left Baseline Corner (px)
    ], dtype=np.float64)

    # 4 Official USAP Top-Down Court Corners in Feet (20 ft wide x 44 ft long)
    usap_corners_ft = np.array([
        [0.0, 0.0],       # Far-Left  (0 ft, 0 ft)
        [20.0, 0.0],      # Far-Right (20 ft, 0 ft)
        [20.0, 44.0],     # Near-Right (20 ft, 44 ft)
        [0.0, 44.0],      # Near-Left  (0 ft, 44 ft)
    ], dtype=np.float64)

    H, _A, _b = solve_homography_4pt_numpy(camera_corners_px, usap_corners_ft)

    print("\n[3] Row 4 — Assignment 1: 4-Point Homography Solved in NumPy:")
    print("    Solved 3x3 Homography Matrix H (Camera Pixels -> USAP Court Feet):")
    for row in H:
        print("      [" + "  ".join(f"{val:12.8f}" for val in row) + "]")

    # Verify all 4 corners map back with zero error
    mapped_corners_ft = apply_homography_matrix(H, camera_corners_px)
    errors = np.linalg.norm(mapped_corners_ft - usap_corners_ft, axis=1)
    max_err = float(np.max(errors))

    print("\n    Verification of 4 Mapped Corners:")
    print("    -----------------------------------------------------------------------")
    print("    Corner  | Camera Pixel (x, y) | Target Court (X, Y) | Mapped (X', Y') | Error")
    print("    -----------------------------------------------------------------------")
    labels = ["Far-L ", "Far-R ", "Near-R", "Near-L"]
    for lbl, src_p, dst_p, map_p, err in zip(labels, camera_corners_px, usap_corners_ft, mapped_corners_ft, errors):
        print(
            f"    {lbl}  | ({src_p[0]:6.1f}, {src_p[1]:5.1f})   | "
            f"({dst_p[0]:4.1f} ft, {dst_p[1]:4.1f} ft)  | "
            f"({map_p[0]:5.2f}, {map_p[1]:5.2f})  | {err:.2e} ft"
        )
    print("    -----------------------------------------------------------------------")
    print(f"    Max Reprojection Error across 4 corners: {max_err:.12f} ft  [PASS]")

    # Also test inverse mapping: where does the exact USAP Net Center (10 ft, 22 ft) appear in pixels?
    H_inv = np.linalg.inv(H)
    net_center_px = apply_homography_matrix(H_inv, np.array([[10.0, 22.0]]))[0]
    remapped_net_ft = apply_homography_matrix(H, net_center_px)[0]
    print(
        f"\n    Bonus Check — USAP Net Center (10.0 ft, 22.0 ft):"
        f"\n      -> Projects to Camera Pixel ({net_center_px[0]:.2f} px, {net_center_px[1]:.2f} px)"
        f"\n      -> Maps back through H to   ({remapped_net_ft[0]:.2f} ft, {remapped_net_ft[1]:.2f} ft)"
    )
    print("=" * 78)

    assert max_err < 1e-9, "4-point homography error exceeded tolerance!"


if __name__ == "__main__":
    main()
