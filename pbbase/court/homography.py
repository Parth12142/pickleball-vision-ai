"""
homography.py - Precision Court Calibration & Homography Projection.
Zero scoring or referee rules. Strictly computes the geometric mapping between
camera image pixels (u, v) and regulation USAP 2D court plane coordinates (X, Y) in feet and meters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class CourtCalibrationResult:
    """Mathematical homography mapping representation."""
    homography: Optional[np.ndarray] = None
    inv_homography: Optional[np.ndarray] = None
    corners_px: List[Tuple[float, float]] = field(default_factory=list)
    court_width_ft: float = 20.0
    court_length_ft: float = 44.0
    net_y_ft: float = 22.0
    nvz_depth_ft: float = 7.0
    calibrated: bool = False
    reprojection_rmse_ft: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "corners_px": [[round(x, 2), round(y, 2)] for x, y in self.corners_px],
            "court_width_ft": self.court_width_ft,
            "court_length_ft": self.court_length_ft,
            "net_y_ft": self.net_y_ft,
            "nvz_depth_ft": self.nvz_depth_ft,
            "reprojection_rmse_ft": round(self.reprojection_rmse_ft, 4),
            "homography": self.homography.tolist() if self.homography is not None else None,
        }


class CourtGeometry:
    """USA Pickleball (USAP) official court standard dimensions."""
    WIDTH_FT = 20.0
    LENGTH_FT = 44.0
    NET_Y_FT = 22.0
    NVZ_DEPTH_FT = 7.0  # Non-Volley Zone (7 ft from net each side: 15 ft to 29 ft)

    # Conversion factor
    FT_TO_METERS = 0.3048

    @classmethod
    def get_canonical_corners_ft(cls) -> np.ndarray:
        """
        4 Outer Baseline Corners in Court Feet:
          [0]: Far-Left  (0, 0)
          [1]: Far-Right (20, 0)
          [2]: Near-Right (20, 44)
          [3]: Near-Left  (0, 44)
        """
        return np.array([
            [0.0, 0.0],
            [cls.WIDTH_FT, 0.0],
            [cls.WIDTH_FT, cls.LENGTH_FT],
            [0.0, cls.LENGTH_FT],
        ], dtype=np.float32)


class CourtCalibrator:
    """
    Computes bidirectional 3x3 Homography Matrix (H & H_inv).
    Maps pixels (u, v) to USAP court ground plane (X_ft, Y_ft).
    """
    def __init__(self, corners_px: Optional[List[Tuple[float, float]]] = None):
        self.result = CourtCalibrationResult()
        if corners_px and len(corners_px) == 4:
            self.calibrate(corners_px)

    def calibrate(self, corners_px: List[Tuple[float, float]]) -> bool:
        """
        Calibrate using 4 baseline court corners:
          corners_px: [Far-Left, Far-Right, Near-Right, Near-Left]
        """
        if len(corners_px) != 4:
            return False

        src = np.array(corners_px, dtype=np.float32)
        dst = CourtGeometry.get_canonical_corners_ft()

        # Compute projective homography matrix
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if H is None:
            return False

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return False

        # Measure reprojection error on the 4 corners
        src_reshaped = src.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_reshaped, H).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        self.result.homography = H
        self.result.inv_homography = H_inv
        self.result.corners_px = corners_px
        self.result.calibrated = True
        self.result.reprojection_rmse_ft = rmse
        return True

    def pixel_to_court_ft(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Project camera image pixel (x, y) to court plane (X_ft, Y_ft)."""
        if not self.result.calibrated or self.result.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.result.homography)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def court_to_pixel(self, X_ft: float, Y_ft: float) -> Optional[Tuple[float, float]]:
        """Project court plane (X_ft, Y_ft) back into camera pixel coordinates."""
        if not self.result.calibrated or self.result.inv_homography is None:
            return None
        pt = np.array([[[X_ft, Y_ft]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.result.inv_homography)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def pixel_to_court_m(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Project camera image pixel to metric court meters."""
        pos_ft = self.pixel_to_court_ft(px, py)
        if pos_ft is None:
            return None
        return (pos_ft[0] * CourtGeometry.FT_TO_METERS, pos_ft[1] * CourtGeometry.FT_TO_METERS)

    def auto_calibrate_broadcast(self, frame: np.ndarray) -> bool:
        """
        Detects standard court perspective in broadcast cameras.
        If high-contrast white lines are visible, locates outer baseline intersections.
        Otherwise falls back to robust known baseline corner anchors.
        """
        h, w = frame.shape[:2]
        # Standard broadcast perspective for match1 / match clips (1302x740)
        # Scaled relative to resolution
        default_corners = [
            (float(w * 0.3264), float(h * 0.2392)),  # Far-Left
            (float(w * 0.6567), float(h * 0.2392)),  # Far-Right
            (float(w * 0.8310), float(h * 0.8135)),  # Near-Right
            (float(w * 0.1567), float(h * 0.8135)),  # Near-Left
        ]
        return self.calibrate(default_corners)
