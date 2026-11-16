"""
homography.py - Precision Court Calibration & Homography Projection.
Zero scoring or referee rules. Strictly computes the geometric mapping between
camera image pixels (u, v) and regulation USAP 2D court plane coordinates (X, Y) in feet and meters.
"""
from dataclasses import dataclass, field
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

CALIBRATION_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "court_calibrations.json")
)


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
        self._pose_ready: bool = False
        self._K_inv: Optional[np.ndarray] = None
        self._R_T: Optional[np.ndarray] = None
        self._cam_pos: Optional[np.ndarray] = None
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
        self._pose_ready = False
        return True

    def solve_3d_pose(self, image_width: int, image_height: int, focal_scale: float = 1.15) -> bool:
        """
        Recovers 3D camera extrinsics (R, t, cam_pos) using cv2.solvePnP from the 4 court corners.
        Enables height-compensated ray-plane intersection for airborne ball tracking.
        """
        if not self.result.calibrated or len(self.result.corners_px) != 4:
            return False

        f = float(focal_scale * max(image_width, image_height))
        K = np.array(
            [[f, 0.0, image_width / 2.0], [0.0, f, image_height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        obj_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [CourtGeometry.WIDTH_FT, 0.0, 0.0],
                [CourtGeometry.WIDTH_FT, CourtGeometry.LENGTH_FT, 0.0],
                [0.0, CourtGeometry.LENGTH_FT, 0.0],
            ],
            dtype=np.float64,
        )
        img_pts = np.array(self.result.corners_px, dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return False

        R, _ = cv2.Rodrigues(rvec)
        cam_pos = (-R.T @ tvec).flatten()
        self._K_inv = np.linalg.inv(K)
        self._R_T = R.T
        self._cam_pos = cam_pos
        self._pose_ready = True
        return True

    def pixel_to_court_at_height_ft(self, px: float, py: float, height_ft: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Projects camera pixel (px, py) onto a horizontal plane at `height_ft` above the court.
        When height_ft <= 0.01 or 3D pose is unavailable, falls back to standard ground homography.
        """
        if height_ft <= 0.01 or not self._pose_ready or self._K_inv is None or self._R_T is None or self._cam_pos is None:
            return self.pixel_to_court_ft(px, py)

        # In our right-handed court frame (X=right, Y=toward camera), +height above court is -Z
        z_target = -float(height_ft)
        ray_cam = self._K_inv @ np.array([float(px), float(py), 1.0], dtype=np.float64)
        ray_world = self._R_T @ ray_cam

        if abs(ray_world[2]) < 1e-6:
            return self.pixel_to_court_ft(px, py)

        s = (z_target - self._cam_pos[2]) / ray_world[2]
        if s <= 0.0:
            return self.pixel_to_court_ft(px, py)

        pt = self._cam_pos + s * ray_world
        return float(pt[0]), float(pt[1])

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

    @classmethod
    def load_calibration_for_clip(cls, clip_name: str) -> Optional[List[Tuple[float, float]]]:
        """Load stored corner coordinates for a clip from court_calibrations.json."""
        if not os.path.exists(CALIBRATION_FILE):
            return None
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            # Match directly or by basename
            base = os.path.basename(clip_name)
            entry = data.get(base) or data.get(clip_name)
            if entry and "corners_px" in entry:
                return [(float(pt[0]), float(pt[1])) for pt in entry["corners_px"]]
        except Exception:
            return None
        return None

    @classmethod
    def save_calibration_for_clip(
        cls,
        clip_name: str,
        corners_px: List[Tuple[float, float]],
        resolution: Tuple[int, int],
        rmse_ft: float = 0.0,
        source: str = "human_pinned_calibrated",
    ) -> bool:
        """Save calibrated corner coordinates for a clip to court_calibrations.json."""
        base = os.path.basename(clip_name)
        data = {}
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        w, h = resolution
        aspect_ratio = "9:16_portrait" if h > w else "16:9_landscape"
        data[base] = {
            "resolution": [int(w), int(h)],
            "aspect_ratio": aspect_ratio,
            "corners_px": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in corners_px],
            "rmse_ft": round(float(rmse_ft), 4),
            "source": source,
        }

        try:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def get_recommended_corners(
        cls, width: int, height: int, clip_name: Optional[str] = None
    ) -> List[Tuple[float, float]]:
        """
        Computes recommended court corners based on resolution and aspect ratio:
        - Checks persistent calibration store first
        - Portrait (h > w): calibrated smartphone camera geometry
        - Landscape (w >= h): calibrated broadcast camera geometry
        """
        if clip_name:
            saved = cls.load_calibration_for_clip(clip_name)
            if saved and len(saved) == 4:
                return saved

        w, h = float(width), float(height)
        if h > w:
            # Vertical smartphone perspective (e.g. 720x1280)
            return [
                (float(w * 0.3403), float(h * 0.40625)),  # Far-Left
                (float(w * 0.6597), float(h * 0.40625)),  # Far-Right
                (float(w * 0.9444), float(h * 0.81250)),  # Near-Right
                (float(w * 0.0556), float(h * 0.81250)),  # Near-Left
            ]
        else:
            # Landscape broadcast perspective (e.g. 1302x740 / 1280x720)
            return [
                (float(w * 0.3264), float(h * 0.2392)),   # Far-Left
                (float(w * 0.6567), float(h * 0.2392)),   # Far-Right
                (float(w * 0.8310), float(h * 0.8135)),   # Near-Right
                (float(w * 0.1567), float(h * 0.8135)),   # Near-Left
            ]

    def auto_calibrate_broadcast(self, frame: np.ndarray, clip_name: Optional[str] = None) -> bool:
        """
        Calibrate court perspective with smart fallback:
        1. Checks stored human calibration in court_calibrations.json
        2. Applies aspect-ratio-aware defaults (portrait vs landscape)
        """
        h, w = frame.shape[:2]
        corners = self.get_recommended_corners(w, h, clip_name=clip_name)
        return self.calibrate(corners)
