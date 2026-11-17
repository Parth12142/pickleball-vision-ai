"""
homography.py - Precision Court Calibration, Planar Homography & 3D Camera Pose Estimation.
Zero scoring or referee rules. Strictly computes the geometric mapping between
camera image pixels (u, v) and regulation USAP 3D court coordinates (X, Y, Z) in feet and meters.
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
    """Mathematical homography and 3D camera pose representation."""
    homography: Optional[np.ndarray] = None
    inv_homography: Optional[np.ndarray] = None
    corners_px: List[Tuple[float, float]] = field(default_factory=list)
    court_width_ft: float = 20.0
    court_length_ft: float = 44.0
    net_y_ft: float = 22.0
    nvz_depth_ft: float = 7.0
    calibrated: bool = False
    reprojection_rmse_ft: float = 0.0
    # 3D Camera Pose fields (populated by solve_3d_pose)
    pose_calibrated: bool = False
    image_size: Optional[Tuple[int, int]] = None
    camera_pos_ft: Optional[Tuple[float, float, float]] = None
    focal_length_px: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "corners_px": [[round(x, 2), round(y, 2)] for x, y in self.corners_px],
            "court_width_ft": self.court_width_ft,
            "court_length_ft": self.court_length_ft,
            "net_y_ft": self.net_y_ft,
            "nvz_depth_ft": self.nvz_depth_ft,
            "reprojection_rmse_ft": round(self.reprojection_rmse_ft, 4),
            "pose_calibrated": self.pose_calibrated,
            "camera_pos_ft": [round(v, 2) for v in self.camera_pos_ft] if self.camera_pos_ft else None,
            "focal_length_px": round(self.focal_length_px, 1),
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

    @classmethod
    def get_canonical_corners_3d_ft(cls) -> np.ndarray:
        """
        4 Outer Baseline Corners in 3D Court Feet (Z = 0 ground plane).
        Right-handed coordinate frame: +X right, +Y toward near camera, -Z upward height.
        """
        return np.array([
            [0.0, 0.0, 0.0],
            [cls.WIDTH_FT, 0.0, 0.0],
            [cls.WIDTH_FT, cls.LENGTH_FT, 0.0],
            [0.0, cls.LENGTH_FT, 0.0],
        ], dtype=np.float64)


class CourtCalibrator:
    """
    Computes bidirectional 3x3 Homography Matrix (H & H_inv) and 3D Camera Pose (K, R, t).
    Maps pixels (u, v) to USAP court plane (X_ft, Y_ft) at arbitrary height above court.
    """
    def __init__(
        self,
        corners_px: Optional[List[Tuple[float, float]]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.result = CourtCalibrationResult()
        self._K: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None
        self._R_T: Optional[np.ndarray] = None
        self._cam_world: Optional[np.ndarray] = None

        if corners_px and len(corners_px) == 4:
            self.calibrate(corners_px, image_size=image_size)

    def calibrate(
        self,
        corners_px: List[Tuple[float, float]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Calibrate using 4 baseline court corners:
          corners_px: [Far-Left, Far-Right, Near-Right, Near-Left]
        Optionally solves 3D camera pose when image_size=(w, h) is provided.
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

        if image_size is not None:
            self.solve_3d_pose(image_size[0], image_size[1])

        return True

    def solve_3d_pose(self, image_width: int, image_height: int) -> bool:
        """
        Recovers full 3D camera intrinsics (optimal focal length f) and extrinsics (R, t)
        from the 4 planar court corners via IPPE + Levenberg-Marquardt PnP optimization.
        Enables parallax-free height-compensated ray projection for airborne balls.
        """
        if not self.result.calibrated or len(self.result.corners_px) != 4:
            return False

        w, h = float(image_width), float(image_height)
        obj_pts = CourtGeometry.get_canonical_corners_3d_ft()
        img_pts = np.array(self.result.corners_px, dtype=np.float64)
        dist = np.zeros(4, dtype=np.float64)

        best_f = 1.25 * max(w, h)
        best_err = float("inf")
        best_rvec = None
        best_tvec = None

        # Coarse-to-fine focal length sweep to minimize corner reprojection error
        f_min = 0.55 * max(w, h)
        f_max = 2.60 * max(w, h)
        for f in np.linspace(f_min, f_max, 80):
            K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
            if err < best_err:
                best_err = err
                best_f = float(f)
                best_rvec = rvec
                best_tvec = tvec

        # Fine local refinement around best_f
        for f in np.linspace(max(f_min, best_f * 0.94), min(f_max, best_f * 1.06), 40):
            K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
            if err < best_err:
                best_err = err
                best_f = float(f)
                best_rvec = rvec
                best_tvec = tvec

        if best_rvec is None or best_tvec is None:
            return False

        K = np.array([[best_f, 0.0, w / 2.0], [0.0, best_f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, rvec=best_rvec, tvec=best_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            rvec, tvec = best_rvec, best_tvec

        R, _ = cv2.Rodrigues(rvec)
        R_T = R.T
        cam_world = (-R_T @ tvec).flatten()

        # Verify camera is physically above court floor (in our RH frame height = -cam_z > 0)
        if -cam_world[2] < 2.0:
            return False

        self._K = K
        self._K_inv = np.linalg.inv(K)
        self._rvec = rvec
        self._tvec = tvec
        self._R_T = R_T
        self._cam_world = cam_world

        self.result.pose_calibrated = True
        self.result.image_size = (int(image_width), int(image_height))
        self.result.camera_pos_ft = (float(cam_world[0]), float(cam_world[1]), float(-cam_world[2]))
        self.result.focal_length_px = best_f
        return True

    def pixel_to_court_ft(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Project camera image pixel (x, y) to Z=0 court ground plane (X_ft, Y_ft)."""
        if not self.result.calibrated or self.result.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.result.homography)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def pixel_to_court_at_height_ft(
        self, px: float, py: float, height_ft: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        """
        Projects camera image pixel (px, py) by intersecting its 3D camera ray with the
        horizontal plane at `height_ft` feet above the court floor.
        Eliminates 3D-to-2D parallax distortion for airborne balls.
        """
        if abs(height_ft) < 1e-4 or not self.result.pose_calibrated or self._K_inv is None or self._R_T is None or self._cam_world is None:
            return self.pixel_to_court_ft(px, py)

        ray_cam = self._K_inv @ np.array([float(px), float(py), 1.0], dtype=np.float64)
        ray_world = self._R_T @ ray_cam

        # Avoid parallel/upward rays above horizon
        target_z = -float(height_ft)  # +height above court is -Z in RH world frame
        dz = ray_world[2]
        if abs(dz) < 1e-6:
            return self.pixel_to_court_ft(px, py)

        s = (target_z - self._cam_world[2]) / dz
        if s <= 0:
            # Ray points above the height plane (near horizon)
            return self.pixel_to_court_ft(px, py)

        pt = self._cam_world + s * ray_world
        return float(pt[0]), float(pt[1])

    def court_3d_to_pixel(
        self, X_ft: float, Y_ft: float, height_ft: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        """Projects a 3D court point (X_ft, Y_ft, height_ft) back into camera pixels (u, v)."""
        if abs(height_ft) < 1e-4 or not self.result.pose_calibrated or self._K is None or self._rvec is None or self._tvec is None:
            return self.court_to_pixel(X_ft, Y_ft)

        pt3d = np.array([[float(X_ft), float(Y_ft), -float(height_ft)]], dtype=np.float64)
        proj, _ = cv2.projectPoints(pt3d, self._rvec, self._tvec, self._K, np.zeros(4, dtype=np.float64))
        return float(proj[0, 0, 0]), float(proj[0, 0, 1])

    def court_to_pixel(self, X_ft: float, Y_ft: float) -> Optional[Tuple[float, float]]:
        """Project Z=0 court ground plane (X_ft, Y_ft) back into camera pixel coordinates."""
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

    def get_court_pixel_scale(self, frame_shape: Optional[Tuple[int, int]] = None) -> float:
        """
        Computes the geometric pixel scale factor of the calibrated court relative to
        the canonical 1302x740 broadcast benchmark (near_w=876px, vert_h=429px).
        Used to normalize pixel velocity gates, bounce acceleration, and spatial margins
        uniformly across 360p, 720p, 1080p, and 1440x2560 clips.
        """
        if self.result.calibrated and len(self.result.corners_px) == 4:
            c = self.result.corners_px
            near_w = float(np.hypot(c[2][0] - c[3][0], c[2][1] - c[3][1]))
            vert_h = float(abs(0.5 * (c[2][1] + c[3][1]) - 0.5 * (c[0][1] + c[1][1])))
            if near_w > 50.0 and vert_h > 30.0:
                scale = 0.5 * (near_w / 876.0) + 0.5 * (vert_h / 429.0)
                return float(np.clip(scale, 0.35, 3.2))
        if frame_shape is not None and len(frame_shape) >= 2:
            h, w = frame_shape[:2]
            return float(np.clip(max(w, h) / 1280.0, 0.35, 3.2))
        if self.result.image_size is not None:
            w, h = self.result.image_size
            return float(np.clip(max(w, h) / 1280.0, 0.35, 3.2))
        return 1.0

    @classmethod
    def load_calibration_for_clip(
        cls,
        clip_name: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Load stored corner coordinates for a clip from court_calibrations.json.
        When target_size=(width, height) is provided and differs from the stored
        calibration resolution, automatically scales corners proportionally.
        """
        if not os.path.exists(CALIBRATION_FILE):
            return None
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            # Match directly or by basename
            base = os.path.basename(clip_name)
            entry = data.get(base) or data.get(clip_name)
            if entry and "corners_px" in entry:
                pts = [(float(pt[0]), float(pt[1])) for pt in entry["corners_px"]]
                if target_size is not None and "resolution" in entry:
                    cal_res = entry["resolution"]
                    if isinstance(cal_res, (list, tuple)) and len(cal_res) == 2:
                        cal_w, cal_h = float(cal_res[0]), float(cal_res[1])
                        tw, th = float(target_size[0]), float(target_size[1])
                        if cal_w > 0 and cal_h > 0 and (abs(cal_w - tw) > 1.0 or abs(cal_h - th) > 1.0):
                            # Ensure matching orientation (both landscape or both portrait)
                            if (tw >= th) == (cal_w >= cal_h):
                                sx = tw / cal_w
                                sy = th / cal_h
                                pts = [(round(px * sx, 2), round(py * sy, 2)) for px, py in pts]
                return pts
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
        - Checks persistent calibration store first (with automatic resolution scaling)
        - Portrait (h > w): calibrated smartphone camera geometry
        - Landscape (w >= h): calibrated broadcast camera geometry
        """
        if clip_name:
            saved = cls.load_calibration_for_clip(clip_name, target_size=(width, height))
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

    @classmethod
    def refine_corners_with_court_lines(
        cls,
        frame: np.ndarray,
        init_corners: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Refines default court corners on uncalibrated clips by detecting high-contrast
        white baseline/sideline segments via HoughLinesP within a local search band.
        """
        if frame is None or frame.size == 0 or len(init_corners) != 4:
            return init_corners
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # White court lines have high luminance (V) and low saturation (S)
            white_mask = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
            masked_gray = cv2.bitwise_and(gray, gray, mask=white_mask)
            edges = cv2.Canny(masked_gray, 50, 150)

            min_line_len = max(30, int(0.08 * min(w, h)))
            lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, threshold=40, minLineLength=min_line_len, maxLineGap=15)
            if lines is None or len(lines) < 4:
                return init_corners

            far_y_init = 0.5 * (init_corners[0][1] + init_corners[1][1])
            near_y_init = 0.5 * (init_corners[2][1] + init_corners[3][1])
            tol_y = 0.06 * h

            far_ys = []
            near_ys = []
            for ln in lines[:, 0, :]:
                x1, y1, x2, y2 = float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])
                dx, dy = x2 - x1, y2 - y1
                length = float(np.hypot(dx, dy))
                if length < min_line_len:
                    continue
                angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
                # Horizontal baseline candidates (within 8 degrees of horizontal)
                if angle_deg < 8.0 or angle_deg > 172.0:
                    mid_y = 0.5 * (y1 + y2)
                    if abs(mid_y - far_y_init) < tol_y:
                        far_ys.append((mid_y, length))
                    elif abs(mid_y - near_y_init) < tol_y:
                        near_ys.append((mid_y, length))

            refined = list(init_corners)
            if far_ys:
                w_far_y = sum(y * l for y, l in far_ys) / max(1e-6, sum(l for _, l in far_ys))
                refined[0] = (refined[0][0], float(w_far_y))
                refined[1] = (refined[1][0], float(w_far_y))
            if near_ys:
                w_near_y = sum(y * l for y, l in near_ys) / max(1e-6, sum(l for _, l in near_ys))
                refined[2] = (refined[2][0], float(w_near_y))
                refined[3] = (refined[3][0], float(w_near_y))
            return refined
        except Exception:
            return init_corners

    def auto_calibrate_broadcast(self, frame: np.ndarray, clip_name: Optional[str] = None) -> bool:
        """
        Calibrate court perspective with 3D camera pose estimation:
        1. Checks stored human calibration in court_calibrations.json (auto-scaled to frame size)
        2. Applies aspect-ratio-aware defaults + white-line refinement for unseen clips
        3. Solves 3D camera pose for height-compensated projection
        """
        h, w = frame.shape[:2]
        has_saved = bool(clip_name and self.load_calibration_for_clip(clip_name, target_size=(w, h)) is not None)
        corners = self.get_recommended_corners(w, h, clip_name=clip_name)
        if not has_saved:
            corners = self.refine_corners_with_court_lines(frame, corners)
        return self.calibrate(corners, image_size=(w, h))
