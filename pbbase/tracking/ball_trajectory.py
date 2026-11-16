"""
ball_trajectory.py - Kinematic Gating, Ball Trajectory Smoothing & Parabolic Interpolation.
Zero scoring or line-call rules: strictly physics-based kinematic tracking.
Includes close-player sneaker/apparel decoy suppression, tight-gate low-conf blur rescue,
and paddle-box coast damping.
"""
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from pbbase.court.homography import CourtCalibrator


class BallTrajectoryTracker:
    """
    Trajectory tracker for fast-moving small pickleball:
    - Kinematic velocity gating with physical momentum prediction
    - Player-box HSV decoy suppression (rejects white sneakers/apparel false detections)
    - Tight-gate low-confidence rescue for motion-blurred paddle impacts
    - Player-box coast damping & ballistic occlusion coasting
    - Retroactive parabolic arc interpolation across gap buffers
    - Metric speed estimation via court planar homography
    """
    def __init__(
        self,
        max_jump_px: float = 85.0,
        trail_len: int = 16,
        max_coast_frames: int = 3,
        gravity_px: float = 0.55,
        drag_coeff: float = 0.015,
        min_init_conf: float = 0.20,
        min_active_conf: float = 0.14,
        min_gated_conf: float = 0.08,
        tight_gate_px: float = 38.0,
    ):
        self.max_jump_px = max_jump_px
        self.trail_len = trail_len
        self.max_coast_frames = max_coast_frames
        self.gravity_px = gravity_px
        self.drag_coeff = drag_coeff
        self.min_init_conf = min_init_conf
        self.min_active_conf = min_active_conf
        self.min_gated_conf = min_gated_conf
        self.tight_gate_px = tight_gate_px

        self.last_pos: Optional[Dict[str, Any]] = None
        self.last_observed_pos: Optional[Dict[str, Any]] = None
        self.velocity: List[float] = [0.0, 0.0]
        self.consecutive_lost: int = 0
        self.coast_near_player: bool = False
        self.gap_buffer: List[Dict[str, Any]] = []
        self.raw_history: List[Optional[Dict[str, Any]]] = []

    def reset(self):
        self.last_pos = None
        self.last_observed_pos = None
        self.velocity = [0.0, 0.0]
        self.consecutive_lost = 0
        self.coast_near_player = False
        self.gap_buffer.clear()
        self.raw_history.clear()

    @staticmethod
    def _is_inside_player_box(
        px: float,
        py: float,
        players: Optional[List[Dict[str, Any]]],
        margin_x: float = 22.0,
        margin_y: float = 8.0,
    ) -> bool:
        """Checks if (px, py) lies inside or adjacent to any active player's bounding box."""
        if not players:
            return False
        for p in players:
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            bx1, by1, bx2, by2 = box
            if (bx1 - margin_x) <= px <= (bx2 + margin_x) and (by1 - margin_y) <= py <= (by2 + margin_y):
                return True
        return False

    @staticmethod
    def _is_achromatic_player_decoy(
        frame: Optional[np.ndarray],
        cx: float,
        cy: float,
        players: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """
        Detects white sneakers, socks, wristbands, or shirt reflections misclassified
        as pickleballs when candidates overlap a player's bounding box.
        Real optic pickleballs exhibit clear yellow-green chromatic contrast (G - B >= 18, S >= 30);
        white apparel decoys are achromatic (low HSV Saturation, G - B near 0).
        """
        if frame is None or frame.size == 0 or not players:
            return False

        in_shoe_zone = False
        in_any_box = False
        for p in players:
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            bx1, by1, bx2, by2 = box
            bh = max(20.0, by2 - by1)
            if (bx1 - 15.0) <= cx <= (bx2 + 15.0) and (by1 - 5.0) <= cy <= (by2 + 10.0):
                in_any_box = True
                if cy >= (by2 - 0.24 * bh):
                    in_shoe_zone = True

        if not in_any_box:
            return False

        h, w = frame.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))
        x1 = max(0, ix - 4)
        x2 = min(w, ix + 5)
        y1 = max(0, iy - 4)
        y2 = min(h, iy + 5)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return False

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        v_flat = hsv[:, :, 2].flatten()
        thresh = np.percentile(v_flat, 70)
        core_mask = hsv[:, :, 2] >= thresh
        if not np.any(core_mask):
            return False

        b_mean = float(np.mean(patch[:, :, 0][core_mask]))
        g_mean = float(np.mean(patch[:, :, 1][core_mask]))
        s_mean = float(np.mean(hsv[:, :, 1][core_mask]))
        gb_diff = g_mean - b_mean

        # Pure white sneaker in bottom 24% foot zone
        if in_shoe_zone and s_mean < 38.0 and gb_diff < 25.0:
            return True

        # Pure white wristband/shirt spot anywhere inside player box
        if s_mean < 28.0 and gb_diff < 10.0:
            return True

        return False

    def _apply_retroactive_parabolic_fit(
        self,
        end_pt: Dict[str, Any],
        court_calibrator: Optional[CourtCalibrator] = None,
    ):
        """
        Retroactively refines coasted points in gap_buffer by fitting a ballistic
        parabolic polynomial between last_observed_pos and newly observed end_pt.
        Also recomputes metric court_ft via homography H so no stale coasted
        coordinates remain in telemetry or on the minimap.
        """
        if not self.gap_buffer or self.last_observed_pos is None:
            self.gap_buffer.clear()
            return

        x0, y0 = self.last_observed_pos["pixel"]
        f0 = self.last_observed_pos["frame_idx"]
        x1, y1 = end_pt["pixel"]
        f1 = end_pt["frame_idx"]
        total_frames = f1 - f0

        if total_frames <= 1:
            self.gap_buffer.clear()
            return

        vy0 = self.velocity[1]
        # Ballistic acceleration: y1 = y0 + vy0 * N + 0.5 * a_y * N^2
        ay = 2.0 * ((y1 - y0) - vy0 * total_frames) / (total_frames * total_frames)

        # Fallback to linear if acceleration is implausibly sharp (|ay| > 12 px/f^2)
        use_linear = abs(ay) > 12.0

        for pt in self.gap_buffer:
            k = pt["frame_idx"] - f0
            frac = k / float(total_frames)
            xk = x0 + frac * (x1 - x0)
            if use_linear:
                yk = y0 + frac * (y1 - y0)
            else:
                yk = y0 + vy0 * k + 0.5 * ay * (k * k)

            pt["pixel"] = (round(float(xk), 1), round(float(yk), 1))
            pt["box"] = [
                round(float(xk - 6.0), 1),
                round(float(yk - 6.0), 1),
                round(float(xk + 6.0), 1),
                round(float(yk + 6.0), 1),
            ]
            if court_calibrator is not None:
                cft = court_calibrator.pixel_to_court_ft(xk, yk)
                pt["court_ft"] = (round(cft[0], 2), round(cft[1], 2)) if cft else None
            pt["retro_parabolic_refined"] = True

        self.gap_buffer.clear()

    def process_candidates(
        self,
        candidates: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        court_calibrator: CourtCalibrator,
        frame: Optional[np.ndarray] = None,
        players: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Associates candidates with ongoing trajectory using kinematic velocity prediction.
        Suppresses white sneaker/apparel false detections near players, performs blurred
        paddle-approach rescue inside tight gates, and dampens coasting near players.
        """
        # Filter out obvious white sneaker / apparel decoys inside player bounding boxes
        clean_cands = []
        for c in (candidates or []):
            cx, cy = c["center_px"]
            if not self._is_achromatic_player_decoy(frame, cx, cy, players):
                clean_cands.append(c)

        # Case 1: Initial track start (no active track)
        if self.last_pos is None:
            valid_cands = [c for c in clean_cands if c.get("conf", 0.0) >= self.min_init_conf]
            if not valid_cands:
                valid_cands = [c for c in clean_cands if c.get("conf", 0.0) >= self.min_active_conf]

            if not valid_cands:
                self.raw_history.append(None)
                return None

            best = max(valid_cands, key=lambda c: c["conf"])
            cx, cy = best["center_px"]
            court_ft = court_calibrator.pixel_to_court_ft(cx, cy)
            point_data = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "pixel": (round(cx, 1), round(cy, 1)),
                "box": [round(float(v), 1) for v in best["box"]],
                "conf": round(float(best["conf"]), 3),
                "speed_mph": 0.0,
                "speed_px_per_frame": 0.0,
                "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                "interpolated": False,
            }
            self.last_pos = point_data
            self.last_observed_pos = point_data
            self.velocity = [0.0, 0.0]
            self.consecutive_lost = 0
            self.coast_near_player = False
            self.gap_buffer.clear()
            self.raw_history.append(point_data)
            return point_data

        # Case 2: Track is currently active -> Predict next position
        pred_x = self.last_pos["pixel"][0] + self.velocity[0]
        pred_y = self.last_pos["pixel"][1] + self.velocity[1] + 0.5 * self.gravity_px

        # Expand gate moderately when recovering immediately after a clamped player-paddle coast
        gate_limit = max(self.max_jump_px, 130.0) if (self.consecutive_lost > 0 and self.coast_near_player) else self.max_jump_px

        best_cand = None
        best_cost = float("inf")

        for c in clean_cands:
            cx, cy = c["center_px"]
            conf = float(c.get("conf", 0.0))
            dist = float(np.hypot(cx - pred_x, cy - pred_y))

            # Confidence tiering: very low conf (0.08-0.14) accepted ONLY inside tight prediction gate
            if conf < self.min_active_conf:
                if conf < self.min_gated_conf or dist > self.tight_gate_px:
                    continue

            if dist < gate_limit:
                cost = dist - 25.0 * conf
                if cost < best_cost:
                    best_cost = cost
                    best_cand = c

        # Subcase 2A: Candidate matched within kinematic gate
        if best_cand is not None:
            cx, cy = best_cand["center_px"]
            vx = cx - self.last_pos["pixel"][0]
            vy = cy - self.last_pos["pixel"][1]
            px_speed = float(np.hypot(vx, vy))

            court_ft = court_calibrator.pixel_to_court_ft(cx, cy)
            speed_mph = 0.0
            if court_ft and self.last_pos.get("court_ft"):
                prev_c = self.last_pos["court_ft"]
                dt = timestamp_s - self.last_pos["timestamp_s"]
                if 0.001 < dt < 0.5:
                    dist_ft = float(np.hypot(court_ft[0] - prev_c[0], court_ft[1] - prev_c[1]))
                    speed_mph = float(np.clip((dist_ft / dt) * 0.681818, 0.0, 75.0))

            point_data = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "pixel": (round(cx, 1), round(cy, 1)),
                "box": [round(float(v), 1) for v in best_cand["box"]],
                "conf": round(float(best_cand["conf"]), 3),
                "speed_mph": round(speed_mph, 1),
                "speed_px_per_frame": round(px_speed, 1),
                "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                "interpolated": False,
            }

            if self.consecutive_lost > 0:
                self._apply_retroactive_parabolic_fit(point_data, court_calibrator=court_calibrator)

            self.velocity = [0.75 * vx + 0.25 * self.velocity[0], 0.75 * vy + 0.25 * self.velocity[1]]
            self.consecutive_lost = 0
            self.coast_near_player = False
            self.last_pos = point_data
            self.last_observed_pos = point_data
            self.raw_history.append(point_data)
            return point_data

        # Subcase 2B: Ball not detected -> Ballistic occlusion coasting
        active_speed = float(np.hypot(self.velocity[0], self.velocity[1]))
        can_coast = (
            self.consecutive_lost < self.max_coast_frames
            and (active_speed > 1.0 or self.consecutive_lost > 0)
        )

        if can_coast:
            self.consecutive_lost += 1

            # Check if ball is entering/inside a player's hitting zone -> dampen coasting to prevent overshoot
            in_player = self._is_inside_player_box(pred_x, pred_y, players) or self._is_inside_player_box(
                self.last_pos["pixel"][0], self.last_pos["pixel"][1], players
            )
            if in_player:
                self.coast_near_player = True
                self.velocity[0] *= 0.30
                self.velocity[1] *= 0.30
            else:
                self.velocity[0] *= (1.0 - self.drag_coeff)
                self.velocity[1] += self.gravity_px

            cx = self.last_pos["pixel"][0] + self.velocity[0]
            cy = self.last_pos["pixel"][1] + self.velocity[1]
            px_speed = float(np.hypot(self.velocity[0], self.velocity[1]))

            court_ft = court_calibrator.pixel_to_court_ft(cx, cy)
            coasted_conf = round(max(0.05, self.last_pos.get("conf", 0.3) * 0.85), 3)
            box_half = 6.0
            coasted_box = [cx - box_half, cy - box_half, cx + box_half, cy + box_half]

            point_data = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "pixel": (round(cx, 1), round(cy, 1)),
                "box": [round(float(v), 1) for v in coasted_box],
                "conf": coasted_conf,
                "speed_mph": self.last_pos.get("speed_mph", 0.0),
                "speed_px_per_frame": round(px_speed, 1),
                "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                "interpolated": True,
                "coast_frame": self.consecutive_lost,
            }

            self.last_pos = point_data
            self.gap_buffer.append(point_data)
            self.raw_history.append(point_data)
            return point_data

        # Subcase 2C: Occlusion exceeded max_coast_frames -> clean track drop
        self.reset()
        self.raw_history.append(None)
        return None


def get_speed_color(speed_px_per_frame: float) -> Tuple[int, int, int]:
    """
    Returns BGR color code based on ball physical pixel speed:
    - Fast Smash (> 22 px/f): Electric Orange (20, 120, 255)
    - Rally Play (10 - 22 px/f): Chartreuse Laser (30, 245, 180)
    - Dink / Drop (< 10 px/f): Ice Cyan (255, 230, 40)
    """
    if speed_px_per_frame >= 22.0:
        return (20, 120, 255)
    elif speed_px_per_frame >= 10.0:
        t = (speed_px_per_frame - 10.0) / 12.0
        b = int(30 * (1 - t) + 20 * t)
        g = int(245 * (1 - t) + 120 * t)
        r = int(180 * (1 - t) + 255 * t)
        return (b, g, r)
    else:
        t = max(0.0, speed_px_per_frame) / 10.0
        b = int(255 * (1 - t) + 30 * t)
        g = int(230 * (1 - t) + 245 * t)
        r = int(40 * (1 - t) + 180 * t)
        return (b, g, r)
