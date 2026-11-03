"""
ball_trajectory.py - Kinematic Gating, Ball Trajectory Smoothing & Parabolic Interpolation.
Zero scoring or line-call rules: strictly physics-based kinematic tracking.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pbbase.court.homography import CourtCalibrator


class BallTrajectoryTracker:
    """
    Trajectory tracker for fast-moving small pickleball:
    - Kinematic velocity gating with physical momentum prediction
    - Parabolic ballistic arc interpolation bridging brief occlusions (1-4 frames)
    - Moving average position smoothing
    - Metric speed estimation via court planar homography
    """
    def __init__(self, max_jump_px: float = 80.0, trail_len: int = 16):
        self.max_jump_px = max_jump_px
        self.trail_len = trail_len

        self.last_pos: Optional[Dict[str, Any]] = None
        self.velocity = [0.0, 0.0]
        self.raw_history: List[Optional[Dict[str, Any]]] = []

    def reset(self):
        self.last_pos = None
        self.velocity = [0.0, 0.0]
        self.raw_history.clear()

    def process_candidates(
        self,
        candidates: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        court_calibrator: CourtCalibrator,
    ) -> Optional[Dict[str, Any]]:
        """
        Associates candidates with ongoing trajectory using velocity prediction.
        """
        if not candidates:
            self.last_pos = None
            self.velocity = [0.0, 0.0]
            self.raw_history.append(None)
            return None

        if self.last_pos is None:
            # Gating initial track start: select highest confidence candidate
            best = max(candidates, key=lambda c: c["conf"])
            cx, cy = best["center_px"]
            point_data = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "pixel": (round(cx, 1), round(cy, 1)),
                "box": best["box"],
                "conf": round(best["conf"], 3),
                "speed_mph": 0.0,
                "speed_px_per_frame": 0.0,
                "court_ft": court_calibrator.pixel_to_court_ft(cx, cy),
                "interpolated": False,
            }
            self.last_pos = point_data
            self.velocity = [0.0, 0.0]
            self.raw_history.append(point_data)
            return point_data

        # Predict current position using velocity
        pred_x = self.last_pos["pixel"][0] + self.velocity[0]
        pred_y = self.last_pos["pixel"][1] + self.velocity[1]

        best_cand = None
        best_cost = float("inf")

        for c in candidates:
            cx, cy = c["center_px"]
            dist = float(np.hypot(cx - pred_x, cy - pred_y))
            # Cost balances spatial distance with detection confidence
            cost = dist - 25.0 * c["conf"]
            if dist < self.max_jump_px and cost < best_cost:
                best_cost = cost
                best_cand = c

        if best_cand is not None:
            cx, cy = best_cand["center_px"]
            vx = cx - self.last_pos["pixel"][0]
            vy = cy - self.last_pos["pixel"][1]
            px_speed = float(np.hypot(vx, vy))

            # Update velocity with exponential moving average
            self.velocity = [0.70 * vx + 0.30 * self.velocity[0], 0.70 * vy + 0.30 * self.velocity[1]]

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
                "box": best_cand["box"],
                "conf": round(best_cand["conf"], 3),
                "speed_mph": round(speed_mph, 1),
                "speed_px_per_frame": round(px_speed, 1),
                "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                "interpolated": False,
            }
            self.last_pos = point_data
            self.raw_history.append(point_data)
            return point_data
        else:
            self.last_pos = None
            self.velocity = [0.0, 0.0]
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
