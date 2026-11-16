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
    - Ballistic occlusion coasting bridging brief occlusions (1-4 frames)
    - Retroactive parabolic arc interpolation across gap buffers
    - Moving average position & velocity smoothing
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
    ):
        self.max_jump_px = max_jump_px
        self.trail_len = trail_len
        self.max_coast_frames = max_coast_frames
        self.gravity_px = gravity_px
        self.drag_coeff = drag_coeff
        self.min_init_conf = min_init_conf

        self.last_pos: Optional[Dict[str, Any]] = None
        self.last_observed_pos: Optional[Dict[str, Any]] = None
        self.velocity: List[float] = [0.0, 0.0]
        self.consecutive_lost: int = 0
        self.gap_buffer: List[Dict[str, Any]] = []
        self.raw_history: List[Optional[Dict[str, Any]]] = []

    def reset(self):
        self.last_pos = None
        self.last_observed_pos = None
        self.velocity = [0.0, 0.0]
        self.consecutive_lost = 0
        self.gap_buffer.clear()
        self.raw_history.clear()

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
        # a_y = 2 * (y1 - y0 - vy0 * N) / N^2
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
    ) -> Optional[Dict[str, Any]]:
        """
        Associates candidates with ongoing trajectory using kinematic velocity prediction.
        Performs ballistic occlusion coasting and retroactive parabolic interpolation.
        """
        # Case 1: Initial track start (no active track)
        if self.last_pos is None:
            if not candidates:
                self.raw_history.append(None)
                return None

            valid_cands = [c for c in candidates if c.get("conf", 0.0) >= self.min_init_conf]
            if not valid_cands:
                valid_cands = candidates

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
            self.gap_buffer.clear()
            self.raw_history.append(point_data)
            return point_data

        # Case 2: Track is currently active -> Predict next position
        pred_x = self.last_pos["pixel"][0] + self.velocity[0]
        pred_y = self.last_pos["pixel"][1] + self.velocity[1] + 0.5 * self.gravity_px

        best_cand = None
        best_cost = float("inf")

        if candidates:
            for c in candidates:
                cx, cy = c["center_px"]
                dist = float(np.hypot(cx - pred_x, cy - pred_y))
                # Gating threshold: reject impossible teleport jumps
                if dist < self.max_jump_px:
                    cost = dist - 25.0 * c.get("conf", 0.0)
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

            # If recovering from brief gap, apply retroactive parabolic fit
            if self.consecutive_lost > 0:
                self._apply_retroactive_parabolic_fit(point_data, court_calibrator=court_calibrator)

            # Update momentum with exponential moving average
            self.velocity = [0.75 * vx + 0.25 * self.velocity[0], 0.75 * vy + 0.25 * self.velocity[1]]
            self.consecutive_lost = 0
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
            # Apply aerodynamic drag and vertical gravity
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
