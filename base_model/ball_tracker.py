"""
ball_tracker.py - Small-Ball Detection, Trajectory Filtering & Bounce Estimation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO

from . import BallTelemetry
from .court_detector import CourtDetector


class BallTracker:
    """
    High-recall small pickleball detection, trajectory smoothing,
    court projection, velocity estimation, and bounce candidate extraction.
    """
    def __init__(self, model_path: str = "weights/ball_best.pt", conf_thresh: float = 0.20):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.trajectory_history: List[Tuple[int, float, float, float]] = []  # (frame_idx, px_x, px_y, timestamp)
        self.bounces: List[Dict[str, Any]] = []

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        court_detector: CourtDetector,
    ) -> Optional[BallTelemetry]:
        # Inference with custom ball detector (class 1 = ball)
        results = self.model.predict(
            frame,
            classes=[1],
            conf=self.conf_thresh,
            verbose=False,
            imgsz=640,
        )[0]

        best_det: Optional[Tuple[List[float], float, Tuple[float, float]]] = None
        highest_conf = 0.0

        if len(results.boxes) > 0:
            for box, conf in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist()):
                c = float(conf)
                # Compute center pixel
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                
                # Filter out absurdly huge boxes (pickleball is compact)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                if bw > 80 or bh > 80:
                    continue

                if c > highest_conf:
                    highest_conf = c
                    best_det = (box, c, (cx, cy))

        if best_det is not None:
            box, conf, center = best_det
            self.trajectory_history.append((frame_idx, center[0], center[1], timestamp_s))
            court_pos = court_detector.pixel_to_court(center[0], center[1])
            speed_mph = self._compute_speed(court_pos, timestamp_s)
            is_bounce = self._check_bounce_inflection(frame_idx, center, court_pos, timestamp_s)
            is_kitchen_bounce = bool(is_bounce and court_pos and (15.0 <= court_pos[1] <= 29.0))

            return BallTelemetry(
                detected=True,
                center_pixel=center,
                bbox_xyxy=box,
                confidence=conf,
                court_pos_ft=court_pos,
                speed_mph=speed_mph,
                interpolated=False,
                is_bounce_candidate=is_bounce,
                is_kitchen_bounce=is_kitchen_bounce,
            )

        # Occlusion / motion-blur interpolation from recent trajectory
        if len(self.trajectory_history) >= 2:
            prev_f, px, py, pt = self.trajectory_history[-1]
            if frame_idx - prev_f <= 3:  # bridge up to 3 missed frames
                prev2_f, px2, py2, pt2 = self.trajectory_history[-2]
                dt_frames = frame_idx - prev_f
                vx = (px - px2) / max(1, prev_f - prev2_f)
                vy = (py - py2) / max(1, prev_f - prev2_f)
                interp_x = px + vx * dt_frames
                interp_y = py + vy * dt_frames

                court_pos = court_detector.pixel_to_court(interp_x, interp_y)
                speed_mph = self._compute_speed(court_pos, timestamp_s)

                return BallTelemetry(
                    detected=False,
                    center_pixel=(interp_x, interp_y),
                    bbox_xyxy=[interp_x - 6, interp_y - 6, interp_x + 6, interp_y + 6],
                    confidence=0.15,
                    court_pos_ft=court_pos,
                    speed_mph=speed_mph,
                    interpolated=True,
                    is_bounce_candidate=False,
                    is_kitchen_bounce=False,
                )

        return None

    def _compute_speed(self, current_court_pos: Optional[Tuple[float, float]], timestamp_s: float) -> float:
        if current_court_pos is None or len(self.trajectory_history) < 2:
            return 0.0
        _, px, py, t_prev = self.trajectory_history[-1]
        dt = timestamp_s - t_prev
        if dt <= 0.001 or dt > 0.5:
            return 0.0
        # Estimate displacement in ft
        prev_court = self.trajectory_history[-2] if len(self.trajectory_history) >= 2 else None
        if not prev_court:
            return 0.0
        dist_ft = float(np.hypot(current_court_pos[0] - prev_court[1], current_court_pos[1] - prev_court[2]))
        speed_ft_s = dist_ft / dt
        return float(np.clip(speed_ft_s * 0.681818, 0.0, 75.0))

    def _check_bounce_inflection(
        self,
        frame_idx: int,
        center_px: Tuple[float, float],
        court_pos: Optional[Tuple[float, float]],
        timestamp_s: float,
    ) -> bool:
        """Detect local vertical velocity inflection point indicating ground contact."""
        if len(self.trajectory_history) < 5:
            return False
        recent = self.trajectory_history[-5:]
        ys = [p[2] for p in recent]
        # In image coordinates, y increases downward.
        # A downward bounce contact has maximum y in image space (lowest vertical point on screen).
        if ys[2] > ys[0] and ys[2] > ys[1] and ys[2] >= ys[3] and ys[2] >= ys[4]:
            if court_pos:
                is_kitchen = 15.0 <= court_pos[1] <= 29.0
                bounce_info = {
                    "frame_idx": frame_idx - 2,
                    "timestamp_s": round(timestamp_s, 3),
                    "pixel": (round(center_px[0], 1), round(center_px[1], 1)),
                    "court_ft": (round(court_pos[0], 2), round(court_pos[1], 2)),
                    "is_kitchen": is_kitchen,
                    "zone": "Kitchen (NVZ)" if is_kitchen else "Out/Court",
                }
                # Prevent duplicate bounce logging within 12 frames
                if not self.bounces or (frame_idx - self.bounces[-1]["frame_idx"]) > 12:
                    self.bounces.append(bounce_info)
                    return True
        return False
