"""
bounce_detector.py - Kinematic Vertical Derivative Filter & Sub-Pixel Parabolic Nadir Fitting.
Zero referee or scoring rules: strictly physics-based detection of court ground impacts,
calculating continuous sub-pixel coordinates (u*, v*) and metric court coordinates (X*, Y*) in feet.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pbbase.court.homography import CourtCalibrator, CourtGeometry


@dataclass
class BounceEvent:
    """Represents a validated court bounce or ball impact event."""
    frame_idx: int
    timestamp_s: float
    sub_frame_offset: float          # Sub-frame offset t* in [-1.5, 1.5]
    pixel_xy: Tuple[float, float]    # Sub-pixel (u*, v*) nadir
    court_xy_ft: Tuple[float, float] # Projected court coordinates (X*, Y*) in feet
    is_ground_bounce: bool           # True = court ground impact; False = airborne paddle volley
    confidence: float                # Detection confidence [0.0, 1.0]
    vertical_accel: float            # d²y/dt² vertical acceleration peak (px/frame²)
    speed_mph: float = 0.0           # Ball speed at impact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_s": round(float(self.timestamp_s), 4),
            "sub_frame_offset": round(float(self.sub_frame_offset), 3),
            "pixel": [round(float(self.pixel_xy[0]), 2), round(float(self.pixel_xy[1]), 2)],
            "court_ft": [round(float(self.court_xy_ft[0]), 2), round(float(self.court_xy_ft[1]), 2)],
            "is_ground_bounce": bool(self.is_ground_bounce),
            "confidence": round(float(self.confidence), 3),
            "vertical_accel": round(float(self.vertical_accel), 2),
            "speed_mph": round(float(self.speed_mph), 1),
        }


class BallBounceDetector:
    """
    Kinematic bounce detector analyzing vertical trajectory acceleration (d²y/dt²),
    inflection sign reversal, parabolic sub-pixel nadir localization, and court ground
    plane disambiguation.
    """
    def __init__(
        self,
        min_accel_thresh: float = 2.0,      # Minimum upward acceleration magnitude (|a_y|) in px/frame²
        window_size: int = 5,               # Parabolic fit window size (must be 5)
        min_frames_between_bounces: int = 6,# Debounce cooldown refractory period
        max_subframe_offset: float = 1.3,   # Max allowable |t*| offset from center frame
    ):
        self.min_accel_thresh = min_accel_thresh
        self.window_size = window_size
        self.min_frames_between_bounces = min_frames_between_bounces
        self.max_subframe_offset = max_subframe_offset

        # Sliding window buffer: stores recent ball points
        self.buffer: List[Dict[str, Any]] = []
        self.last_bounce_frame: int = -999
        self.detected_bounces: List[BounceEvent] = []

    def reset(self):
        """Reset internal buffers and history."""
        self.buffer.clear()
        self.last_bounce_frame = -999
        self.detected_bounces.clear()

    @staticmethod
    def fit_parabola_5pt(ys: List[float]) -> Tuple[float, float, float]:
        """
        Fast closed-form least-squares quadratic fit for 5 uniformly-spaced points
        at t in {-2, -1, 0, 1, 2}.
        Fits: y(t) = a * t^2 + b * t + c.
        Returns: (a, b, c).
        """
        y_m2, y_m1, y_0, y_p1, y_p2 = ys
        a = (2.0 * (y_m2 + y_p2) - (y_m1 + y_p1) - 2.0 * y_0) / 14.0
        b = (-2.0 * y_m2 - y_m1 + y_p1 + 2.0 * y_p2) / 10.0
        c = (-3.0 * (y_m2 + y_p2) + 12.0 * (y_m1 + y_p1) + 17.0 * y_0) / 35.0
        return a, b, c

    @staticmethod
    def fit_linear_5pt(xs: List[float]) -> Tuple[float, float]:
        """
        Fast closed-form least-squares linear fit for 5 uniformly-spaced points
        at t in {-2, -1, 0, 1, 2}.
        Fits: x(t) = m * t + c.
        Returns: (m, c).
        """
        x_m2, x_m1, x_0, x_p1, x_p2 = xs
        m = (-2.0 * x_m2 - x_m1 + x_p1 + 2.0 * x_p2) / 10.0
        c = (x_m2 + x_m1 + x_0 + x_p1 + x_p2) / 5.0
        return m, c

    def _is_paddle_volley(
        self,
        pixel_xy: Tuple[float, float],
        court_xy_ft: Tuple[float, float],
        players: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Disambiguates ground bounce from paddle volley:
        - Ground bounces occur at court ground level ($Z \approx 0$).
        - Paddle volleys occur in mid-air near player torso/paddle height ($Z > 2.0 ft$).
        Returns True if likely a paddle hit; False if likely a court ground bounce.
        """
        if not players:
            return False

        px, py = pixel_xy

        for p in players:
            box = p.get("box")  # [x1, y1, x2, y2]
            if not box or len(box) < 4:
                continue

            bx1, by1, bx2, by2 = box
            player_h = by2 - by1

            # Player foot is near by2; waist/chest is in upper 70% of box
            # If the ball is inside the player's upper box and above the feet by > 25% of height
            in_x = (bx1 - 25.0) <= px <= (bx2 + 25.0)
            in_y_torso = by1 <= py <= (by2 - 0.25 * player_h)

            if in_x and in_y_torso:
                # Ball contact happened at torso/shoulder height of player -> Paddle Volley
                return True

        return False

    def update(
        self,
        ball: Optional[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        court_calibrator: CourtCalibrator,
        players: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[BounceEvent]:
        """
        Streams a new ball detection frame. When a local 5-frame window has passed,
        evaluates whether the center frame was a physical court bounce.
        """
        # Store point in buffer
        if ball is not None and "pixel" in ball:
            item = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "pixel": ball["pixel"],
                "speed_mph": ball.get("speed_mph", 0.0),
                "conf": ball.get("conf", 1.0),
                "interpolated": ball.get("interpolated", False),
                "players": players,
            }
        else:
            item = None

        self.buffer.append(item)

        # Keep buffer bounded to window size + 4
        if len(self.buffer) > 15:
            self.buffer.pop(0)

        # Need at least 5 frames in buffer to evaluate the center point (index -3)
        if len(self.buffer) < 5:
            return None

        # Check candidate window: buffer[-5:]
        win = self.buffer[-5:]
        if any(pt is None for pt in win):
            return None

        # Verify consecutive frame indices
        frames = [pt["frame_idx"] for pt in win]
        if frames != list(range(frames[0], frames[0] + 5)):
            return None

        center_idx = frames[2]

        # Debounce check
        if (center_idx - self.last_bounce_frame) < self.min_frames_between_bounces:
            return None

        ys = [float(pt["pixel"][1]) for pt in win]
        xs = [float(pt["pixel"][0]) for pt in win]

        # In image coordinates, y increases downward.
        # Approaching ground: y increases -> v_y = y[t] - y[t-1] > 0
        # Rebounding up: y decreases -> v_y = y[t+1] - y[t] < 0
        # Local nadir: y[2] >= y[1] and y[2] >= y[3] (or y[2] is local maximum)
        is_local_max_y = (ys[2] >= ys[1]) and (ys[2] >= ys[3])

        # Vertical velocity change across center
        vy_in = ys[2] - ys[0]    # positive if downward
        vy_out = ys[4] - ys[2]   # negative if upward

        velocity_reversal = (vy_in > 0.5) and (vy_out < -0.5)

        if not (is_local_max_y or velocity_reversal):
            return None

        # Closed-form 5-point quadratic fit: y(t) = a * t^2 + b * t + c
        a, b, c = self.fit_parabola_5pt(ys)

        # Downward concave parabola has a < 0 (peak at t* = -b / (2a))
        if a >= -0.15:
            # Curvature not sharply upward after collision
            return None

        # Compute continuous sub-frame vertex offset t*
        t_star = -b / (2.0 * a)

        if abs(t_star) > self.max_subframe_offset:
            return None

        # Sub-pixel nadir coordinates
        sub_y = a * (t_star ** 2) + b * t_star + c
        mx, cx = self.fit_linear_5pt(xs)
        sub_x = mx * t_star + cx

        # Vertical acceleration magnitude: d²y/dt² = 2*a
        vert_accel = abs(2.0 * a)
        if vert_accel < self.min_accel_thresh:
            return None

        # Project sub-pixel point to court coordinates
        court_xy = court_calibrator.pixel_to_court_ft(sub_x, sub_y)
        if court_xy is None:
            return None

        # Disambiguate ground bounce vs paddle volley
        center_players = win[2].get("players")
        is_paddle = self._is_paddle_volley((sub_x, sub_y), court_xy, center_players)
        is_ground = not is_paddle

        # Confidence metric based on curvature strength and fit quality
        curvature_score = min(1.0, vert_accel / 8.0)
        offset_penalty = max(0.0, 1.0 - abs(t_star) / self.max_subframe_offset)
        conf = float(np.clip(0.5 * curvature_score + 0.5 * offset_penalty, 0.2, 0.99))

        center_pt = win[2]
        center_ts = center_pt["timestamp_s"] + (t_star / 30.0)

        event = BounceEvent(
            frame_idx=center_idx,
            timestamp_s=center_ts,
            sub_frame_offset=t_star,
            pixel_xy=(sub_x, sub_y),
            court_xy_ft=court_xy,
            is_ground_bounce=is_ground,
            confidence=conf,
            vertical_accel=vert_accel,
            speed_mph=center_pt.get("speed_mph", 0.0),
        )

        self.last_bounce_frame = center_idx
        self.detected_bounces.append(event)
        return event

    def detect_bounces_in_trajectory(
        self,
        trajectory: List[Optional[Dict[str, Any]]],
        court_calibrator: CourtCalibrator,
        players_per_frame: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[BounceEvent]:
        """
        Batch-processes an entire trajectory stream and returns all detected bounce events.
        """
        self.reset()
        bounces: List[BounceEvent] = []

        for idx, pt in enumerate(trajectory):
            ts = pt.get("timestamp_s", idx / 30.0) if pt else (idx / 30.0)
            players = players_per_frame[idx] if (players_per_frame and idx < len(players_per_frame)) else None
            ev = self.update(pt, idx, ts, court_calibrator, players=players)
            if ev is not None:
                bounces.append(ev)

        return bounces
