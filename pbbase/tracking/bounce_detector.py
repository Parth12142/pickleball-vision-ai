"""
bounce_detector.py - Kinematic Vertical Derivative Filter & Sub-Pixel Parabolic Nadir Fitting.
Zero referee or scoring rules: strictly physics-based detection of court ground impacts,
calculating continuous sub-pixel coordinates (u*, v*) and metric court coordinates (X*, Y*) in feet.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from scipy.signal import find_peaks, savgol_filter

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
        scale_px: float = 1.0,
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
        shoe_margin_x = 35.0 * scale_px
        torso_margin_x = 48.0 * scale_px

        # First check if impact is right at floor level beside any player's shoes
        for p in players:
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            bx1, by1, bx2, by2 = box
            player_h = max(20.0 * scale_px, by2 - by1)
            if (bx1 - shoe_margin_x) <= px <= (bx2 + shoe_margin_x) and (by2 - 0.18 * player_h) <= py <= (by2 + 12.0 * scale_px):
                # Floor-level impact next to player's feet -> Ground Bounce
                return False

        for p in players:
            box = p.get("box")  # [x1, y1, x2, y2]
            if not box or len(box) < 4:
                continue

            bx1, by1, bx2, by2 = box
            player_h = max(20.0 * scale_px, by2 - by1)

            # Torso/chest + outstretched paddle reach zone (excludes top 5% head edge to prevent foreground overlap)
            in_x = (bx1 - torso_margin_x) <= px <= (bx2 + torso_margin_x)
            in_y_torso = (by1 + 0.05 * player_h) <= py <= (by2 - 0.20 * player_h)

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

        # Estimate effective FPS and court pixel scale for invariant kinematic thresholds
        dt_span = float(win[4]["timestamp_s"] - win[0]["timestamp_s"])
        eff_fps = float(np.clip(4.0 / dt_span, 15.0, 120.0)) if dt_span > 1e-3 else 30.0
        fps_ratio = eff_fps / 30.0
        fps_accel_scale = fps_ratio ** 2
        scale_px = court_calibrator.get_court_pixel_scale() if court_calibrator is not None else 1.0
        norm_px_scale = max(0.42, scale_px)

        # Debounce check (scaled by FPS so refractory period is ~0.20s regardless of 30/60fps)
        eff_min_frames = max(4, int(round(self.min_frames_between_bounces * fps_ratio)))
        if (center_idx - self.last_bounce_frame) < eff_min_frames:
            return None

        ys = [float(pt["pixel"][1]) for pt in win]
        xs = [float(pt["pixel"][0]) for pt in win]

        # In image coordinates, y increases downward.
        # Approaching ground: y increases -> v_y > 0; Rebounding up: y decreases -> v_y < 0
        vy_in = (ys[2] - ys[0]) * fps_ratio / norm_px_scale    # positive if descending into center
        vy_out = (ys[4] - ys[2]) * fps_ratio / norm_px_scale   # negative if ascending out of center

        is_true_nadir = (ys[2] >= ys[1]) and (ys[2] >= ys[3]) and (vy_in > 0.45) and (vy_out < -0.45)
        velocity_reversal = (vy_in > 0.90) and (vy_out < -0.90)

        if not (is_true_nadir or velocity_reversal):
            return None

        # Closed-form 5-point quadratic fit: y(t) = a * t^2 + b * t + c
        a, b, c = self.fit_parabola_5pt(ys)
        norm_a = a * fps_accel_scale / norm_px_scale

        # Downward concave parabola has a < 0 (peak at t* = -b / (2a))
        if norm_a >= -0.14:
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

        # Normalized vertical acceleration magnitude (30fps / 720p equivalent)
        vert_accel = abs(2.0 * norm_a)
        if vert_accel < self.min_accel_thresh:
            return None

        # Project sub-pixel point to court coordinates
        court_xy = court_calibrator.pixel_to_court_ft(sub_x, sub_y)
        if court_xy is None:
            return None

        # Disambiguate ground bounce vs paddle volley
        center_players = win[2].get("players")
        is_paddle = self._is_paddle_volley((sub_x, sub_y), court_xy, center_players, scale_px=norm_px_scale)
        is_ground = not is_paddle

        # Confidence metric based on curvature strength and fit quality
        curvature_score = min(1.0, vert_accel / 8.0)
        offset_penalty = max(0.0, 1.0 - abs(t_star) / self.max_subframe_offset)
        conf = float(np.clip(0.5 * curvature_score + 0.5 * offset_penalty, 0.2, 0.99))

        center_pt = win[2]
        center_ts = center_pt["timestamp_s"] + (t_star / eff_fps)

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


class SavitzkyGolayBounceAnalyzer:
    """
    Savitzky-Golay Ball Trajectory Smoother & Ground Bounce Detector.

    Reconstructs clean continuous ball flight arcs from noisy YOLO detections by:
      1. Extracting true non-interpolated (`interpolated=False`) detections and rejecting
         single-frame telecine duplicate freezes so coasting errors never warp trajectories.
      2. Interpolating short occlusion gaps and computing FPS-normalized `savgol_filter`
         1st/2nd vertical derivatives to pinpoint sharp downward-to-upward impact nadirs.
      3. Disambiguating true floor bounces (`is_ground_bounce=True`) from chest-high volleys
         using 3D ball height and player bounding-box vertical ratios.
      4. Producing piecewise Savitzky-Golay smoothed pixel & court coordinates locked in step
         between each ground impact / turnaround vertex so the ball orb and trail flow smoothly.
    """

    def __init__(
        self,
        polyorder: int = 3,
        min_accel_30fps: float = 2.1,
        min_prominence_px: float = 3.5,
        min_bounce_interval_s: float = 0.25,
        max_gap_s: float = 0.30,
    ) -> None:
        self.polyorder = polyorder
        self.min_accel_30fps = min_accel_30fps
        self.min_prominence_px = min_prominence_px
        self.min_bounce_interval_s = min_bounce_interval_s
        self.max_gap_s = max_gap_s

    @staticmethod
    def _odd_win(desired: int, length: int, poly: int) -> int:
        w = min(desired, length if length % 2 == 1 else length - 1)
        if w % 2 == 0:
            w -= 1
        return max(poly + 2 if (poly + 2) % 2 == 1 else poly + 3, w)

    @staticmethod
    def _extract_clean_anchors(
        stream: List[Dict[str, Any]],
        max_gap_frames: int,
        scale_px: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract genuine observed pixels, deduplicate telecine freezes, and fill short same-track gaps."""
        n = len(stream)
        obs_idx: List[int] = []
        obs_x: List[float] = []
        obs_y: List[float] = []
        obs_tid: List[Optional[int]] = []

        # Collect genuine non-interpolated anchors
        raw_obs: List[Tuple[int, float, float, Optional[int]]] = []
        for r in stream:
            fi = int(r["frame_idx"])
            b = r.get("ball")
            if b is not None and b.get("pixel") and not b.get("interpolated", False):
                raw_obs.append((fi, float(b["pixel"][0]), float(b["pixel"][1]), b.get("track_id")))

        # Collapse consecutive telecine freeze repeats smoothly across plateau intervals
        idx_r = 0
        while idx_r < len(raw_obs):
            fi0, px0, py0, tid0 = raw_obs[idx_r]
            end_r = idx_r
            while (
                end_r + 1 < len(raw_obs)
                and raw_obs[end_r + 1][0] == raw_obs[end_r][0] + 1
                and raw_obs[end_r + 1][1] == px0
                and raw_obs[end_r + 1][2] == py0
            ):
                end_r += 1
            # Anchor at start of freeze plateau (interpolator spans smoothly to next changed frame)
            obs_idx.append(fi0)
            obs_x.append(px0)
            obs_y.append(py0)
            obs_tid.append(tid0)
            idx_r = end_r + 1

        x_clean = np.full(n, np.nan, dtype=float)
        y_clean = np.full(n, np.nan, dtype=float)
        valid = np.zeros(n, dtype=bool)

        if not obs_idx:
            return x_clean, y_clean, valid

        norm_s = max(0.45, scale_px)
        for k in range(len(obs_idx) - 1):
            i0, i1 = obs_idx[k], obs_idx[k + 1]
            df = i1 - i0
            dist_px = math.hypot(obs_x[k + 1] - obs_x[k], obs_y[k + 1] - obs_y[k])
            tid_changed = (obs_tid[k] is not None and obs_tid[k + 1] is not None and obs_tid[k] != obs_tid[k + 1])
            max_bridge_px = max(75.0 * norm_s, 42.0 * norm_s * df)
            x_clean[i0], y_clean[i0] = obs_x[k], obs_y[k]
            valid[i0] = True
            if df <= max_gap_frames and not tid_changed and dist_px <= max_bridge_px:
                span = np.arange(i0, i1 + 1)
                t = (span - i0) / max(1, df)
                x_clean[span] = obs_x[k] + t * (obs_x[k + 1] - obs_x[k])
                y_clean[span] = obs_y[k] + t * (obs_y[k + 1] - obs_y[k])
                valid[span] = True

        # Trailing anchor
        last_i = obs_idx[-1]
        if 0 <= last_i < n:
            x_clean[last_i] = obs_x[-1]
            y_clean[last_i] = obs_y[-1]
            valid[last_i] = True

        # Preserve final interpolated tail only if within 3 frames and physically close
        tail_step_lim = max(35.0, 45.0 * norm_s)
        for r in stream:
            fi = int(r["frame_idx"])
            if not valid[fi] and r.get("ball") and r["ball"].get("pixel"):
                px, py = float(r["ball"]["pixel"][0]), float(r["ball"]["pixel"][1])
                if fi > 0 and valid[fi - 1] and math.hypot(px - x_clean[fi - 1], py - y_clean[fi - 1]) < tail_step_lim:
                    x_clean[fi], y_clean[fi] = px, py
                    valid[fi] = True

        return x_clean, y_clean, valid

    @staticmethod
    def _snap_to_exact_impact_frame(
        fi_sg: int,
        st: int,
        en: int,
        y_clean: np.ndarray,
    ) -> int:
        """
        Snap a Savitzky-Golay peak frame `fi_sg` to the exact physical ground impact frame.
        Symmetric SG filtering across a bounce followed closely by a paddle return can shift
        the polynomial peak forward by +1..2 frames (or backward on 60fps telecine).
        Within `[fi_sg - 3, fi_sg + 2]`, checks:
          1. Sharp downward velocity arrest (`dy_prev >= 6.0` and `dy_next <= max(1.5, 0.25 * dy_prev)`),
             capturing exact floor impact when the ball continues drifting toward the camera after bouncing.
          2. Earliest frame reaching the local raw vertical nadir (`yt >= y_max - 2.2` and `dy_next <= 1.2`).
        """
        w0 = max(st, fi_sg - 3)
        w1 = min(en, fi_sg + 2)
        window_y = y_clean[w0 : w1 + 1]
        if len(window_y) == 0 or np.all(np.isnan(window_y)):
            return fi_sg

        y_max = float(np.nanmax(window_y))
        # Pass 1: check for sharp downward velocity arrest (e.g. fast baseline/service bounce)
        for t in range(w0, min(en, fi_sg + 1)):
            yt = float(y_clean[t])
            yp1 = float(y_clean[max(st, t - 1)])
            yn1 = float(y_clean[min(en, t + 1)])
            dy_prev = yt - yp1
            dy_next = yn1 - yt
            if dy_prev >= 6.0 and dy_next <= max(1.5, 0.25 * dy_prev) and yt >= (y_max - 8.5):
                return t

        # Pass 2: earliest frame at the local raw vertical maximum where descent stops
        for t in range(w0, w1 + 1):
            yt = float(y_clean[t])
            yn = float(y_clean[min(en, t + 1)])
            yp = float(y_clean[max(st, t - 2)])
            if yt >= (y_max - 2.2) and (yn - yt) <= 1.2 and (yt - yp) >= 0.0:
                return t
        return int(w0 + int(np.nanargmax(window_y)))

    @staticmethod
    def _is_airborne_volley(
        px_xy: Tuple[float, float],
        height_ft: Optional[float],
        players: List[Dict[str, Any]],
        scale_px: float = 1.0,
    ) -> bool:
        """Identify upper/mid-torso airborne paddle volleys so they are excluded from ground bounces."""
        if height_ft is not None and height_ft >= 2.2:
            return True

        px, py = px_xy
        margin_x = 45.0 * scale_px
        margin_y = 10.0 * scale_px
        for p in players:
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            bx1, by1, bx2, by2 = box
            ph = max(20.0 * scale_px, by2 - by1)
            if (bx1 - margin_x) <= px <= (bx2 + margin_x) and by1 <= py <= (by2 + margin_y):
                rel_y = (py - by1) / ph
                if rel_y < 0.78 and (height_ft is None or height_ft >= 1.35):
                    return True
        return False

    def analyze_stream(
        self,
        stream: List[Dict[str, Any]],
        calibrator: CourtCalibrator,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Run Savitzky-Golay bounce detection and piecewise ball trajectory smoothing over `stream`.
        Normalizes prominence and vertical acceleration by both FPS and court pixel scale.
        Returns `{bounces, all_events, smoothed_stream}`.
        """
        n = len(stream)
        if n == 0:
            return {"bounces": [], "all_events": [], "smoothed_stream": []}

        fps = float(fps or 30.0)
        scale_px = calibrator.get_court_pixel_scale() if calibrator is not None else 1.0
        norm_px_scale = max(0.42, scale_px)
        eff_prominence = self.min_prominence_px * norm_px_scale

        max_gap = max(6, int(round(self.max_gap_s * fps)))
        base_win = 9 if fps < 45.0 else 15

        x_clean, y_clean, valid = self._extract_clean_anchors(stream, max_gap_frames=max_gap, scale_px=norm_px_scale)

        # Find contiguous valid intervals
        segments: List[Tuple[int, int]] = []
        i = 0
        while i < n:
            if valid[i]:
                j = i
                while j + 1 < n and valid[j + 1]:
                    j += 1
                segments.append((i, j))
                i = j + 1
            else:
                i += 1

        all_events: List[BounceEvent] = []
        ground_bounces: List[BounceEvent] = []
        vertex_frames: List[int] = []

        min_dist = max(4, int(round(fps * self.min_bounce_interval_s)))
        fps_scale = (fps / 30.0) ** 2

        for st, en in segments:
            seg_len = en - st + 1
            if seg_len < 7:
                continue
            win = self._odd_win(base_win, seg_len, self.polyorder)
            xs_sg = savgol_filter(x_clean[st : en + 1], win, self.polyorder)
            ys_sg = savgol_filter(y_clean[st : en + 1], win, self.polyorder)
            ay_sg = savgol_filter(y_clean[st : en + 1], win, self.polyorder, deriv=2)

            peaks_prom, _ = find_peaks(ys_sg, distance=min_dist, prominence=eff_prominence)
            # Also detect perspective velocity-arrest bounces (balls coming toward the camera
            # where forward +dy cancels upward rebound -dy, leaving a sharp -ay_sg arrest knee)
            min_raw_ay = (self.min_accel_30fps * 1.15 * norm_px_scale) / max(0.25, fps_scale)
            peaks_acc, _ = find_peaks(-ay_sg, distance=min_dist, height=min_raw_ay)

            cand_peaks = list(peaks_prom)
            for pa in peaks_acc:
                if all(abs(int(pa) - int(pp)) >= min_dist for pp in cand_peaks):
                    fi_cand = self._snap_to_exact_impact_frame(st + int(pa), st, en, y_clean)
                    dy_in = float(y_clean[fi_cand] - y_clean[max(st, fi_cand - 2)])
                    dy_out = float(y_clean[min(en, fi_cand + 2)] - y_clean[fi_cand])
                    if dy_in >= (12.0 * norm_px_scale / max(0.5, fps / 30.0)) and dy_out <= max(0.5 * norm_px_scale, 0.12 * dy_in):
                        cand_peaks.append(pa)
            cand_peaks.sort()

            for p in cand_peaks:
                norm_accel = (abs(float(ay_sg[p])) * fps_scale) / norm_px_scale
                if ay_sg[p] >= 0.0 or norm_accel < self.min_accel_30fps:
                    continue

                fi_sg = st + int(p)
                fi = self._snap_to_exact_impact_frame(fi_sg, st, en, y_clean)
                if fi < 2 or fi > (n - 3):
                    continue
                if vertex_frames and (fi - vertex_frames[-1]) < min_dist:
                    continue

                p_snap = max(0, min(seg_len - 1, fi - st))
                t_star = 0.0
                if 0 < p_snap < seg_len - 1:
                    ym, y0, yp = ys_sg[p_snap - 1], ys_sg[p_snap], ys_sg[p_snap + 1]
                    denom = 2.0 * (ym - 2.0 * y0 + yp)
                    if abs(denom) > 1e-6:
                        t_star = float(np.clip((ym - yp) / denom, -0.5, 0.5))

                sub_x = float(x_clean[fi])
                sub_y = float(y_clean[fi])

                court_xy = calibrator.pixel_to_court_ft(sub_x, sub_y)
                if court_xy is None:
                    continue
                # Reject impossible far-off-court / backdrop wall impacts outside athletic apron
                if court_xy[0] < -6.0 or court_xy[0] > 26.0 or court_xy[1] < -8.0 or court_xy[1] > 52.0:
                    continue

                rec = stream[fi]
                ball_rec = rec.get("ball") or {}
                h_ft = float(ball_rec["height_ft"]) if ball_rec.get("height_ft") is not None else None
                is_volley = self._is_airborne_volley((sub_x, sub_y), h_ft, rec.get("players", []), scale_px=norm_px_scale)
                is_ground = not is_volley

                conf = float(np.clip(0.45 + 0.35 * min(1.0, norm_accel / 8.0) + 0.20 * (1.0 - abs(t_star)), 0.25, 0.99))
                ts = float(rec.get("timestamp_s", fi / fps)) + (t_star / fps)
                ev = BounceEvent(
                    frame_idx=fi,
                    timestamp_s=ts,
                    sub_frame_offset=t_star,
                    pixel_xy=(sub_x, sub_y),
                    court_xy_ft=court_xy,
                    is_ground_bounce=is_ground,
                    confidence=conf,
                    vertical_accel=norm_accel,
                    speed_mph=float(ball_rec.get("speed_mph", 0.0)),
                )
                all_events.append(ev)
                vertex_frames.append(fi)
                if is_ground:
                    ground_bounces.append(ev)

        # Corroborate with any sharp 5-point ground bounces from Pass 1 not near existing SG peaks
        for idx, rec in enumerate(stream):
            b5 = rec.get("bounce")
            if b5 is not None and b5.get("is_ground_bounce", True) and b5.get("pixel") and b5.get("court_ft"):
                fi_5 = int(b5.get("frame_idx", idx))
                if fi_5 < 2 or fi_5 > (n - 3):
                    continue
                cft_5 = (float(b5["court_ft"][0]), float(b5["court_ft"][1]))
                if cft_5[0] < -6.0 or cft_5[0] > 26.0 or cft_5[1] < -8.0 or cft_5[1] > 52.0:
                    continue
                if any(abs(fi_5 - vf) < min_dist for vf in vertex_frames):
                    continue
                px_5 = (float(b5["pixel"][0]), float(b5["pixel"][1]))
                ball_rec = rec.get("ball") or {}
                if ball_rec.get("interpolated", False):
                    continue
                h_ft = float(ball_rec["height_ft"]) if ball_rec.get("height_ft") is not None else None
                if not self._is_airborne_volley(px_5, h_ft, rec.get("players", []), scale_px=norm_px_scale):
                    ev5 = BounceEvent(
                        frame_idx=fi_5,
                        timestamp_s=float(b5.get("timestamp_s", fi_5 / fps)),
                        sub_frame_offset=float(b5.get("sub_frame_offset", 0.0)),
                        pixel_xy=px_5,
                        court_xy_ft=cft_5,
                        is_ground_bounce=True,
                        confidence=float(b5.get("confidence", 0.7)),
                        vertical_accel=float(b5.get("vertical_accel", 2.5)),
                        speed_mph=float(b5.get("speed_mph", 0.0)),
                    )
                    all_events.append(ev5)
                    ground_bounces.append(ev5)
                    vertex_frames.append(fi_5)

        all_events.sort(key=lambda e: e.frame_idx)
        ground_bounces.sort(key=lambda e: e.frame_idx)
        vertex_frames.sort()

        # Piecewise Savitzky-Golay smoothing split strictly at detected impact vertices
        # so arcs flow seamlessly while ground bounce V-corners stay crisp!
        x_smooth = x_clean.copy()
        y_smooth = y_clean.copy()
        vertex_set = set(vertex_frames)

        arc_win = 7 if fps < 45.0 else 11
        for st, en in segments:
            cuts = [st] + sorted(vf for vf in vertex_set if st < vf < en) + [en]
            for idx_c in range(len(cuts) - 1):
                a0, a1 = cuts[idx_c], cuts[idx_c + 1]
                alen = a1 - a0 + 1
                if alen >= 5:
                    w_arc = self._odd_win(arc_win, alen, min(2, self.polyorder))
                    sx = savgol_filter(x_clean[a0 : a1 + 1], w_arc, min(2, self.polyorder))
                    sy = savgol_filter(y_clean[a0 : a1 + 1], w_arc, min(2, self.polyorder))
                    # Pin shared vertices tightly while smoothing flight interior
                    if alen >= 7:
                        x_smooth[a0 + 1 : a1] = sx[1:-1]
                        y_smooth[a0 + 1 : a1] = sy[1:-1]
                    else:
                        x_smooth[a0 : a1 + 1] = sx
                        y_smooth[a0 : a1 + 1] = sy

        # Re-anchor confirmed ground bounces exactly to their impact coordinate
        bounce_map = {ev.frame_idx: ev for ev in ground_bounces}
        for fi, ev in bounce_map.items():
            x_smooth[fi], y_smooth[fi] = ev.pixel_xy

        box_half = max(3.5, round(6.0 * norm_px_scale, 1))
        smoothed_stream: List[Dict[str, Any]] = []
        for idx, rec in enumerate(stream):
            new_rec = dict(rec)
            if valid[idx]:
                raw_ball = rec.get("ball")
                was_missing = (raw_ball is None) or (not raw_ball.get("pixel"))
                b_orig = dict(raw_ball or {})
                sx, sy = float(x_smooth[idx]), float(y_smooth[idx])
                b_orig["trail_pixel"] = [round(sx, 2), round(sy, 2)]
                # Keep observed non-interpolated ball head locked to raw ball center (zero lag!)
                if was_missing or b_orig.get("interpolated", False):
                    b_orig["frame_idx"] = idx
                    b_orig["timestamp_s"] = rec.get("timestamp_s", round(idx / fps, 3))
                    b_orig["pixel"] = [round(sx, 2), round(sy, 2)]
                    b_orig.setdefault("box", [round(sx - box_half, 1), round(sy - box_half, 1), round(sx + box_half, 1), round(sy + box_half, 1)])
                    b_orig.setdefault("conf", 0.0)
                    b_orig.setdefault("speed_mph", 0.0)
                    b_orig.setdefault("speed_px_per_frame", 0.0)
                    b_orig["interpolated"] = True
                c_ft = calibrator.pixel_to_court_ft(sx, sy)
                if idx in bounce_map:
                    ev_b = bounce_map[idx]
                    b_orig["court_ft"] = [round(float(ev_b.court_xy_ft[0]), 2), round(float(ev_b.court_xy_ft[1]), 2)]
                    if "minimap_court_ft" in b_orig:
                        b_orig["minimap_court_ft"] = b_orig["court_ft"]
                elif b_orig.get("court_ft") is None and c_ft is not None:
                    b_orig["court_ft"] = [round(float(c_ft[0]), 2), round(float(c_ft[1]), 2)]
                new_rec["ball"] = b_orig
            if idx in bounce_map:
                new_rec["bounce"] = bounce_map[idx].to_dict()
            else:
                new_rec["bounce"] = None
            smoothed_stream.append(new_rec)

        return {
            "bounces": [ev.to_dict() for ev in ground_bounces],
            "all_events": [ev.to_dict() for ev in all_events],
            "smoothed_stream": smoothed_stream,
        }
