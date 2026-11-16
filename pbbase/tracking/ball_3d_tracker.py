"""
ball_3d_tracker.py - 3D Height-Compensated Ballistic Trajectory Reconstructor.
Solves the 2D ground-homography parallax problem for airborne pickleballs by:
  1. Segmenting the rally into shot arcs bounded by Ground Bounces (Z=0) and Paddle Contacts (Z~3ft).
  2. Computing the parabolic gravity height profile Z(t) along each arc.
  3. Intersecting 3D camera pixel rays (from solvePnP) with the horizontal plane Z = Z(t).
  4. Interpolating brief detection gaps and applying Savitzky-Golay / moving-window smoothing
     so the top-down minimap displays physically true court (X, Y) ball trajectories.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pbbase.court.homography import CourtCalibrator


GRAVITY_FT_S2 = 32.174


@dataclass
class ContactAnchor:
    """Represents a known height anchor (ground bounce or paddle hit) in the rally."""
    frame_idx: int
    timestamp_s: float
    height_ft: float
    court_xy_ft: Tuple[float, float]
    kind: str  # 'ground_bounce', 'paddle_hit', 'rally_start', 'rally_end'


class Ball3DReconstructor:
    """
    Reconstructs true 3D-compensated ground-plane (X, Y) trajectories for the 2D minimap.
    """
    def __init__(
        self,
        paddle_height_ft: float = 3.0,
        max_gap_frames: int = 10,
        smooth_window: int = 5,
    ):
        self.paddle_height_ft = paddle_height_ft
        self.max_gap_frames = max_gap_frames
        self.smooth_window = smooth_window

    def _find_nearest_player(
        self,
        pixel_xy: Tuple[float, float],
        players: Optional[List[Dict[str, Any]]],
        max_dist_px: float = 110.0,
    ) -> Optional[Dict[str, Any]]:
        if not players:
            return None
        px, py = pixel_xy
        best_p = None
        best_d = max_dist_px
        for p in players:
            box = p.get("box")
            if not box or len(box) < 4:
                continue
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            d = float(np.hypot(px - cx, py - cy))
            if d < best_d:
                best_d = d
                best_p = p
        return best_p

    def _interpolate_pixel_gaps(
        self,
        telemetry_stream: List[Dict[str, Any]],
    ) -> List[Optional[Tuple[float, float]]]:
        """Builds a continuous pixel trajectory list, linearly interpolating short None gaps."""
        n = len(telemetry_stream)
        pixels: List[Optional[Tuple[float, float]]] = [None] * n
        for i, rec in enumerate(telemetry_stream):
            b = rec.get("ball")
            if b is not None and b.get("pixel") is not None:
                pixels[i] = (float(b["pixel"][0]), float(b["pixel"][1]))

        # Linearly fill short gaps
        i = 0
        while i < n:
            if pixels[i] is None:
                start = i - 1
                j = i
                while j < n and pixels[j] is None:
                    j += 1
                end = j
                gap_len = end - start - 1
                if start >= 0 and end < n and 1 <= gap_len <= self.max_gap_frames:
                    x0, y0 = pixels[start]
                    x1, y1 = pixels[end]
                    for k in range(start + 1, end):
                        alpha = (k - start) / (end - start)
                        pixels[k] = (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))
                i = end
            else:
                i += 1

        return pixels

    def _collect_anchors(
        self,
        telemetry_stream: List[Dict[str, Any]],
        pixels: List[Optional[Tuple[float, float]]],
        court_calibrator: CourtCalibrator,
        bounces: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ContactAnchor]:
        anchors: List[ContactAnchor] = []
        used_frames = set()

        # 1. Add all bounces & paddle hits from BallBounceDetector
        if bounces:
            for b in bounces:
                f_idx = int(b["frame_idx"])
                if not (0 <= f_idx < len(telemetry_stream)):
                    continue
                is_ground = bool(b.get("is_ground_bounce", True))
                ts = float(telemetry_stream[f_idx].get("timestamp_s", f_idx / 30.0))

                if is_ground:
                    h_ft = 0.0
                    c_ft = (float(b["court_ft"][0]), float(b["court_ft"][1]))
                    kind = "ground_bounce"
                else:
                    h_ft = self.paddle_height_ft
                    px_xy = (float(b["pixel"][0]), float(b["pixel"][1]))
                    # Project at paddle height or snap to nearest hitting player if close
                    c_proj = court_calibrator.pixel_to_court_at_height_ft(px_xy[0], px_xy[1], h_ft)
                    nearest_p = self._find_nearest_player(px_xy, telemetry_stream[f_idx].get("players"))
                    if nearest_p and nearest_p.get("court_ft") and c_proj:
                        # Blend 50% player foot position + 50% height-compensated ray projection
                        pf = nearest_p["court_ft"]
                        c_ft = (0.5 * float(pf[0]) + 0.5 * c_proj[0], 0.5 * float(pf[1]) + 0.5 * c_proj[1])
                    elif c_proj:
                        c_ft = c_proj
                    else:
                        c_ft = (float(b["court_ft"][0]), float(b["court_ft"][1]))
                    kind = "paddle_hit"

                anchors.append(ContactAnchor(f_idx, ts, h_ft, c_ft, kind))
                used_frames.add(f_idx)

        # 2. Detect additional paddle returns (direction reversals / upward-to-downward peaks near players)
        n = len(pixels)
        for k in range(4, n - 4):
            if any(abs(k - uf) < 8 for uf in used_frames):
                continue
            if any(pixels[m] is None for m in range(k - 3, k + 4)):
                continue

            v_in_x = pixels[k][0] - pixels[k - 3][0]
            v_in_y = pixels[k][1] - pixels[k - 3][1]
            v_out_x = pixels[k + 3][0] - pixels[k][0]
            v_out_y = pixels[k + 3][1] - pixels[k][1]

            norm_in = float(np.hypot(v_in_x, v_in_y))
            norm_out = float(np.hypot(v_out_x, v_out_y))
            if norm_in < 4.0 or norm_out < 4.0:
                continue

            cos_angle = (v_in_x * v_out_x + v_in_y * v_out_y) / (norm_in * norm_out)
            # Significant direction change (> ~65 deg) near a player indicates a paddle return
            if cos_angle < 0.42:
                nearest_p = self._find_nearest_player(pixels[k], telemetry_stream[k].get("players"), max_dist_px=100.0)
                if nearest_p is not None:
                    ts = float(telemetry_stream[k].get("timestamp_s", k / 30.0))
                    h_ft = self.paddle_height_ft
                    c_proj = court_calibrator.pixel_to_court_at_height_ft(pixels[k][0], pixels[k][1], h_ft)
                    if nearest_p.get("court_ft") and c_proj:
                        pf = nearest_p["court_ft"]
                        c_ft = (0.5 * float(pf[0]) + 0.5 * c_proj[0], 0.5 * float(pf[1]) + 0.5 * c_proj[1])
                    elif c_proj:
                        c_ft = c_proj
                    else:
                        continue
                    anchors.append(ContactAnchor(k, ts, h_ft, c_ft, "paddle_hit"))
                    used_frames.add(k)

        # 3. Ensure rally start and end anchors exist
        valid_indices = [i for i, p in enumerate(pixels) if p is not None]
        if valid_indices:
            first_i = valid_indices[0]
            if not any(abs(first_i - uf) <= 1 for uf in used_frames):
                ts = float(telemetry_stream[first_i].get("timestamp_s", first_i / 30.0))
                nearest_p = self._find_nearest_player(pixels[first_i], telemetry_stream[first_i].get("players"), max_dist_px=140.0)
                h_ft = self.paddle_height_ft if nearest_p else 2.5
                c_proj = court_calibrator.pixel_to_court_at_height_ft(pixels[first_i][0], pixels[first_i][1], h_ft)
                if nearest_p and nearest_p.get("court_ft") and c_proj:
                    pf = nearest_p["court_ft"]
                    c_ft = (0.4 * float(pf[0]) + 0.6 * c_proj[0], 0.4 * float(pf[1]) + 0.6 * c_proj[1])
                elif c_proj:
                    c_ft = c_proj
                else:
                    c_ft = (10.0, 22.0)
                anchors.append(ContactAnchor(first_i, ts, h_ft, c_ft, "rally_start"))

            last_i = valid_indices[-1]
            if not any(abs(last_i - uf) <= 1 for uf in used_frames):
                ts = float(telemetry_stream[last_i].get("timestamp_s", last_i / 30.0))
                h_ft = 1.0
                c_proj = court_calibrator.pixel_to_court_at_height_ft(pixels[last_i][0], pixels[last_i][1], h_ft)
                c_ft = c_proj if c_proj else (10.0, 22.0)
                anchors.append(ContactAnchor(last_i, ts, h_ft, c_ft, "rally_end"))

        anchors.sort(key=lambda a: a.frame_idx)
        return anchors

    def reconstruct(
        self,
        telemetry_stream: List[Dict[str, Any]],
        court_calibrator: CourtCalibrator,
        bounces: Optional[List[Dict[str, Any]]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reconstructs 3D height-compensated (X, Y) coordinates for every frame in telemetry_stream.
        Populates `frame["ball"]["minimap_court_ft"]` and `frame["ball"]["est_height_ft"]`.
        """
        if not telemetry_stream or not court_calibrator.result.calibrated:
            return telemetry_stream

        if image_size is not None and not court_calibrator._pose_ready:
            court_calibrator.solve_3d_pose(image_size[0], image_size[1])

        pixels = self._interpolate_pixel_gaps(telemetry_stream)
        anchors = self._collect_anchors(telemetry_stream, pixels, court_calibrator, bounces=bounces)
        n = len(telemetry_stream)

        if len(anchors) < 2:
            # Fallback: project at nominal waist height 2.5 ft
            for i, rec in enumerate(telemetry_stream):
                if pixels[i] is not None and rec.get("ball") is not None:
                    c_ft = court_calibrator.pixel_to_court_at_height_ft(pixels[i][0], pixels[i][1], 2.5)
                    if c_ft:
                        rec["ball"]["minimap_court_ft"] = [round(c_ft[0], 2), round(c_ft[1], 2)]
                        rec["ball"]["est_height_ft"] = 2.5
            return telemetry_stream

        # Compute per-frame height Z(t) and ray-projected (X, Y) across each segment
        xs = np.full(n, np.nan, dtype=np.float64)
        ys = np.full(n, np.nan, dtype=np.float64)
        zs = np.full(n, np.nan, dtype=np.float64)

        for seg_idx in range(len(anchors) - 1):
            a1 = anchors[seg_idx]
            a2 = anchors[seg_idx + 1]
            dt_total = max(0.033, a2.timestamp_s - a1.timestamp_s)

            # Vertical initial velocity under gravity to connect Z1 -> Z2 in dt_total
            vz0 = (a2.height_ft - a1.height_ft) / dt_total + 0.5 * GRAVITY_FT_S2 * dt_total

            for f_idx in range(a1.frame_idx, a2.frame_idx + 1):
                if pixels[f_idx] is None:
                    continue
                t_rel = float(telemetry_stream[f_idx].get("timestamp_s", f_idx / 30.0)) - a1.timestamp_s
                alpha = np.clip(t_rel / dt_total, 0.0, 1.0)

                # Parabolic height Z(t), physically clamped to [0, 16] ft
                z_t = a1.height_ft + vz0 * t_rel - 0.5 * GRAVITY_FT_S2 * (t_rel ** 2)
                z_t = float(np.clip(z_t, 0.0, 16.0))

                # Ray intersection at Z(t)
                ray_xy = court_calibrator.pixel_to_court_at_height_ft(pixels[f_idx][0], pixels[f_idx][1], z_t)

                # Linear ground-track prior connecting exact contact anchors
                lin_x = a1.court_xy_ft[0] + alpha * (a2.court_xy_ft[0] - a1.court_xy_ft[0])
                lin_y = a1.court_xy_ft[1] + alpha * (a2.court_xy_ft[1] - a1.court_xy_ft[1])

                if ray_xy is not None:
                    # Near anchors (alpha near 0 or 1), trust the exact anchor prior strongly;
                    # mid-flight, fuse 65% 3D ray projection with 35% linear ballistic path to prevent
                    # longitudinal parallax overshoot from small focal/gravity approximations.
                    endpoint_weight = 1.0 - 0.65 * np.sin(np.pi * alpha)
                    x_final = endpoint_weight * lin_x + (1.0 - endpoint_weight) * ray_xy[0]
                    y_final = endpoint_weight * lin_y + (1.0 - endpoint_weight) * ray_xy[1]
                else:
                    x_final, y_final = lin_x, lin_y

                xs[f_idx] = x_final
                ys[f_idx] = y_final
                zs[f_idx] = z_t

        # Gentle moving-average smoothing (window=5) on valid segments to eliminate micro-jitter
        half_w = self.smooth_window // 2
        xs_smooth = xs.copy()
        ys_smooth = ys.copy()
        for i in range(n):
            if np.isnan(xs[i]):
                continue
            lo = max(0, i - half_w)
            hi = min(n, i + half_w + 1)
            win_x = xs[lo:hi]
            win_y = ys[lo:hi]
            valid = ~np.isnan(win_x)
            if np.sum(valid) >= 2:
                xs_smooth[i] = float(np.mean(win_x[valid]))
                ys_smooth[i] = float(np.mean(win_y[valid]))

        # Write reconstructed minimap_court_ft into telemetry_stream
        for i, rec in enumerate(telemetry_stream):
            if np.isnan(xs_smooth[i]) or np.isnan(ys_smooth[i]):
                continue
            c_ft = [round(float(xs_smooth[i]), 2), round(float(ys_smooth[i]), 2)]
            h_ft = round(float(zs[i]), 2) if not np.isnan(zs[i]) else 0.0

            if rec.get("ball") is not None:
                rec["ball"]["minimap_court_ft"] = c_ft
                rec["ball"]["est_height_ft"] = h_ft
            elif pixels[i] is not None:
                # Create lightweight interpolated ball entry for gap frames
                rec["ball"] = {
                    "frame_idx": i,
                    "timestamp_s": rec.get("timestamp_s", round(i / 30.0, 3)),
                    "pixel": [round(pixels[i][0], 1), round(pixels[i][1], 1)],
                    "box": [round(pixels[i][0] - 6, 1), round(pixels[i][1] - 6, 1), round(pixels[i][0] + 6, 1), round(pixels[i][1] + 6, 1)],
                    "conf": 0.25,
                    "speed_mph": 0.0,
                    "speed_px_per_frame": 0.0,
                    "court_ft": c_ft,
                    "minimap_court_ft": c_ft,
                    "est_height_ft": h_ft,
                    "interpolated": True,
                }

        return telemetry_stream
