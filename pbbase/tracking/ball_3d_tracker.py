"""
ball_3d_tracker.py - 3D Camera-Ray Ballistic Reconstruction for Accurate 2D Minimap Tracking.
Eliminates 3D-to-2D planar homography parallax distortion on airborne pickleballs by:
  1. Scrubbing isolated detector teleport outliers.
  2. Segmenting rallies into shot arcs bounded by ground bounces (Z=0) and paddle hits (Z=player paddle height).
  3. Solving bounded 3D ballistic trajectories (X(t), Y(t), Z(t)) against calibrated camera extrinsics (solvePnP).
  4. Projecting observed ball pixels onto the solved physical height plane Z(t) and filling occlusion gaps.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import least_squares

from pbbase.court.homography import CourtCalibrator, CourtGeometry
from pbbase.tracking.bounce_detector import BounceEvent


@dataclass
class ShotAnchor:
    """Represents a shot boundary (ground bounce, paddle contact, or segment endpoint)."""
    frame_idx: int
    timestamp_s: float
    kind: str  # 'ground_bounce' | 'paddle_hit' | 'endpoint'
    court_xy_ft: Tuple[float, float]
    height_ft: float
    pixel_xy: Optional[Tuple[float, float]] = None


class Ball3DReconstructor:
    """
    Reconstructs physically accurate top-down court positions (X_ft, Y_ft) and
    estimated flight height (height_ft) across full rally telemetry streams.
    """

    def __init__(
        self,
        gravity_ft_s2: float = 30.0,       # Effective gravity incl. pickleball aerodynamic lift/drag
        max_outlier_jump_px: float = 135.0,
        paddle_prox_px: float = 95.0,
    ):
        self.gravity_ft_s2 = gravity_ft_s2
        self.max_outlier_jump_px = max_outlier_jump_px
        self.paddle_prox_px = paddle_prox_px

    @staticmethod
    def _dist_point_to_box(px: float, py: float, box: List[float]) -> float:
        x1, y1, x2, y2 = box
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _estimate_paddle_height_ft(py: float, box: List[float]) -> float:
        """Estimate 3D height of ball contact along player bounding box."""
        _, y1, _, y2 = box
        bh = max(20.0, y2 - y1)
        # Foot is at y2 (height ~0), head is at y1 (height ~5.8 ft)
        frac = (y2 - py) / bh
        return float(np.clip(frac * 5.8, 0.6, 6.2))

    @staticmethod
    def _find_closest_player_box(
        px: float, py: float, players: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Finds the player most physically consistent with touching the ball at (px, py)."""
        valid = [p for p in players if p.get("box")]
        if not valid:
            return None

        # Score combines distance to box AND preference for torso center vs distant foreground cap edge
        def score_player(p: Dict[str, Any]) -> float:
            x1, y1, x2, y2 = p["box"]
            dx = max(x1 - px, 0.0, px - x2)
            dy = max(y1 - py, 0.0, py - y2)
            box_d = float(np.hypot(dx, dy))
            bh = max(20.0, y2 - y1)
            torso_y = y1 + 0.45 * bh
            center_x = 0.5 * (x1 + x2)
            center_d = float(np.hypot(center_x - px, torso_y - py))
            return box_d * 10.0 + center_d * 0.25

        return min(valid, key=score_player)

    def _scrub_outliers(
        self, telemetry_stream: List[Dict[str, Any]]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Extracts pixel centers (u, v) from telemetry_stream and removes isolated
        teleport false positives that jump > max_outlier_jump_px and do not persist.
        """
        n = len(telemetry_stream)
        raw_pts: List[Optional[Tuple[float, float]]] = []
        is_interp: List[bool] = []
        for rec in telemetry_stream:
            b = rec.get("ball")
            if b and b.get("pixel"):
                raw_pts.append((float(b["pixel"][0]), float(b["pixel"][1])))
                is_interp.append(bool(b.get("interpolated", False)))
            else:
                raw_pts.append(None)
                is_interp.append(False)

        cleaned = list(raw_pts)
        # Iterate and check isolated spikes
        valid_indices = [i for i, p in enumerate(cleaned) if p is not None]
        for idx_pos, i in enumerate(valid_indices):
            p_curr = cleaned[i]
            # Look backward and forward up to 3 valid observations
            prev_pts = [cleaned[valid_indices[j]] for j in range(max(0, idx_pos - 2), idx_pos) if cleaned[valid_indices[j]] is not None]
            next_pts = [cleaned[valid_indices[j]] for j in range(idx_pos + 1, min(len(valid_indices), idx_pos + 3)) if cleaned[valid_indices[j]] is not None]

            if prev_pts and next_pts:
                p_prev = prev_pts[-1]
                p_next = next_pts[0]
                d_prev = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                d_next = np.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
                d_bridge = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])

                # Isolated outlier jump (prev -> next is smooth, but curr is far away from both)
                if d_prev > self.max_outlier_jump_px and d_next > self.max_outlier_jump_px and d_bridge < self.max_outlier_jump_px * 1.4:
                    cleaned[i] = None

        return cleaned

    def _find_shot_anchors(
        self,
        telemetry_stream: List[Dict[str, Any]],
        clean_pts: List[Optional[Tuple[float, float]]],
        bounces: List[Dict[str, Any]],
        calibrator: CourtCalibrator,
    ) -> Tuple[List[ShotAnchor], List[Dict[str, Any]]]:
        """
        Finds ordered ShotAnchors partitioning the rally into individual ballistic flights:
          - Verified Ground Bounces (Z = 0.0 ft)
          - Player Paddle Contacts (Z ~ 1.5 - 5.8 ft)
          - Post-bounce upward->downward far-court player hits
          - Rally start/end boundary points
        Also returns updated bounce dictionaries if a foot-level ground bounce was detected.
        """
        n = len(telemetry_stream)
        valid_frames = [i for i, p in enumerate(clean_pts) if p is not None]
        if not valid_frames:
            return [], bounces

        anchors: Dict[int, ShotAnchor] = {}
        updated_bounces = [dict(b) for b in bounces]

        # 1. Process detected bounces (from BallBounceDetector)
        ground_bounce_frames: List[int] = []
        for b in updated_bounces:
            f_idx = int(b["frame_idx"])
            if f_idx >= n:
                continue
            px = tuple(b["pixel"])
            ts = float(b.get("timestamp_s", f_idx / 30.0))
            players = telemetry_stream[f_idx].get("players", [])
            is_ground = bool(b.get("is_ground_bounce", True))

            # Check height above nearest player's feet if flagged as paddle hit
            p_height = 0.0
            if not is_ground:
                best_p = self._find_closest_player_box(px[0], px[1], players)
                if best_p and self._dist_point_to_box(px[0], px[1], best_p["box"]) < 120.0:
                    p_height = self._estimate_paddle_height_ft(px[1], best_p["box"])
                else:
                    p_height = 2.5

            # Recompute accurate court coordinates at estimated height
            if is_ground:
                cft = calibrator.pixel_to_court_ft(px[0], px[1]) or tuple(b["court_ft"])
                ground_bounce_frames.append(f_idx)
            else:
                cft = calibrator.pixel_to_court_at_height_ft(px[0], px[1], p_height) or tuple(b["court_ft"])

            anchors[f_idx] = ShotAnchor(
                frame_idx=f_idx,
                timestamp_s=ts,
                kind="ground_bounce" if is_ground else "paddle_hit",
                court_xy_ft=(float(cft[0]), float(cft[1])),
                height_ft=p_height,
                pixel_xy=(float(px[0]), float(px[1])),
            )

        # 2. Detect post-bounce far-court return hits (local pixel-y minimum after a ground bounce)
        for gb_f in ground_bounce_frames:
            gb_anchor = anchors[gb_f]
            # Look forward 5..26 frames after far-court ground bounce (Y < 24 ft)
            if gb_anchor.court_xy_ft[1] < 24.0:
                best_min_y = float("inf")
                best_f = None
                for k in range(gb_f + 5, min(n - 3, gb_f + 28)):
                    if clean_pts[k] is None:
                        continue
                    # Don't search past another existing anchor
                    if any(a_f > gb_f and a_f <= k for a_f in anchors.keys()):
                        break
                    # Check if local minimum in y (upward rebound ending and heading back to near court)
                    pre_pts = [clean_pts[j] for j in range(max(0, k - 4), k) if clean_pts[j] is not None]
                    post_pts = [clean_pts[j] for j in range(k + 1, min(n, k + 5)) if clean_pts[j] is not None]
                    if not pre_pts or not post_pts:
                        continue
                    vy_in = (clean_pts[k][1] - pre_pts[0][1]) / max(1, k - max(0, k - 4))
                    vy_out = (post_pts[-1][1] - clean_pts[k][1]) / max(1, min(n, k + 5) - 1 - k)
                    if vy_in < -0.8 and vy_out > 0.3 and clean_pts[k][1] < best_min_y:
                        # Confirm proximity to a player on the far side
                        players = telemetry_stream[k].get("players", [])
                        if any(self._dist_point_to_box(clean_pts[k][0], clean_pts[k][1], p["box"]) < self.paddle_prox_px for p in players if p.get("box")):
                            best_min_y = clean_pts[k][1]
                            best_f = k

                if best_f is not None and not any(abs(best_f - af) < 6 for af in anchors):
                    px = clean_pts[best_f]
                    players = telemetry_stream[best_f].get("players", [])
                    best_p = self._find_closest_player_box(px[0], px[1], players)
                    p_h = self._estimate_paddle_height_ft(px[1], best_p["box"]) if best_p else 3.2
                    cft = calibrator.pixel_to_court_at_height_ft(px[0], px[1], p_h) or calibrator.pixel_to_court_ft(px[0], px[1])
                    if cft:
                        anchors[best_f] = ShotAnchor(
                            frame_idx=best_f,
                            timestamp_s=float(telemetry_stream[best_f].get("timestamp_s", best_f / 30.0)),
                            kind="paddle_hit",
                            court_xy_ft=(float(cft[0]), float(cft[1])),
                            height_ft=p_h,
                            pixel_xy=px,
                        )

        # 3. Detect fast-hit occlusion reversals near players (e.g. smash/pass occluded briefly behind hitter)
        for i in range(4, n - 4):
            if any(abs(i - af) < 7 for af in anchors):
                continue
            if clean_pts[i] is None:
                continue

            # Check pre/post valid neighbors across short/medium gaps (up to 18 frames)
            pre_idx = next((j for j in range(i - 3, max(-1, i - 8), -1) if clean_pts[j] is not None), None)
            if pre_idx is None:
                pre_idx = next((j for j in range(i - 1, max(-1, i - 4), -1) if clean_pts[j] is not None), None)

            post_idx = next((j for j in range(i + 3, min(n, i + 18)) if clean_pts[j] is not None), None)
            if post_idx is None:
                post_idx = next((j for j in range(i + 1, min(n, i + 4)) if clean_pts[j] is not None), None)

            if pre_idx is None or post_idx is None:
                continue

            gap = post_idx - i
            p_curr = clean_pts[i]
            players = telemetry_stream[i].get("players", [])
            near_players = [
                p for p in players
                if p.get("box") and self._dist_point_to_box(p_curr[0], p_curr[1], p["box"]) < 65.0
            ]
            if not near_players:
                continue

            v_in = np.array([p_curr[0] - clean_pts[pre_idx][0], p_curr[1] - clean_pts[pre_idx][1]]) / max(1, i - pre_idx)
            v_out = np.array([clean_pts[post_idx][0] - p_curr[0], clean_pts[post_idx][1] - p_curr[1]]) / max(1, gap)
            n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)

            if n_in > 3.0 and n_out > 3.0:
                cos_theta = float(np.dot(v_in, v_out) / (n_in * n_out))
                # Distinguish true paddle hits from smooth mid-air gravity apexes:
                # 1) Occlusion gap >= 5 frames right at player
                # 2) Upward-concavity bounce/hit (vy_in > 1.5 and vy_out < -1.5, anti-gravity)
                # 3) Lateral reversal (vx flips sign) or steep acute kink (cos < -0.55) with non-parabolic jerk
                is_gap_hit = (gap >= 5 and (v_in[1] * v_out[1] < 0 or cos_theta < 0.35))
                is_antigravity_hit = (v_in[1] > 1.5 and v_out[1] < -1.5 and cos_theta < 0.45)
                is_lateral_kink = (v_in[0] * v_out[0] < -1.0 and cos_theta < 0.25)
                speed_jump = max(n_in, n_out) / max(0.5, min(n_in, n_out))
                is_acute_impulse = (cos_theta < -0.55 and speed_jump > 1.45)

                if is_gap_hit or is_antigravity_hit or is_lateral_kink or is_acute_impulse:
                    best_p = self._find_closest_player_box(p_curr[0], p_curr[1], near_players)
                    if best_p is None:
                        continue
                    p_h = self._estimate_paddle_height_ft(p_curr[1], best_p["box"])
                    cft = calibrator.pixel_to_court_at_height_ft(p_curr[0], p_curr[1], p_h) or calibrator.pixel_to_court_ft(p_curr[0], p_curr[1])
                    if cft:
                        anchors[i] = ShotAnchor(
                            frame_idx=i,
                            timestamp_s=float(telemetry_stream[i].get("timestamp_s", i / 30.0)),
                            kind="paddle_hit",
                            court_xy_ft=(float(cft[0]), float(cft[1])),
                            height_ft=p_h,
                            pixel_xy=p_curr,
                        )

        # 4. Ensure start and end endpoints are anchored
        f_start = valid_frames[0]
        if not any(abs(f_start - af) < 5 for af in anchors):
            px = clean_pts[f_start]
            players = telemetry_stream[f_start].get("players", [])
            best_p = min([p for p in players if p.get("box")], key=lambda p: self._dist_point_to_box(px[0], px[1], p["box"]), default=None)
            h_start = self._estimate_paddle_height_ft(px[1], best_p["box"]) if best_p and self._dist_point_to_box(px[0], px[1], best_p["box"]) < 140.0 else 3.0
            cft = calibrator.pixel_to_court_at_height_ft(px[0], px[1], h_start) or calibrator.pixel_to_court_ft(px[0], px[1])
            if cft:
                anchors[f_start] = ShotAnchor(
                    frame_idx=f_start,
                    timestamp_s=float(telemetry_stream[f_start].get("timestamp_s", f_start / 30.0)),
                    kind="endpoint",
                    court_xy_ft=(float(cft[0]), float(cft[1])),
                    height_ft=h_start,
                    pixel_xy=px,
                )

        f_end = valid_frames[-1]
        if not any(abs(f_end - af) < 4 for af in anchors):
            px = clean_pts[f_end]
            h_end = 1.5
            cft = calibrator.pixel_to_court_at_height_ft(px[0], px[1], h_end) or calibrator.pixel_to_court_ft(px[0], px[1])
            if cft:
                anchors[f_end] = ShotAnchor(
                    frame_idx=f_end,
                    timestamp_s=float(telemetry_stream[f_end].get("timestamp_s", f_end / 30.0)),
                    kind="endpoint",
                    court_xy_ft=(float(cft[0]), float(cft[1])),
                    height_ft=h_end,
                    pixel_xy=px,
                )

        sorted_anchors = [anchors[k] for k in sorted(anchors.keys())]
        return sorted_anchors, updated_bounces

    def _reconstruct_segment(
        self,
        telemetry_stream: List[Dict[str, Any]],
        clean_pts: List[Optional[Tuple[float, float]]],
        a1: ShotAnchor,
        a2: ShotAnchor,
        calibrator: CourtCalibrator,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Reconstructs 3D trajectory (X_ft, Y_ft, Z_ft) between two consecutive anchors a1 -> a2.
        Combines 3D ballistic projectile fitting (when camera 3D pose is available) with
        height-compensated pixel ray intersection for physically exact ground tracks.
        Returns mapping: frame_idx -> (X_ft, Y_ft, height_ft).
        """
        k1, k2 = a1.frame_idx, a2.frame_idx
        t1, t2 = a1.timestamp_s, a2.timestamp_s
        dt_total = max(1e-3, t2 - t1)

        # Gather observed non-null pixels in [k1, k2]
        obs_k, obs_t, obs_u, obs_v = [], [], [], []
        for k in range(k1, k2 + 1):
            if clean_pts[k] is not None:
                obs_k.append(k)
                obs_t.append(float(telemetry_stream[k].get("timestamp_s", k / 30.0)))
                obs_u.append(clean_pts[k][0])
                obs_v.append(clean_pts[k][1])

        obs_t_arr = np.array(obs_t, dtype=np.float64)
        obs_u_arr = np.array(obs_u, dtype=np.float64)
        obs_v_arr = np.array(obs_v, dtype=np.float64)

        x1_init, y1_init = a1.court_xy_ft
        x2_init, y2_init = a2.court_xy_ft
        z1_init = float(a1.height_ft)
        z2_init = float(a2.height_ft)
        g_eff = self.gravity_ft_s2
        k_d = 0.25

        # Fit 3D projectile parameters if 3D pose is calibrated and enough points exist
        if calibrator.result.pose_calibrated and len(obs_k) >= 4:
            # Pin endpoints strictly if ground bounce; lightly constrain paddle/endpoint heights
            w_end1 = 6.0 if a1.kind == "ground_bounce" else 1.5
            w_end2 = 6.0 if a2.kind == "ground_bounce" else 1.5
            z1_min = 0.0 if a1.kind == "ground_bounce" else 0.5
            z1_max = 0.05 if a1.kind == "ground_bounce" else 7.5
            z2_min = 0.0 if a2.kind == "ground_bounce" else 0.5
            z2_max = 0.05 if a2.kind == "ground_bounce" else 7.5

            def residuals(params):
                px1, py1, pz1, px2, py2, pz2, kd, g = params
                tau = (obs_t_arr - t1) / dt_total
                dt = obs_t_arr - t1
                if abs(kd) > 1e-3:
                    s_tau = (1.0 - np.exp(-kd * tau)) / (1.0 - np.exp(-kd))
                else:
                    s_tau = tau

                X = px1 + (px2 - px1) * s_tau
                Y = py1 + (py2 - py1) * s_tau
                vz0 = (pz2 - pz1) / dt_total + 0.5 * g * dt_total
                Z = np.maximum(0.0, pz1 + vz0 * dt - 0.5 * g * (dt ** 2))

                proj_u = np.empty_like(X)
                proj_v = np.empty_like(Y)
                for idx in range(len(X)):
                    uv = calibrator.court_3d_to_pixel(X[idx], Y[idx], Z[idx])
                    if uv:
                        proj_u[idx], proj_v[idx] = uv
                    else:
                        proj_u[idx], proj_v[idx] = obs_u_arr[idx], obs_v_arr[idx]

                res_px = np.concatenate([proj_u - obs_u_arr, proj_v - obs_v_arr])
                priors = np.array([
                    (px1 - x1_init) * w_end1,
                    (py1 - y1_init) * w_end1,
                    (pz1 - z1_init) * (10.0 if a1.kind == "ground_bounce" else 2.5),
                    (px2 - x2_init) * w_end2,
                    (py2 - y2_init) * w_end2,
                    (pz2 - z2_init) * (10.0 if a2.kind == "ground_bounce" else 2.5),
                    (kd - 0.25) * 4.0,
                    (g - 30.0) * 0.6,
                ])
                return np.concatenate([res_px, priors])

            x0 = [
                np.clip(x1_init, -15.0, 35.0),
                np.clip(y1_init, -15.0, 58.0),
                np.clip(z1_init, z1_min, z1_max),
                np.clip(x2_init, -15.0, 35.0),
                np.clip(y2_init, -15.0, 58.0),
                np.clip(z2_init, z2_min, z2_max),
                0.25,
                30.0,
            ]
            lb = [-15.0, -18.0, z1_min, -15.0, -18.0, z2_min, 0.0, 20.0]
            ub = [38.0, 60.0, z1_max, 38.0, 60.0, z2_max, 1.2, 38.0]

            try:
                opt = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=80)
                x1_init, y1_init, z1_init, x2_init, y2_init, z2_init, k_d, g_eff = opt.x
            except Exception:
                pass

        # Evaluate trajectory across every frame k in [k1, k2]
        vz0 = (z2_init - z1_init) / dt_total + 0.5 * g_eff * dt_total
        segment_out: Dict[int, Tuple[float, float, float]] = {}

        for k in range(k1, k2 + 1):
            t = float(telemetry_stream[k].get("timestamp_s", k / 30.0))
            dt = max(0.0, t - t1)
            tau = np.clip(dt / dt_total, 0.0, 1.0)
            s_tau = (1.0 - np.exp(-k_d * tau)) / (1.0 - np.exp(-k_d)) if abs(k_d) > 1e-3 else tau

            x_fit = x1_init + (x2_init - x1_init) * s_tau
            y_fit = y1_init + (y2_init - y1_init) * s_tau
            z_fit = float(max(0.0, z1_init + vz0 * dt - 0.5 * g_eff * (dt ** 2)))

            px = clean_pts[k]
            if px is not None and calibrator.result.pose_calibrated:
                ray_xy = calibrator.pixel_to_court_at_height_ft(px[0], px[1], z_fit)
                if ray_xy is not None:
                    rx, ry = ray_xy
                    # Check ray sanity (avoid extreme horizon blowups)
                    if abs(rx - x_fit) < 8.0 and abs(ry - y_fit) < 12.0:
                        # Lateral X from camera ray captures genuine ball curve; Y stays smoothly anchored
                        x_final = 0.65 * rx + 0.35 * x_fit
                        y_final = 0.25 * ry + 0.75 * y_fit
                        segment_out[k] = (float(x_final), float(y_final), z_fit)
                        continue

            segment_out[k] = (float(x_fit), float(y_fit), z_fit)

        return segment_out

    def reconstruct_stream(
        self,
        telemetry_stream: List[Dict[str, Any]],
        bounces: List[Dict[str, Any]],
        calibrator: CourtCalibrator,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Post-processes the full rally telemetry_stream to populate accurate 3D-compensated
        `minimap_court_ft` and `height_ft` on every frame (including occlusion gap fills).
        """
        if not telemetry_stream or not calibrator.result.calibrated:
            return telemetry_stream, bounces

        clean_pts = self._scrub_outliers(telemetry_stream)
        anchors, updated_bounces = self._find_shot_anchors(telemetry_stream, clean_pts, bounces, calibrator)

        if len(anchors) < 2:
            # Not enough anchors for ballistic segments; apply fallback smoothing
            return telemetry_stream, updated_bounces

        reconstructed: Dict[int, Tuple[float, float, float]] = {}
        for i in range(len(anchors) - 1):
            seg = self._reconstruct_segment(
                telemetry_stream, clean_pts, anchors[i], anchors[i + 1], calibrator
            )
            reconstructed.update(seg)

        # Gentle 3-point weighted temporal smoothing (preserving pinned bounce positions)
        pinned_frames = {a.frame_idx for a in anchors if a.kind == "ground_bounce"}
        smoothed: Dict[int, Tuple[float, float, float]] = {}
        keys = sorted(reconstructed.keys())
        for idx in keys:
            if idx in pinned_frames:
                smoothed[idx] = reconstructed[idx]
                continue
            pts_w = []
            for offset, w in [(-1, 0.25), (0, 0.50), (1, 0.25)]:
                if (idx + offset) in reconstructed:
                    pts_w.append((reconstructed[idx + offset], w))
            w_sum = sum(w for _, w in pts_w)
            sx = sum(pt[0] * w for pt, w in pts_w) / w_sum
            sy = sum(pt[1] * w for pt, w in pts_w) / w_sum
            sz = sum(pt[2] * w for pt, w in pts_w) / w_sum
            smoothed[idx] = (round(float(sx), 2), round(float(sy), 2), round(float(sz), 2))

        # Inject reconstructed coordinates into telemetry_stream
        for k, (mx, my, mz) in smoothed.items():
            rec = telemetry_stream[k]
            ball = rec.get("ball")
            if ball is not None:
                ball["minimap_court_ft"] = [mx, my]
                ball["height_ft"] = mz
            else:
                # Reconstructed occlusion gap fill for continuous smooth minimap trajectory
                px_proj = calibrator.court_3d_to_pixel(mx, my, mz)
                rec["ball"] = {
                    "frame_idx": k,
                    "timestamp_s": rec.get("timestamp_s", round(k / 30.0, 3)),
                    "pixel": [round(px_proj[0], 1), round(px_proj[1], 1)] if px_proj else None,
                    "box": [round(px_proj[0] - 6, 1), round(px_proj[1] - 6, 1), round(px_proj[0] + 6, 1), round(px_proj[1] + 6, 1)] if px_proj else [0, 0, 0, 0],
                    "conf": 0.0,
                    "speed_mph": 0.0,
                    "speed_px_per_frame": 0.0,
                    "court_ft": [mx, my],
                    "minimap_court_ft": [mx, my],
                    "height_ft": mz,
                    "interpolated": True,
                }

        return telemetry_stream, updated_bounces
