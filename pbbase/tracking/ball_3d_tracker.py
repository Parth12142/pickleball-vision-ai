"""
ball_3d_tracker.py - 3D Camera-Ray Ballistic Reconstruction for Unified 1-to-1 Minimap Tracking.
Ensures the video overlay ball and 2D minimap dot always represent the EXACT SAME ball:
  1. Works strictly from genuine model detections (ignoring stale/coasted interpolations)
     and scrubs isolated teleport outliers.
  2. Segments rallies by Ground Bounces (Z=0) and Player Paddle Contacts, resolving exact
     contact vertices inside close-player hitting envelopes during motion-blur gaps.
  3. Fits bounded 3D gravity arcs (X(t), Y(t), Z(t)) via solvePnP extrinsics.
  4. On every observed frame, projects the EXACT model pixel (u_obs, v_obs) at solved height Z(t)
     so video reticle and minimap dot stay locked 1-to-1 (< 2 px discrepancy).
  5. On occluded frames, computes synchronized 3D point (X, Y, Z) -> both video pixel & minimap.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import least_squares

from pbbase.court.homography import CourtCalibrator, CourtGeometry


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
    estimated flight height (height_ft) while keeping 1-to-1 sync with model tracking.
    """

    def __init__(
        self,
        gravity_ft_s2: float = 30.0,
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
        Extracts ONLY genuine observed pixel centers (interpolated=False) from
        telemetry_stream and removes isolated teleport spikes (> max_outlier_jump_px).
        """
        cleaned: List[Optional[Tuple[float, float]]] = []
        for rec in telemetry_stream:
            b = rec.get("ball")
            if b and b.get("pixel") and not b.get("interpolated", False):
                cleaned.append((float(b["pixel"][0]), float(b["pixel"][1])))
            else:
                cleaned.append(None)

        valid_indices = [i for i, p in enumerate(cleaned) if p is not None]
        for idx_pos, i in enumerate(valid_indices):
            p_curr = cleaned[i]
            prev_pts = [cleaned[valid_indices[j]] for j in range(max(0, idx_pos - 2), idx_pos) if cleaned[valid_indices[j]] is not None]
            next_pts = [cleaned[valid_indices[j]] for j in range(idx_pos + 1, min(len(valid_indices), idx_pos + 3)) if cleaned[valid_indices[j]] is not None]

            if prev_pts and next_pts:
                p_prev = prev_pts[-1]
                p_next = next_pts[0]
                d_prev = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                d_next = np.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
                d_bridge = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])

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
        Finds ordered ShotAnchors partitioning the rally into individual ballistic flights.
        Handles close-player occlusion hits cleanly by resolving contact vertices inside
        the hitting player's envelope rather than letting gap noise pull anchors astray.
        """
        n = len(telemetry_stream)
        valid_frames = [i for i, p in enumerate(clean_pts) if p is not None]
        if not valid_frames:
            return [], bounces

        anchors: Dict[int, ShotAnchor] = {}
        updated_bounces = [dict(b) for b in bounces]

        # 1. Ground bounces and low/torso bounces from BallBounceDetector
        ground_bounce_frames: List[int] = []
        for b in updated_bounces:
            f_idx = int(b["frame_idx"])
            if f_idx >= n:
                continue
            px = tuple(b["pixel"])
            ts = float(b.get("timestamp_s", f_idx / 30.0))
            players = telemetry_stream[f_idx].get("players", [])
            is_ground = bool(b.get("is_ground_bounce", True))

            p_height = 0.0
            if not is_ground:
                best_p = self._find_closest_player_box(px[0], px[1], players)
                if best_p and self._dist_point_to_box(px[0], px[1], best_p["box"]) < 120.0:
                    p_height = self._estimate_paddle_height_ft(px[1], best_p["box"])
                else:
                    p_height = 2.5

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
            if gb_anchor.court_xy_ft[1] < 24.0:
                best_min_y = float("inf")
                best_f = None
                for k in range(gb_f + 5, min(n - 3, gb_f + 28)):
                    if clean_pts[k] is None:
                        continue
                    if any(a_f > gb_f and a_f <= k for a_f in anchors.keys()):
                        break
                    pre_pts = [clean_pts[j] for j in range(max(0, k - 4), k) if clean_pts[j] is not None]
                    post_pts = [clean_pts[j] for j in range(k + 1, min(n, k + 5)) if clean_pts[j] is not None]
                    if not pre_pts or not post_pts:
                        continue
                    vy_in = (clean_pts[k][1] - pre_pts[0][1]) / max(1, k - max(0, k - 4))
                    vy_out = (post_pts[-1][1] - clean_pts[k][1]) / max(1, min(n, k + 5) - 1 - k)
                    if vy_in < -0.8 and vy_out > 0.3 and clean_pts[k][1] < best_min_y:
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

        # 3. Detect fast-hit occlusion reversals & close-player strikes
        for i in range(4, n - 4):
            if any(abs(i - af) < 7 for af in anchors):
                continue
            if clean_pts[i] is None:
                continue

            pre_idx = next((j for j in range(i - 3, max(-1, i - 8), -1) if clean_pts[j] is not None), None)
            if pre_idx is None:
                pre_idx = next((j for j in range(i - 1, max(-1, i - 4), -1) if clean_pts[j] is not None), None)

            post_idx = next((j for j in range(i + 1, min(n, i + 18)) if clean_pts[j] is not None), None)
            if pre_idx is None or post_idx is None:
                continue

            # Find a second post point slightly further out to verify outgoing velocity vector
            post_idx2 = next((j for j in range(post_idx + 2, min(n, post_idx + 7)) if clean_pts[j] is not None), post_idx)

            gap = post_idx - i
            p_curr = clean_pts[i]
            players = telemetry_stream[i].get("players", [])
            near_players = [
                p for p in players
                if p.get("box") and self._dist_point_to_box(p_curr[0], p_curr[1], p["box"]) < 105.0
            ]
            if not near_players:
                continue

            v_in = np.array([p_curr[0] - clean_pts[pre_idx][0], p_curr[1] - clean_pts[pre_idx][1]]) / max(1, i - pre_idx)
            if post_idx2 > post_idx:
                v_out = np.array([clean_pts[post_idx2][0] - clean_pts[post_idx][0], clean_pts[post_idx2][1] - clean_pts[post_idx][1]]) / max(1, post_idx2 - post_idx)
            else:
                v_out = np.array([clean_pts[post_idx][0] - p_curr[0], clean_pts[post_idx][1] - p_curr[1]]) / max(1, gap)

            n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            if n_in > 3.0 and n_out > 2.5:
                cos_theta = float(np.dot(v_in, v_out) / (n_in * n_out))
                is_gap_hit = (gap >= 3 and (v_in[1] * v_out[1] < 0 or cos_theta < 0.35))
                is_antigravity_hit = (v_in[1] > 1.5 and v_out[1] < -1.5 and cos_theta < 0.45)
                is_lateral_kink = (v_in[0] * v_out[0] < -1.0 and cos_theta < 0.25)
                speed_jump = max(n_in, n_out) / max(0.5, min(n_in, n_out))
                is_acute_impulse = (cos_theta < -0.55 and speed_jump > 1.45)

                if is_gap_hit or is_antigravity_hit or is_lateral_kink or is_acute_impulse:
                    # Resolve true hit frame & coordinate inside the player's hitting zone if gap > 1
                    hit_frame = i
                    hit_px = p_curr
                    if gap >= 2:
                        # Interpolate contact time between i and post_idx based on gap midpoint toward closest player
                        mid_k = i + max(1, min(gap - 1, gap // 3 if gap >= 6 else gap // 2))
                        hit_frame = mid_k
                        # Predict ball position entering player's torso plane
                        best_p = self._find_closest_player_box(p_curr[0], p_curr[1], telemetry_stream[min(n - 1, hit_frame)].get("players", near_players))
                        if best_p and best_p.get("box"):
                            bx1, by1, bx2, by2 = best_p["box"]
                            pred_px = (
                                float(np.clip(p_curr[0] + v_in[0] * (hit_frame - i), bx1 - 15.0, bx2 + 15.0)),
                                float(np.clip(p_curr[1] + v_in[1] * (hit_frame - i), by1, by2 - 0.15 * (by2 - by1))),
                            )
                            hit_px = pred_px
                            players_hit = telemetry_stream[hit_frame].get("players", near_players)
                            best_p = self._find_closest_player_box(hit_px[0], hit_px[1], players_hit)
                    else:
                        best_p = self._find_closest_player_box(hit_px[0], hit_px[1], near_players)

                    if best_p is None or any(abs(hit_frame - af) < 6 for af in anchors):
                        continue

                    p_h = self._estimate_paddle_height_ft(hit_px[1], best_p["box"])
                    # Anchor tightly at player's actual court position (with paddle offset) to prevent parallax confusion
                    if best_p.get("court_ft"):
                        cft = (float(best_p["court_ft"][0]), float(best_p["court_ft"][1]))
                    else:
                        cft = calibrator.pixel_to_court_at_height_ft(hit_px[0], hit_px[1], p_h) or calibrator.pixel_to_court_ft(hit_px[0], hit_px[1])

                    if cft:
                        anchors[hit_frame] = ShotAnchor(
                            frame_idx=hit_frame,
                            timestamp_s=float(telemetry_stream[hit_frame].get("timestamp_s", hit_frame / 30.0)),
                            kind="paddle_hit",
                            court_xy_ft=(float(cft[0]), float(cft[1])),
                            height_ft=p_h,
                            pixel_xy=hit_px,
                        )

        # 4. Ensure start and end endpoints are anchored
        f_start = valid_frames[0]
        if not any(abs(f_start - af) < 5 for af in anchors):
            px = clean_pts[f_start]
            players = telemetry_stream[f_start].get("players", [])
            best_p = self._find_closest_player_box(px[0], px[1], players)
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
        prev_anchor_state: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[int, Tuple[float, float, float, bool]]:
        """
        Reconstructs 3D trajectory (X_ft, Y_ft, Z_ft, is_observed) between anchors a1 -> a2.
        Enforces 1-to-1 consistency and strict C0 continuity across shared shot anchors.
        """
        k1, k2 = a1.frame_idx, a2.frame_idx
        t1, t2 = a1.timestamp_s, a2.timestamp_s
        dt_total = max(1e-3, t2 - t1)

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

        if prev_anchor_state is not None:
            x1_init, y1_init, z1_init = prev_anchor_state
        else:
            x1_init, y1_init = a1.court_xy_ft
            z1_init = float(a1.height_ft)

        x2_init, y2_init = a2.court_xy_ft
        z2_init = float(a2.height_ft)
        g_eff = self.gravity_ft_s2
        k_d = 0.25

        if calibrator.result.pose_calibrated and len(obs_k) >= 3:
            w_end1 = 22.0 if prev_anchor_state is not None else (6.0 if a1.kind == "ground_bounce" else 2.8)
            w_end2 = 6.0 if a2.kind == "ground_bounce" else 2.8
            wz1 = 22.0 if prev_anchor_state is not None else (10.0 if a1.kind == "ground_bounce" else 4.5)
            wz2 = 10.0 if a2.kind == "ground_bounce" else 4.5
            z1_min = 0.0 if a1.kind == "ground_bounce" else (0.5 if prev_anchor_state is not None else (1.2 if a1.height_ft >= 1.5 else 0.5))
            z1_max = 0.05 if a1.kind == "ground_bounce" else 7.5
            z2_min = 0.0 if a2.kind == "ground_bounce" else (1.2 if a2.height_ft >= 1.5 else 0.5)
            z2_max = 0.05 if a2.kind == "ground_bounce" else 7.5

            def residuals(params):
                px1, py1, pz1, px2, py2, pz2, kd, g = params
                tau = (obs_t_arr - t1) / dt_total
                dt = obs_t_arr - t1
                s_tau = (1.0 - np.exp(-kd * tau)) / (1.0 - np.exp(-kd)) if abs(kd) > 1e-3 else tau

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
                    (pz1 - z1_init) * wz1,
                    (px2 - x2_init) * w_end2,
                    (py2 - y2_init) * w_end2,
                    (pz2 - z2_init) * wz2,
                    (kd - 0.25) * 4.0,
                    (g - 30.0) * 0.6,
                ])
                return np.concatenate([res_px, priors])

            if prev_anchor_state is not None:
                px1_c = float(np.clip(x1_init, -15.0, 38.0))
                py1_c = float(np.clip(y1_init, -18.0, 60.0))
                pz1_c = float(np.clip(z1_init, z1_min, z1_max))
                x0 = [
                    px1_c,
                    py1_c,
                    pz1_c,
                    np.clip(x2_init, -15.0, 35.0),
                    np.clip(y2_init, -15.0, 58.0),
                    np.clip(z2_init, z2_min, z2_max),
                    0.25,
                    30.0,
                ]
                lb = [px1_c - 0.05, py1_c - 0.05, max(0.0, pz1_c - 0.05), -15.0, -18.0, z2_min, 0.0, 20.0]
                ub = [px1_c + 0.05, py1_c + 0.05, min(7.5, pz1_c + 0.05), 38.0, 60.0, z2_max, 1.2, 38.0]
            else:
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

        vz0 = (z2_init - z1_init) / dt_total + 0.5 * g_eff * dt_total
        segment_out: Dict[int, Tuple[float, float, float, bool]] = {}

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
                # 1-to-1 Pixel-to-Ray locking at solved flight height z_fit:
                # Ensures the minimap projects the EXACT same ball seen in camera frame (reproj_err -> ~0 px)
                ray_xy = calibrator.pixel_to_court_at_height_ft(px[0], px[1], z_fit)
                if ray_xy is not None:
                    rx, ry = ray_xy
                    if abs(rx - x_fit) < 5.5 and abs(ry - y_fit) < 8.0:
                        segment_out[k] = (float(rx), float(ry), z_fit, True)
                        continue
                    elif abs(rx - x_fit) < 10.0 and abs(ry - y_fit) < 14.0:
                        segment_out[k] = (float(0.65 * rx + 0.35 * x_fit), float(0.65 * ry + 0.35 * y_fit), z_fit, True)
                        continue

            segment_out[k] = (float(x_fit), float(y_fit), z_fit, False)

        return segment_out

    def reconstruct_stream(
        self,
        telemetry_stream: List[Dict[str, Any]],
        bounces: List[Dict[str, Any]],
        calibrator: CourtCalibrator,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Post-processes the full rally telemetry_stream so camera reticle AND 2D minimap
        remain 1-to-1 locked onto the exact same physical 3D ball across all frames.
        """
        if not telemetry_stream or not calibrator.result.calibrated:
            return telemetry_stream, bounces

        clean_pts = self._scrub_outliers(telemetry_stream)
        anchors, updated_bounces = self._find_shot_anchors(telemetry_stream, clean_pts, bounces, calibrator)

        if len(anchors) < 2:
            return telemetry_stream, updated_bounces

        reconstructed: Dict[int, Tuple[float, float, float, bool]] = {}
        prev_state: Optional[Tuple[float, float, float]] = None
        for i in range(len(anchors) - 1):
            seg = self._reconstruct_segment(
                telemetry_stream,
                clean_pts,
                anchors[i],
                anchors[i + 1],
                calibrator,
                prev_anchor_state=prev_state,
            )
            reconstructed.update(seg)
            end_k = anchors[i + 1].frame_idx
            if end_k in seg:
                prev_state = (seg[end_k][0], seg[end_k][1], seg[end_k][2])
            else:
                prev_state = None

        # Preserve exact ray-locked observed positions; lightly smooth only extreme spikes
        keys = sorted(reconstructed.keys())
        for k in keys:
            mx, my, mz, is_obs = reconstructed[k]
            mx_r, my_r, mz_r = round(float(mx), 2), round(float(my), 2), round(float(mz), 2)
            rec = telemetry_stream[k]
            ball = rec.get("ball")

            if is_obs and ball is not None and not ball.get("interpolated", False):
                # Observed frame: keep EXACT tracked pixel, sync minimap_court_ft to 3D ray intersection
                ball["minimap_court_ft"] = [mx_r, my_r]
                ball["court_ft"] = [mx_r, my_r]
                ball["height_ft"] = mz_r
            else:
                # Occluded / Coasted frame: synchronize BOTH camera pixel AND minimap from unified 3D curve!
                px_proj = calibrator.court_3d_to_pixel(mx, my, mz)
                px_list = [round(px_proj[0], 1), round(px_proj[1], 1)] if px_proj else (ball.get("pixel") if ball else None)
                box_list = (
                    [round(px_list[0] - 6.0, 1), round(px_list[1] - 6.0, 1), round(px_list[0] + 6.0, 1), round(px_list[1] + 6.0, 1)]
                    if px_list else [0.0, 0.0, 0.0, 0.0]
                )
                if ball is not None:
                    ball["pixel"] = px_list
                    ball["box"] = box_list
                    ball["court_ft"] = [mx_r, my_r]
                    ball["minimap_court_ft"] = [mx_r, my_r]
                    ball["height_ft"] = mz_r
                    ball["interpolated"] = True
                else:
                    rec["ball"] = {
                        "frame_idx": k,
                        "timestamp_s": rec.get("timestamp_s", round(k / 30.0, 3)),
                        "pixel": px_list,
                        "box": box_list,
                        "conf": 0.0,
                        "speed_mph": 0.0,
                        "speed_px_per_frame": 0.0,
                        "court_ft": [mx_r, my_r],
                        "minimap_court_ft": [mx_r, my_r],
                        "height_ft": mz_r,
                        "interpolated": True,
                    }

        return telemetry_stream, updated_bounces
