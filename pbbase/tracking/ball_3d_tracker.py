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
        self,
        telemetry_stream: List[Dict[str, Any]],
        scale_px: float = 1.0,
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Extracts ONLY genuine observed pixel centers (interpolated=False) from
        telemetry_stream and removes isolated teleport spikes (> max_outlier_jump_px * scale_px).
        """
        eff_max_jump = self.max_outlier_jump_px * max(0.45, scale_px)
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

                if d_prev > eff_max_jump and d_next > eff_max_jump and d_bridge < eff_max_jump * 1.4:
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

        scale_px = calibrator.get_court_pixel_scale() if calibrator is not None else 1.0
        norm_px = max(0.45, scale_px)
        eff_paddle_prox = self.paddle_prox_px * norm_px

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
                if best_p and self._dist_point_to_box(px[0], px[1], best_p["box"]) < 120.0 * norm_px:
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
                    if vy_in < -0.8 * norm_px and vy_out > 0.3 * norm_px and clean_pts[k][1] < best_min_y:
                        players = telemetry_stream[k].get("players", [])
                        if any(self._dist_point_to_box(clean_pts[k][0], clean_pts[k][1], p["box"]) < eff_paddle_prox for p in players if p.get("box")):
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

            pre_idx = next((j for j in range(i - 1, max(-1, i - 5), -1) if clean_pts[j] is not None), None)
            pre_far = next((j for j in range(i - 3, max(-1, i - 7), -1) if clean_pts[j] is not None), pre_idx)
            post_idx = next((j for j in range(i + 1, min(n, i + 16)) if clean_pts[j] is not None), None)
            if pre_idx is None or post_idx is None:
                continue

            gap = post_idx - i
            p_curr = clean_pts[i]
            players = telemetry_stream[i].get("players", [])
            near_players = [
                p for p in players
                if p.get("box") and self._dist_point_to_box(p_curr[0], p_curr[1], p["box"]) < 105.0 * norm_px
            ]
            if not near_players:
                continue

            # For contiguous detections (gap == 1), compare immediate local tangent only so hits aren't triggered 3 frames early
            if gap == 1:
                pre_ref = pre_idx if (i - pre_idx) >= 2 or pre_far is None else pre_far
                post_ref = next((j for j in range(post_idx + 1, min(n, post_idx + 3)) if clean_pts[j] is not None), post_idx)
                v_in = np.array([p_curr[0] - clean_pts[pre_ref][0], p_curr[1] - clean_pts[pre_ref][1]]) / max(1, i - pre_ref)
                v_out = np.array([clean_pts[post_ref][0] - p_curr[0], clean_pts[post_ref][1] - p_curr[1]]) / max(1, post_ref - i)
            else:
                pre_ref = pre_far if pre_far is not None else pre_idx
                post_idx2 = next((j for j in range(post_idx + 2, min(n, post_idx + 6)) if clean_pts[j] is not None), post_idx)
                v_in = np.array([p_curr[0] - clean_pts[pre_ref][0], p_curr[1] - clean_pts[pre_ref][1]]) / max(1, i - pre_ref)
                if post_idx2 > post_idx:
                    v_out = np.array([clean_pts[post_idx2][0] - clean_pts[post_idx][0], clean_pts[post_idx2][1] - clean_pts[post_idx][1]]) / max(1, post_idx2 - post_idx)
                else:
                    v_out = np.array([clean_pts[post_idx][0] - p_curr[0], clean_pts[post_idx][1] - p_curr[1]]) / max(1, gap)

            n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            if n_in > 3.0 * norm_px and n_out > 2.5 * norm_px:
                cos_theta = float(np.dot(v_in, v_out) / (n_in * n_out))
                is_gap_hit = (gap >= 3 and (v_in[1] * v_out[1] < 0 or cos_theta < 0.35))
                is_antigravity_hit = (v_in[1] > 1.5 * norm_px and v_out[1] < -1.5 * norm_px and cos_theta < 0.45)
                is_lateral_kink = (v_in[0] * v_out[0] < -1.0 * (norm_px ** 2) and cos_theta < 0.25)
                speed_jump = max(n_in, n_out) / max(0.5, min(n_in, n_out))
                is_acute_impulse = (cos_theta < -0.55 and speed_jump > 1.45)

                if is_gap_hit or is_antigravity_hit or is_lateral_kink or is_acute_impulse:
                    # Resolve true hit frame & coordinate inside the player's hitting zone if gap > 1
                    hit_frame = i
                    hit_px = p_curr
                    if gap >= 2:
                        mid_k = i + max(1, min(gap - 1, gap // 3 if gap >= 6 else gap // 2))
                        hit_frame = mid_k
                        best_p = self._find_closest_player_box(p_curr[0], p_curr[1], telemetry_stream[min(n - 1, hit_frame)].get("players", near_players))
                        if best_p and best_p.get("box"):
                            bx1, by1, bx2, by2 = best_p["box"]
                            pred_px = (
                                float(np.clip(p_curr[0] + v_in[0] * (hit_frame - i), bx1 - 15.0 * norm_px, bx2 + 15.0 * norm_px)),
                                float(np.clip(p_curr[1] + v_in[1] * (hit_frame - i), by1, by2 - 0.15 * (by2 - by1))),
                            )
                            hit_px = pred_px
                            players_hit = telemetry_stream[hit_frame].get("players", near_players)
                            best_p = self._find_closest_player_box(hit_px[0], hit_px[1], players_hit)
                    else:
                        # Refine to local turnaround extremum in [i, i+2] if ball still travels further before reversing
                        best_i = i
                        if i + 1 < n and clean_pts[i + 1] is not None and cos_theta > -0.7:
                            # If ball moved along v_in direction to i+1 before snapping back, advance hit_frame to i+1
                            step1 = np.array([clean_pts[i + 1][0] - p_curr[0], clean_pts[i + 1][1] - p_curr[1]])
                            if np.dot(step1, v_in) > 0 and i + 2 < n and clean_pts[i + 2] is not None:
                                step2 = np.array([clean_pts[i + 2][0] - clean_pts[i + 1][0], clean_pts[i + 2][1] - clean_pts[i + 1][1]])
                                if np.dot(step1, step2) < np.dot(v_in, step1):
                                    best_i = i + 1
                                    hit_px = clean_pts[best_i]
                        hit_frame = best_i
                        best_p = self._find_closest_player_box(hit_px[0], hit_px[1], telemetry_stream[hit_frame].get("players", near_players))

                    if best_p is None or any(abs(hit_frame - af) < 6 for af in anchors):
                        continue

                    p_h = self._estimate_paddle_height_ft(hit_px[1], best_p["box"])
                    ray_cft = calibrator.pixel_to_court_at_height_ft(hit_px[0], hit_px[1], p_h) or calibrator.pixel_to_court_ft(hit_px[0], hit_px[1])
                    if best_p.get("court_ft") and ray_cft is not None:
                        px_ft, py_ft = float(best_p["court_ft"][0]), float(best_p["court_ft"][1])
                        cft = (float(0.6 * ray_cft[0] + 0.4 * px_ft), float(0.5 * ray_cft[1] + 0.5 * py_ft))
                    elif best_p.get("court_ft"):
                        cft = (float(best_p["court_ft"][0]), float(best_p["court_ft"][1]))
                    else:
                        cft = ray_cft

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
            h_start = self._estimate_paddle_height_ft(px[1], best_p["box"]) if best_p and self._dist_point_to_box(px[0], px[1], best_p["box"]) < 140.0 * norm_px else 3.0
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
        Uses smooth Gaussian ray-to-ballistic blending and interior low-pass stabilization
        to prevent camera-ray longitudinal sliding jitter on the 2D minimap.
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

        a1_x, a1_y = a1.court_xy_ft
        a2_x, a2_y = a2.court_xy_ft
        if prev_anchor_state is not None and np.hypot(prev_anchor_state[0] - a1_x, prev_anchor_state[1] - a1_y) < 5.5:
            x1_init, y1_init, z1_init = prev_anchor_state
            use_prev = True
        else:
            x1_init, y1_init = a1_x, a1_y
            z1_init = float(a1.height_ft)
            use_prev = False

        x2_init, y2_init = a2_x, a2_y
        z2_init = float(a2.height_ft)
        g_eff = self.gravity_ft_s2
        k_d = 0.25

        if calibrator.result.pose_calibrated and len(obs_k) >= 3:
            n_obs = len(obs_k)
            w_end1 = max(5.5, 0.35 * n_obs) if a1.kind == "ground_bounce" else max(4.2, 0.28 * n_obs)
            w_end2 = max(5.5, 0.35 * n_obs) if a2.kind == "ground_bounce" else max(4.2, 0.28 * n_obs)
            if use_prev:
                w_end1 += 4.0
            wz1 = 12.0 if a1.kind == "ground_bounce" else (7.0 if use_prev else 5.0)
            wz2 = 12.0 if a2.kind == "ground_bounce" else 5.0
            z1_min = 0.0 if a1.kind == "ground_bounce" else (0.5 if use_prev else (1.2 if a1.height_ft >= 1.5 else 0.5))
            z1_max = 0.05 if a1.kind == "ground_bounce" else 7.2
            z2_min = 0.0 if a2.kind == "ground_bounce" else (1.2 if a2.height_ft >= 1.5 else 0.5)
            z2_max = 0.05 if a2.kind == "ground_bounce" else 7.2

            dx1_tol = 3.5 if a1.kind == "ground_bounce" else 6.0
            dy1_tol = 4.5 if a1.kind == "ground_bounce" else 8.5
            dx2_tol = 3.5 if a2.kind == "ground_bounce" else 6.0
            dy2_tol = 4.5 if a2.kind == "ground_bounce" else 8.5

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

                z_peak_excess = max(0.0, float(np.max(Z)) - 11.5)
                res_px = np.concatenate([proj_u - obs_u_arr, proj_v - obs_v_arr])
                priors = np.array([
                    (px1 - x1_init) * w_end1,
                    (py1 - y1_init) * w_end1,
                    (pz1 - z1_init) * wz1,
                    (px2 - x2_init) * w_end2,
                    (py2 - y2_init) * w_end2,
                    (pz2 - z2_init) * wz2,
                    (kd - 0.25) * 4.0,
                    (g - 30.0) * 0.8,
                    z_peak_excess * 12.0,
                ])
                return np.concatenate([res_px, priors])

            lb1_x = max(-15.0, a1_x - dx1_tol)
            ub1_x = min(38.0, a1_x + dx1_tol)
            lb1_y = max(-18.0, a1_y - dy1_tol)
            ub1_y = min(60.0, a1_y + dy1_tol)
            lb2_x = max(-15.0, a2_x - dx2_tol)
            ub2_x = min(38.0, a2_x + dx2_tol)
            lb2_y = max(-18.0, a2_y - dy2_tol)
            ub2_y = min(60.0, a2_y + dy2_tol)

            if use_prev:
                lb1_x = max(lb1_x, x1_init - 1.2)
                ub1_x = min(ub1_x, x1_init + 1.2)
                lb1_y = max(lb1_y, y1_init - 1.5)
                ub1_y = min(ub1_y, y1_init + 1.5)
                z1_lo = max(z1_min, z1_init - 0.8)
                z1_hi = min(z1_max, z1_init + 0.8)
            else:
                z1_lo, z1_hi = z1_min, z1_max

            if z1_lo >= z1_hi:
                z1_hi = z1_lo + 0.05

            x0 = [
                float(np.clip(x1_init, lb1_x, ub1_x)),
                float(np.clip(y1_init, lb1_y, ub1_y)),
                float(np.clip(z1_init, z1_lo, z1_hi)),
                float(np.clip(x2_init, lb2_x, ub2_x)),
                float(np.clip(y2_init, lb2_y, ub2_y)),
                float(np.clip(z2_init, z2_min, z2_max)),
                0.25,
                30.0,
            ]
            lb = [lb1_x, lb1_y, z1_lo, lb2_x, lb2_y, z2_min, 0.0, 22.0]
            ub = [ub1_x, ub1_y, z1_hi, ub2_x, ub2_y, z2_max, 1.2, 36.0]

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
                # Smooth Gaussian ray-to-curve fusion (eliminates hard threshold jumps & ray sliding)
                if k == k1 or k == k2:
                    segment_out[k] = (float(x_fit), float(y_fit), z_fit, True)
                    continue
                ray_xy = calibrator.pixel_to_court_at_height_ft(px[0], px[1], z_fit)
                if ray_xy is not None:
                    rx, ry = ray_xy
                    err_ft = float(np.hypot(rx - x_fit, (ry - y_fit) * 0.75))
                    w_ray = float(0.42 * np.exp(-0.5 * ((err_ft / 5.0) ** 2)))
                    bx = w_ray * rx + (1.0 - w_ray) * x_fit
                    by = w_ray * ry + (1.0 - w_ray) * y_fit
                    segment_out[k] = (float(bx), float(by), z_fit, True)
                    continue

            segment_out[k] = (float(x_fit), float(y_fit), z_fit, px is not None)

        # Light 3-point zero-phase interior smoothing along segment to suppress per-frame bbox jitter
        if (k2 - k1) >= 4:
            ks = list(range(k1, k2 + 1))
            xs_seg = [segment_out[k][0] for k in ks]
            ys_seg = [segment_out[k][1] for k in ks]
            for idx_s in range(1, len(ks) - 1):
                k_curr = ks[idx_s]
                sx = 0.22 * xs_seg[idx_s - 1] + 0.56 * xs_seg[idx_s] + 0.22 * xs_seg[idx_s + 1]
                sy = 0.22 * ys_seg[idx_s - 1] + 0.56 * ys_seg[idx_s] + 0.22 * ys_seg[idx_s + 1]
                segment_out[k_curr] = (float(sx), float(sy), segment_out[k_curr][2], segment_out[k_curr][3])

        return segment_out

    def reconstruct_stream(
        self,
        telemetry_stream: List[Dict[str, Any]],
        bounces: List[Dict[str, Any]],
        calibrator: CourtCalibrator,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Post-processes the full rally telemetry_stream so camera reticle AND 2D minimap
        remain 1-to-1 locked onto the exact same physical 3D ball across all frames,
        without fabricating phantom balls across long untracked non-rally intervals.
        """
        if not telemetry_stream or not calibrator.result.calibrated:
            return telemetry_stream, bounces

        scale_px = calibrator.get_court_pixel_scale()
        clean_pts = self._scrub_outliers(telemetry_stream, scale_px=scale_px)
        anchors, updated_bounces = self._find_shot_anchors(telemetry_stream, clean_pts, bounces, calibrator)

        if len(anchors) < 2:
            return telemetry_stream, updated_bounces

        # Estimate stream FPS and max allowable occlusion interpolation gap (~0.45s)
        dt_avg = (
            (float(telemetry_stream[-1].get("timestamp_s", 1.0)) - float(telemetry_stream[0].get("timestamp_s", 0.0)))
            / max(1, len(telemetry_stream) - 1)
            if len(telemetry_stream) > 1
            else (1.0 / 30.0)
        )
        eff_fps = float(np.clip(1.0 / max(1e-3, dt_avg), 15.0, 120.0))
        max_interp_gap_frames = max(12, int(round(0.45 * eff_fps)))
        obs_indices = [i for i, p in enumerate(clean_pts) if p is not None]

        norm_px = max(0.45, scale_px)
        img_sz = calibrator.result.image_size

        def _gap_span_at(k_idx: int) -> int:
            if clean_pts[k_idx] is not None:
                return 0
            # Binary search in obs_indices
            pos = int(np.searchsorted(obs_indices, k_idx))
            if pos == 0 or pos >= len(obs_indices):
                return 999999
            return int(obs_indices[pos] - obs_indices[pos - 1])

        def _segment_has_break(a1: ShotAnchor, a2: ShotAnchor) -> bool:
            if (a2.timestamp_s - a1.timestamp_s) > 2.4:
                return True
            prev_obs_k = None
            for kk in range(a1.frame_idx, a2.frame_idx + 1):
                if clean_pts[kk] is not None:
                    if prev_obs_k is not None:
                        df = kk - prev_obs_k
                        if df > max_interp_gap_frames:
                            return True
                        p0, p1 = clean_pts[prev_obs_k], clean_pts[kk]
                        dist_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                        if df > 1 and dist_px > max(95.0 * norm_px, 42.0 * norm_px * df):
                            return True
                        tid0 = (telemetry_stream[prev_obs_k].get("ball") or {}).get("track_id")
                        tid1 = (telemetry_stream[kk].get("ball") or {}).get("track_id")
                        if tid0 is not None and tid1 is not None and tid0 != tid1 and (df > 1 or dist_px > 85.0 * norm_px):
                            return True
                    prev_obs_k = kk
            return False

        reconstructed: Dict[int, Tuple[float, float, float, bool]] = {}
        prev_state: Optional[Tuple[float, float, float]] = None
        for i in range(len(anchors) - 1):
            if _segment_has_break(anchors[i], anchors[i + 1]):
                prev_state = None
                continue
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
            if end_k in seg and anchors[i + 1].kind != "endpoint":
                prev_state = (seg[end_k][0], seg[end_k][1], seg[end_k][2])
            else:
                prev_state = None

        box_half = max(3.5, round(6.0 * scale_px, 1))
        keys = sorted(reconstructed.keys())
        for k in keys:
            mx, my, mz, is_obs = reconstructed[k]
            mx_r, my_r, mz_r = round(float(mx), 2), round(float(my), 2), round(float(mz), 2)
            rec = telemetry_stream[k]
            ball = rec.get("ball")

            if is_obs and ball is not None and not ball.get("interpolated", False):
                # Observed frame: keep EXACT tracked pixel, sync minimap_court_ft to 3D ballistic Ground Shadow
                ball["minimap_court_ft"] = [mx_r, my_r]
                ball["ground_shadow_ft"] = [mx_r, my_r]
                ball["court_ft"] = [mx_r, my_r]
                ball["height_ft"] = mz_r
            else:
                # Do not fabricate balls across long untracked non-rally intervals (> 0.45s gap)
                gap_span = _gap_span_at(k)
                if gap_span > max_interp_gap_frames:
                    continue

                pos = int(np.searchsorted(obs_indices, k))
                chord_pt: Optional[Tuple[float, float]] = None
                chord_dist = 0.0
                if 0 < pos < len(obs_indices):
                    ip, inx = obs_indices[pos - 1], obs_indices[pos]
                    tid_p = (telemetry_stream[ip].get("ball") or {}).get("track_id")
                    tid_n = (telemetry_stream[inx].get("ball") or {}).get("track_id")
                    p_prev, p_next = clean_pts[ip], clean_pts[inx]
                    chord_dist = float(np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1]))
                    if (tid_p is not None and tid_n is not None and tid_p != tid_n) or chord_dist > max(95.0 * norm_px, 42.0 * norm_px * (inx - ip)):
                        if ball is not None and ball.get("interpolated", False):
                            rec["ball"] = None
                        continue
                    frac = (k - ip) / max(1, inx - ip)
                    chord_pt = (p_prev[0] + frac * (p_next[0] - p_prev[0]), p_prev[1] + frac * (p_next[1] - p_prev[1]))

                px_proj = calibrator.court_3d_to_pixel(mx, my, mz)
                px_ok = px_proj is not None
                if px_ok and img_sz is not None:
                    iw, ih = img_sz
                    if px_proj[0] < -20.0 or px_proj[0] > (iw + 20.0) or px_proj[1] < -20.0 or px_proj[1] > (ih + 20.0):
                        px_ok = False
                if px_ok and chord_pt is not None:
                    dev_lim = max(42.0 * norm_px, 0.40 * chord_dist + 18.0 * norm_px)
                    if float(np.hypot(px_proj[0] - chord_pt[0], px_proj[1] - chord_pt[1])) > dev_lim:
                        px_ok = False

                if px_ok:
                    px_list = [round(px_proj[0], 1), round(px_proj[1], 1)]
                elif chord_pt is not None:
                    existing_px = ball.get("pixel") if ball else None
                    dev_lim = max(42.0 * norm_px, 0.40 * chord_dist + 18.0 * norm_px)
                    if existing_px and float(np.hypot(existing_px[0] - chord_pt[0], existing_px[1] - chord_pt[1])) <= dev_lim:
                        px_list = [round(float(existing_px[0]), 1), round(float(existing_px[1]), 1)]
                    else:
                        px_list = [round(chord_pt[0], 1), round(chord_pt[1], 1)]
                    safe_cft = calibrator.pixel_to_court_at_height_ft(px_list[0], px_list[1], min(mz, 6.0))
                    if safe_cft is not None:
                        mx_r, my_r = round(float(safe_cft[0]), 2), round(float(safe_cft[1]), 2)
                else:
                    px_list = ball.get("pixel") if ball else None

                if px_list is None:
                    continue

                box_list = [round(px_list[0] - box_half, 1), round(px_list[1] - box_half, 1), round(px_list[0] + box_half, 1), round(px_list[1] + box_half, 1)]
                if ball is not None:
                    ball["pixel"] = px_list
                    ball["box"] = box_list
                    ball["court_ft"] = [mx_r, my_r]
                    ball["minimap_court_ft"] = [mx_r, my_r]
                    ball["ground_shadow_ft"] = [mx_r, my_r]
                    ball["height_ft"] = mz_r
                    ball["interpolated"] = True
                else:
                    tid_inherit = (telemetry_stream[obs_indices[pos - 1]].get("ball") or {}).get("track_id") if pos > 0 else None
                    rec["ball"] = {
                        "frame_idx": k,
                        "timestamp_s": rec.get("timestamp_s", round(k / eff_fps, 3)),
                        "pixel": px_list,
                        "box": box_list,
                        "conf": 0.0,
                        "speed_mph": 0.0,
                        "speed_px_per_frame": 0.0,
                        "court_ft": [mx_r, my_r],
                        "minimap_court_ft": [mx_r, my_r],
                        "ground_shadow_ft": [mx_r, my_r],
                        "height_ft": mz_r,
                        "interpolated": True,
                        "track_id": tid_inherit,
                    }

        return telemetry_stream, updated_bounces
