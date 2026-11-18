"""
pipeline.py - Master Unified Base Perception Pipeline (Zero-Rule Core).
Streams video frames through:
  1. Court Geometry Calibration & 3D Camera Pose Estimation (solvePnP)
  2. Multi-Player Detection & Canonical Tracking
  3. Micro-Ball Detection & Kinematic Trajectory
  4. Kinematic Bounce Detection & Sub-Pixel Localization (Week 3)
  5. 3D Ballistic Ray-Height Trajectory Reconstruction (Parallax-Free Minimap)
  6. 2D Orthographic Court Minimap & Broadcast HUD Composition (Week 3)
Emits clean, structured perception telemetry JSON and diagnostic preview video.
"""
import json
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from pbbase.court.homography import CourtCalibrator
from pbbase.court.minimap import CourtMinimap
from pbbase.models.detector import PlayerDetector, BallDetector
from pbbase.tracking.player_tracker import CanonicalPlayerTracker
from pbbase.tracking.ball_trajectory import BallTrajectoryTracker, get_speed_color
from pbbase.tracking.bounce_detector import BallBounceDetector, SavitzkyGolayBounceAnalyzer
from pbbase.tracking.ball_3d_tracker import Ball3DReconstructor
from pbbase.data.schema import SCHEMA_VERSION, SCHEMA_PRODUCER


PLAYER_THEMES = [
    {"primary": (0, 145, 255), "name": "P1"},   # Coral / Orange
    {"primary": (255, 175, 0), "name": "P2"},   # Electric Cyan
    {"primary": (80, 220, 0), "name": "P3"},    # Emerald Green
    {"primary": (0, 215, 255), "name": "P4"},   # Bright Gold
]


def _default_ball_weights() -> str:
    tuned_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights", "ball_yolo26_tuned.pt")
    )
    if os.path.exists(tuned_path):
        return "weights/ball_yolo26_tuned.pt"
    best_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights", "ball_yolo26_best.pt")
    )
    if os.path.exists(best_path):
        return "weights/ball_yolo26_best.pt"
    return "weights/ball_colab_dukcv.pt"


def _scrub_unconfirmed_coast_tails(telemetry_stream: List[Dict[str, Any]]) -> None:
    """
    Remove speculative open-loop coasting (`interpolated=True`) frames not bounded
    on both sides by real model detections (`interpolated=False`) of the same track.
    """
    n = len(telemetry_stream)
    i = 0
    while i < n:
        b = telemetry_stream[i].get("ball")
        if b is None:
            i += 1
            continue
        tid = b.get("track_id")
        j = i
        while j + 1 < n:
            nb = telemetry_stream[j + 1].get("ball")
            if nb is None:
                break
            ntid = nb.get("track_id")
            if tid is not None and ntid is not None and ntid != tid:
                break
            j += 1

        obs_indices = [k for k in range(i, j + 1) if not telemetry_stream[k]["ball"].get("interpolated", False)]
        if not obs_indices:
            for k in range(i, j + 1):
                telemetry_stream[k]["ball"] = None
        else:
            first_obs, last_obs = obs_indices[0], obs_indices[-1]
            for k in range(i, first_obs):
                telemetry_stream[k]["ball"] = None
            for k in range(last_obs + 1, j + 1):
                telemetry_stream[k]["ball"] = None
        i = j + 1


def _prune_ghost_ball_tracks(telemetry_stream: List[Dict[str, Any]], scale_px: float = 1.0) -> None:
    """
    Remove short, uncorroborated orphan ball fragments (low-conf bleacher flickers
    or tiny stagnant spots between rallies) so dead-ball search intervals stay clean.
    """
    n = len(telemetry_stream)
    if n < 12:
        return
    norm_s = max(0.45, scale_px)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        b = telemetry_stream[i].get("ball")
        if b is None:
            i += 1
            continue
        tid = b.get("track_id")
        j = i
        while j + 1 < n:
            nb = telemetry_stream[j + 1].get("ball")
            if nb is None:
                break
            ntid = nb.get("track_id")
            if tid is not None and ntid is not None and ntid != tid:
                break
            j += 1
        runs.append((i, j))
        i = j + 1

    for r_idx, (st, en) in enumerate(runs):
        obs = [k for k in range(st, en + 1) if telemetry_stream[k]["ball"].get("pixel") and not telemetry_stream[k]["ball"].get("interpolated", False)]
        if not obs:
            for k in range(st, en + 1):
                telemetry_stream[k]["ball"] = None
            continue
        max_conf = max(float(telemetry_stream[k]["ball"].get("conf", 0.0)) for k in obs)
        p0 = telemetry_stream[obs[0]]["ball"]["pixel"]
        pend = telemetry_stream[obs[-1]]["ball"]["pixel"]
        max_span = max(float(np.hypot(telemetry_stream[k]["ball"]["pixel"][0] - p0[0], telemetry_stream[k]["ball"]["pixel"][1] - p0[1])) for k in obs)

        drop_run = False
        if len(obs) == 1 and max_conf < 0.72:
            drop_run = True
        elif len(obs) <= 4:
            if max_conf < 0.50:
                drop_run = True
            elif max_span < (14.0 * norm_s) and max_conf < 0.68:
                drop_run = True
            else:
                prev_px = telemetry_stream[runs[r_idx - 1][1]]["ball"]["pixel"] if r_idx > 0 and telemetry_stream[runs[r_idx - 1][1]].get("ball") else None
                next_px = telemetry_stream[runs[r_idx + 1][0]]["ball"]["pixel"] if (r_idx + 1) < len(runs) and telemetry_stream[runs[r_idx + 1][0]].get("ball") else None
                jump_prev = float(np.hypot(p0[0] - prev_px[0], p0[1] - prev_px[1])) if prev_px else 0.0
                jump_next = float(np.hypot(next_px[0] - pend[0], next_px[1] - pend[1])) if next_px else 0.0
                if (prev_px and jump_prev > 115.0 * norm_s) and (next_px and jump_next > 115.0 * norm_s) and max_conf < 0.62:
                    drop_run = True

        if drop_run:
            for k in range(st, en + 1):
                telemetry_stream[k]["ball"] = None


class BasePerceptionPipeline:
    def __init__(
        self,
        player_weights: str = "weights/yolo26n.pt",
        ball_weights: Optional[str] = None,
        player_conf: float = 0.25,
        ball_conf: float = 0.08,
        ball_imgsz: Optional[int] = None,
        multiscale: bool = False,
        ball_patch_slices: Optional[List[Tuple[float, float, float, float]]] = None,
        ball_max_coast: int = 3,
        ball_max_jump_px: float = 95.0,
        court_corners: Optional[List[Tuple[float, float]]] = None,
        render_minimap: bool = False,
        minimap_position: str = "bottom-right",
    ):
        resolved_ball_weights = ball_weights or _default_ball_weights()
        resolved_ball_imgsz = ball_imgsz if ball_imgsz is not None else (1312 if "best" in resolved_ball_weights else 1280)

        print(f"[BasePerceptionPipeline] Loading Player Model: {player_weights}")
        self.player_detector = PlayerDetector(player_weights, conf_thresh=player_conf)
        print(f"[BasePerceptionPipeline] Loading Ball Model  : {resolved_ball_weights} (imgsz={resolved_ball_imgsz}, multiscale={multiscale})")
        self.ball_detector = BallDetector(
            resolved_ball_weights,
            conf_thresh=ball_conf,
            imgsz=resolved_ball_imgsz,
            multiscale=multiscale,
            patch_slices=ball_patch_slices,
            patch_imgsz=min(1280, max(640, resolved_ball_imgsz)),
        )

        self._init_court_corners = court_corners
        self.court_calibrator = CourtCalibrator(corners_px=court_corners)
        self.player_tracker = CanonicalPlayerTracker(max_players=4)
        self.ball_tracker = BallTrajectoryTracker(
            max_jump_px=ball_max_jump_px,
            max_coast_frames=ball_max_coast,
        )
        self.bounce_detector = BallBounceDetector()
        self.sg_bounce_analyzer = SavitzkyGolayBounceAnalyzer()
        self.ball_3d_reconstructor = Ball3DReconstructor()
        self.court_minimap = CourtMinimap(court_scale=9.0, padding_x=54, padding_y=50, trail_len=20)
        self.render_minimap = render_minimap
        self.minimap_position = minimap_position

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        out_json: Optional[str] = None,
        out_preview: Optional[str] = None,
        court_corners: Optional[List[Tuple[float, float]]] = None,
        render_minimap: Optional[bool] = None,
        minimap_position: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        do_render_minimap = render_minimap if render_minimap is not None else self.render_minimap
        m_pos = minimap_position if minimap_position is not None else self.minimap_position

        # Reset stateful trackers for new video stream
        self.player_tracker.reset()
        self.ball_tracker.reset()
        self.bounce_detector.reset()
        self.court_minimap.reset()
        # Reset court calibration so a previous clip's corners never leak into this one
        self.court_calibrator = CourtCalibrator()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        effective_corners = court_corners if (court_corners and len(court_corners) == 4) else self._init_court_corners
        if effective_corners and len(effective_corners) == 4:
            self.court_calibrator.calibrate(effective_corners, image_size=(w, h))

        frames_to_read = min(total_frames, max_frames) if max_frames else total_frames
        print(f"[BasePerceptionPipeline] Processing Pass 1 (Perception): {video_path} ({w}x{h} @ {fps:.1f} FPS, {frames_to_read} frames)")

        t0 = time.time()
        telemetry_stream: List[Dict[str, Any]] = []
        frame_idx = 0
        player_counts = []

        # Feature timing accumulators (seconds)
        t_player_total = 0.0
        t_ball_total = 0.0
        t_bounce_total = 0.0
        t_court_total = 0.0
        t_render_total = 0.0

        # =====================================================================
        # PASS 1: Object Detection, Canonical Tracking & Bounce Kinetics
        # =====================================================================
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break

            timestamp_s = frame_idx / fps

            # 1. Calibrate Court + 3D Pose on first frame
            t_c_start = time.time()
            if not self.court_calibrator.result.calibrated:
                self.court_calibrator.auto_calibrate_broadcast(frame, clip_name=os.path.basename(video_path))
            elif not self.court_calibrator.result.pose_calibrated:
                self.court_calibrator.solve_3d_pose(w, h)
            t_court_total += (time.time() - t_c_start)

            # 2. Detect & Track Players
            t_p_start = time.time()
            raw_players = self.player_detector.detect(frame)
            active_players = self.player_tracker.update(raw_players, (h, w), timestamp_s, self.court_calibrator, frame=frame)
            player_counts.append(len(active_players))
            t_player_total += (time.time() - t_p_start)

            # 3. Detect & Track Ball
            t_b_start = time.time()
            raw_ball_cands = self.ball_detector.detect(frame)
            ball_result = self.ball_tracker.process_candidates(
                raw_ball_cands,
                frame_idx,
                timestamp_s,
                self.court_calibrator,
                frame=frame,
                players=active_players,
            )
            t_ball_total += (time.time() - t_b_start)

            # 4. Kinematic Bounce Detection & Sub-Pixel Nadir Fitting (Week 3)
            t_bnc_start = time.time()
            bounce_event = self.bounce_detector.update(
                ball_result,
                frame_idx,
                timestamp_s,
                self.court_calibrator,
                players=active_players,
            )
            t_bounce_total += (time.time() - t_bnc_start)

            frame_record = {
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "court_calibrated": self.court_calibrator.result.calibrated,
                "players": active_players,
                "ball": ball_result,
                "bounce": bounce_event.to_dict() if bounce_event is not None else None,
            }
            telemetry_stream.append(frame_record)

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {frame_idx}/{frames_to_read} frames ({frame_idx / max(0.01, elapsed):.1f} FPS)...")

        cap.release()

        # Scrub speculative dead-end coasting & short orphan ghost tracks BEFORE SG / 3D fitting
        _scrub_unconfirmed_coast_tails(telemetry_stream)
        scale_px_post = max(0.45, self.court_calibrator.get_court_pixel_scale((h, w)))
        _prune_ghost_ball_tracks(telemetry_stream, scale_px=scale_px_post)

        # =====================================================================
        # SAVITZKY-GOLAY BOUNCE REFINEMENT & 3D BALLISTIC RECONSTRUCTION
        # =====================================================================
        t_3d_start = time.time()
        sg_analysis = self.sg_bounce_analyzer.analyze_stream(
            telemetry_stream, self.court_calibrator, fps=fps
        )
        telemetry_stream = sg_analysis.get("smoothed_stream", telemetry_stream)
        sg_bounces = sg_analysis.get("bounces", [b.to_dict() for b in self.bounce_detector.detected_bounces])
        telemetry_stream, all_bounces = self.ball_3d_reconstructor.reconstruct_stream(
            telemetry_stream, sg_bounces, self.court_calibrator
        )
        t_3d_ms = round((time.time() - t_3d_start) * 1000, 1)

        # Ensure zero unconfirmed coasting tails or 1-frame isolated flashes survive reconstruction
        _scrub_unconfirmed_coast_tails(telemetry_stream)
        n_stream = len(telemetry_stream)
        if n_stream >= 6:
            iso_jump_px = 95.0 * scale_px_post
            for idx_s in range(n_stream):
                rec_s = telemetry_stream[idx_s]
                b_s = rec_s.get("ball")
                if b_s is None or not b_s.get("pixel") or rec_s.get("bounce") is not None:
                    continue
                if float(b_s.get("conf", 0.0)) >= 0.72:
                    continue
                bx_s, by_s = b_s["pixel"][0], b_s["pixel"][1]
                tid_s = b_s.get("track_id")
                prev_b = telemetry_stream[idx_s - 1].get("ball") if idx_s > 0 else None
                next_b = telemetry_stream[idx_s + 1].get("ball") if (idx_s + 1) < n_stream else None
                isolated_prev = (
                    prev_b is None
                    or not prev_b.get("pixel")
                    or (tid_s is not None and prev_b.get("track_id") is not None and prev_b.get("track_id") != tid_s)
                    or float(np.hypot(bx_s - prev_b["pixel"][0], by_s - prev_b["pixel"][1])) > iso_jump_px
                )
                isolated_next = (
                    next_b is None
                    or not next_b.get("pixel")
                    or (tid_s is not None and next_b.get("track_id") is not None and next_b.get("track_id") != tid_s)
                    or float(np.hypot(bx_s - next_b["pixel"][0], by_s - next_b["pixel"][1])) > iso_jump_px
                )
                if isolated_prev and isolated_next:
                    rec_s["ball"] = None
        _scrub_unconfirmed_coast_tails(telemetry_stream)

        # Recompute ball continuity metrics & camera history after 3D gap reconstruction
        frames_with_ball = sum(1 for r in telemetry_stream if r.get("ball") is not None)
        raw_frames_with_ball = sum(
            1 for r in telemetry_stream if r.get("ball") is not None and not r["ball"].get("interpolated", False)
        )
        ball_pts_history: List[Optional[Tuple[float, float, float, Optional[int]]]] = []
        for r in telemetry_stream:
            b = r.get("ball")
            if b is not None and b.get("pixel"):
                ball_pts_history.append((b["pixel"][0], b["pixel"][1], b.get("speed_px_per_frame", 0.0), b.get("track_id")))
            else:
                ball_pts_history.append(None)

        # =====================================================================
        # PASS 2: Render H.264 Preview Video with Parallax-Free 3D Minimap HUD
        # =====================================================================
        if out_preview:
            print(f"[BasePerceptionPipeline] Rendering Pass 2 (3D Minimap HUD Preview)...")
            os.makedirs(os.path.dirname(os.path.abspath(out_preview)), exist_ok=True)
            raw_preview_tmp = f"{out_preview}.tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_preview_tmp, fourcc, fps, (w, h))

            self.court_minimap.configure_for_frame(w, h)
            self.court_minimap.reset()
            cap_vis = cv2.VideoCapture(video_path)
            cumulative_bounces = 0

            for r_idx in range(len(telemetry_stream)):
                ok_v, v_frame = cap_vis.read()
                if not ok_v:
                    break

                t_r_start = time.time()
                rec = telemetry_stream[r_idx]
                rec_bounce = rec.get("bounce")
                if rec_bounce is not None:
                    cumulative_bounces += 1
                    if rec_bounce.get("is_ground_bounce", True) and rec_bounce.get("court_ft"):
                        self.court_minimap.add_bounce(
                            (float(rec_bounce["court_ft"][0]), float(rec_bounce["court_ft"][1])),
                            r_idx,
                        )

                annotated = self._render_frame(
                    v_frame,
                    rec,
                    ball_pts_history[: r_idx + 1],
                    r_idx,
                    frames_to_read,
                    fps,
                    cumulative_bounces=cumulative_bounces,
                    render_minimap=do_render_minimap,
                    minimap_position=m_pos,
                )
                writer.write(annotated)
                t_render_total += (time.time() - t_r_start)

            cap_vis.release()
            writer.release()
            self._encode_h264(raw_preview_tmp, out_preview)
            print(f"[BasePerceptionPipeline] Saved H.264 preview to: {out_preview}")

        elapsed_total = time.time() - t0

        corners_list = [[round(x, 1), round(y, 1)] for x, y in self.court_calibrator.result.corners_px]
        court_schema = {
            "calibrated": self.court_calibrator.result.calibrated,
            "reprojection_rmse_ft": round(self.court_calibrator.result.reprojection_rmse_ft, 4),
            "pose_calibrated": self.court_calibrator.result.pose_calibrated,
            "camera_pos_ft": [round(v, 2) for v in self.court_calibrator.result.camera_pos_ft] if self.court_calibrator.result.camera_pos_ft else None,
            "dimensions_ft": {
                "width": 20.0,
                "length": 44.0,
                "net_y": 22.0,
                "nvz_depth": 7.0,
            },
            "corners_px": corners_list,
        }

        bench_schema = {
            "ball_detection_rate_pct": round((frames_with_ball / max(1, frame_idx)) * 100, 1),
            "raw_ball_detection_rate_pct": round((raw_frames_with_ball / max(1, frame_idx)) * 100, 1),
            "frames_with_ball": frames_with_ball,
            "raw_frames_with_ball": raw_frames_with_ball,
            "mean_players_per_frame": round(float(np.mean(player_counts)) if player_counts else 0.0, 2),
            "multiscale_inference": self.ball_detector.multiscale or (min(w, h) <= 480),
            "total_bounces_detected": len(all_bounces),
            "ball_3d_reconstruction_ms": t_3d_ms,
            "stage_latencies_ms": {
                "player_ms": round((t_player_total / max(1, frame_idx)) * 1000, 1),
                "ball_ms": round((t_ball_total / max(1, frame_idx)) * 1000, 1),
                "bounce_ms": round((t_bounce_total / max(1, frame_idx)) * 1000, 2),
                "court_ms": round((t_court_total / max(1, frame_idx)) * 1000, 2),
                "render_ms": round((t_render_total / max(1, frame_idx)) * 1000, 1),
            },
        }

        summary = {
            "schema_version": SCHEMA_VERSION,
            "producer": SCHEMA_PRODUCER,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_path,
            "total_frames_processed": frame_idx,
            "fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "elapsed_s": round(elapsed_total, 2),
            "processing_fps": round(frame_idx / max(0.001, elapsed_total), 2),
            "court": court_schema,
            "benchmark_summary": bench_schema,
            "bounces": all_bounces,
            "total_bounces_detected": len(all_bounces),
            # Top-level backward compatibility keys for reports & CLI runners
            "court_calibrated": self.court_calibrator.result.calibrated,
            "court_reprojection_rmse_ft": round(self.court_calibrator.result.reprojection_rmse_ft, 4),
            "multiscale_inference": self.ball_detector.multiscale,
            "mean_players_per_frame": bench_schema["mean_players_per_frame"],
            "frames_with_ball": frames_with_ball,
            "ball_detection_rate_pct": bench_schema["ball_detection_rate_pct"],
            "player_ms": bench_schema["stage_latencies_ms"]["player_ms"],
            "ball_ms": bench_schema["stage_latencies_ms"]["ball_ms"],
            "bounce_ms": bench_schema["stage_latencies_ms"]["bounce_ms"],
            "court_ms": bench_schema["stage_latencies_ms"]["court_ms"],
            "render_ms": bench_schema["stage_latencies_ms"]["render_ms"],
            "telemetry_stream": telemetry_stream,
        }

        if out_json:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[BasePerceptionPipeline] Saved structured perception JSON to: {out_json}")

        return summary

    def _render_frame(
        self,
        frame: np.ndarray,
        record: Dict[str, Any],
        ball_history: List[Optional[Tuple[float, float, float]]],
        frame_idx: int,
        total_frames: int,
        fps: float,
        cumulative_bounces: Optional[int] = None,
        render_minimap: bool = False,
        minimap_position: str = "bottom-right",
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        is_compact = w < 960
        is_hi_res = max(w, h) >= 1800
        line_thick = 4 if is_hi_res else 2

        # 1. Court Polygon Overlay
        corners = self.court_calibrator.result.corners_px
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 220), thickness=line_thick, lineType=cv2.LINE_AA)

            # Center Net Line
            p_net_l = self.court_calibrator.court_to_pixel(0.0, 22.0)
            p_net_r = self.court_calibrator.court_to_pixel(20.0, 22.0)
            if p_net_l and p_net_r:
                cv2.line(vis, (int(p_net_l[0]), int(p_net_l[1])), (int(p_net_r[0]), int(p_net_r[1])), (255, 255, 255), line_thick, cv2.LINE_AA)

        # 2. Player Halos & Badges
        for p in record["players"]:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            pid = p["id"]
            theme = PLAYER_THEMES[(pid - 1) % len(PLAYER_THEMES)]
            col = theme["primary"]
            cx = (x1 + x2) // 2
            bw = max(20, x2 - x1)

            # Ground AR Halo
            axes = (int(bw * 0.65), max(5, int(bw * 0.18)))
            cv2.ellipse(vis, (cx, min(h - 1, y2)), axes, 0, 0, 360, col, line_thick, lineType=cv2.LINE_AA)

            # Player Floating Pill
            lbl = f"P{pid}"
            if p.get("court_ft"):
                lbl += f" ({p['court_ft'][0]:.1f},{p['court_ft'][1]:.1f})ft"
            if p.get("speed_mph", 0.0) > 1.0:
                lbl += f" {p['speed_mph']:.1f}mph"

            p_font = 0.27 if is_compact else (0.72 if is_hi_res else 0.38)
            char_w = 5 if is_compact else (14 if is_hi_res else 8)
            pill_h = 12 if is_compact else (34 if is_hi_res else 18)
            pill_w = len(lbl) * char_w + (20 if is_hi_res else 12)
            bx1 = max(4, min(w - pill_w - 4, cx - pill_w // 2))
            by1 = max(4, y1 - (pill_h + 6 if is_hi_res else pill_h + 3))
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + pill_h), (15, 23, 42), -1)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + pill_h), col, 2 if is_hi_res else 1, cv2.LINE_AA)
            cv2.putText(vis, lbl, (bx1 + (9 if is_hi_res else 5), by1 + pill_h - (8 if is_hi_res else 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, p_font, (240, 240, 240), 2 if is_hi_res else 1, cv2.LINE_AA)

        # 3. Ball Trajectory Ribbon & Reticle
        ball = record["ball"]
        if ball is not None and ball.get("pixel"):
            trail_len = 16
            scale_px = self.court_calibrator.get_court_pixel_scale((h, w))
            max_trail_step_px = max(72.0, 108.0 * scale_px * (30.0 / max(15.0, fps)))
            valid_pts = []
            for i in range(len(ball_history) - 1, max(-1, len(ball_history) - 1 - trail_len), -1):
                pt = ball_history[i]
                if pt is None:
                    break
                if valid_pts:
                    nxt = valid_pts[-1]
                    tid_p = pt[3] if len(pt) > 3 else None
                    tid_n = nxt[3] if len(nxt) > 3 else None
                    if tid_p is not None and tid_n is not None and tid_p != tid_n:
                        break
                    if float(np.hypot(nxt[0] - pt[0], nxt[1] - pt[1])) > max_trail_step_px:
                        break
                valid_pts.append(pt)
            valid_pts.reverse()

            if len(valid_pts) >= 2:
                overlay = vis.copy()
                for k in range(len(valid_pts) - 1):
                    p1 = (int(valid_pts[k][0]), int(valid_pts[k][1]))
                    p2 = (int(valid_pts[k + 1][0]), int(valid_pts[k + 1][1]))
                    spd = valid_pts[k + 1][2]
                    col = get_speed_color(spd)
                    cv2.line(overlay, p1, p2, col, max(2, int((k + 1) / len(valid_pts) * 5)), cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.50, vis, 0.50, 0, vis)

            bx, by = int(ball["pixel"][0]), int(ball["pixel"][1])
            spd_px = ball.get("speed_px_per_frame", 0.0)
            ret_col = get_speed_color(spd_px)
            is_interpolated = ball.get("interpolated", False)
            rad = 14 if is_hi_res else 8
            if is_interpolated:
                cv2.circle(vis, (bx, by), rad, (120, 200, 255), 2 if is_hi_res else 1, cv2.LINE_AA)
                cv2.circle(vis, (bx, by), 4 if is_hi_res else 2, (120, 200, 255), -1, cv2.LINE_AA)
                cv2.putText(vis, "3D-EST", (bx + (18 if is_hi_res else 10), by + (20 if is_hi_res else 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65 if is_hi_res else 0.35, (120, 200, 255), 2 if is_hi_res else 1, cv2.LINE_AA)
            else:
                cv2.circle(vis, (bx, by), rad, ret_col, 3 if is_hi_res else 2, cv2.LINE_AA)
                cv2.circle(vis, (bx, by), 4 if is_hi_res else 2, (255, 255, 255), -1, cv2.LINE_AA)

            if ball.get("speed_mph", 0.0) > 1.0:
                cv2.putText(vis, f"{ball['speed_mph']:.1f} MPH", (bx + (20 if is_hi_res else 12), by - (8 if is_hi_res else 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.78 if is_hi_res else 0.42, (0, 255, 220), 2 if is_hi_res else 1, cv2.LINE_AA)

        # 4. Bounce Ripple Indicator on Camera Frame (Week 3)
        bounce = record.get("bounce")
        if bounce is not None and bounce.get("pixel"):
            b_px, b_py = int(bounce["pixel"][0]), int(bounce["pixel"][1])
            b_col = (80, 220, 80) if bounce.get("is_ground_bounce", True) else (60, 60, 230)
            cv2.circle(vis, (b_px, b_py), 28 if is_hi_res else 16, b_col, 3 if is_hi_res else 2, cv2.LINE_AA)
            cv2.circle(vis, (b_px, b_py), 14 if is_hi_res else 8, (255, 255, 255), 2 if is_hi_res else 1, cv2.LINE_AA)
            c_xy = bounce.get("court_ft")
            if c_xy:
                b_lbl = f"BOUNCE ({c_xy[0]:.1f}, {c_xy[1]:.1f})ft"
                cv2.putText(vis, b_lbl, (b_px - (70 if is_hi_res else 40), b_py - (36 if is_hi_res else 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75 if is_hi_res else 0.42, b_col, 2 if is_hi_res else 1, cv2.LINE_AA)

        # 5. Top-Down Court Minimap HUD Composition (Week 3)
        if render_minimap and self.court_calibrator.result.calibrated:
            eff_pos = "top-right" if ((h > w or w < 900) and minimap_position == "bottom-right") else minimap_position
            minimap_canvas = self.court_minimap.render(
                players=record.get("players"),
                ball=record.get("ball"),
                frame_idx=frame_idx,
            )
            vis = self.court_minimap.compose_onto_frame(
                vis,
                minimap_canvas,
                position=eff_pos,
                alpha=0.88,
            )

        # 6. Clean Telemetry Header HUD (scales compactly for <= 720px clips)
        ball_status = "3D-EST" if (ball and ball.get("interpolated")) else ("LOCKED" if ball else "SEARCH")
        total_bncs = cumulative_bounces if cumulative_bounces is not None else len(self.bounce_detector.detected_bounces)
        status_line = f"F: {frame_idx + 1}/{total_frames} | Players: {len(record['players'])}/4 | Ball: {ball_status} | Bounces: {total_bncs}"
        sec = frame_idx / fps

        hud_title = "PICKVIZ (3D MINIMAP)" if render_minimap else "PICKVIZ (3D PERCEPTION)"
        if is_compact:
            cv2.rectangle(vis, (6, 6), (242, 44), (15, 23, 42), -1)
            cv2.rectangle(vis, (6, 6), (242, 44), (0, 255, 220), 1)
            cv2.putText(vis, hud_title, (11, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (0, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(vis, status_line, (11, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (250, 250, 250), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Time: {int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*100):02d} | Zero Rules Core",
                        (11, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (160, 160, 160), 1, cv2.LINE_AA)
        elif is_hi_res:
            cv2.rectangle(vis, (24, 24), (820, 160), (15, 23, 42), -1)
            cv2.rectangle(vis, (24, 24), (820, 160), (0, 255, 220), 2)
            cv2.putText(vis, hud_title, (44, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 220), 2, cv2.LINE_AA)
            cv2.putText(vis, status_line, (44, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Time: {int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*100):02d} | Zero Rules (Pure Perception)",
                        (44, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (160, 160, 160), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(vis, (16, 16), (410, 84), (15, 23, 42), -1)
            cv2.rectangle(vis, (16, 16), (410, 84), (0, 255, 220), 1)
            cv2.putText(vis, hud_title, (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(vis, status_line, (26, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (250, 250, 250), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Time: {int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*100):02d} | Zero Rules (Pure Perception)",
                        (26, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA)

        return vis

    def _encode_h264(self, raw_path: str, target_path: str):
        encode_h264(raw_path, target_path)


def encode_h264(raw_path: str, target_path: str):
    """Transcode a raw OpenCV MP4 file into H.264/yuv420p web-playable MP4."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y", "-v", "error", "-i", raw_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            target_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return
        except Exception as e:
            print(f"[warning] FFmpeg encode failed ({e}). Retaining raw video.")

    if os.path.exists(raw_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(raw_path, target_path)
