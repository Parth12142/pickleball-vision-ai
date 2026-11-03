"""
pipeline.py - Master Unified Base Perception Pipeline (Zero-Rule Core).
Streams video frames through:
  1. Court Geometry Calibration
  2. Multi-Player Detection & Canonical Tracking
  3. Micro-Ball Detection & Kinematic Trajectory
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
from pbbase.models.detector import PlayerDetector, BallDetector
from pbbase.tracking.player_tracker import CanonicalPlayerTracker
from pbbase.tracking.ball_trajectory import BallTrajectoryTracker, get_speed_color


PLAYER_THEMES = [
    {"primary": (0, 145, 255), "name": "P1"},   # Coral / Orange
    {"primary": (255, 175, 0), "name": "P2"},   # Electric Cyan
    {"primary": (80, 220, 0), "name": "P3"},    # Emerald Green
    {"primary": (0, 215, 255), "name": "P4"},   # Bright Gold
]


class BasePerceptionPipeline:
    def __init__(
        self,
        player_weights: str = "weights/yolo26n.pt",
        ball_weights: str = "weights/ball_colab_dukcv.pt",
        player_conf: float = 0.25,
        ball_conf: float = 0.15,
        ball_imgsz: int = 1280,
    ):
        print(f"[BasePerceptionPipeline] Loading Player Model: {player_weights}")
        self.player_detector = PlayerDetector(player_weights, conf_thresh=player_conf)
        print(f"[BasePerceptionPipeline] Loading Ball Model  : {ball_weights} (imgsz={ball_imgsz})")
        self.ball_detector = BallDetector(ball_weights, conf_thresh=ball_conf, imgsz=ball_imgsz)

        self.court_calibrator = CourtCalibrator()
        self.player_tracker = CanonicalPlayerTracker(max_players=4)
        self.ball_tracker = BallTrajectoryTracker(max_jump_px=80.0)

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        out_json: Optional[str] = None,
        out_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_read = min(total_frames, max_frames) if max_frames else total_frames
        print(f"[BasePerceptionPipeline] Processing: {video_path} ({w}x{h} @ {fps:.1f} FPS, {frames_to_read} frames)")

        # Prepare preview writer if requested
        writer = None
        raw_preview_tmp = None
        if out_preview:
            os.makedirs(os.path.dirname(os.path.abspath(out_preview)), exist_ok=True)
            raw_preview_tmp = f"{out_preview}.tmp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_preview_tmp, fourcc, fps, (w, h))

        t0 = time.time()
        telemetry_stream: List[Dict[str, Any]] = []
        frame_idx = 0
        ball_pts_history: List[Tuple[float, float, float]] = []  # (cx, cy, speed_px)
        frames_with_ball = 0
        player_counts = []

        # Feature timing accumulators (seconds)
        t_player_total = 0.0
        t_ball_total = 0.0
        t_court_total = 0.0
        t_render_total = 0.0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break

            timestamp_s = frame_idx / fps

            # 1. Calibrate Court on first frame
            t_c_start = time.time()
            if not self.court_calibrator.result.calibrated:
                self.court_calibrator.auto_calibrate_broadcast(frame)
            t_court_total += (time.time() - t_c_start)

            # 2. Detect & Track Players
            t_p_start = time.time()
            raw_players = self.player_detector.detect(frame)
            active_players = self.player_tracker.update(raw_players, (h, w), timestamp_s, self.court_calibrator)
            player_counts.append(len(active_players))
            t_player_total += (time.time() - t_p_start)

            # 3. Detect & Track Ball
            t_b_start = time.time()
            raw_ball_cands = self.ball_detector.detect(frame)
            ball_result = self.ball_tracker.process_candidates(raw_ball_cands, frame_idx, timestamp_s, self.court_calibrator)
            if ball_result is not None:
                frames_with_ball += 1
                bx, by = ball_result["pixel"]
                bspeed = ball_result.get("speed_px_per_frame", 0.0)
                ball_pts_history.append((bx, by, bspeed))
            else:
                ball_pts_history.append(None)
            t_ball_total += (time.time() - t_b_start)

            # Telemetry record for this frame
            frame_record = {
                "frame_idx": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "court_calibrated": self.court_calibrator.result.calibrated,
                "players": active_players,
                "ball": ball_result,
            }
            telemetry_stream.append(frame_record)

            # 4. Render diagnostic overlay if preview requested
            if writer:
                t_r_start = time.time()
                annotated = self._render_frame(frame, frame_record, ball_pts_history, frame_idx, frames_to_read, fps)
                writer.write(annotated)
                t_render_total += (time.time() - t_r_start)

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {frame_idx}/{frames_to_read} frames ({frame_idx / max(0.01, elapsed):.1f} FPS)...")

        cap.release()
        elapsed_total = time.time() - t0

        if writer:
            writer.release()
            self._encode_h264(raw_preview_tmp, out_preview)
            print(f"[BasePerceptionPipeline] Saved H.264 preview to: {out_preview}")

        summary = {
            "video_path": video_path,
            "total_frames_processed": frame_idx,
            "fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "elapsed_s": round(elapsed_total, 2),
            "processing_fps": round(frame_idx / max(0.001, elapsed_total), 2),
            "court_calibrated": self.court_calibrator.result.calibrated,
            "court_reprojection_rmse_ft": round(self.court_calibrator.result.reprojection_rmse_ft, 4),
            "mean_players_per_frame": round(float(np.mean(player_counts)) if player_counts else 0.0, 2),
            "frames_with_ball": frames_with_ball,
            "ball_detection_rate_pct": round((frames_with_ball / max(1, frame_idx)) * 100, 1),
            "player_ms": round((t_player_total / max(1, frame_idx)) * 1000, 1),
            "ball_ms": round((t_ball_total / max(1, frame_idx)) * 1000, 1),
            "court_ms": round((t_court_total / max(1, frame_idx)) * 1000, 2),
            "render_ms": round((t_render_total / max(1, frame_idx)) * 1000, 1),
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
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Court Polygon Overlay
        corners = self.court_calibrator.result.corners_px
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 220), thickness=2, lineType=cv2.LINE_AA)

            # Center Net Line
            p_net_l = self.court_calibrator.court_to_pixel(0.0, 22.0)
            p_net_r = self.court_calibrator.court_to_pixel(20.0, 22.0)
            if p_net_l and p_net_r:
                cv2.line(vis, (int(p_net_l[0]), int(p_net_l[1])), (int(p_net_r[0]), int(p_net_r[1])), (255, 255, 255), 2, cv2.LINE_AA)

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
            cv2.ellipse(vis, (cx, min(h - 1, y2)), axes, 0, 0, 360, col, 2, lineType=cv2.LINE_AA)

            # Player Floating Pill
            lbl = f"P{pid}"
            if p.get("court_ft"):
                lbl += f" ({p['court_ft'][0]:.1f}, {p['court_ft'][1]:.1f})ft"
            if p.get("speed_mph", 0.0) > 1.0:
                lbl += f" {p['speed_mph']:.1f}mph"

            pill_w = len(lbl) * 8 + 16
            bx1 = max(4, cx - pill_w // 2)
            by1 = max(4, y1 - 22)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + 18), (15, 23, 42), -1)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + 18), col, 1, cv2.LINE_AA)
            cv2.putText(vis, lbl, (bx1 + 6, by1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

        # 3. Ball Trajectory Ribbon & Reticle
        ball = record["ball"]
        if ball is not None and ball["pixel"]:
            trail_len = 16
            valid_pts = []
            for i in range(len(ball_history) - 1, max(-1, len(ball_history) - 1 - trail_len), -1):
                pt = ball_history[i]
                if pt is None:
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
            cv2.circle(vis, (bx, by), 8, ret_col, 2, cv2.LINE_AA)
            cv2.circle(vis, (bx, by), 2, (255, 255, 255), -1, cv2.LINE_AA)

            if ball.get("speed_mph", 0.0) > 1.0:
                cv2.putText(vis, f"{ball['speed_mph']:.1f} MPH", (bx + 12, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 220), 1, cv2.LINE_AA)

        # 4. Clean Telemetry HUD
        cv2.rectangle(vis, (16, 16), (380, 80), (15, 23, 42), -1)
        cv2.rectangle(vis, (16, 16), (380, 80), (0, 255, 220), 1)
        cv2.putText(vis, "PICKLEBALL BASE VISION MODEL", (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 220), 1, cv2.LINE_AA)
        status_line = f"F: {frame_idx + 1}/{total_frames} | Players: {len(record['players'])}/4 | Ball: {'LOCKED' if ball else 'SEARCH'}"
        cv2.putText(vis, status_line, (26, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (250, 250, 250), 1, cv2.LINE_AA)
        sec = frame_idx / fps
        cv2.putText(vis, f"Time: {int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*100):02d} | Zero Rules (Pure Perception)",
                    (26, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

        return vis

    def _encode_h264(self, raw_path: str, target_path: str):
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
