"""
unified_engine.py - Unified Master Pickleball Base Vision Model.
"""
import json
import os
import time
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from . import (
    BallTelemetry,
    CourtGeometryResult,
    FramePerceptionResult,
    PlayerTelemetry,
)
from .court_detector import CourtDetector
from .player_tracker import PlayerTracker
from .ball_tracker import BallTracker


class PickleballBaseModel:
    """
    Unified Base Vision Perception Model for Pickleball.
    Carries court detection/homography, multi-player tracking, and small-ball trajectory.
    100% decoupled from downstream rules, fouls, and scoring.
    """
    def __init__(
        self,
        player_model_path: str = "yolo11n.pt",
        ball_model_path: str = "weights/ball_best.pt",
        player_conf: float = 0.28,
        ball_conf: float = 0.20,
    ):
        print(f"[PickleballBaseModel] Initializing Player Perception: {player_model_path}")
        self.player_tracker = PlayerTracker(player_model_path, conf_thresh=player_conf)
        print(f"[PickleballBaseModel] Initializing Ball Perception  : {ball_model_path}")
        self.ball_tracker = BallTracker(ball_model_path, conf_thresh=ball_conf)
        self.court_detector = CourtDetector()

    def calibrate_court(self, first_frame: np.ndarray, manual_corners: Optional[List] = None) -> bool:
        if manual_corners:
            return self.court_detector.calibrate(manual_corners)
        return self.court_detector.auto_detect(first_frame)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> FramePerceptionResult:
        if not self.court_detector.calibrated:
            self.court_detector.auto_detect(frame)

        players = self.player_tracker.track_frame(frame, timestamp_s, self.court_detector)
        ball = self.ball_tracker.detect_and_track(frame, frame_idx, timestamp_s, self.court_detector)

        return FramePerceptionResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            court=self.court_detector.get_result(),
            players=players,
            ball=ball,
            bounces_detected=self.ball_tracker.bounces,
        )

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        out_json: Optional[str] = None,
        out_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process full video and export clean structured perception telemetry.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_read = min(total_frames, max_frames) if max_frames else total_frames
        print(f"[PickleballBaseModel] Processing: {video_path} ({w}x{h} @ {fps:.1f} FPS, {frames_to_read} frames)")

        # Prepare preview video writer if requested
        writer = None
        raw_tmp_preview = None
        if out_preview:
            os.makedirs(os.path.dirname(os.path.abspath(out_preview)), exist_ok=True)
            raw_tmp_preview = f"{out_preview}.raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(raw_tmp_preview, fourcc, fps, (w, h))

        t0 = time.time()
        telemetry_stream: List[Dict[str, Any]] = []
        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break

            timestamp_s = frame_idx / fps
            result = self.process_frame(frame, frame_idx, timestamp_s)
            telemetry_stream.append(result.to_dict())

            if writer:
                annotated = self._render_diagnostic_frame(frame, result)
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{frames_to_read} frames ({frame_idx / (time.time() - t0):.1f} fps)...")

        cap.release()
        if writer:
            writer.release()
            self._transcode_to_web_h264(raw_tmp_preview, out_preview)
            print(f"[PickleballBaseModel] Saved web-compatible diagnostic preview (H.264) to: {out_preview}")

        elapsed_s = time.time() - t0
        summary = {
            "video_path": video_path,
            "total_frames_processed": frame_idx,
            "fps": round(fps, 2),
            "elapsed_s": round(elapsed_s, 2),
            "processing_fps": round(frame_idx / max(0.001, elapsed_s), 2),
            "court_calibrated": self.court_detector.calibrated,
            "total_bounces_detected": len(self.ball_tracker.bounces),
            "telemetry_stream": telemetry_stream,
        }

        if out_json:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[PickleballBaseModel] Saved clean perception telemetry to: {out_json}")

        return summary

    def _transcode_to_web_h264(self, raw_path: str, target_path: str) -> None:
        """Transcode raw mp4v video to universal web H.264/AVC1 with faststart."""
        try:
            import imageio_ffmpeg
            import subprocess
            import shutil
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            tmp_target = f"{target_path}.transcode.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-v", "error", "-i", raw_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                tmp_target
            ]
            subprocess.run(cmd, check=True)
            shutil.move(tmp_target, target_path)
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception as e:
            print(f"[warning] H.264 transcoding fallback ({e}). Keeping raw video.")
            if os.path.exists(raw_path):
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(raw_path, target_path)

    def _render_diagnostic_frame(self, frame: np.ndarray, result: FramePerceptionResult) -> np.ndarray:
        """Render clean, high-precision perception overlay with Kitchen (NVZ) detection."""
        vis = frame.copy()
        
        # 1. Render Court Boundary Polygon
        if result.court.corners_pixel and len(result.court.corners_pixel) == 4:
            pts = np.array(result.court.corners_pixel, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 223), thickness=2, lineType=cv2.LINE_AA)

        # 2. Render Kitchen (Non-Volley Zone / NVZ) Detection
        if self.court_detector.calibrated:
            k_polys = self.court_detector.get_kitchen_polygons_px()
            if "full_kitchen" in k_polys:
                k_pts = np.array(k_polys["full_kitchen"], dtype=np.int32).reshape((-1, 1, 2))
                # Soft translucent amber/orange fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [k_pts], color=(20, 100, 220))
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [k_pts], isClosed=True, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)

            # Draw NVZ Boundary Lines & Net
            k_lines = self.court_detector.get_kitchen_lines_px()
            if "far_nvz_line" in k_lines and k_lines["far_nvz_line"][0] and k_lines["far_nvz_line"][1]:
                p1 = (int(k_lines["far_nvz_line"][0][0]), int(k_lines["far_nvz_line"][0][1]))
                p2 = (int(k_lines["far_nvz_line"][1][0]), int(k_lines["far_nvz_line"][1][1]))
                cv2.line(vis, p1, p2, (0, 165, 255), 2, cv2.LINE_AA)
            if "near_nvz_line" in k_lines and k_lines["near_nvz_line"][0] and k_lines["near_nvz_line"][1]:
                p1 = (int(k_lines["near_nvz_line"][0][0]), int(k_lines["near_nvz_line"][0][1]))
                p2 = (int(k_lines["near_nvz_line"][1][0]), int(k_lines["near_nvz_line"][1][1]))
                cv2.line(vis, p1, p2, (0, 165, 255), 2, cv2.LINE_AA)
            if "net_line" in k_lines and k_lines["net_line"][0] and k_lines["net_line"][1]:
                p1 = (int(k_lines["net_line"][0][0]), int(k_lines["net_line"][0][1]))
                p2 = (int(k_lines["net_line"][1][0]), int(k_lines["net_line"][1][1]))
                cv2.line(vis, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

            # Perspective on-court Kitchen Label (centered on physical net)
            if "net_line" in k_lines and k_lines["net_line"][0] and k_lines["net_line"][1]:
                p1 = k_lines["net_line"][0]
                p2 = k_lines["net_line"][1]
                cx = int((p1[0] + p2[0]) / 2.0)
                cy = int((p1[1] + p2[1]) / 2.0)
                cv2.putText(vis, "NON-VOLLEY ZONE (KITCHEN)", (cx - 105, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # 3. Render Tracked Players & Kitchen Intrusion
        kitchen_alert_active = False
        for p in result.players:
            x1, y1, x2, y2 = [int(v) for v in p.bbox_xyxy]
            fx, fy = int(p.foot_pixel[0]), int(p.foot_pixel[1])

            if p.kitchen_intrusion:
                kitchen_alert_active = True
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(vis, (fx, fy), 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, (fx, fy), 14, (0, 140, 255), 2, cv2.LINE_AA)
                tag = f"P{p.id} KITCHEN FAULT ({p.court_zone})"
                cv2.putText(vis, tag, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 107, 0), 2, cv2.LINE_AA)
                cv2.circle(vis, (fx, fy), 5, (0, 255, 223), -1, cv2.LINE_AA)
                tag = f"P{p.id} [{p.court_zone}]"
                if p.court_pos_ft:
                    tag += f" ({p.court_pos_ft[0]:.1f}, {p.court_pos_ft[1]:.1f})ft"
                cv2.putText(vis, tag, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Render Ball & Trajectory
        if result.ball and result.ball.center_pixel:
            bx, by = int(result.ball.center_pixel[0]), int(result.ball.center_pixel[1])
            color = (0, 255, 0) if not result.ball.interpolated else (0, 165, 255)
            cv2.circle(vis, (bx, by), 6, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (bx, by), 9, (255, 255, 255), 1, cv2.LINE_AA)
            if result.ball.speed_mph > 1.0:
                cv2.putText(vis, f"{result.ball.speed_mph:.1f} MPH", (bx + 12, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 223), 1, cv2.LINE_AA)
            if result.ball.is_kitchen_bounce:
                cv2.putText(vis, "KITCHEN BOUNCE (DINK)", (bx - 50, by - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)

        # 5. Render HUD Diagnostic Badge
        cv2.rectangle(vis, (16, 16), (360, 78), (15, 23, 42), -1)
        border_col = (0, 0, 255) if kitchen_alert_active else (0, 255, 223)
        cv2.rectangle(vis, (16, 16), (360, 78), border_col, 1)
        cv2.putText(vis, "PICKLEBALL BASE VISION MODEL", (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 223), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Frame: {result.frame_idx} | Players: {len(result.players)} | Ball: {'YES' if result.ball and result.ball.detected else 'NO'}",
                    (26, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        kitchen_txt = "KITCHEN (NVZ): FAULT DETECTED" if kitchen_alert_active else "KITCHEN (NVZ): ACTIVE / SAFE"
        kitchen_txt_col = (0, 80, 255) if kitchen_alert_active else (0, 255, 120)
        cv2.putText(vis, kitchen_txt, (26, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, kitchen_txt_col, 1, cv2.LINE_AA)

        return vis
