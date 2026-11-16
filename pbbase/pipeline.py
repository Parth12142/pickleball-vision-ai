"""
pipeline.py - Master Unified Base Perception Pipeline (Zero-Rule Core).
Streams video frames through:
  1. Court Geometry Calibration
  2. Multi-Player Detection & Canonical Tracking
  3. Micro-Ball Detection & Kinematic Trajectory
  4. Kinematic Bounce Detection & Sub-Pixel Localization (Week 3)
  5. 2D Orthographic Court Minimap & Broadcast HUD Composition (Week 3)
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
from pbbase.tracking.bounce_detector import BallBounceDetector, BounceEvent
from pbbase.tracking.ball_3d_tracker import Ball3DReconstructor
from pbbase.data.schema import SCHEMA_VERSION, SCHEMA_PRODUCER


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
        multiscale: bool = False,
        ball_patch_slices: Optional[List[Tuple[float, float, float, float]]] = None,
        ball_max_coast: int = 3,
        ball_max_jump_px: float = 85.0,
        court_corners: Optional[List[Tuple[float, float]]] = None,
        render_minimap: bool = True,
        minimap_position: str = "bottom-right",
    ):
        print(f"[BasePerceptionPipeline] Loading Player Model: {player_weights}")
        self.player_detector = PlayerDetector(player_weights, conf_thresh=player_conf)
        print(f"[BasePerceptionPipeline] Loading Ball Model  : {ball_weights} (imgsz={ball_imgsz}, multiscale={multiscale})")
        self.ball_detector = BallDetector(
            ball_weights,
            conf_thresh=ball_conf,
            imgsz=ball_imgsz,
            multiscale=multiscale,
            patch_slices=ball_patch_slices,
        )

        self.court_calibrator = CourtCalibrator(corners_px=court_corners)
        self.player_tracker = CanonicalPlayerTracker(max_players=4)
        self.ball_tracker = BallTrajectoryTracker(
            max_jump_px=ball_max_jump_px,
            max_coast_frames=ball_max_coast,
        )
        self.bounce_detector = BallBounceDetector()
        self.ball_3d_reconstructor = Ball3DReconstructor()
        self.court_minimap = CourtMinimap(court_scale=8.5, padding=45, trail_len=20)
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

        if court_corners and len(court_corners) == 4:
            self.court_calibrator.calibrate(court_corners)

        do_render_minimap = render_minimap if render_minimap is not None else self.render_minimap
        m_pos = minimap_position if minimap_position is not None else self.minimap_position

        # Reset stateful trackers for new video stream
        self.player_tracker.reset()
        self.ball_tracker.reset()
        self.bounce_detector.reset()
        self.court_minimap.reset()

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
        ball_pts_history: List[Optional[Tuple[float, float, float]]] = []  # (cx, cy, speed_px)
        frames_with_ball = 0
        player_counts = []

        # Feature timing accumulators (seconds)
        t_player_total = 0.0
        t_ball_total = 0.0
        t_bounce_total = 0.0
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
                self.court_calibrator.auto_calibrate_broadcast(frame, clip_name=os.path.basename(video_path))
            if self.court_calibrator.result.calibrated and not self.court_calibrator._pose_ready:
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
            ball_result = self.ball_tracker.process_candidates(raw_ball_cands, frame_idx, timestamp_s, self.court_calibrator)
            if ball_result is not None:
                frames_with_ball += 1
                bx, by = ball_result["pixel"]
                bspeed = ball_result.get("speed_px_per_frame", 0.0)
                ball_pts_history.append((bx, by, bspeed))
            else:
                ball_pts_history.append(None)
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

            # Telemetry record for this frame
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

        # 5. 3D Height-Compensated Ballistic Reconstruction for Minimap
        all_bounces_dicts = [b.to_dict() for b in self.bounce_detector.detected_bounces]
        self.ball_3d_reconstructor.reconstruct(
            telemetry_stream,
            self.court_calibrator,
            bounces=all_bounces_dicts,
            image_size=(w, h),
        )

        # 6. Pass 2: Render diagnostic overlay & 3D-compensated Minimap HUD if preview requested
        if writer:
            self.court_minimap.reset()
            cap_vis = cv2.VideoCapture(video_path)
            vis_idx = 0
            vis_ball_history: List[Optional[Tuple[float, float, float]]] = []
            while cap_vis.isOpened() and vis_idx < len(telemetry_stream):
                ok, vframe = cap_vis.read()
                if not ok:
                    break
                rec = telemetry_stream[vis_idx]
                b = rec.get("ball")
                if b is not None and b.get("pixel") is not None:
                    vis_ball_history.append((b["pixel"][0], b["pixel"][1], b.get("speed_px_per_frame", 0.0)))
                else:
                    vis_ball_history.append(None)

                bnc = rec.get("bounce")
                if bnc is not None and bnc.get("is_ground_bounce", False):
                    self.court_minimap.add_bounce(tuple(bnc["court_ft"]), vis_idx)

                t_r_start = time.time()
                annotated = self._render_frame(
                    vframe,
                    rec,
                    vis_ball_history,
                    vis_idx,
                    frames_to_read,
                    fps,
                    render_minimap=do_render_minimap,
                    minimap_position=m_pos,
                )
                writer.write(annotated)
                t_render_total += (time.time() - t_r_start)
                vis_idx += 1

            cap_vis.release()
            writer.release()
            self._encode_h264(raw_preview_tmp, out_preview)
            print(f"[BasePerceptionPipeline] Saved H.264 preview to: {out_preview}")

        elapsed_total = time.time() - t0

        corners_list = [[round(x, 1), round(y, 1)] for x, y in self.court_calibrator.result.corners_px]
        court_schema = {
            "calibrated": self.court_calibrator.result.calibrated,
            "reprojection_rmse_ft": round(self.court_calibrator.result.reprojection_rmse_ft, 4),
            "dimensions_ft": {
                "width": 20.0,
                "length": 44.0,
                "net_y": 22.0,
                "nvz_depth": 7.0,
            },
            "corners_px": corners_list,
        }

        all_bounces = [b.to_dict() for b in self.bounce_detector.detected_bounces]

        bench_schema = {
            "ball_detection_rate_pct": round((frames_with_ball / max(1, frame_idx)) * 100, 1),
            "frames_with_ball": frames_with_ball,
            "mean_players_per_frame": round(float(np.mean(player_counts)) if player_counts else 0.0, 2),
            "multiscale_inference": self.ball_detector.multiscale,
            "total_bounces_detected": len(all_bounces),
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
        render_minimap: bool = True,
        minimap_position: str = "bottom-right",
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
            is_interpolated = ball.get("interpolated", False)
            if is_interpolated:
                cv2.circle(vis, (bx, by), 8, (120, 200, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, (bx, by), 2, (120, 200, 255), -1, cv2.LINE_AA)
                cv2.putText(vis, "COAST", (bx + 10, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 200, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(vis, (bx, by), 8, ret_col, 2, cv2.LINE_AA)
                cv2.circle(vis, (bx, by), 2, (255, 255, 255), -1, cv2.LINE_AA)

            if ball.get("speed_mph", 0.0) > 1.0:
                cv2.putText(vis, f"{ball['speed_mph']:.1f} MPH", (bx + 12, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 220), 1, cv2.LINE_AA)

        # 4. Bounce Ripple Indicator on Camera Frame (Week 3)
        bounce = record.get("bounce")
        if bounce is not None and bounce.get("pixel"):
            b_px, b_py = int(bounce["pixel"][0]), int(bounce["pixel"][1])
            b_col = (80, 220, 80) if bounce.get("is_ground_bounce", True) else (60, 60, 230)
            cv2.circle(vis, (b_px, b_py), 16, b_col, 2, cv2.LINE_AA)
            cv2.circle(vis, (b_px, b_py), 8, (255, 255, 255), 1, cv2.LINE_AA)
            c_xy = bounce.get("court_ft")
            if c_xy:
                b_lbl = f"BOUNCE ({c_xy[0]:.1f}, {c_xy[1]:.1f})ft"
                cv2.putText(vis, b_lbl, (b_px - 40, b_py - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)

        # 5. Top-Down Court Minimap HUD Composition (Week 3)
        if render_minimap and self.court_calibrator.result.calibrated:
            minimap_canvas = self.court_minimap.render(
                players=record.get("players"),
                ball=record.get("ball"),
                frame_idx=frame_idx,
            )
            vis = self.court_minimap.compose_onto_frame(
                vis,
                minimap_canvas,
                position=minimap_position,
                alpha=0.88,
            )

        # 6. Clean Telemetry Header HUD
        cv2.rectangle(vis, (16, 16), (395, 84), (15, 23, 42), -1)
        cv2.rectangle(vis, (16, 16), (395, 84), (0, 255, 220), 1)
        cv2.putText(vis, "PICKLEBALL BASE VISION MODEL", (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 220), 1, cv2.LINE_AA)
        ball_status = "COAST" if (ball and ball.get("interpolated")) else ("LOCKED" if ball else "SEARCH")
        total_bncs = len(self.bounce_detector.detected_bounces)
        status_line = f"F: {frame_idx + 1}/{total_frames} | Players: {len(record['players'])}/4 | Ball: {ball_status} | Bounces: {total_bncs}"
        cv2.putText(vis, status_line, (26, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (250, 250, 250), 1, cv2.LINE_AA)
        sec = frame_idx / fps
        cv2.putText(vis, f"Time: {int(sec//60):02d}:{int(sec%60):02d}.{int((sec%1)*100):02d} | Zero Rules (Pure Perception)",
                    (26, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA)

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
