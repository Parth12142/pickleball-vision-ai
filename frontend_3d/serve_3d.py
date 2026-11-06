#!/usr/bin/env python3
"""
serve_3d.py — Pickleball Vision AI 3D Open-World Experience Server (Bruno Simon Concept)
Serves `frontend_3d/` at http://localhost:3001 (completely isolated from port 3000),
mounts `/outputs/`, `/clips/`, and `/results/`, and provides court calibration & pipeline APIs.
"""
import http.server
import json
import mimetypes
import os
import shutil
import socketserver
import sys
import threading
import urllib.parse

PORT = 3001
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_3D_DIR = os.path.abspath(os.path.dirname(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import pbbase.patch_env
from pbbase.court.homography import CourtCalibrator
from pbbase.pipeline import BasePerceptionPipeline

PIPELINE_LOCK = threading.Lock()
PIPELINE_STATUS = {
    "running": False,
    "clip": None,
    "progress_pct": 0,
    "status": "idle",
    "result": None,
    "error": None,
}


def extract_frame0_path(clip_name: str) -> str:
    base = os.path.basename(clip_name)
    clip_path = os.path.join(ROOT_DIR, "clips", base)
    if not os.path.exists(clip_path):
        return None
    cache_path = os.path.join(ROOT_DIR, "outputs", f"frame0_{base}.jpg")
    if os.path.exists(cache_path):
        return cache_path

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cv2.imwrite(cache_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return cache_path


def run_pipeline_worker(clip_name: str = "match1.mp4", max_frames: int = 289):
    global PIPELINE_STATUS
    base = os.path.basename(clip_name)
    stem = os.path.splitext(base)[0]
    clip_path = os.path.join(ROOT_DIR, "clips", base)
    out_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, f"{stem}_base_telemetry.json")
    out_preview = os.path.join(out_dir, f"{stem}_base_preview.mp4")

    with PIPELINE_LOCK:
        PIPELINE_STATUS.update({
            "running": True,
            "clip": base,
            "progress_pct": 15,
            "status": f"Loading YOLO26 & Ball Models for {base}...",
            "result": None,
            "error": None,
        })

    try:
        player_weights = os.path.join(ROOT_DIR, "weights/yolo26n.pt")
        ball_weights = os.path.join(ROOT_DIR, "weights/ball_yolo26_best.pt")
        if not os.path.exists(ball_weights):
            ball_weights = os.path.join(ROOT_DIR, "weights/ball_colab_dukcv.pt")

        pipeline = BasePerceptionPipeline(
            player_weights=player_weights,
            ball_weights=ball_weights,
            ball_imgsz=1312,
            multiscale=False,
            ball_max_coast=3,
        )

        with PIPELINE_LOCK:
            PIPELINE_STATUS["progress_pct"] = 40
            PIPELINE_STATUS["status"] = f"Tracking 9s Rally ({max_frames} frames)..."

        summary = pipeline.process_video(
            video_path=clip_path,
            max_frames=max_frames,
            out_json=out_json,
            out_preview=out_preview,
        )

        with PIPELINE_LOCK:
            PIPELINE_STATUS.update({
                "running": False,
                "progress_pct": 100,
                "status": "completed",
                "result": {
                    "clip": base,
                    "frames_processed": summary["total_frames_processed"],
                    "processing_fps": summary["processing_fps"],
                    "ball_detect_pct": summary["ball_detection_rate_pct"],
                    "mean_players": summary["mean_players_per_frame"],
                    "court_rmse_ft": summary["court_reprojection_rmse_ft"],
                    "preview_url": f"/outputs/{stem}_base_preview.mp4",
                    "telemetry_url": f"/outputs/{stem}_base_telemetry.json",
                },
                "error": None,
            })
    except Exception as e:
        with PIPELINE_LOCK:
            PIPELINE_STATUS.update({
                "running": False,
                "progress_pct": 0,
                "status": "error",
                "result": None,
                "error": str(e),
            })


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class World3DRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed = urllib.parse.urlparse(path).path
        if (
            parsed.startswith("/outputs/")
            or parsed.startswith("/results/")
            or parsed.startswith("/clips/")
        ):
            rel = parsed.lstrip("/")
            return os.path.join(ROOT_DIR, rel)
        rel = parsed.lstrip("/")
        if not rel:
            rel = "index.html"
        return os.path.join(FRONTEND_3D_DIR, rel)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_file_with_range(self, filepath):
        if not os.path.isfile(filepath):
            self.send_error(404, "File not found")
            return

        file_size = os.path.getsize(filepath)
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type:
            mime_type = "application/octet-stream"

        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(filepath, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return

        range_spec = range_header.split("bytes=", 1)[1].strip()
        start_str, end_str = range_spec.split("-", 1) if "-" in range_spec else (range_spec, "")
        try:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except ValueError:
            self.send_error(416, "Requested Range Not Satisfiable")
            return

        if start >= file_size or end >= file_size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        content_length = (end - start) + 1
        self.send_response(206)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(filepath, "rb") as f:
            f.seek(start)
            bytes_left = content_length
            chunk_size = 64 * 1024
            while bytes_left > 0:
                to_read = min(bytes_left, chunk_size)
                chunk = f.read(to_read)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                bytes_left -= len(chunk)

    def do_GET(self):
        url_obj = urllib.parse.urlparse(self.path)
        parsed = url_obj.path
        query_params = urllib.parse.parse_qs(url_obj.query)

        if parsed == "/api/frame0":
            clip_name = query_params.get("clip", ["match1.mp4"])[0]
            cache_path = extract_frame0_path(clip_name)
            if not cache_path or not os.path.exists(cache_path):
                return self.send_json(404, {"ok": False, "error": "Frame0 not found"})
            return self.send_file_with_range(cache_path)

        if parsed == "/api/court-calibration":
            clip_name = query_params.get("clip", ["match1.mp4"])[0]
            base = os.path.basename(clip_name)
            saved = CourtCalibrator.load_calibration_for_clip(base)
            corners = saved or CourtCalibrator.get_recommended_corners(1302, 740, clip_name=base)
            calibrator = CourtCalibrator(corners)
            return self.send_json(200, {
                "ok": True,
                "clip": base,
                "corners_px": corners,
                "rmse_ft": round(calibrator.result.reprojection_rmse_ft, 4),
            })

        if parsed == "/api/pipeline-status":
            with PIPELINE_LOCK:
                return self.send_json(200, {"ok": True, **PIPELINE_STATUS})

        filepath = self.translate_path(self.path)
        if os.path.isfile(filepath):
            if filepath.lower().endswith((".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".gif")):
                return self.send_file_with_range(filepath)

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path).path
        try:
            data = self.read_json_body()
        except Exception:
            return self.send_json(400, {"ok": False, "error": "Invalid JSON"})

        if parsed == "/api/court-calibration":
            clip_name = data.get("clip", "match1.mp4")
            base = os.path.basename(clip_name)
            corners_raw = data.get("corners_px", [])
            if len(corners_raw) != 4:
                return self.send_json(400, {"ok": False, "error": "4 corners required"})
            pts = [(float(p[0]), float(p[1])) for p in corners_raw]
            calibrator = CourtCalibrator(pts)
            CourtCalibrator.save_calibration_for_clip(
                clip_name=base,
                corners_px=pts,
                resolution=(1302, 740),
                rmse_ft=calibrator.result.reprojection_rmse_ft,
                source="human_pinned_3d_world",
            )
            return self.send_json(200, {
                "ok": True,
                "clip": base,
                "corners_px": pts,
                "rmse_ft": round(calibrator.result.reprojection_rmse_ft, 4),
            })

        if parsed == "/api/run-pipeline":
            with PIPELINE_LOCK:
                if PIPELINE_STATUS.get("running"):
                    return self.send_json(409, {"ok": False, "error": "Already running"})
            t = threading.Thread(target=run_pipeline_worker, args=("match1.mp4", 289), daemon=True)
            t.start()
            return self.send_json(202, {"ok": True, "message": "Started 9s rally tracking"})

        return self.send_json(404, {"ok": False, "error": "Not found"})


if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), World3DRequestHandler) as httpd:
        print("=" * 72)
        print(" 🏎️ PICKLEBALL 3D OPEN-WORLD EXPERIENCE (BRUNO SIMON CONCEPT)")
        print("=" * 72)
        print(f" 🌐 3D World URL  : http://localhost:{PORT}")
        print(f" 🏟️ Classic URL   : http://localhost:3000 (Untouched)")
        print("=" * 72)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down 3D server.")
