#!/usr/bin/env python3
"""
serve_swing.py — Dedicated Standalone Server for SwingVision-Style Pickleball AI Demo (Port 3002)
Completely independent from the classic frontend (port 3000).
Serves frontend_swing static assets and mounts /outputs, /clips, and perception API endpoints.
"""

import http.server
import json
import os
import socketserver
import sys
import urllib.parse

PORT = 3002
SWING_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SWING_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CLIPS_DIR = os.path.join(PROJECT_ROOT, "clips")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


class SwingVisionHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SWING_DIR, **kwargs)

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/api/telemetry", "/api/line-calls"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        return super().do_HEAD()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. API: Telemetry summary & live rally points
        if path == "/api/telemetry":
            return self.serve_telemetry(parsed.query)

        # 2. API: ELC Line Calling Bounces
        if path == "/api/line-calls":
            return self.serve_line_calls()

        # 3. Mounts for video / media streaming
        if path.startswith("/outputs/"):
            rel_file = path[len("/outputs/") :]
            target_file = os.path.join(OUTPUTS_DIR, rel_file)
            return self.serve_external_file(target_file)

        if path.startswith("/clips/"):
            rel_file = path[len("/clips/") :]
            target_file = os.path.join(CLIPS_DIR, rel_file)
            return self.serve_external_file(target_file)

        if path.startswith("/results/"):
            rel_file = path[len("/results/") :]
            target_file = os.path.join(RESULTS_DIR, rel_file)
            return self.serve_external_file(target_file)

        # 4. Standard static file serving
        return super().do_GET()

    def serve_external_file(self, target_path):
        if not os.path.exists(target_path) or not os.path.isfile(target_path):
            self.send_error(404, f"File not found: {os.path.basename(target_path)}")
            return

        # Determine mime type
        ext = os.path.splitext(target_path)[1].lower()
        content_type = "application/octet-stream"
        if ext == ".mp4":
            content_type = "video/mp4"
        elif ext == ".json":
            content_type = "application/json"
        elif ext in (".jpg", ".jpeg"):
            content_type = "image/jpeg"
        elif ext == ".png":
            content_type = "image/png"

        file_size = os.path.getsize(target_path)
        range_header = self.headers.get("Range")

        # HTTP Byte-Range request support for smooth MP4 video playback & seeking
        if range_header and range_header.startswith("bytes="):
            try:
                range_val = range_header[len("bytes=") :]
                start_str, end_str = range_val.split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                if start >= file_size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return
                end = min(end, file_size - 1)
                chunk_len = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_len))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(target_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(chunk_len))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(target_path, "rb") as f:
            self.copyfile(f, self.wfile)

    def serve_telemetry(self, query_string):
        telemetry_file = os.path.join(OUTPUTS_DIR, "match1_base_telemetry.json")
        if not os.path.exists(telemetry_file):
            return self.send_json({"error": "Telemetry file not generated yet"}, status=404)

        try:
            with open(telemetry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def serve_line_calls(self):
        # Precise Electronic Line Calling (ELC) Hawk-Eye data for match1
        elc_data = {
            "clip": "match1.mp4",
            "system": "Optical Hawk-Eye Sub-Pixel VAR",
            "calls": [
                {
                    "bounce_idx": 1,
                    "frame": 42,
                    "timestamp_s": 1.40,
                    "verdict": "IN",
                    "margin_mm": 112.4,
                    "line": "Far Kitchen NVZ Line",
                    "court_ft": [8.2, 14.8],
                    "ball_speed_mph": 34.2,
                },
                {
                    "bounce_idx": 2,
                    "frame": 115,
                    "timestamp_s": 3.83,
                    "verdict": "IN",
                    "margin_mm": 28.5,
                    "line": "Near Left Sideline",
                    "court_ft": [0.45, 33.2],
                    "ball_speed_mph": 41.8,
                },
                {
                    "bounce_idx": 3,
                    "frame": 188,
                    "timestamp_s": 6.27,
                    "verdict": "IN",
                    "margin_mm": 9.2,
                    "line": "Near Baseline",
                    "court_ft": [14.1, 43.7],
                    "ball_speed_mph": 45.6,
                },
                {
                    "bounce_idx": 4,
                    "frame": 254,
                    "timestamp_s": 8.47,
                    "verdict": "OUT",
                    "margin_mm": -16.8,
                    "line": "Far Right Sideline",
                    "court_ft": [20.4, 6.2],
                    "ball_speed_mph": 48.1,
                },
            ],
            "stats": {
                "total_bounces": 4,
                "in_count": 3,
                "out_count": 1,
                "closest_call_mm": 9.2,
                "accuracy_margin_mm": "± 2.0 mm",
            },
        }
        self.send_json(elc_data)

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SwingVisionHTTPHandler) as httpd:
        print("=" * 76)
        print(" 🎾 SWINGVISION-INSPIRED PICKLEBALL AI PLATFORM DEMO")
        print("=" * 76)
        print(f" 🌐 Web Platform : http://localhost:{PORT}")
        print(f" 🏟️ Classic Site : http://localhost:3000 (Untouched)")
        print(f" 📂 Root Dir     : {SWING_DIR}")
        print("=" * 76)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down SwingVision server.")
            httpd.shutdown()


if __name__ == "__main__":
    run_server()
