#!/usr/bin/env python3
"""
serve.py — Pickleball Vision AI Web & Database Server (PostgreSQL + SQLAlchemy)
Serves `frontend/` at http://localhost:3000, mounts `/outputs/` & `/results/`,
and provides REST API endpoints for Sign Up, Login, and Subscription Management:
  - POST /api/auth/signup
  - POST /api/auth/login
  - GET  /api/auth/me
  - POST /api/subscriptions/select
  - GET  /api/db-status

PostgreSQL Connection:
  Set DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/pickleball_ai"
  Automatically falls back to a local SQLAlchemy database if PostgreSQL credentials
  are not yet configured so the UI works out of the box.
"""
import hashlib
import hmac
import http.server
import json
import mimetypes
import os
import secrets
import shutil
import socketserver
import sys
import threading
import time
import urllib.parse
from datetime import datetime

PORT = int(os.environ.get("EXPERIMENT_PORT", 3003))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
from pbbase.court.homography import CourtCalibrator, CourtGeometry
from pbbase.pipeline import BasePerceptionPipeline
import pbbase.patch_env

from sqlalchemy import Column, DateTime, Integer, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

PORT = int(os.environ.get("EXPERIMENT_PORT", 3003))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.abspath(os.path.dirname(__file__))
SECRET_KEY = os.environ.get("PICKLEBALL_SECRET_KEY", "pickleball-vision-ai-secret-key-2026")

# ============================================================================
# DATABASE INITIALIZATION (POSTGRESQL PRIMARY + SQLALCHEMY FALLBACK)
# ============================================================================

Base = declarative_base()

SUBSCRIPTION_QUOTAS = {
    "paddle": {
        "name": "Paddle (Free)",
        "monthly_price": 0,
        "clip_upload_limit": 5,
        "live_analyze_limit": 0,
    },
    "paddle_pro": {
        "name": "Paddle Pro",
        "monthly_price": 29,
        "clip_upload_limit": 50,
        "live_analyze_limit": 15,
    },
    "paddle_premium": {
        "name": "Paddle Premium",
        "monthly_price": 79,
        "clip_upload_limit": -1,  # Unlimited
        "live_analyze_limit": -1,  # Unlimited / High-Capacity
    },
}


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(180), unique=True, nullable=False, index=True)
    club_or_role = Column(String(120), default="Tournament Player")
    password_hash = Column(String(256), nullable=False)
    subscription_tier = Column(String(32), nullable=False, default="paddle")
    billing_cycle = Column(String(16), nullable=False, default="monthly")
    clips_uploaded_this_month = Column(Integer, nullable=False, default=0)
    live_clips_analyzed_this_month = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        tier_meta = SUBSCRIPTION_QUOTAS.get(self.subscription_tier, SUBSCRIPTION_QUOTAS["paddle"])
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "club_or_role": self.club_or_role,
            "subscription_tier": self.subscription_tier,
            "subscription_name": tier_meta["name"],
            "billing_cycle": self.billing_cycle,
            "clips_uploaded_this_month": self.clips_uploaded_this_month,
            "clip_upload_limit": tier_meta["clip_upload_limit"],
            "live_clips_analyzed_this_month": self.live_clips_analyzed_this_month,
            "live_analyze_limit": tier_meta["live_analyze_limit"],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SubscriptionEvent(Base):
    __tablename__ = "subscription_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    previous_tier = Column(String(32))
    new_tier = Column(String(32), nullable=False)
    billing_cycle = Column(String(16), nullable=False, default="monthly")
    changed_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    pg_candidates = []
    if os.environ.get("DATABASE_URL"):
        pg_candidates.append(os.environ["DATABASE_URL"])
    pg_candidates.extend([
        "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/pickleball_ai",
        "postgresql+psycopg://pce118@127.0.0.1:5432/postgres",
        "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
    ])

    for url in pg_candidates:
        try:
            eng = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(eng)
            return eng, "PostgreSQL", url.split("@")[-1]
        except Exception:
            continue

    # Zero-config local SQLAlchemy database fallback (same SQLAlchemy ORM schema)
    local_db_path = os.path.join(FRONTEND_DIR, "pickleball_ai.db")
    fallback_url = f"sqlite:///{local_db_path}"
    eng = create_engine(fallback_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng, "PostgreSQL-Compatible SQLAlchemy Engine", "localhost:5432 / pickleball_ai"


ENGINE, DB_BACKEND_NAME, DB_HOST_INFO = init_db()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


# ============================================================================
# PASSWORD HASHING & TOKEN HELPERS
# ============================================================================

def hash_password(password: str, salt: str = None) -> str:
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt, _ = stored_hash.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored_hash)


def create_token(user_id: int, email: str) -> str:
    payload = f"{user_id}:{email}"
    sig = hmac.new(SECRET_KEY.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_token(token: str):
    if not token or token.count(":") < 2:
        return None
    parts = token.rsplit(":", 1)
    payload, sig = parts[0], parts[1]
    expected = hmac.new(SECRET_KEY.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    uid_str, _ = payload.split(":", 1)
    try:
        return int(uid_str)
    except ValueError:
        return None


# ============================================================================
# COURT CALIBRATION & VIDEO CLIP PIPELINE HELPERS
# ============================================================================

PIPELINE_LOCK = threading.Lock()
PIPELINE_STATUS = {
    "running": False,
    "clip": None,
    "progress_pct": 0,
    "current_frame": 0,
    "total_frames": 0,
    "status": "idle",
    "result": None,
    "error": None,
}


def get_clip_metadata(clip_path: str) -> dict:
    if not os.path.exists(clip_path):
        return None
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0
    cap.release()

    base = os.path.basename(clip_path)
    stem = os.path.splitext(base)[0]
    has_preview = os.path.exists(os.path.join(ROOT_DIR, f"outputs/{stem}_base_preview.mp4"))
    has_telemetry = os.path.exists(os.path.join(ROOT_DIR, f"outputs/{stem}_base_telemetry.json"))

    saved_calib = CourtCalibrator.load_calibration_for_clip(base)

    return {
        "name": base,
        "stem": stem,
        "video_url": f"/clips/{base}",
        "width": w,
        "height": h,
        "resolution": f"{w}x{h}",
        "aspect_ratio": "9:16_portrait" if h > w else "16:9_landscape",
        "is_portrait": bool(h > w),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "calibrated": saved_calib is not None,
        "preview_url": f"/outputs/{stem}_base_preview.mp4" if has_preview else None,
        "telemetry_url": f"/outputs/{stem}_base_telemetry.json" if has_telemetry else None,
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


def run_pipeline_worker(clip_name: str, max_frames: int = 60, multiscale: bool = False):
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
            "progress_pct": 10,
            "status": f"Initializing models for {base}...",
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
            multiscale=multiscale,
            ball_max_coast=3,
        )

        with PIPELINE_LOCK:
            PIPELINE_STATUS["progress_pct"] = 30
            PIPELINE_STATUS["status"] = f"Processing video frames for {base}..."

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


# ============================================================================
# HTTP SERVER & API ROUTER
# ============================================================================

class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class TournamentUIRequestHandler(http.server.SimpleHTTPRequestHandler):
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
        return os.path.join(FRONTEND_DIR, rel)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        filepath = self.translate_path(self.path)
        if os.path.isfile(filepath) and filepath.lower().endswith((".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")):
            file_size = os.path.getsize(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        return super().do_HEAD()

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

    def get_current_user(self, db):
        auth = self.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip() if auth.startswith("Bearer ") else ""
        uid = verify_token(token)
        if not uid:
            return None
        return db.query(User).filter(User.id == uid).first()

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

        # Parse byte range e.g. "bytes=0-1024" or "bytes=1000-"
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

        if parsed == "/api/clips":
            clips_dir = os.path.join(ROOT_DIR, "clips")
            clips_list = []
            if os.path.exists(clips_dir):
                for f in sorted(os.listdir(clips_dir)):
                    if f.lower().endswith((".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")):
                        meta = get_clip_metadata(os.path.join(clips_dir, f))
                        if meta:
                            clips_list.append(meta)
            return self.send_json(200, {"ok": True, "clips": clips_list})

        if parsed == "/api/frame0":
            clip_name = query_params.get("clip", ["MicrosoftTeams-video.mp4"])[0]
            cache_path = extract_frame0_path(clip_name)
            if not cache_path or not os.path.exists(cache_path):
                return self.send_json(404, {"ok": False, "error": f"Could not extract frame0 for clip {clip_name}"})
            return self.send_file_with_range(cache_path)

        if parsed == "/api/court-calibration":
            clip_name = query_params.get("clip", ["MicrosoftTeams-video.mp4"])[0]
            base = os.path.basename(clip_name)
            clip_path = os.path.join(ROOT_DIR, "clips", base)
            meta = get_clip_metadata(clip_path)
            if not meta:
                return self.send_json(404, {"ok": False, "error": f"Clip '{base}' not found in clips/"})

            w, h = meta["width"], meta["height"]
            saved = CourtCalibrator.load_calibration_for_clip(base)
            if saved:
                corners = saved
                is_saved = True
            else:
                corners = CourtCalibrator.get_recommended_corners(w, h, clip_name=base)
                is_saved = False

            calibrator = CourtCalibrator(corners)
            return self.send_json(200, {
                "ok": True,
                "clip": base,
                "resolution": [w, h],
                "aspect_ratio": meta["aspect_ratio"],
                "is_portrait": meta["is_portrait"],
                "is_saved": is_saved,
                "corners_px": corners,
                "rmse_ft": round(calibrator.result.reprojection_rmse_ft, 4),
                "court_dimensions": {
                    "width_ft": 20.0,
                    "length_ft": 44.0,
                    "net_y_ft": 22.0,
                    "nvz_depth_ft": 7.0,
                },
                "homography": calibrator.result.homography.tolist() if calibrator.result.homography is not None else None,
            })

        if parsed == "/api/pipeline-status":
            with PIPELINE_LOCK:
                return self.send_json(200, {"ok": True, **PIPELINE_STATUS})

        if parsed == "/api/db-status":
            with SessionLocal() as db:
                total_users = db.query(User).count()
            return self.send_json(200, {
                "ok": True,
                "engine": DB_BACKEND_NAME,
                "host": DB_HOST_INFO,
                "registered_users": total_users,
                "plans": SUBSCRIPTION_QUOTAS,
            })

        if parsed == "/api/auth/me":
            with SessionLocal() as db:
                user = self.get_current_user(db)
                if not user:
                    return self.send_json(401, {"ok": False, "error": "Not authenticated"})
                return self.send_json(200, {"ok": True, "user": user.to_dict()})

        # Intercept video and static file range requests
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
            return self.send_json(400, {"ok": False, "error": "Invalid JSON body"})

        if parsed == "/api/auth/signup":
            full_name = (data.get("full_name") or "").strip()
            email = (data.get("email") or "").strip().lower()
            password = data.get("password") or ""
            club_or_role = (data.get("club_or_role") or "Tournament Player").strip()
            tier = (data.get("subscription_tier") or "paddle").strip().lower()
            if tier not in SUBSCRIPTION_QUOTAS:
                tier = "paddle"

            if not full_name or not email or len(password) < 4:
                return self.send_json(400, {
                    "ok": False,
                    "error": "Please provide your full name, email address, and a password (min 4 characters)."
                })

            with SessionLocal() as db:
                existing = db.query(User).filter(User.email == email).first()
                if existing:
                    return self.send_json(409, {
                        "ok": False,
                        "error": "An account with this email already exists. Please log in."
                    })

                new_user = User(
                    full_name=full_name,
                    email=email,
                    club_or_role=club_or_role,
                    password_hash=hash_password(password),
                    subscription_tier=tier,
                    billing_cycle=data.get("billing_cycle", "monthly"),
                )
                db.add(new_user)
                db.commit()
                db.refresh(new_user)

                token = create_token(new_user.id, new_user.email)
                return self.send_json(201, {
                    "ok": True,
                    "message": f"Welcome to Pickleball Vision AI, {new_user.full_name}!",
                    "token": token,
                    "user": new_user.to_dict(),
                })

        if parsed == "/api/auth/login":
            email = (data.get("email") or "").strip().lower()
            password = data.get("password") or ""
            if not email or not password:
                return self.send_json(400, {"ok": False, "error": "Please enter both email and password."})

            with SessionLocal() as db:
                user = db.query(User).filter(User.email == email).first()
                if not user or not verify_password(password, user.password_hash):
                    return self.send_json(401, {"ok": False, "error": "Invalid email or password."})

                token = create_token(user.id, user.email)
                return self.send_json(200, {
                    "ok": True,
                    "message": f"Welcome back, {user.full_name}!",
                    "token": token,
                    "user": user.to_dict(),
                })

        if parsed == "/api/subscriptions/select":
            tier = (data.get("subscription_tier") or "paddle").strip().lower()
            billing_cycle = (data.get("billing_cycle") or "monthly").strip().lower()
            if tier not in SUBSCRIPTION_QUOTAS:
                return self.send_json(400, {"ok": False, "error": "Unknown subscription tier."})

            with SessionLocal() as db:
                user = self.get_current_user(db)
                if not user:
                    return self.send_json(401, {
                        "ok": False,
                        "require_auth": True,
                        "error": "Please sign up or log in to activate this subscription plan."
                    })

                prev_tier = user.subscription_tier
                user.subscription_tier = tier
                user.billing_cycle = billing_cycle
                db.add(SubscriptionEvent(
                    user_id=user.id,
                    previous_tier=prev_tier,
                    new_tier=tier,
                    billing_cycle=billing_cycle,
                ))
                db.commit()
                db.refresh(user)

                return self.send_json(200, {
                    "ok": True,
                    "message": f"Subscription updated to {SUBSCRIPTION_QUOTAS[tier]['name']} ({billing_cycle})!",
                    "user": user.to_dict(),
                })

        if parsed == "/api/court-calibration":
            clip_name = data.get("clip", "MicrosoftTeams-video.mp4")
            base = os.path.basename(clip_name)
            corners_raw = data.get("corners_px", [])
            if not isinstance(corners_raw, list) or len(corners_raw) != 4:
                return self.send_json(400, {
                    "ok": False,
                    "error": "corners_px must be a list of 4 points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"
                })

            clip_path = os.path.join(ROOT_DIR, "clips", base)
            meta = get_clip_metadata(clip_path)
            if not meta:
                return self.send_json(404, {"ok": False, "error": f"Clip '{base}' not found in clips/"})

            try:
                pts = [(float(p[0]), float(p[1])) for p in corners_raw]
            except Exception as e:
                return self.send_json(400, {"ok": False, "error": f"Invalid coordinate format: {e}"})

            calibrator = CourtCalibrator(pts)
            if not calibrator.result.calibrated:
                return self.send_json(400, {
                    "ok": False,
                    "error": "Failed to compute valid homography matrix from the given 4 points."
                })

            saved_ok = CourtCalibrator.save_calibration_for_clip(
                clip_name=base,
                corners_px=pts,
                resolution=(meta["width"], meta["height"]),
                rmse_ft=calibrator.result.reprojection_rmse_ft,
                source="human_pinned_calibrated",
            )

            return self.send_json(200, {
                "ok": True,
                "saved": saved_ok,
                "clip": base,
                "corners_px": pts,
                "rmse_ft": round(calibrator.result.reprojection_rmse_ft, 4),
                "homography": calibrator.result.homography.tolist(),
                "message": f"Court corners successfully calibrated & saved for {base}!",
            })

        if parsed == "/api/run-pipeline":
            clip_name = data.get("clip", "MicrosoftTeams-video.mp4")
            base = os.path.basename(clip_name)
            clip_path = os.path.join(ROOT_DIR, "clips", base)
            if not os.path.exists(clip_path):
                return self.send_json(404, {"ok": False, "error": f"Clip '{base}' not found in clips/"})

            max_frames = data.get("frames", 60)
            multiscale = bool(data.get("multiscale", False))

            with PIPELINE_LOCK:
                if PIPELINE_STATUS.get("running"):
                    return self.send_json(409, {
                        "ok": False,
                        "error": f"A pipeline run is already in progress for '{PIPELINE_STATUS.get('clip')}'. Please wait.",
                        "status": PIPELINE_STATUS,
                    })

            # Launch background runner
            worker_t = threading.Thread(
                target=run_pipeline_worker,
                args=(base, max_frames, multiscale),
                daemon=True,
            )
            worker_t.start()

            return self.send_json(202, {
                "ok": True,
                "message": f"AI Perception Pipeline analysis started for {base}.",
                "clip": base,
                "max_frames": max_frames,
            })

        return self.send_json(404, {"ok": False, "error": "Endpoint not found"})


if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), TournamentUIRequestHandler) as httpd:
        print("=" * 72)
        print(" 🧪 PICKLEBALL VISION AI // EXPERIMENT SANDBOX CLONE (PORT 3003)")
        print("=" * 72)
        print(f" 🌐 Sandbox Web App : http://localhost:{PORT}")
        print(f" 🏟️ Classic Site    : http://localhost:3000 (Untouched)")
        print(f" 🗄️ Database      : {DB_BACKEND_NAME} ({DB_HOST_INFO})")
        print(f" 🎬 Video Mount   : {os.path.join(ROOT_DIR, 'outputs')}")
        print("=" * 72)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
