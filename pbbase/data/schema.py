"""
schema.py - Standardized Unified Perception JSON Telemetry Schema (v1.0.0).
Defines the canonical perception telemetry contract for Players, Ball, Court, and Bounces.
Zero scoring or line-call rules: strictly physical spatial perception.
"""
from typing import Dict, List, Optional, Tuple, Any
import datetime

SCHEMA_VERSION = "1.0.0"
SCHEMA_PRODUCER = "Pickleball Base Perception Engine (Zero-Rule Core)"


def validate_telemetry_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that a telemetry dictionary strictly conforms to the Unified Perception Schema v1.0.0.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # 1. Top-Level Required Keys
    required_top = [
        "schema_version",
        "producer",
        "timestamp",
        "video_path",
        "total_frames_processed",
        "fps",
        "resolution",
        "elapsed_s",
        "processing_fps",
        "court",
        "benchmark_summary",
        "telemetry_stream",
    ]
    for k in required_top:
        if k not in data:
            errors.append(f"Missing top-level key: '{k}'")

    if errors:
        return False, errors

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Invalid schema_version: expected '{SCHEMA_VERSION}', got '{data.get('schema_version')}'")

    # 2. Court Object Validation
    court = data.get("court", {})
    for ck in ["calibrated", "reprojection_rmse_ft", "dimensions_ft", "corners_px"]:
        if ck not in court:
            errors.append(f"Missing court key: '{ck}'")

    # 3. Benchmark Summary Validation
    bench = data.get("benchmark_summary", {})
    for bk in ["ball_detection_rate_pct", "mean_players_per_frame", "multiscale_inference", "stage_latencies_ms"]:
        if bk not in bench:
            errors.append(f"Missing benchmark_summary key: '{bk}'")

    # 4. Optional Top-Level Bounces Validation (Week 3)
    if "bounces" in data:
        bounces = data["bounces"]
        if not isinstance(bounces, list):
            errors.append("'bounces' must be a list")
        else:
            for b_idx, b in enumerate(bounces):
                for bk in ["frame_idx", "timestamp_s", "pixel", "court_ft", "is_ground_bounce", "vertical_accel"]:
                    if bk not in b:
                        errors.append(f"Bounce {b_idx}: missing key '{bk}'")

    # 5. Telemetry Stream Validation
    stream = data.get("telemetry_stream", [])
    if not isinstance(stream, list):
        errors.append("telemetry_stream must be a list")
        return False, errors

    for idx, frame in enumerate(stream):
        for fk in ["frame_idx", "timestamp_s", "court_calibrated", "players", "ball"]:
            if fk not in frame:
                errors.append(f"Frame {idx}: missing key '{fk}'")

        # Validate players
        players = frame.get("players", [])
        if not isinstance(players, list):
            errors.append(f"Frame {idx}: 'players' must be a list")
        else:
            for p in players:
                for pk in ["id", "slot", "box", "foot_px", "court_ft", "speed_mph", "side", "is_coasting"]:
                    if pk not in p:
                        errors.append(f"Frame {idx}, Player {p.get('id', '?')}: missing key '{pk}'")

        # Validate ball (can be None/null or dict)
        ball = frame.get("ball")
        if ball is not None:
            for bk in ["pixel", "box", "conf", "speed_mph", "speed_px_per_frame", "court_ft", "interpolated"]:
                if bk not in ball:
                    errors.append(f"Frame {idx}, Ball: missing key '{bk}'")

        # Validate bounce event if present (Week 3)
        bounce = frame.get("bounce")
        if bounce is not None:
            for b_key in ["frame_idx", "timestamp_s", "pixel", "court_ft", "is_ground_bounce"]:
                if b_key not in bounce:
                    errors.append(f"Frame {idx}, Bounce: missing key '{b_key}'")

        # Only check first 50 frames to prevent massive error spam in validation
        if idx >= 50 and errors:
            break

    return len(errors) == 0, errors
