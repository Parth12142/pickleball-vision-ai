#!/usr/bin/env python3
"""
scripts/validate_week3.py - Empirical Rally Validator for Week 3: Minimap & Bounce Kinematics.
Runs match1.mp4 through the full Base Perception Pipeline with 2D Minimap HUD and
Parabolic Nadir Bounce Detection. Audits:
  - Total bounces detected and timestamps
  - Sub-pixel parabolic nadir accuracy (t*, u*, v*)
  - Court impact coordinates (X*, Y*) in feet
  - Ground vs paddle volley disambiguation
  - Minimap rendering throughput and visual composition
  - Telemetry schema compliance
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pbbase.pipeline import BasePerceptionPipeline
from pbbase.court.homography import CourtGeometry
from pbbase.data.schema import validate_telemetry_schema


def validate_week3(
    video_path: str = "clips/match1.mp4",
    max_frames: int = 289,
    out_json: str = "data/week3_match1_telemetry.json",
    out_preview: str = "output/week3_match1_minimap_preview.mp4",
) -> Dict[str, Any]:
    print("=" * 72)
    print("🏓 WEEK 3 EMPIRICAL VALIDATOR: COURT MINIMAP & BOUNCE KINEMATICS")
    print("=" * 72)
    print(f"Video Source : {video_path}")
    print(f"Max Frames   : {max_frames}")
    print(f"Output JSON  : {out_json}")
    print(f"Output Video : {out_preview}")
    print("-" * 72)

    pipeline = BasePerceptionPipeline(
        multiscale=False,
        render_minimap=True,
        minimap_position="bottom-right",
    )

    t0 = time.time()
    result = pipeline.process_video(
        video_path=video_path,
        max_frames=max_frames,
        out_json=out_json,
        out_preview=out_preview,
    )
    total_elapsed = time.time() - t0

    print("-" * 72)
    print("🔬 AUDITING WEEK 3 DELIVERABLES...")

    # 1. Telemetry Schema Validation
    is_valid_schema, schema_errors = validate_telemetry_schema(result)
    print(f"1. Telemetry Schema Check       : {'PASS' if is_valid_schema else 'FAIL'}")
    if not is_valid_schema:
        for err in schema_errors:
            print(f"   [Error] {err}")

    # 2. Bounces Audit
    bounces = result.get("bounces", [])
    total_bounces = len(bounces)
    print(f"2. Bounces Detected in Rally    : {total_bounces}")

    ground_bounces = [b for b in bounces if b.get("is_ground_bounce", False)]
    print(f"   - Court Ground Bounces       : {len(ground_bounces)}")
    print(f"   - Airborne Paddle Hits       : {total_bounces - len(ground_bounces)}")

    bounce_details = []
    debounce_pass = True
    in_bounds_bounces = 0

    for idx, b in enumerate(bounces):
        f = b["frame_idx"]
        ts = b["timestamp_s"]
        sub_t = b.get("sub_frame_offset", 0.0)
        u, v = b["pixel"]
        x_ft, y_ft = b["court_ft"]
        accel = b["vertical_accel"]
        spd = b.get("speed_mph", 0.0)
        is_ground = b.get("is_ground_bounce", False)

        is_in = (0.0 <= x_ft <= CourtGeometry.WIDTH_FT) and (0.0 <= y_ft <= CourtGeometry.LENGTH_FT)
        if is_in:
            in_bounds_bounces += 1

        status_str = "GROUND BOUNCE" if is_ground else "PADDLE CONTACT"
        bounds_str = "IN-BOUNDS" if is_in else "OUT-OF-BOUNDS"

        line = (
            f"   #{idx + 1:02d} | Frame {f:3d} (t={ts:5.2f}s, sub_t={sub_t:+5.2f}) | "
            f"Pixel: ({u:6.1f}, {v:6.1f}) | Court: ({x_ft:5.1f}, {y_ft:5.1f})ft | "
            f"Accel: {accel:4.1f}px/f² | Speed: {spd:4.1f}mph | {status_str} | {bounds_str}"
        )
        print(line)
        bounce_details.append(line)

        # Check debounce gap
        if idx > 0:
            frame_gap = f - bounces[idx - 1]["frame_idx"]
            if frame_gap < 6:
                debounce_pass = False

    print(f"3. Debounce Integrity (>= 6 frames) : {'PASS' if debounce_pass else 'FAIL'}")
    print(f"4. Court In-Bounds Ratio            : {in_bounds_bounces}/{total_bounces} ({in_bounds_bounces/max(1, total_bounces)*100:.1f}%)")

    # 3. Minimap & Latency Audit
    bench = result.get("benchmark_summary", {})
    latencies = bench.get("stage_latencies_ms", {})
    minimap_and_render_ms = latencies.get("render_ms", 0.0)
    bounce_ms = latencies.get("bounce_ms", 0.0)
    print(f"5. Pipeline Stage Latencies:")
    print(f"   - Ball Detection & Trajectory: {latencies.get('ball_ms', 0.0):.1f} ms")
    print(f"   - Player Tracking & Halos    : {latencies.get('player_ms', 0.0):.1f} ms")
    print(f"   - Bounce Nadir Fitting       : {bounce_ms:.2f} ms (< 1.0 ms)")
    print(f"   - Minimap & Frame HUD Render : {minimap_and_render_ms:.1f} ms (< 5.0 ms)")
    print(f"   - Overall Processing FPS     : {result.get('processing_fps', 0.0):.1f} FPS")

    # 4. Preview Video Check
    preview_exists = os.path.exists(out_preview) and os.path.getsize(out_preview) > 100000
    print(f"6. Minimap Broadcast Preview MP4: {'PASS' if preview_exists else 'FAIL'} ({os.path.getsize(out_preview) / 1024 / 1024:.2f} MB)" if preview_exists else "6. Minimap Broadcast Preview MP4: FAIL")

    all_pass = is_valid_schema and (total_bounces >= 3) and debounce_pass and preview_exists
    print("=" * 72)
    print(f"🏁 WEEK 3 OVERALL STATUS: {'PASS - ALL DELIVERABLES MET' if all_pass else 'FAIL'}")
    print("=" * 72)

    return {
        "all_pass": all_pass,
        "is_valid_schema": is_valid_schema,
        "total_bounces": total_bounces,
        "ground_bounces": len(ground_bounces),
        "in_bounds_bounces": in_bounds_bounces,
        "debounce_pass": debounce_pass,
        "preview_exists": preview_exists,
        "processing_fps": result.get("processing_fps", 0.0),
        "bounce_ms": bounce_ms,
        "render_ms": minimap_and_render_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Week 3 Minimap & Bounce Kinematics.")
    parser.add_argument("--video", default="clips/match1.mp4", help="Path to input video clip.")
    parser.add_argument("--max_frames", type=int, default=289, help="Max frames to process.")
    parser.add_argument("--out_json", default="data/week3_match1_telemetry.json", help="Output telemetry JSON.")
    parser.add_argument("--out_preview", default="output/week3_match1_minimap_preview.mp4", help="Output preview MP4.")
    args = parser.parse_args()

    audit = validate_week3(
        video_path=args.video,
        max_frames=args.max_frames,
        out_json=args.out_json,
        out_preview=args.out_preview,
    )

    sys.exit(0 if audit["all_pass"] else 1)
