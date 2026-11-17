#!/usr/bin/env python3
"""
scripts/render_broadcast_animation.py - Render Broadcast Animation + Live Scoreboard Videos.

Pipeline:
  1. Applies `SavitzkyGolayBounceAnalyzer` (`scipy.signal.savgol_filter`) to raw telemetry to
     reconstruct smooth deduplicated ball flight arcs and detect accurate ground bounces.
  2. Runs `MatchScoreTracker` (`ScoringFSM` + `TwoBounceEnforcer` + `NVZFootFaultDetector`)
     over the Savitzky-Golay refined stream to emit synchronized USAP Scoreboard states.
  3. Renders H.264 showcase videos into `outputs/` featuring yellow wireframe lines, rotating
     player foot halos, smooth magenta ball trails, BOUNCE sub-zone green flashes + concentric
     ripples, translucent bottom-left minimap, and bottom-right live USAP Scoreboard.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pbbase.court.broadcast_fx import BroadcastAnimator
from pbbase.court.homography import CourtCalibrator
from pbbase.pipeline import encode_h264
from pbbase.rules import MatchScoreTracker
from pbbase.tracking.bounce_detector import SavitzkyGolayBounceAnalyzer


DEFAULT_CLIPS = [
    {
        "name": "match1",
        "video": "clips/match1.mp4",
        "telemetry": "data/match1_full_telemetry.json",
        "out_video": "outputs/match1.mp4",
    },
    {
        "name": "match2",
        "video": "clips/match2.mp4",
        "telemetry": "data/match2_telemetry.json",
        "out_video": "outputs/match2.mp4",
    },
    {
        "name": "match_mlp",
        "video": "clips/match_mlp.mp4",
        "telemetry": "data/match_mlp_telemetry.json",
        "out_video": "outputs/match_mlp.mp4",
    },
    {
        "name": "cut",
        "video": "clips/cut.mp4",
        "telemetry": "data/cut_telemetry.json",
        "out_video": "outputs/cut.mp4",
    },
    {
        "name": "screencast",
        "video": "clips/screencast.mp4",
        "telemetry": "data/screencast_telemetry.json",
        "out_video": "outputs/screencast.mp4",
    },
]


def render_broadcast_video(
    clip_name: str,
    video_path: str,
    telemetry_path: str,
    out_video: str,
) -> Dict[str, Any]:
    with open(telemetry_path, "r") as f:
        telem = json.load(f)

    court_info = telem.get("court", {})
    corners_px = [tuple(pt) for pt in court_info.get("corners_px", [])]
    calibrator = CourtCalibrator(corners_px=corners_px if len(corners_px) == 4 else None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or telem.get("fps", 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not calibrator.result.calibrated and len(corners_px) == 4:
        calibrator.calibrate(corners_px, image_size=(w, h))

    # 1. Savitzky-Golay trajectory smoothing & ground bounce localization
    sg_analyzer = SavitzkyGolayBounceAnalyzer()
    sg_res = sg_analyzer.analyze_stream(telem["telemetry_stream"], calibrator=calibrator, fps=fps)
    refined_telem = dict(telem)
    refined_telem["telemetry_stream"] = sg_res["smoothed_stream"]
    refined_telem["bounces"] = sg_res["all_events"]

    # 2. Live USAP Rules Engine + Scoreboard state
    rule_res = MatchScoreTracker().analyze_clip_telemetry(refined_telem, assume_serve_start=None)
    frame_rules = rule_res["frame_rules"]

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    raw_tmp = out_video + ".tmp.mp4"
    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    bounces_by_f = {int(b["frame_idx"]): b for b in sg_res["bounces"]}
    stream = sg_res["smoothed_stream"]
    animator = BroadcastAnimator(calibrator=calibrator, trail_len=20, total_frames=len(stream), fps=fps, clip_name=clip_name)

    for idx, rec in enumerate(stream):
        ret, frame = cap.read()
        if not ret:
            break

        fi = int(rec["frame_idx"])
        vis = animator.render_frame(
            frame=frame,
            frame_idx=fi,
            players=rec.get("players", []),
            ball=rec.get("ball"),
            bounce_event=bounces_by_f.get(fi),
            rule_frame=frame_rules[idx],
        )
        writer.write(vis)

    cap.release()
    writer.release()
    encode_h264(raw_tmp, out_video)

    return {
        "clip": clip_name,
        "out_video": out_video,
        "frames": len(stream),
        "ground_bounce_frames": sorted(bounces_by_f.keys()),
        "score_call": rule_res["fsm_summary"]["serve_context"]["score_call"],
        "two_bounce_summary": rule_res["two_bounce_summary"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Savitzky-Golay Broadcast Animation + Scoreboard Videos")
    parser.add_argument("--clips", nargs="+", default=["match1", "match2", "match_mlp", "cut", "screencast"], help="Clip names to render")
    args = parser.parse_args()

    wanted = set(args.clips)
    results: List[Dict[str, Any]] = []
    for cfg in DEFAULT_CLIPS:
        if cfg["name"] not in wanted:
            continue
        if not os.path.isfile(cfg["video"]) or not os.path.isfile(cfg["telemetry"]):
            print(f"[SKIP] Missing video or telemetry for {cfg['name']}")
            continue
        results.append(
            render_broadcast_video(
                clip_name=cfg["name"],
                video_path=cfg["video"],
                telemetry_path=cfg["telemetry"],
                out_video=cfg["out_video"],
            )
        )
    print(json.dumps(results, indent=2))
