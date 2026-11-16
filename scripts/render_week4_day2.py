#!/usr/bin/env python3
"""
scripts/render_week4_day2.py - Week 4 Day 2 (W4-01 & W4-02 Halfway Deliverable).

Applies MatchScoreTracker + ScoringFSM to match1.mp4 perception telemetry and renders:
  - Browser-playable H.264 broadcast output video (`outputs/match1_week4_day2.mp4`)
    with Top-Strip Status Bar, Bottom-Left Broadcast Scorebug (0-0-2 -> SIDE-OUT -> 0-0-1),
    Diagonal Service Box highlight during serve, Player SRV/RCV badges, and Bottom-Right
    2D Court Minimap with synced ball trajectory and bounce rings.
  - Complete rule-annotated JSON telemetry (`outputs/match1_week4_day2_telemetry.json`).
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pbbase.court.homography import CourtCalibrator, CourtGeometry
from pbbase.court.minimap import CourtMinimap, PLAYER_COLORS
from pbbase.pipeline import encode_h264
from pbbase.rules import MatchScoreTracker
from pbbase.tracking.ball_trajectory import get_speed_color


def _draw_target_service_quadrant(
    frame: np.ndarray,
    calibrator: CourtCalibrator,
    minimap: CourtMinimap,
    minimap_img: np.ndarray,
    bounds_ft: Tuple[float, float, float, float],
    valid_hit: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Highlight the official diagonal target service quadrant on camera floor & minimap."""
    x0, x1, y0, y1 = bounds_ft
    color = (60, 210, 120) if valid_hit else (82, 196, 160)

    # 1. Camera floor translucent highlight via inverse homography
    if calibrator.result.calibrated:
        corners_ft = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts_px = []
        for cx_ft, cy_ft in corners_ft:
            u, v = calibrator.court_to_pixel(cx_ft, cy_ft)
            pts_px.append([int(round(u)), int(round(v))])
        poly = np.array(pts_px, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0.0, frame)
        cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)

        lbl_u, lbl_v = calibrator.court_to_pixel((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        cv2.putText(
            frame,
            "TARGET SERVE BOX",
            (int(lbl_u) - 70, int(lbl_v) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 255, 245),
            1,
            cv2.LINE_AA,
        )

    # 2. Minimap quadrant highlight
    m0 = minimap.court_to_canvas(x0, y0)
    m1 = minimap.court_to_canvas(x1, y1)
    m_over = minimap_img.copy()
    cv2.rectangle(m_over, m0, m1, color, -1)
    cv2.addWeighted(m_over, 0.28, minimap_img, 0.72, 0.0, minimap_img)
    cv2.rectangle(minimap_img, m0, m1, color, 2)

    return frame, minimap_img


def _draw_week4_hud(
    frame: np.ndarray,
    rule_frame: Dict[str, Any],
    fps: float,
    total_frames: int,
) -> np.ndarray:
    """Render Top Status Ribbon + Bottom-Left USAP Broadcast Scorebug with zero player/ball overlap."""
    fh, fw = frame.shape[:2]
    f_idx = rule_frame["frame_idx"]
    score_call = rule_frame["score_call"]
    r_state = rule_frame["rally_state"]
    is_sideout = r_state == "RALLY_ENDED_SIDE_OUT"

    # 1. Top Full-Width Header Strip (16, 8) -> (fw - 16, 46): 0 player/ball overlaps verified
    top_over = frame.copy()
    cv2.rectangle(top_over, (16, 8), (fw - 16, 46), (12, 19, 30), -1)
    cv2.addWeighted(top_over, 0.84, frame, 0.16, 0.0, frame)
    border_col = (80, 220, 140) if is_sideout else (160, 196, 82)
    cv2.rectangle(frame, (16, 8), (fw - 16, 46), border_col, 1, cv2.LINE_AA)

    sec = f_idx / fps
    top_left_txt = (
        f"WEEK 4 DAY 2 (W4-01 & W4-02)  |  MATCH1.MP4  |  "
        f"F:{f_idx + 1}/{total_frames}  {int(sec // 60):02d}:{sec % 60:05.2f}"
    )
    cv2.putText(frame, top_left_txt, (28, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, rule_frame["status_banner"], (640, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (196, 245, 220), 1, cv2.LINE_AA)

    # 2. Bottom-Left Broadcast Scorebug (16, fh - 132) -> (520, fh - 18): 0 overlaps verified
    bx1, by1, bx2, by2 = 16, fh - 132, 520, fh - 18
    bug_over = frame.copy()
    cv2.rectangle(bug_over, (bx1, by1), (bx2, by2), (12, 19, 30), -1)
    cv2.addWeighted(bug_over, 0.88, frame, 0.12, 0.0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_col, 2 if is_sideout else 1, cv2.LINE_AA)

    # Title row inside scorebug
    tag_col = (140, 240, 180) if is_sideout else (196, 230, 140)
    cv2.putText(
        frame,
        "USAP AUTONOMOUS SCOREBUG (0-0-2 & SIDE-OUT)",
        (bx1 + 14, by1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        tag_col,
        1,
        cv2.LINE_AA,
    )

    # Score badge pill on right of scorebug
    pill_bg = (30, 110, 72) if is_sideout else (25, 65, 115)
    cv2.rectangle(frame, (bx2 - 124, by1 + 10), (bx2 - 12, by1 + 52), pill_bg, -1)
    cv2.rectangle(frame, (bx2 - 124, by1 + 10), (bx2 - 12, by1 + 52), (180, 230, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, score_call, (bx2 - 110, by1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Team rows
    srv_team = rule_frame["serving_team"]
    mark_a = "*" if srv_team == "A" else " "
    mark_b = "*" if srv_team == "B" else " "
    col_a = (255, 255, 255) if srv_team == "A" else (165, 180, 196)
    col_b = (255, 255, 255) if srv_team == "B" else (165, 180, 196)

    cv2.putText(
        frame,
        f"{mark_a} TEAM A (P1/P2) : {rule_frame['score_a']}",
        (bx1 + 14, by1 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        col_a,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{mark_b} TEAM B (P4/P3) : {rule_frame['score_b']}",
        (bx1 + 14, by1 + 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        col_b,
        1,
        cv2.LINE_AA,
    )

    # Bottom server / side-out line
    state_line = (
        f"SRV #{rule_frame['server_number']}: {rule_frame['active_server_id']} ({rule_frame['server_box']})  ->  "
        f"RCV: {rule_frame['active_receiver_id']} ({rule_frame['target_receiver_box']})  |  SIDE-OUTS: {rule_frame['side_out_count']}"
    )
    cv2.putText(frame, state_line, (bx1 + 14, by1 + 101), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (210, 224, 238), 1, cv2.LINE_AA)

    return frame


def render_week4_day2(
    video_path: str = "clips/match1.mp4",
    telemetry_path: str = "data/match1_full_telemetry.json",
    out_video: str = "outputs/match1_week4_day2.mp4",
    out_json: str = "outputs/match1_week4_day2_telemetry.json",
    save_keyframes_dir: Optional[str] = None,
) -> Dict[str, Any]:
    with open(telemetry_path, "r") as f:
        telem = json.load(f)

    tracker = MatchScoreTracker()
    rule_result = tracker.analyze_clip_telemetry(telem)
    frame_rules = rule_result["frame_rules"]
    serve_bounce_f = rule_result["serve_bounce_frame"] or 26

    # Calibrate 2D/camera homography using exact telemetry corners
    court_info = telem.get("court", {})
    corners_px = [tuple(pt) for pt in court_info.get("corners_px", [])]
    calibrator = CourtCalibrator(corners_px=corners_px if len(corners_px) == 4 else None)
    minimap = CourtMinimap(court_scale=9.0, padding_x=54, padding_y=50, trail_len=20)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or telem.get("fps", 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not calibrator.result.calibrated and len(corners_px) == 4:
        calibrator.calibrate(corners_px, image_size=(w, h))

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    raw_tmp = out_video + ".tmp.mp4"
    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stream = telem["telemetry_stream"]
    bounces_by_frame = {int(b["frame_idx"]): b for b in telem.get("bounces", [])}
    ball_history: List[Optional[Tuple[float, float, float]]] = []
    minimap.reset()

    if save_keyframes_dir:
        os.makedirs(save_keyframes_dir, exist_ok=True)
    keyframe_targets = {12, 26, 150, 263, 272}

    total_frames = len(stream)
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rec = stream[idx]
        r_frame = frame_rules[idx]
        vis = frame.copy()

        # Highlight target service box cleanly during serve & serve landing frames
        show_serve_box = idx <= (serve_bounce_f + 8)
        target_quad = tuple(r_frame["target_quadrant_ft"])

        # Track players & tag active server / receiver
        for p in rec.get("players", []):
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            pid_str = str(p.get("slot") or f"P{p.get('id', '?')}")
            slot_idx = max(0, int(pid_str[1:]) - 1) if pid_str.startswith("P") and pid_str[1:].isdigit() else 0
            col = PLAYER_COLORS[slot_idx % len(PLAYER_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

            role_tag = ""
            if pid_str == r_frame["active_server_id"]:
                role_tag = " [NEXT SRV]" if r_frame["rally_state"] == "RALLY_ENDED_SIDE_OUT" else " [SRV]"
            elif pid_str == r_frame["active_receiver_id"]:
                role_tag = " [RCV]"

            lbl = f"{pid_str}{role_tag}"
            if p.get("court_ft"):
                lbl += f" ({p['court_ft'][0]:.1f},{p['court_ft'][1]:.1f})ft"

            pill_w = len(lbl) * 7 + 16
            cx = (x1 + x2) // 2
            bx1 = max(4, min(w - pill_w - 4, cx - pill_w // 2))
            by1 = max(50, y1 - 22)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + 18), (15, 23, 42), -1)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + 18), col, 1, cv2.LINE_AA)
            cv2.putText(vis, lbl, (bx1 + 6, by1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

        # Ball trail & bounce ripples
        ball = rec.get("ball")
        if ball is not None and ball.get("pixel"):
            bx, by = float(ball["pixel"][0]), float(ball["pixel"][1])
            spd_px = float(ball.get("speed_px_per_frame", 0.0))
            ball_history.append((bx, by, spd_px))

            valid_pts = [pt for pt in ball_history[-16:] if pt is not None]
            if len(valid_pts) >= 2:
                over = vis.copy()
                for k in range(len(valid_pts) - 1):
                    p1 = (int(valid_pts[k][0]), int(valid_pts[k][1]))
                    p2 = (int(valid_pts[k + 1][0]), int(valid_pts[k + 1][1]))
                    cv2.line(over, p1, p2, get_speed_color(valid_pts[k + 1][2]), max(2, int((k + 1) / len(valid_pts) * 5)), cv2.LINE_AA)
                cv2.addWeighted(over, 0.50, vis, 0.50, 0.0, vis)

            cv2.circle(vis, (int(bx), int(by)), 8, (80, 240, 180), 2, cv2.LINE_AA)
            cv2.circle(vis, (int(bx), int(by)), 2, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            ball_history.append(None)

        bounce = rec.get("bounce") or bounces_by_frame.get(idx)
        if bounce is not None:
            if bounce.get("is_ground_bounce", True):
                minimap.add_bounce(
                    court_xy_ft=tuple(bounce["court_ft"]),
                    frame_idx=idx,
                )
            if bounce.get("pixel"):
                bpx, bpy = int(bounce["pixel"][0]), int(bounce["pixel"][1])
                cxy = bounce["court_ft"]
                in_rect = (0.0 <= cxy[0] <= CourtGeometry.WIDTH_FT) and (0.0 <= cxy[1] <= CourtGeometry.LENGTH_FT)
                b_col = (80, 220, 120) if in_rect else (70, 80, 235)
                cv2.circle(vis, (bpx, bpy), 18, b_col, 2, cv2.LINE_AA)
                tag = "SERVE IN" if idx == serve_bounce_f else ("OOB FAULT" if not in_rect else "BOUNCE")
                cv2.putText(vis, f"{tag} ({cxy[0]:.1f},{cxy[1]:.1f})ft", (max(10, bpx - 65), max(64, bpy - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, b_col, 1, cv2.LINE_AA)

        minimap_canvas = minimap.render(players=rec.get("players"), ball=rec.get("ball"), frame_idx=idx)
        if show_serve_box:
            vis, minimap_canvas = _draw_target_service_quadrant(
                vis,
                calibrator,
                minimap,
                minimap_canvas,
                target_quad,
                valid_hit=(idx >= serve_bounce_f and bool(rule_result["serve_valid_diagonal"])),
            )

        vis = minimap.compose_onto_frame(vis, minimap_canvas, position="bottom-right", alpha=0.88)
        vis = _draw_week4_hud(vis, r_frame, fps=fps, total_frames=total_frames)

        writer.write(vis)
        if save_keyframes_dir and idx in keyframe_targets:
            cv2.imwrite(os.path.join(save_keyframes_dir, f"day2_frame_{idx:03d}.jpg"), vis)

    cap.release()
    writer.release()
    encode_h264(raw_tmp, out_video)

    with open(out_json, "w") as f:
        json.dump(rule_result, f, indent=2)

    return {
        "out_video": out_video,
        "out_json": out_json,
        "total_frames": total_frames,
        "fsm_summary": rule_result["fsm_summary"],
        "serve_valid_diagonal": rule_result["serve_valid_diagonal"],
        "terminal_event_frame": rule_result["terminal_event_frame"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Week 4 Day 2 (W4-01 & W4-02) Output on match1.mp4")
    parser.add_argument("--video", default="clips/match1.mp4")
    parser.add_argument("--telemetry", default="data/match1_full_telemetry.json")
    parser.add_argument("--out_video", default="outputs/match1_week4_day2.mp4")
    parser.add_argument("--out_json", default="outputs/match1_week4_day2_telemetry.json")
    parser.add_argument("--keyframes_dir", default=None)
    args = parser.parse_args()

    res = render_week4_day2(
        video_path=args.video,
        telemetry_path=args.telemetry,
        out_video=args.out_video,
        out_json=args.out_json,
        save_keyframes_dir=args.keyframes_dir,
    )
    print(json.dumps(res, indent=2))
