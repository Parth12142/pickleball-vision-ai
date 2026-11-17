#!/usr/bin/env python3
"""
scripts/render_week4_day3.py - Week 4 Day 3 (W4-03 Two-Bounce Rule + W4-04 NVZ Kitchen Detector).

Renders H.264 broadcast verification videos into `outputs/` WITHOUT touching website files and
with NO corner minimap overlay (clean on-court homography Kitchen NVZ zones, serve target box,
player foot clearance tags, live Two-Bounce status pills, and USAP Broadcast Scorebug).
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
from pbbase.court.minimap import PLAYER_COLORS
from pbbase.pipeline import encode_h264
from pbbase.rules import MatchScoreTracker, NVZ_FAR_LINE_Y_FT, NVZ_NEAR_LINE_Y_FT
from pbbase.tracking.ball_trajectory import get_speed_color


DEFAULT_CLIPS = [
    {
        "name": "match1",
        "video": "clips/match1.mp4",
        "telemetry": "data/match1_full_telemetry.json",
        "out_video": "outputs/match1_week4_day3.mp4",
        "out_json": "outputs/match1_week4_day3_telemetry.json",
    },
    {
        "name": "match_mlp",
        "video": "clips/match_mlp.mp4",
        "telemetry": "outputs/match_mlp_base_telemetry.json",
        "out_video": "outputs/match_mlp_week4_day3.mp4",
        "out_json": "outputs/match_mlp_week4_day3_telemetry.json",
    },
    {
        "name": "cut",
        "video": "clips/cut.mp4",
        "telemetry": "outputs/cut_base_telemetry.json",
        "out_video": "outputs/cut_week4_day3.mp4",
        "out_json": "outputs/cut_week4_day3_telemetry.json",
    },
]


def _draw_court_zone_poly(
    frame: np.ndarray,
    calibrator: CourtCalibrator,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color_bgr: Tuple[int, int, int],
    alpha: float,
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw a calibrated translucent floor zone onto the camera frame."""
    if not calibrator.result.calibrated:
        return frame

    pts_px = []
    for cx_ft, cy_ft in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        u, v = calibrator.court_to_pixel(cx_ft, cy_ft)
        pts_px.append([int(round(u)), int(round(v))])
    poly = np.array(pts_px, dtype=np.int32)

    over = frame.copy()
    cv2.fillPoly(over, [poly], color_bgr)
    cv2.addWeighted(over, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.polylines(frame, [poly], True, color_bgr, thickness, cv2.LINE_AA)

    if label:
        lu, lv = calibrator.court_to_pixel((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        cv2.putText(
            frame,
            label,
            (int(lu) - len(label) * 4, int(lv) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (248, 250, 252),
            1,
            cv2.LINE_AA,
        )
    return frame


def _draw_nvz_zones_on_court(
    frame: np.ndarray,
    calibrator: CourtCalibrator,
    rule_frame: Dict[str, Any],
) -> np.ndarray:
    """Render Far & Near NVZ (Kitchen) floor overlays colored dynamically by live NVZ state."""
    p_states = rule_frame.get("nvz_player_states", {})

    def _zone_style(team: str) -> Tuple[Tuple[int, int, int], float, Optional[str]]:
        t_states = [st for st in p_states.values() if st.get("team") == team]
        if any(st.get("status") == "NVZ_FAULT" for st in t_states):
            return (60, 70, 245), 0.28, "NVZ FAULT!"
        if any(st.get("status") == "IN_KITCHEN_LEGAL" for st in t_states):
            return (80, 215, 130), 0.20, "KITCHEN LEGAL"
        if any(st.get("status") == "NVZ_EDGE" for st in t_states):
            return (55, 185, 250), 0.16, "NVZ EDGE WATCH"
        return (180, 170, 70), 0.08, None

    col_b, alpha_b, lbl_b = _zone_style("B")
    col_a, alpha_a, lbl_a = _zone_style("A")

    frame = _draw_court_zone_poly(
        frame, calibrator, 0.0, CourtGeometry.WIDTH_FT, NVZ_FAR_LINE_Y_FT, CourtGeometry.NET_Y_FT,
        col_b, alpha_b, thickness=2, label=lbl_b,
    )
    frame = _draw_court_zone_poly(
        frame, calibrator, 0.0, CourtGeometry.WIDTH_FT, CourtGeometry.NET_Y_FT, NVZ_NEAR_LINE_Y_FT,
        col_a, alpha_a, thickness=2, label=lbl_a,
    )
    return frame


def _draw_day3_hud(
    frame: np.ndarray,
    clip_name: str,
    rule_frame: Dict[str, Any],
    fps: float,
    total_frames: int,
) -> np.ndarray:
    """Render responsive Two-Row Top Broadcast Ribbon & Bottom-Left USAP Scorebug."""
    fh, fw = frame.shape[:2]
    is_portrait = fh > fw
    sc = 1.45 if is_portrait else 1.0

    f_idx = rule_frame["frame_idx"]
    sec = f_idx / max(fps, 1.0)
    r_state = rule_frame["rally_state"]
    tb_phase = rule_frame.get("two_bounce_phase", "")
    is_mid_rally = tb_phase == "MID_RALLY_OPEN"
    fault_type = rule_frame.get("fault_type")
    is_sideout = r_state == "RALLY_ENDED_SIDE_OUT"
    is_rule_fault = fault_type in ("SERVE_VOLLEY_FAULT", "RETURN_VOLLEY_FAULT") or (
        isinstance(fault_type, str) and fault_type.startswith("NVZ_")
    )

    border_col = (60, 75, 245) if is_rule_fault else ((80, 220, 140) if is_sideout else (160, 210, 90))

    # 1. Responsive Top Ribbon (2 rows on landscape, stacked compactly on portrait)
    rib_x1, rib_y1, rib_x2 = 14, 8, fw - 14
    rib_y2 = int(8 + (132 if is_portrait else 64))
    top_over = frame.copy()
    cv2.rectangle(top_over, (rib_x1, rib_y1), (rib_x2, rib_y2), (10, 17, 28), -1)
    cv2.addWeighted(top_over, 0.84, frame, 0.16, 0.0, frame)
    cv2.rectangle(frame, (rib_x1, rib_y1), (rib_x2, rib_y2), border_col, 1, cv2.LINE_AA)

    row1_left = (
        f"WEEK 4 DAY 3 (W4-03 TWO-BOUNCE & W4-04 NVZ) | {clip_name.upper()} | "
        f"F:{f_idx + 1}/{total_frames}  {int(sec // 60):02d}:{sec % 60:05.2f}"
    )
    tb_col = (80, 80, 245) if tb_phase == "VIOLATION_TWO_BOUNCE_FAULT" else (
        (95, 235, 165) if tb_phase in ("TWO_BOUNCE_SATISFIED", "MID_RALLY_OPEN") else (90, 215, 250)
    )
    nvz_txt = rule_frame.get("nvz_status_text", "")
    nvz_in = rule_frame.get("nvz_players_in_zone", [])
    nvz_col = (80, 80, 245) if "FAULT" in nvz_txt else (
        (95, 235, 165) if nvz_in else ((80, 210, 255) if "WATCH" in nvz_txt else (196, 224, 240))
    )

    if is_portrait:
        cv2.putText(frame, row1_left, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 232, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, rule_frame["status_banner"], (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (190, 245, 218), 1, cv2.LINE_AA)
        cv2.putText(frame, rule_frame.get("two_bounce_text", ""), (28, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, tb_col, 1, cv2.LINE_AA)
        cv2.putText(frame, nvz_txt, (740, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, nvz_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, row1_left, (26, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, rule_frame["status_banner"], (max(630, fw // 2 + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 245, 218), 1, cv2.LINE_AA)
        cv2.putText(frame, rule_frame.get("two_bounce_text", ""), (26, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, tb_col, 1, cv2.LINE_AA)
        cv2.putText(frame, nvz_txt, (max(630, fw // 2 + 10), 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, nvz_col, 1, cv2.LINE_AA)

    # 2. Bottom-Left USAP Autonomous Broadcast Scorebug
    bug_h = int(round(140 * sc))
    bug_w = min(int(round(552 * sc)), fw - 32)
    bx1, by1 = 16, fh - 16 - bug_h
    bx2, by2 = 16 + bug_w, fh - 16
    bug_over = frame.copy()
    cv2.rectangle(bug_over, (bx1, by1), (bx2, by2), (10, 17, 28), -1)
    cv2.addWeighted(bug_over, 0.88, frame, 0.12, 0.0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_col, 2 if (is_sideout or is_rule_fault or is_portrait) else 1, cv2.LINE_AA)

    pad_x = int(round(14 * sc))
    cv2.putText(
        frame,
        "PICKVIZ USAP AUTONOMOUS REFEREE (W4 DAY 3)",
        (bx1 + pad_x, by1 + int(round(22 * sc))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43 * sc,
        (165, 235, 195),
        1 if sc == 1.0 else 2,
        cv2.LINE_AA,
    )

    score_call = rule_frame["score_call"]
    pill_bg = (30, 110, 72) if is_sideout else (25, 65, 115)
    px1, py1 = bx2 - int(round(126 * sc)), by1 + int(round(10 * sc))
    px2, py2 = bx2 - int(round(12 * sc)), by1 + int(round(52 * sc))
    cv2.rectangle(frame, (px1, py1), (px2, py2), pill_bg, -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (180, 230, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, score_call, (px1 + int(round(14 * sc)), py1 + int(round(30 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * sc, (255, 255, 255), 2, cv2.LINE_AA)

    srv_team = rule_frame["serving_team"]
    mark_a = "*" if (srv_team == "A" and not is_mid_rally) else " "
    mark_b = "*" if (srv_team == "B" and not is_mid_rally) else " "
    col_a = (255, 255, 255) if (srv_team == "A" or is_mid_rally) else (165, 180, 196)
    col_b = (255, 255, 255) if (srv_team == "B" or is_mid_rally) else (165, 180, 196)
    cv2.putText(frame, f"{mark_a} TEAM A (NEAR) : {rule_frame['score_a']}", (bx1 + pad_x, by1 + int(round(48 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sc, col_a, 1 if sc == 1.0 else 2, cv2.LINE_AA)
    cv2.putText(frame, f"{mark_b} TEAM B (FAR)  : {rule_frame['score_b']}", (bx1 + pad_x, by1 + int(round(70 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sc, col_b, 1 if sc == 1.0 else 2, cv2.LINE_AA)

    if is_mid_rally:
        srv_line = f"MODE: MID-RALLY LIVE  |  SIDE-OUTS: {rule_frame['side_out_count']}  |  NO FAULTS"
    else:
        srv_line = (
            f"SRV #{rule_frame['server_number']}: {rule_frame['active_server_id']} ({rule_frame['server_box']}) -> "
            f"RCV: {rule_frame['active_receiver_id']} ({rule_frame['target_receiver_box']}) | SIDE-OUTS: {rule_frame['side_out_count']}"
        )
    cv2.putText(frame, srv_line, (bx1 + pad_x, by1 + int(round(96 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.39 * sc, (210, 224, 238), 1, cv2.LINE_AA)

    tb_cnt = int(rule_frame.get("two_bounce_count", 0))
    cv2.putText(frame, "2-BOUNCE:", (bx1 + pad_x, by1 + int(round(122 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * sc, (190, 210, 230), 1, cv2.LINE_AA)
    for idx_pip in range(2):
        cx_p = bx1 + pad_x + int(round((92 + idx_pip * 22) * sc))
        cy_p = by1 + int(round(117 * sc))
        filled = idx_pip < tb_cnt
        pcol = (85, 235, 160) if filled else (75, 95, 120)
        cv2.circle(frame, (cx_p, cy_p), int(round(7 * sc)), pcol, -1 if filled else 1, cv2.LINE_AA)

    nvz_short = "IN KITCHEN LEGAL" if nvz_in else ("NVZ FAULT!" if "FAULT" in nvz_txt else "NVZ CLEAR")
    cv2.putText(frame, f"|  {nvz_short}", (bx1 + pad_x + int(round(138 * sc)), by1 + int(round(122 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * sc, nvz_col, 1, cv2.LINE_AA)

    return frame


def render_day3_clip(
    clip_name: str,
    video_path: str,
    telemetry_path: str,
    out_video: str,
    out_json: str,
) -> Dict[str, Any]:
    with open(telemetry_path, "r") as f:
        telem = json.load(f)

    tracker = MatchScoreTracker()
    rule_result = tracker.analyze_clip_telemetry(telem)
    frame_rules = rule_result["frame_rules"]
    serve_bounce_f = rule_result["serve_bounce_frame"]
    tb_sum = rule_result["two_bounce_summary"]
    ret_bounce_f = tb_sum.get("return_bounce_frame")

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

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    raw_tmp = out_video + ".tmp.mp4"
    writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stream = telem["telemetry_stream"]
    bounces_by_frame = {int(b["frame_idx"]): b for b in telem.get("bounces", [])}
    ball_history: List[Optional[Tuple[float, float, float]]] = []
    total_frames = len(stream)

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rec = stream[idx]
        r_frame = frame_rules[idx]
        vis = frame.copy()

        # 1. Direct floor overlays: Kitchen (NVZ) zones + serve box highlight (no corner minimap!)
        vis = _draw_nvz_zones_on_court(vis, calibrator, r_frame)
        if serve_bounce_f is not None and idx <= (serve_bounce_f + 10):
            x0, x1, y0, y1 = tuple(r_frame["target_quadrant_ft"])
            s_ok = idx >= serve_bounce_f and bool(rule_result["serve_valid_diagonal"])
            vis = _draw_court_zone_poly(
                vis, calibrator, x0, x1, y0, y1,
                (60, 220, 130) if s_ok else (82, 196, 160),
                alpha=0.18,
                thickness=2,
                label="SERVE IN BOX" if s_ok else "TARGET SERVE BOX",
            )

        # 2. Player bounding boxes + foot markers + NVZ clearance tags
        nvz_p_states = r_frame.get("nvz_player_states", {})
        for p in rec.get("players", []):
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            pid_str = str(p.get("slot") or f"P{p.get('id', '?')}")
            slot_idx = max(0, int(pid_str[1:]) - 1) if pid_str.startswith("P") and pid_str[1:].isdigit() else 0
            p_nvz = nvz_p_states.get(pid_str, {})
            n_stat = p_nvz.get("status", "CLEAR")

            if n_stat == "NVZ_FAULT":
                col = (60, 70, 245)
            elif n_stat == "IN_KITCHEN_LEGAL":
                col = (80, 225, 130)
            elif n_stat == "NVZ_EDGE":
                col = (60, 195, 255)
            else:
                col = PLAYER_COLORS[slot_idx % len(PLAYER_COLORS)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            if p.get("foot_px"):
                fx, fy = int(round(p["foot_px"][0])), int(round(p["foot_px"][1]))
                cv2.circle(vis, (fx, fy), 6, col, -1, cv2.LINE_AA)

            role_tag = ""
            if serve_bounce_f is not None:
                if pid_str == r_frame["active_server_id"]:
                    role_tag = " [NEXT SRV]" if r_frame["rally_state"] == "RALLY_ENDED_SIDE_OUT" else " [SRV]"
                elif pid_str == r_frame["active_receiver_id"]:
                    role_tag = " [RCV]"

            clr_ft = p_nvz.get("clearance_ft")
            nvz_chip = ""
            if n_stat == "NVZ_FAULT":
                nvz_chip = " NVZ FAULT!"
            elif n_stat == "IN_KITCHEN_LEGAL":
                nvz_chip = " IN NVZ LEGAL"
            elif n_stat == "NVZ_EDGE" and clr_ft is not None:
                nvz_chip = f" EDGE+{clr_ft:.1f}ft"
            elif clr_ft is not None:
                nvz_chip = f" +{clr_ft:.1f}ft"

            lbl = f"{pid_str}{role_tag}{nvz_chip}"
            lbl_sc = 0.50 if h > w else 0.36
            pill_w = int(round(len(lbl) * (10 if h > w else 7) + 14))
            pill_h = 24 if h > w else 18
            cx = (x1 + x2) // 2
            bx1 = max(4, min(w - pill_w - 4, cx - pill_w // 2))
            by1 = max((144 if h > w else 76), y1 - (pill_h + 4))
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + pill_h), (12, 19, 32), -1)
            cv2.rectangle(vis, (bx1, by1), (bx1 + pill_w, by1 + pill_h), col, 1, cv2.LINE_AA)
            cv2.putText(vis, lbl, (bx1 + 6, by1 + pill_h - 6), cv2.FONT_HERSHEY_SIMPLEX, lbl_sc, (242, 245, 250), 1, cv2.LINE_AA)

        # 3. Ball trail & annotated Two-Bounce / OOB / Volley rings
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
        if bounce is not None and bounce.get("pixel"):
            bpx, bpy = int(bounce["pixel"][0]), int(bounce["pixel"][1])
            cxy = bounce["court_ft"]
            in_rect = (0.0 <= cxy[0] <= CourtGeometry.WIDTH_FT) and (0.0 <= cxy[1] <= CourtGeometry.LENGTH_FT)
            is_ground = bounce.get("is_ground_bounce", True)

            if serve_bounce_f is not None and idx == serve_bounce_f:
                tag, b_col = "1/2 SERVE BOUNCE IN", (80, 230, 130)
            elif ret_bounce_f is not None and idx == ret_bounce_f:
                tag, b_col = "2/2 RETURN CLEAR (VOLLEY OPEN)", (80, 235, 190)
            elif not in_rect:
                tag, b_col = "OOB SIDE-OUT", (70, 80, 240)
            elif not is_ground:
                tag, b_col = "VOLLEY CONTACT", (90, 210, 255)
            else:
                tag, b_col = "GROUND BOUNCE", (80, 220, 120)

            cv2.circle(vis, (bpx, bpy), 18, b_col, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{tag} ({cxy[0]:.1f},{cxy[1]:.1f})ft", (max(10, bpx - 80), max(84, bpy - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)

        vis = _draw_day3_hud(vis, clip_name, r_frame, fps=fps, total_frames=total_frames)
        writer.write(vis)

    cap.release()
    writer.release()
    encode_h264(raw_tmp, out_video)

    with open(out_json, "w") as f:
        json.dump(rule_result, f, indent=2)

    return {
        "clip": clip_name,
        "out_video": out_video,
        "out_json": out_json,
        "total_frames": total_frames,
        "score_call": rule_result["fsm_summary"]["serve_context"]["score_call"],
        "two_bounce_summary": rule_result["two_bounce_summary"],
        "nvz_summary": rule_result["nvz_summary"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Week 4 Day 3 output videos into outputs/")
    parser.add_argument("--clips", nargs="+", default=["match1", "match_mlp", "cut"], help="Clip names to render")
    args = parser.parse_args()

    wanted = set(args.clips)
    results = []
    for cfg in DEFAULT_CLIPS:
        if cfg["name"] not in wanted:
            continue
        if not os.path.isfile(cfg["video"]) or not os.path.isfile(cfg["telemetry"]):
            print(f"[SKIP] Missing video or telemetry for {cfg['name']}")
            continue
        results.append(
            render_day3_clip(
                clip_name=cfg["name"],
                video_path=cfg["video"],
                telemetry_path=cfg["telemetry"],
                out_video=cfg["out_video"],
                out_json=cfg["out_json"],
            )
        )
    print(json.dumps(results, indent=2))
