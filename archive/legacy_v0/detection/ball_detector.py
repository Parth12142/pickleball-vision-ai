import os
import time
import cv2
import numpy as np
from ultralytics import YOLO


def _catmull_rom_subdivide(points, steps_per_segment=4):
    """Generates a smooth parametric Catmull-Rom spline through trajectory points."""
    if len(points) < 2:
        return points
    if len(points) == 2:
        pts = []
        for alpha in np.linspace(0, 1, steps_per_segment):
            x = (1 - alpha) * points[0][0] + alpha * points[1][0]
            y = (1 - alpha) * points[0][1] + alpha * points[1][1]
            pts.append((x, y))
        return pts

    # Extended points with virtual ends for boundary tangency
    P = [points[0]] + list(points) + [points[-1]]
    spline_curve = []

    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            # Catmull-Rom basis functions
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            spline_curve.append((x, y))

    spline_curve.append(points[-1])
    return spline_curve


def _get_speed_color(speed):
    """Speed-responsive color palette as specified in README:

    - Fast Smash (> 22 px/f): Electric Orange (BGR: 20, 120, 255)
    - Rally Play (10-22 px/f): Neon Chartreuse Laser Yellow (BGR: 30, 245, 180)
    - Drop / Dink (< 10 px/f): Ice Cyan (BGR: 255, 230, 40)
    """
    if speed >= 22.0:
        return (20, 120, 255)       # Electric Orange
    elif speed >= 10.0:
        # Blend from Chartreuse to Orange
        t = (speed - 10.0) / 12.0
        b = int(30 * (1 - t) + 20 * t)
        g = int(245 * (1 - t) + 120 * t)
        r = int(180 * (1 - t) + 255 * t)
        return (b, g, r)
    else:
        # Blend from Cyan to Chartreuse
        t = max(0.0, speed) / 10.0
        b = int(255 * (1 - t) + 30 * t)
        g = int(230 * (1 - t) + 245 * t)
        r = int(40 * (1 - t) + 180 * t)
        return (b, g, r)


def run(frames, fps, size, cfg):
    weights = cfg.get("weights", "weights/ball_best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Ball weights not found: {weights}\n"
            "Download best.pt and put it in weights/, "
            "or set modules.ball.enabled: false in config.yaml"
        )
    imgsz = cfg.get("imgsz", 1312)
    conf = cfg.get("conf", 0.15)
    max_jump_px = cfg.get("max_jump_px", 75.0)

    t = time.perf_counter()
    model = YOLO(weights)
    if frames:
        model.predict(frames[0], imgsz=imgsz, verbose=False)
    load_s = time.perf_counter() - t

    ids = [k for k, v in model.names.items() if str(v).lower() == "ball"]
    if not ids:
        raise ValueError(f"No class called 'ball' in this model: {model.names}")
    ball_id = ids[0]

    raw_candidates = []
    t = time.perf_counter()

    for f in frames:
        r = model.predict(f, imgsz=imgsz, conf=conf, verbose=False)[0]
        cands = []
        for box in r.boxes:
            if int(box.cls[0]) != ball_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            # Filter unrealistic candidate sizes
            if bw > 55 or bh > 55 or bw < 2 or bh < 2:
                continue
            c = float(box.conf[0])
            cands.append({
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "box": [x1, y1, x2, y2],
                "conf": c,
            })
        raw_candidates.append(cands)

    tracked_points = [None] * len(frames)
    last_pos = None
    velocity = [0.0, 0.0]

    for i, cands in enumerate(raw_candidates):
        if not cands:
            last_pos = None
            velocity = [0.0, 0.0]
            continue

        if last_pos is None:
            # Gating new track start: choose candidate with highest confidence
            best = max(cands, key=lambda c: c["conf"])
            best["speed"] = 0.0
            tracked_points[i] = best
            last_pos = best
            velocity = [0.0, 0.0]
        else:
            pred_x = last_pos["x"] + velocity[0]
            pred_y = last_pos["y"] + velocity[1]

            best_cand = None
            best_cost = float("inf")

            for c in cands:
                dist = float(np.hypot(c["x"] - pred_x, c["y"] - pred_y))
                cost = dist - 25.0 * c["conf"]
                if dist < max_jump_px and cost < best_cost:
                    best_cost = cost
                    best_cand = c

            if best_cand is not None:
                new_vx = best_cand["x"] - last_pos["x"]
                new_vy = best_cand["y"] - last_pos["y"]
                speed = float(np.hypot(new_vx, new_vy))
                best_cand["speed"] = speed
                velocity = [0.7 * new_vx + 0.3 * velocity[0], 0.7 * new_vy + 0.3 * velocity[1]]
                tracked_points[i] = best_cand
                last_pos = best_cand
            else:
                last_pos = None
                velocity = [0.0, 0.0]

    # Parabolic gap filling (only for short gaps of 1-4 frames)
    interpolated = list(tracked_points)
    i = 0
    n = len(interpolated)
    g_est = 0.35

    while i < n:
        if interpolated[i] is None:
            gap_start = i
            while i < n and interpolated[i] is None:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_start > 0 and gap_end < n and gap_len <= 4:
                p1 = interpolated[gap_start - 1]
                p2 = interpolated[gap_end]
                total_dist = float(np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"]))
                # Check that distance per frame is physically realistic
                if total_dist <= max_jump_px * (gap_len + 1):
                    step_speed = total_dist / (gap_len + 1)
                    for k in range(gap_start, gap_end):
                        alpha = (k - gap_start + 1) / (gap_len + 1)
                        sag = 4.0 * alpha * (1.0 - alpha) * (g_est * (gap_len ** 1.3))
                        interpolated[k] = {
                            "x": p1["x"] + alpha * (p2["x"] - p1["x"]),
                            "y": p1["y"] + alpha * (p2["y"] - p1["y"]) + sag,
                            "conf": 0.45,
                            "speed": step_speed,
                            "interpolated": True,
                        }
        else:
            i += 1

    # Light smoothing pass
    smoothed = list(interpolated)
    window = 2
    for k in range(len(interpolated)):
        if interpolated[k] is not None:
            nbrs = [
                interpolated[j]
                for j in range(max(0, k - window), min(len(interpolated), k + window + 1))
                if interpolated[j] is not None
            ]
            if len(nbrs) >= 2:
                smoothed[k] = {
                    "x": float(np.mean([p["x"] for p in nbrs])),
                    "y": float(np.mean([p["y"] for p in nbrs])),
                    "conf": interpolated[k]["conf"],
                    "speed": interpolated[k].get("speed", 0.0),
                    "interpolated": interpolated[k].get("interpolated", False),
                }

    # Recompute speed from smoothed positions for clean telemetry
    for k in range(1, len(smoothed)):
        if smoothed[k] is not None and smoothed[k - 1] is not None:
            dx = smoothed[k]["x"] - smoothed[k - 1]["x"]
            dy = smoothed[k]["y"] - smoothed[k - 1]["y"]
            smoothed[k]["speed"] = float(np.hypot(dx, dy))

    process_s = time.perf_counter() - t
    found = sum(1 for d in smoothed if d is not None)

    return {
        "data": {
            "points": smoothed,
            "trail": cfg.get("trail", 16),
            "max_jump_px": max_jump_px,
        },
        "timing": {"load_s": load_s, "process_s": process_s},
        "stats": {
            "frames_with_ball": found,
            "detection_rate": round(found / len(frames), 3) if frames else 0.0,
        },
    }


def draw(frame, i, data):
    pts = data["points"]
    trail_len = data.get("trail", 16)
    max_jump = data.get("max_jump_px", 75.0)

    curr = pts[i] if i < len(pts) else None

    # Only draw laser trail if ball is currently tracked
    if curr is not None:
        # Collect strictly contiguous valid points ending at frame i
        contiguous_pts = []
        for idx in range(i, max(-1, i - trail_len), -1):
            p = pts[idx]
            if p is None:
                break
            if contiguous_pts:
                dist = float(np.hypot(p["x"] - contiguous_pts[-1][0], p["y"] - contiguous_pts[-1][1]))
                if dist > max_jump * 1.5:
                    break
            contiguous_pts.append((p["x"], p["y"], p.get("speed", 0.0)))

        contiguous_pts.reverse()  # Chronological order: tail -> head

        # 1. Aerodynamic Spline Ribbon Trail (Hawk-Eye / Toptracer Style)
        if len(contiguous_pts) >= 3:
            curve_points = [(p[0], p[1]) for p in contiguous_pts]
            smooth_curve = _catmull_rom_subdivide(curve_points, steps_per_segment=3)
            m = len(smooth_curve)

            current_speed = curr.get("speed", 0.0)
            glow_color = _get_speed_color(current_speed)

            glow_overlay = frame.copy()
            for k in range(m - 1):
                pt1 = (int(smooth_curve[k][0]), int(smooth_curve[k][1]))
                pt2 = (int(smooth_curve[k + 1][0]), int(smooth_curve[k + 1][1]))
                progress = (k + 1) / m

                # Aerodynamic Tapering: thin at tail -> thicker at head
                thickness = max(1, int((progress ** 1.3) * 4))

                # Soft diffused outer ribbon
                cv2.line(glow_overlay, pt1, pt2, glow_color, thickness + 2, lineType=cv2.LINE_AA)
                # Focused luminous inner core
                cv2.line(frame, pt1, pt2, (255, 255, 255), max(1, int(progress * 1.8)), lineType=cv2.LINE_AA)

            cv2.addWeighted(glow_overlay, 0.45, frame, 0.55, 0, frame)

        # 2. Sleek Broadcast AR Ball Highlight & Rotating Compass Reticle
        cx, cy = int(curr["x"]), int(curr["y"])
        current_speed = curr.get("speed", 0.0)
        reticle_col = _get_speed_color(current_speed)

        # Soft luminous ball bloom
        bloom = frame.copy()
        cv2.circle(bloom, (cx, cy), 13, reticle_col, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(bloom, 0.25, frame, 0.75, 0, frame)

        # Thin minimalist AR circle
        cv2.circle(frame, (cx, cy), 9, (20, 20, 20), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 9, reticle_col, 1, lineType=cv2.LINE_AA)

        # Center pip (from README spec)
        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Rotating 4-point compass ticks
        rot_angle = np.radians((i * 4) % 360)
        for offset in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ang = rot_angle + offset
            tx1 = int(cx + 9 * np.cos(ang))
            ty1 = int(cy + 9 * np.sin(ang))
            tx2 = int(cx + 12 * np.cos(ang))
            ty2 = int(cy + 12 * np.sin(ang))
            cv2.line(frame, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
