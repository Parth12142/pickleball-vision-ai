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
            # Catmull-Rom basis functions (centripetal formulation approximation)
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


def run(frames, fps, size, cfg):
    weights = cfg.get("weights", "weights/ball_best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Ball weights not found: {weights}\n"
            "Download best.pt from Colab and put it there, "
            "or set modules.ball.enabled: false in config.yaml"
        )
    imgsz = cfg.get("imgsz", 1312)
    conf = cfg.get("conf", 0.15)
    max_jump_px = cfg.get("max_jump_px", 75.0)

    t = time.perf_counter()
    model = YOLO(weights)
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
            if bw > 50 or bh > 50 or bw < 2 or bh < 2:
                continue
            c = float(box.conf[0])
            cands.append({"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0, "conf": c})
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
                dist = np.hypot(c["x"] - pred_x, c["y"] - pred_y)
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

    # Parabolic gap filling
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

            if gap_start > 0 and gap_end < n and gap_len <= 5:
                p1 = interpolated[gap_start - 1]
                p2 = interpolated[gap_end]
                total_dist = np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
                if total_dist <= max_jump_px * (gap_len + 1):
                    for k in range(gap_start, gap_end):
                        alpha = (k - gap_start + 1) / (gap_len + 1)
                        sag = 4.0 * alpha * (1.0 - alpha) * (g_est * (gap_len ** 1.5))
                        interpolated[k] = {
                            "x": p1["x"] + alpha * (p2["x"] - p1["x"]),
                            "y": p1["y"] + alpha * (p2["y"] - p1["y"]) + sag,
                            "conf": 0.5,
                            "speed": float(total_dist / (gap_len + 1)),
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
                }

    process_s = time.perf_counter() - t
    found = sum(1 for d in smoothed if d is not None)

    return {
        "data": {
            "points": smoothed,
            "trail": cfg.get("trail", 18),
            "max_jump_px": max_jump_px,
        },
        "timing": {"load_s": load_s, "process_s": process_s},
        "stats": {
            "frames_with_ball": found,
            "detection_rate": round(found / len(frames), 3),
        },
    }


def draw(frame, i, data):
    pts = data["points"]
    trail_len = data.get("trail", 18)
    max_jump = data.get("max_jump_px", 75.0)

    start_idx = max(0, i - trail_len)
    raw_segment = pts[start_idx : i + 1]

    # Collect valid consecutive points
    valid_pts = []
    for p in raw_segment:
        if p is not None:
            if valid_pts:
                dist = np.hypot(p["x"] - valid_pts[-1][0], p["y"] - valid_pts[-1][1])
                if dist > max_jump * 1.5:
                    valid_pts = []  # Discontinuity/cut: reset
            valid_pts.append((p["x"], p["y"]))

    # 1. Aerodynamic Spline Ribbon Trail (Hawk-Eye / Toptracer Style)
    if len(valid_pts) >= 3:
        smooth_curve = _catmull_rom_subdivide(valid_pts, steps_per_segment=4)
        m = len(smooth_curve)

        glow_overlay = frame.copy()
        for k in range(m - 1):
            pt1 = (int(smooth_curve[k][0]), int(smooth_curve[k][1]))
            pt2 = (int(smooth_curve[k + 1][0]), int(smooth_curve[k + 1][1]))
            progress = (k + 1) / m

            # Aerodynamic Tapering: 1px at tail -> 5px at head
            thickness = max(1, int((progress ** 1.3) * 5))

            # Broadcast Golden/Amber Glow Palette
            # Tail: warm deep amber -> Head: luminous bright neon yellow
            r_val = int(255)
            g_val = int(160 + 90 * progress)
            b_val = int(20 + 180 * (progress ** 2))
            glow_col = (b_val, g_val, r_val)  # BGR

            # Soft diffused outer ribbon
            cv2.line(glow_overlay, pt1, pt2, glow_col, thickness + 3, lineType=cv2.LINE_AA)
            # Focused inner core
            cv2.line(frame, pt1, pt2, (240, 255, 255), max(1, int(progress * 2)), lineType=cv2.LINE_AA)

        cv2.addWeighted(glow_overlay, 0.40, frame, 0.60, 0, frame)

    # 2. Sleek Broadcast AR Ball Highlight & Rotating Compass Reticle
    curr = pts[i]
    if curr is not None:
        cx, cy = int(curr["x"]), int(curr["y"])

        # Soft luminous ball highlight / bloom
        bloom = frame.copy()
        cv2.circle(bloom, (cx, cy), 14, (100, 240, 255), -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(bloom, 0.28, frame, 0.72, 0, frame)

        # Thin minimalist AR circle
        cv2.circle(frame, (cx, cy), 9, (20, 20, 20), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 9, (120, 245, 255), 1, lineType=cv2.LINE_AA)

        # Subtle 4-point rotating tick animation (rotates 4 degrees every frame)
        rot_angle = np.radians((i * 4) % 360)
        for offset in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ang = rot_angle + offset
            tx1 = int(cx + 9 * np.cos(ang))
            ty1 = int(cy + 9 * np.sin(ang))
            tx2 = int(cx + 12 * np.cos(ang))
            ty2 = int(cy + 12 * np.sin(ang))
            cv2.line(frame, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
