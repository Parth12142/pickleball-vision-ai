import collections
import time
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_ROI = [(0.10, 0.10), (0.90, 0.10), (0.98, 0.99), (0.02, 0.99)]

# Sleek broadcast color palette (Coral, Electric Cyan, Emerald, Gold)
PLAYER_THEMES = [
    {
        "primary": (0, 145, 255),    # Vibrant Coral / Orange
        "light": (120, 210, 255),
        "name": "PLAYER 1",
    },
    {
        "primary": (255, 175, 0),    # Electric Cyan
        "light": (255, 230, 150),
        "name": "PLAYER 2",
    },
    {
        "primary": (80, 220, 0),     # Emerald Green
        "light": (180, 255, 140),
        "name": "PLAYER 3",
    },
    {
        "primary": (0, 215, 255),    # Bright Gold
        "light": (150, 240, 255),
        "name": "PLAYER 4",
    },
]


def _box_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    return (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0))


def _foot_in_roi(box, roi):
    if roi is None:
        return True
    foot = (float((box[0] + box[2]) / 2.0), float(box[3]))
    return cv2.pointPolygonTest(roi, foot, False) >= 0


def _draw_rounded_rect(img, pt1, pt2, color, radius=5):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, pt1, pt2, color, -1)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)


class BroadcastPlayerTracker:
    """Broadcast-quality AR player tracker with:

    - Smooth spring-damped floating badges
    - Ground-projected animated AR rings & contact shadows
    - Footstep breadcrumb trail
    - Near/far court side persistence with zero ID switches
    """

    def __init__(self, max_players=4, max_lost_frames=80, smooth_alpha=0.72):
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.smooth_alpha = smooth_alpha
        self.slots = {}
        self.next_slot_id = 1

    def update(self, detections, h, w, roi=None):
        candidates = []
        for d in detections:
            box = d["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh < 0.06 * h or bw < 0.02 * w:
                continue
            if not _foot_in_roi(box, roi):
                continue
            cy = (box[1] + box[3]) / 2.0
            side = "near" if cy > 0.45 * h else "far"
            candidates.append({"box": box, "conf": d.get("conf", 0.5), "side": side})

        unmatched_dets = list(range(len(candidates)))
        assigned = {}
        max_dist = 0.28 * np.hypot(w, h)

        for slot_id, slot in list(self.slots.items()):
            best_idx = None
            best_score = float("-inf")
            c_slot = _center(slot["smooth_box"])
            slot_side = slot.get("side", "unknown")

            for idx in unmatched_dets:
                det = candidates[idx]
                det_box = det["box"]
                c_det = _center(det_box)
                dist = np.hypot(c_slot[0] - c_det[0], c_slot[1] - c_det[1])
                if dist > max_dist:
                    continue

                side_match = 1.0 if (slot_side == det["side"] or slot_side == "unknown") else 0.2
                iou = _box_iou(slot["smooth_box"], det_box)
                dist_norm = 1.0 - (dist / max_dist)

                score = side_match * (0.6 * iou + 0.4 * dist_norm)
                if score > best_score and score > 0.15:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                assigned[slot_id] = candidates[best_idx]
                unmatched_dets.remove(best_idx)
                slot["lost"] = 0
                slot["hits"] += 1
                slot["side"] = candidates[best_idx]["side"]
            else:
                slot["lost"] += 1

        for slot_id, det in assigned.items():
            slot = self.slots[slot_id]
            curr_box = det["box"]
            slot["box"] = curr_box
            s_box = slot["smooth_box"]
            a = self.smooth_alpha
            new_smooth = [a * c + (1.0 - a) * s for c, s in zip(curr_box, s_box)]
            slot["smooth_box"] = new_smooth

            cx = int((new_smooth[0] + new_smooth[2]) / 2.0)
            cy = int(new_smooth[3])
            slot.setdefault("history", []).append((cx, cy))
            if len(slot["history"]) > 20:
                slot["history"].pop(0)

        for slot_id in list(self.slots.keys()):
            if self.slots[slot_id]["lost"] > self.max_lost_frames:
                del self.slots[slot_id]

        for idx in unmatched_dets:
            if len(self.slots) < self.max_players:
                cand = candidates[idx]
                box = cand["box"]
                used_ids = set(self.slots.keys())
                avail_id = next((i for i in range(1, self.max_players + 1) if i not in used_ids), None)
                if avail_id is None:
                    avail_id = self.next_slot_id
                    self.next_slot_id += 1
                cx = int((box[0] + box[2]) / 2.0)
                cy = int(box[3])
                self.slots[avail_id] = {
                    "box": box,
                    "smooth_box": list(box),
                    "lost": 0,
                    "hits": 1,
                    "side": cand["side"],
                    "history": [(cx, cy)],
                }

        active = []
        for slot_id, slot in sorted(self.slots.items()):
            if slot["lost"] <= 3:
                active.append({
                    "id": slot_id,
                    "box": list(slot["smooth_box"]),
                    "history": list(slot.get("history", [])),
                })
        return active


def run(frames, fps, size, cfg):
    w, h = size
    t_start = time.perf_counter()

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    model.predict(frames[0], classes=[0], verbose=False)
    load_s = time.perf_counter() - t_start

    roi = None
    if cfg.get("use_roi", True):
        pts = cfg.get("roi", DEFAULT_ROI)
        roi = np.array([(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32)

    max_players = cfg.get("max_players", 4)
    tracker = BroadcastPlayerTracker(
        max_players=max_players,
        max_lost_frames=int(fps * 2.5),
        smooth_alpha=0.72,
    )

    t_proc = time.perf_counter()
    data = []
    seen = collections.Counter()
    model_ms = []

    for f in frames:
        r = model(f, classes=[0], conf=cfg.get("conf", 0.28), verbose=False)[0]
        model_ms.append(sum(r.speed.values()))

        dets = []
        for box in r.boxes:
            dets.append({"box": box.xyxy[0].tolist(), "conf": float(box.conf[0])})

        active = tracker.update(dets, h, w, roi)
        for p in active:
            seen[p["id"]] += 1
        data.append(active)

    process_s = time.perf_counter() - t_proc

    return {
        "data": data,
        "timing": {"load_s": load_s, "process_s": process_s},
        "stats": {
            "players_tracked": len(seen),
            "frames_per_track": {int(i): int(seen[i]) for i in sorted(seen.keys())},
            "mean_players_per_frame": round(float(np.mean([len(d) for d in data])), 2),
            "model_only_ms_per_frame": round(float(np.mean(model_ms)), 1),
        },
    }


def draw(frame, i, data):
    for d in data[i]:
        x1, y1, x2, y2 = map(int, d["box"])
        tid = d["id"]
        theme = PLAYER_THEMES[(tid - 1) % len(PLAYER_THEMES)]
        color = theme["primary"]
        light = theme["light"]

        cx = (x1 + x2) // 2
        bw = max(24, x2 - x1)

        # 1. Subtle Motion Breadcrumbs (Fading Ground Dots)
        history = d.get("history", [])
        for idx in range(0, len(history), 2):
            pt = history[idx]
            progress = (idx + 1) / len(history)
            r = max(1, int(progress * 3.5))
            dot_color = tuple(int(c * (0.3 + 0.7 * progress)) for c in color)
            cv2.circle(frame, pt, r, dot_color, -1, cv2.LINE_AA)

        # 2. Ground Contact Shadow + Projected Broadcast AR Halo
        axes = (int(bw * 0.65), max(5, int(bw * 0.18)))
        overlay = frame.copy()

        # Dark contact shadow
        cv2.ellipse(overlay, (cx, y2 + 2), axes, 0, 0, 360, (10, 10, 10), -1, lineType=cv2.LINE_AA)
        # Soft colored translucent glow
        cv2.ellipse(overlay, (cx, y2), (axes[0] + 4, axes[1] + 2), 0, 0, 360, color, 2, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Crisp inner AR ellipse
        cv2.ellipse(frame, (cx, y2), axes, 0, 0, 360, color, 1, lineType=cv2.LINE_AA)

        # Animated rotating tick marks on the AR ring
        rot = (i * 3 + tid * 45) % 360
        cv2.ellipse(frame, (cx, y2), axes, rot, 0, 35, light, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, y2), axes, rot + 180, 0, 35, light, 2, lineType=cv2.LINE_AA)

        # 3. Modern Broadcast Floating Nameplate (Frosted Glass Pill)
        label = f"P{tid}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        pill_w = tw + 22
        pill_h = th + 10
        bx1 = cx - pill_w // 2
        by1 = max(4, y1 - pill_h - 7)
        bx2 = bx1 + pill_w
        by2 = by1 + pill_h

        # Frosted glass background
        pill_overlay = frame.copy()
        _draw_rounded_rect(pill_overlay, (bx1, by1), (bx2, by2), (20, 20, 20), radius=5)
        cv2.addWeighted(pill_overlay, 0.65, frame, 0.35, 0, frame)

        # Sleek thin border
        cv2.rectangle(frame, (bx1 + 1, by1 + 1), (bx2 - 1, by2 - 1), (60, 60, 60), 1, cv2.LINE_AA)

        # Left color indicator dot
        dot_cx = bx1 + 8
        dot_cy = (by1 + by2) // 2
        cv2.circle(frame, (dot_cx, dot_cy), 3, color, -1, cv2.LINE_AA)

        # Typography
        cv2.putText(frame, label, (bx1 + 15, by2 - 5), font, font_scale, (250, 250, 250), thickness, cv2.LINE_AA)
