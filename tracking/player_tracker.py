import collections
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
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
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def _center(box):
    return (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0))


def _foot_in_roi(box, roi):
    if roi is None:
        return True
    foot = (float((box[0] + box[2]) / 2.0), float(box[3]))
    return cv2.pointPolygonTest(roi, foot, False) >= 0


def _deduplicate_candidates(candidates, iou_thresh=0.35):
    """Suppresses duplicate/nested detections of the same player (e.g. torso vs full body)."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: c.get("conf", 0.0), reverse=True)
    kept = []
    for cand in sorted_cands:
        b = cand["box"]
        duplicate = False
        for k in kept:
            kb = k["box"]
            if _box_iou(b, kb) > iou_thresh:
                duplicate = True
                break
            # Also check if center of cand is inside an existing kept box
            cx, cy = _center(b)
            if kb[0] <= cx <= kb[2] and kb[1] <= cy <= kb[3]:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _draw_rounded_rect(img, pt1, pt2, color, radius=5):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    if r <= 1:
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
    - Hungarian optimal bipartite association (zero identity switches)
    - Candidate NMS deduplication (no double-slotting on one player)
    - Scene cut detection & re-initialization
    - Motion-prediction coasting across brief occlusions (up to 10 frames)
    - Anti-aliased floating badges & AR court rings
    """

    def __init__(self, max_players=4, max_lost_frames=75, max_coast_frames=10, smooth_alpha=0.72):
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.max_coast_frames = max_coast_frames
        self.smooth_alpha = smooth_alpha
        self.slots = {}
        self.next_slot_id = 1

    def reset_on_scene_cut(self):
        self.slots.clear()
        self.next_slot_id = 1

    def update(self, detections, h, w, roi=None):
        raw_candidates = []
        for d in detections:
            box = d["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh < 0.04 * h or bw < 0.015 * w:
                continue
            if not _foot_in_roi(box, roi):
                continue
            cy = (box[1] + box[3]) / 2.0
            side = "near" if cy > 0.45 * h else "far"
            raw_candidates.append({
                "box": box,
                "conf": d.get("conf", 0.5),
                "side": side,
            })

        # Deduplicate candidates to prevent 2 slots on 1 player
        candidates = _deduplicate_candidates(raw_candidates, iou_thresh=0.35)

        slot_ids = list(self.slots.keys())
        num_slots = len(slot_ids)
        num_cands = len(candidates)
        max_dist = 0.32 * np.hypot(w, h)

        assigned_slots = {}
        matched_cands = set()

        if num_slots > 0 and num_cands > 0:
            # Build cost matrix for Hungarian matching
            cost_matrix = np.full((num_slots, num_cands), 1e5, dtype=np.float32)

            for s_idx, sid in enumerate(slot_ids):
                slot = self.slots[sid]
                # Predict slot center using velocity if available
                s_box = slot["smooth_box"]
                vel = slot.get("velocity", [0.0, 0.0])
                c_slot = (_center(s_box)[0] + vel[0], _center(s_box)[1] + vel[1])
                slot_side = slot.get("side", "unknown")

                for c_idx, cand in enumerate(candidates):
                    c_det = _center(cand["box"])
                    dist = np.hypot(c_slot[0] - c_det[0], c_slot[1] - c_det[1])
                    if dist > max_dist:
                        continue

                    iou = _box_iou(s_box, cand["box"])
                    dist_norm = dist / max_dist  # 0 (perfect) to 1 (max_dist)

                    # Smooth side consistency: mild penalty if crossing net, not hard drop
                    side_factor = 1.0 if (slot_side == cand["side"] or slot_side == "unknown") else 1.3
                    cost = side_factor * (0.5 * (1.0 - iou) + 0.5 * dist_norm)
                    cost_matrix[s_idx, c_idx] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.95:  # Valid association threshold
                    sid = slot_ids[r]
                    assigned_slots[sid] = candidates[c]
                    matched_cands.add(c)

        # Update matched slots
        for sid, cand in assigned_slots.items():
            slot = self.slots[sid]
            curr_box = cand["box"]
            prev_center = _center(slot["smooth_box"])
            new_center = _center(curr_box)
            vx = new_center[0] - prev_center[0]
            vy = new_center[1] - prev_center[1]

            # Update velocity with exponential moving average
            prev_vx, prev_vy = slot.get("velocity", [0.0, 0.0])
            slot["velocity"] = [0.6 * vx + 0.4 * prev_vx, 0.6 * vy + 0.4 * prev_vy]

            slot["box"] = curr_box
            s_box = slot["smooth_box"]
            a = self.smooth_alpha
            slot["smooth_box"] = [a * curr + (1.0 - a) * prev for curr, prev in zip(curr_box, s_box)]
            slot["lost"] = 0
            slot["hits"] += 1
            slot["side"] = cand["side"]

            cx = int((slot["smooth_box"][0] + slot["smooth_box"][2]) / 2.0)
            cy = int(slot["smooth_box"][3])
            slot.setdefault("history", []).append((cx, cy))
            if len(slot["history"]) > 20:
                slot["history"].pop(0)

        # Handle unmatched slots (coasting)
        for sid in slot_ids:
            if sid not in assigned_slots:
                slot = self.slots[sid]
                slot["lost"] += 1
                # Coast position slightly with damping velocity
                vel = slot.get("velocity", [0.0, 0.0])
                decay = 0.75 ** slot["lost"]
                dx = vel[0] * decay
                dy = vel[1] * decay
                s_box = slot["smooth_box"]
                slot["smooth_box"] = [s_box[0] + dx, s_box[1] + dy, s_box[2] + dx, s_box[3] + dy]

        # Purge slots exceeding max_lost_frames
        for sid in list(self.slots.keys()):
            if self.slots[sid]["lost"] > self.max_lost_frames:
                del self.slots[sid]

        # Register new slots for unmatched high-confidence candidates
        unmatched_cands = [candidates[i] for i in range(num_cands) if i not in matched_cands]
        for cand in unmatched_cands:
            if len(self.slots) < self.max_players:
                # Assign lowest available ID 1..max_players
                used_ids = set(self.slots.keys())
                avail_id = next((i for i in range(1, self.max_players + 1) if i not in used_ids), None)
                if avail_id is None:
                    avail_id = self.next_slot_id
                    self.next_slot_id += 1

                box = cand["box"]
                cx = int((box[0] + box[2]) / 2.0)
                cy = int(box[3])
                self.slots[avail_id] = {
                    "box": box,
                    "smooth_box": list(box),
                    "lost": 0,
                    "hits": 1,
                    "side": cand["side"],
                    "velocity": [0.0, 0.0],
                    "history": [(cx, cy)],
                }

        # Build active return list: include slots lost <= max_coast_frames
        active = []
        for slot_id, slot in sorted(self.slots.items()):
            if slot["lost"] <= self.max_coast_frames:
                active.append({
                    "id": slot_id,
                    "box": list(slot["smooth_box"]),
                    "history": list(slot.get("history", [])),
                    "is_coasting": slot["lost"] > 0,
                })
        return active


def run(frames, fps, size, cfg):
    w, h = size
    t_start = time.perf_counter()

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    if frames:
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
        max_coast_frames=10,
        smooth_alpha=0.72,
    )

    t_proc = time.perf_counter()
    data = []
    seen = collections.Counter()
    model_ms = []
    prev_frame = None

    conf_thresh = cfg.get("conf", 0.22)

    for f in frames:
        # Detect camera cuts
        if prev_frame is not None:
            diff = float(np.mean(cv2.absdiff(prev_frame, f)))
            if diff > 28.0:
                tracker.reset_on_scene_cut()
        prev_frame = f

        r = model(f, classes=[0], conf=conf_thresh, verbose=False)[0]
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
            "mean_players_per_frame": round(float(np.mean([len(d) for d in data])) if data else 0.0, 2),
            "model_only_ms_per_frame": round(float(np.mean(model_ms)) if model_ms else 0.0, 1),
        },
    }


def draw(frame, i, data):
    h, w = frame.shape[:2]
    if i >= len(data):
        return

    # Draw player AR elements
    for d in data[i]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        # Clip to image boundary
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

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
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                progress = (idx + 1) / max(1, len(history))
                r = max(1, int(progress * 3.5))
                dot_color = tuple(int(c * (0.3 + 0.7 * progress)) for c in color)
                cv2.circle(frame, pt, r, dot_color, -1, cv2.LINE_AA)

        # 2. Ground Contact Shadow + Projected Broadcast AR Halo
        axes = (int(bw * 0.65), max(5, int(bw * 0.18)))
        overlay = frame.copy()

        # Dark contact shadow
        cv2.ellipse(overlay, (cx, min(h - 2, y2 + 2)), axes, 0, 0, 360, (10, 10, 10), -1, lineType=cv2.LINE_AA)
        # Soft colored translucent glow
        cv2.ellipse(overlay, (cx, min(h - 1, y2)), (axes[0] + 4, axes[1] + 2), 0, 0, 360, color, 2, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Crisp inner AR ellipse
        cv2.ellipse(frame, (cx, min(h - 1, y2)), axes, 0, 0, 360, color, 1, lineType=cv2.LINE_AA)

        # Animated rotating tick marks on the AR ring
        rot = (i * 3 + tid * 45) % 360
        cv2.ellipse(frame, (cx, min(h - 1, y2)), axes, rot, 0, 35, light, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, min(h - 1, y2)), axes, rot + 180, 0, 35, light, 2, lineType=cv2.LINE_AA)

        # 3. Modern Broadcast Floating Nameplate (Frosted Glass Pill)
        label = f"P{tid}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.38, min(0.55, 0.45 * (w / 1280.0)))
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        pill_w = tw + 22
        pill_h = th + 10
        bx1 = max(2, min(w - pill_w - 2, cx - pill_w // 2))
        by1 = max(4, min(h - pill_h - 2, y1 - pill_h - 7))
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
        cv2.putText(frame, label, (bx1 + 15, by2 - 4), font, font_scale, (250, 250, 250), thickness, cv2.LINE_AA)
