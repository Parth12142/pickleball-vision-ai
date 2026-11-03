"""
player_tracker.py - Multi-Person Tracking with Court Perspective & Persistent Canonical Slots.
Pure spatial tracking: NO scoring rules or foul judgments.
"""
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from pbbase.court.homography import CourtCalibrator


def _box_iou(b1: List[float], b2: List[float]) -> float:
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


def _center(box: List[float]) -> Tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


class CanonicalPlayerTracker:
    """
    Robust Multi-Object Player Tracker with:
    - Canonical 4-Player Slots (P1, P2 near court; P3, P4 far court)
    - Hungarian optimal bipartite assignment
    - Candidate NMS deduplication (prevents torso vs full-body duplicates)
    - Motion-prediction coasting across brief player swings/occlusions
    - Ground foot projection via CourtCalibrator
    """
    def __init__(
        self,
        max_players: int = 4,
        max_lost_frames: int = 60,
        max_coast_frames: int = 10,
        smooth_alpha: float = 0.70,
    ):
        self.max_players = max_players
        self.max_lost_frames = max_lost_frames
        self.max_coast_frames = max_coast_frames
        self.smooth_alpha = smooth_alpha

        self.slots: Dict[int, Dict[str, Any]] = {}
        self.next_slot_id = 1
        self.prev_court_pos: Dict[int, Tuple[float, float, float]] = {}  # id -> (X_ft, Y_ft, timestamp)

    def reset(self):
        """Reset all active tracking slots (e.g., on camera cut)."""
        self.slots.clear()
        self.next_slot_id = 1
        self.prev_court_pos.clear()

    def update(
        self,
        raw_detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        timestamp_s: float,
        court_calibrator: CourtCalibrator,
    ) -> List[Dict[str, Any]]:
        h, w = frame_shape[:2]

        # 1. Filter out spectators and bench sitters outside court envelope
        valid_cands = []
        court_corners = court_calibrator.result.corners_px
        court_poly = np.array(court_corners, dtype=np.int32) if len(court_corners) == 4 else None

        for det in raw_detections:
            box = det["box"]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            foot_x, foot_y = det["foot_px"]

            # Aspect ratio & minimum size filter
            if bh < 0.05 * h or bw < 0.015 * w:
                continue

            # Court proximity check: allow athletic play up to 50px outside baseline/sidelines
            if court_poly is not None:
                dist = cv2.pointPolygonTest(court_poly, (foot_x, foot_y), measureDist=True)
                if dist < -55.0:
                    continue

            side = "near" if foot_y > 0.45 * h else "far"
            valid_cands.append({
                "box": box,
                "conf": det["conf"],
                "foot_px": (foot_x, foot_y),
                "side": side,
            })

        # 2. Candidate NMS Deduplication (highest confidence first)
        valid_cands.sort(key=lambda c: c["conf"], reverse=True)
        deduped = []
        for cand in valid_cands:
            b = cand["box"]
            is_dup = False
            for kept in deduped:
                if _box_iou(b, kept["box"]) > 0.35:
                    is_dup = True
                    break
                cx, cy = _center(b)
                kb = kept["box"]
                if kb[0] <= cx <= kb[2] and kb[1] <= cy <= kb[3]:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(cand)

        # 3. Hungarian Association with existing slots
        slot_ids = list(self.slots.keys())
        num_slots = len(slot_ids)
        num_cands = len(deduped)
        max_dist = 0.35 * np.hypot(w, h)

        assigned_slots = {}
        matched_cands = set()

        if num_slots > 0 and num_cands > 0:
            cost_matrix = np.full((num_slots, num_cands), 1e5, dtype=np.float32)

            for s_idx, sid in enumerate(slot_ids):
                slot = self.slots[sid]
                s_box = slot["smooth_box"]
                vel = slot.get("velocity", [0.0, 0.0])
                c_pred = (_center(s_box)[0] + vel[0], _center(s_box)[1] + vel[1])
                slot_side = slot.get("side", "unknown")

                for c_idx, cand in enumerate(deduped):
                    c_det = _center(cand["box"])
                    dist = np.hypot(c_pred[0] - c_det[0], c_pred[1] - c_det[1])
                    if dist > max_dist:
                        continue

                    iou = _box_iou(s_box, cand["box"])
                    dist_norm = dist / max_dist
                    side_factor = 1.0 if (slot_side == cand["side"] or slot_side == "unknown") else 1.35
                    cost = side_factor * (0.5 * (1.0 - iou) + 0.5 * dist_norm)
                    cost_matrix[s_idx, c_idx] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.95:
                    sid = slot_ids[r]
                    assigned_slots[sid] = deduped[c]
                    matched_cands.add(c)

        # 4. Update matched slots
        for sid, cand in assigned_slots.items():
            slot = self.slots[sid]
            curr_box = cand["box"]
            prev_center = _center(slot["smooth_box"])
            new_center = _center(curr_box)
            vx = new_center[0] - prev_center[0]
            vy = new_center[1] - prev_center[1]

            prev_vx, prev_vy = slot.get("velocity", [0.0, 0.0])
            slot["velocity"] = [0.6 * vx + 0.4 * prev_vx, 0.6 * vy + 0.4 * prev_vy]
            slot["box"] = curr_box

            a = self.smooth_alpha
            s_box = slot["smooth_box"]
            slot["smooth_box"] = [a * c + (1.0 - a) * p for c, p in zip(curr_box, s_box)]
            slot["lost"] = 0
            slot["hits"] += 1
            slot["side"] = cand["side"]

            fx = (slot["smooth_box"][0] + slot["smooth_box"][2]) / 2.0
            fy = slot["smooth_box"][3]
            slot.setdefault("history", []).append((int(fx), int(fy)))
            if len(slot["history"]) > 25:
                slot["history"].pop(0)

        # 5. Handle unmatched slots (coasting)
        for sid in slot_ids:
            if sid not in assigned_slots:
                slot = self.slots[sid]
                slot["lost"] += 1
                vel = slot.get("velocity", [0.0, 0.0])
                decay = 0.75 ** slot["lost"]
                dx, dy = vel[0] * decay, vel[1] * decay
                s_box = slot["smooth_box"]
                slot["smooth_box"] = [s_box[0] + dx, s_box[1] + dy, s_box[2] + dx, s_box[3] + dy]

        # Purge stale tracks
        for sid in list(self.slots.keys()):
            if self.slots[sid]["lost"] > self.max_lost_frames:
                del self.slots[sid]

        # 6. Allocate slots for unmatched high-confidence candidates
        unmatched_cands = [deduped[i] for i in range(num_cands) if i not in matched_cands]
        for cand in unmatched_cands:
            if len(self.slots) < self.max_players:
                used_ids = set(self.slots.keys())
                avail_id = next((i for i in range(1, self.max_players + 1) if i not in used_ids), None)
                if avail_id is None:
                    avail_id = self.next_slot_id
                    self.next_slot_id += 1

                box = cand["box"]
                fx = (box[0] + box[2]) / 2.0
                fy = box[3]
                self.slots[avail_id] = {
                    "box": box,
                    "smooth_box": list(box),
                    "lost": 0,
                    "hits": 1,
                    "side": cand["side"],
                    "velocity": [0.0, 0.0],
                    "history": [(int(fx), int(fy))],
                }

        # 7. Compute Metric Court Coordinates & Speed
        active_telemetry = []
        for sid, slot in sorted(self.slots.items()):
            if slot["lost"] <= self.max_coast_frames:
                box = [round(v, 2) for v in slot["smooth_box"]]
                fx = (box[0] + box[2]) / 2.0
                fy = box[3]
                court_ft = court_calibrator.pixel_to_court_ft(fx, fy)

                speed_mph = 0.0
                if court_ft and sid in self.prev_court_pos:
                    px_prev, py_prev, t_prev = self.prev_court_pos[sid]
                    dt = timestamp_s - t_prev
                    if 0.001 < dt < 0.5:
                        dist_ft = float(np.hypot(court_ft[0] - px_prev, court_ft[1] - py_prev))
                        speed_ft_s = dist_ft / dt
                        speed_mph = float(np.clip(speed_ft_s * 0.681818, 0.0, 22.0))

                if court_ft:
                    self.prev_court_pos[sid] = (court_ft[0], court_ft[1], timestamp_s)

                active_telemetry.append({
                    "id": sid,
                    "box": box,
                    "foot_px": (round(fx, 1), round(fy, 1)),
                    "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                    "speed_mph": round(speed_mph, 1),
                    "is_coasting": slot["lost"] > 0,
                    "side": slot["side"],
                    "history": list(slot.get("history", [])),
                })

        return active_telemetry
