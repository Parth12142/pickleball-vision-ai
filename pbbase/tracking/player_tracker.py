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


def _extract_appearance_feature(frame: Optional[np.ndarray], box: List[float]) -> Optional[np.ndarray]:
    """
    Extracts a 16-bin normalized Hue histogram from the torso region (15% to 55% of box height).
    Captures jersey/shirt color signature to prevent partner identity swaps during kitchen cross-overs.
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 4, min(w, x2))
    bh = max(10, y2 - y1)

    ty1 = max(0, min(h - 1, y1 + int(0.15 * bh)))
    ty2 = max(ty1 + 4, min(h, y1 + int(0.55 * bh)))

    torso = frame[ty1:ty2, x1:x2]
    if torso.size == 0 or torso.shape[0] < 4 or torso.shape[1] < 4:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _compare_appearance(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
    """Computes Bhattacharyya distance between two appearance histograms in [0.0, 1.0]."""
    if hist1 is None or hist2 is None:
        return 0.5
    try:
        dist = cv2.compareHist(
            hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
        )
        if np.isnan(dist) or np.isinf(dist):
            return 0.5
        return float(np.clip(dist, 0.0, 1.0))
    except Exception:
        return 0.5


class CanonicalPlayerTracker:
    """
    Robust Multi-Object Player Tracker with:
    - Canonical 4-Player Doubles Slots (P1, P2 near court; P3, P4 far court)
    - Metric Court Envelope Spectator Rejection (rejects bench warmers and spectators)
    - Net Partition Barrier (infinite cost across the 22ft net line)
    - Multi-Objective Hungarian matching (motion prediction + IoU + Torso HSV appearance)
    - Zero Partner ID Swaps during cross-overs and kitchen stacking
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
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        h, w = frame_shape[:2]

        # 1. Filter out spectators and bench sitters outside court athletic perimeter
        valid_cands = []
        court_calibrated = court_calibrator.result.calibrated
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

            # Court Perimeter Envelope Spectator Rejection
            if court_calibrated:
                court_ft = court_calibrator.pixel_to_court_ft(foot_x, foot_y)
                if court_ft is not None:
                    cx_ft, cy_ft = court_ft
                    # Physical pickleball athletic envelope:
                    # Court is [0, 20] x [0, 44] ft.
                    # Sideline clearance 6.0 ft, baseline clearance 8.0 ft.
                    if cx_ft < -6.0 or cx_ft > 26.0 or cy_ft < -8.0 or cy_ft > 52.0:
                        continue
            elif court_poly is not None:
                scale_px = court_calibrator.get_court_pixel_scale((h, w)) if court_calibrator is not None else 1.0
                dist = cv2.pointPolygonTest(court_poly, (foot_x, foot_y), measureDist=True)
                if dist < (-55.0 * scale_px):
                    continue

            # Net side determination
            if court_calibrated:
                court_ft = court_calibrator.pixel_to_court_ft(foot_x, foot_y)
                side = "near" if (court_ft and court_ft[1] > 22.0) else ("near" if foot_y > 0.45 * h else "far")
            else:
                side = "near" if foot_y > 0.45 * h else "far"

            # Appearance signature
            app_feat = _extract_appearance_feature(frame, box)

            valid_cands.append({
                "box": box,
                "conf": det["conf"],
                "foot_px": (foot_x, foot_y),
                "side": side,
                "appearance": app_feat,
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

        # 3. Hungarian Multi-Objective Association with existing slots
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
                slot_app = slot.get("appearance")

                for c_idx, cand in enumerate(deduped):
                    # Strict Net Partition Barrier: Player cannot jump across the net!
                    if slot_side != "unknown" and cand["side"] != "unknown" and slot_side != cand["side"]:
                        continue

                    c_det = _center(cand["box"])
                    dist = np.hypot(c_pred[0] - c_det[0], c_pred[1] - c_det[1])
                    if dist > max_dist:
                        continue

                    iou = _box_iou(s_box, cand["box"])
                    dist_norm = dist / max_dist
                    app_dist = _compare_appearance(slot_app, cand.get("appearance"))

                    # Multi-objective weighted cost function with appearance preservation
                    app_penalty = 0.35 if app_dist > 0.65 else 0.0
                    cost = 0.30 * dist_norm + 0.25 * (1.0 - iou) + 0.45 * app_dist + app_penalty
                    cost_matrix[s_idx, c_idx] = cost

            cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=1e5)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.88:
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

            # Running appearance blend
            if cand.get("appearance") is not None:
                if slot.get("appearance") is None:
                    slot["appearance"] = cand["appearance"]
                else:
                    slot["appearance"] = 0.80 * slot["appearance"] + 0.20 * cand["appearance"]

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

        # 6. Allocate canonical slots for unmatched high-confidence candidates
        unmatched_cands = [deduped[i] for i in range(num_cands) if i not in matched_cands]
        for cand in unmatched_cands:
            if len(self.slots) < self.max_players:
                used_ids = set(self.slots.keys())
                cand_side = cand.get("side", "near")
                preferred = [1, 2, 3, 4] if cand_side == "near" else [3, 4, 1, 2]
                avail_id = next((i for i in preferred if i not in used_ids and i <= self.max_players), None)
                if avail_id is None:
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
                    "appearance": cand.get("appearance"),
                    "velocity": [0.0, 0.0],
                    "history": [(int(fx), int(fy))],
                }

        # 7. Compute Metric Court Coordinates & Speed with Split-Step Damping & Court-Space EMA
        active_telemetry = []
        for sid, slot in sorted(self.slots.items()):
            if slot["lost"] <= self.max_coast_frames:
                box = [round(v, 2) for v in slot["smooth_box"]]
                fx = (box[0] + box[2]) / 2.0
                fy = box[3]
                raw_court_ft = court_calibrator.pixel_to_court_ft(fx, fy)
                court_ft = raw_court_ft

                speed_mph = 0.0
                if raw_court_ft is not None:
                    rx, ry = float(raw_court_ft[0]), float(raw_court_ft[1])
                    # Enforce physical net partition boundary on ground projection
                    if slot["side"] == "far":
                        ry = min(21.6, ry)
                    elif slot["side"] == "near":
                        ry = max(22.4, ry)
                    rx = float(np.clip(rx, -5.0, 25.0))
                    ry = float(np.clip(ry, -7.0, 51.0))

                    if sid in self.prev_court_pos:
                        px_prev, py_prev, t_prev = self.prev_court_pos[sid]
                        dt = timestamp_s - t_prev
                        step_ft = float(np.hypot(rx - px_prev, ry - py_prev))
                        if 0.001 < dt < 0.35 and step_ft < 5.0:
                            # Dampen sudden upward y2 bounding-box jumps (split-steps / lunges) in depth Y
                            alpha_x = 0.65
                            alpha_y = 0.38 if (ry < py_prev - 0.6 and abs(rx - px_prev) < 0.8) else 0.55
                            sx = alpha_x * rx + (1.0 - alpha_x) * px_prev
                            sy = alpha_y * ry + (1.0 - alpha_y) * py_prev
                        else:
                            sx, sy = rx, ry

                        if 0.001 < dt < 0.5:
                            dist_ft = float(np.hypot(sx - px_prev, sy - py_prev))
                            speed_ft_s = dist_ft / dt
                            speed_mph = float(np.clip(speed_ft_s * 0.681818, 0.0, 22.0))
                        court_ft = (sx, sy)
                    else:
                        court_ft = (rx, ry)

                    self.prev_court_pos[sid] = (court_ft[0], court_ft[1], timestamp_s)

                active_telemetry.append({
                    "id": sid,
                    "slot": f"P{sid}",
                    "box": box,
                    "foot_px": (round(fx, 1), round(fy, 1)),
                    "court_ft": (round(court_ft[0], 2), round(court_ft[1], 2)) if court_ft else None,
                    "speed_mph": round(speed_mph, 1),
                    "is_coasting": slot["lost"] > 0,
                    "side": slot["side"],
                    "history": list(slot.get("history", [])),
                })

        return active_telemetry
