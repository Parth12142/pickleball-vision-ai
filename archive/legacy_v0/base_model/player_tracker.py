"""
player_tracker.py - Multi-Person Detection & Tracking with Court Projection.
"""
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from . import PlayerTelemetry
from .court_detector import CourtDetector


class PlayerTracker:
    """
    High-accuracy player detection, persistent ID tracking (ByteTrack),
    and foot-to-court metric coordinate projection.
    """
    # Normalized court ROI bounding polygon (rejects spectators & bench)
    DEFAULT_ROI_FRAC = [(0.10, 0.10), (0.90, 0.10), (0.98, 0.99), (0.02, 0.99)]

    def __init__(self, model_path: str = "yolo11n.pt", conf_thresh: float = 0.28):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.prev_positions: Dict[int, Tuple[float, float, float]] = {}  # id -> (X_ft, Y_ft, timestamp)
        self.stable_id_map: Dict[int, int] = {}
        self.next_canonical_id = 1

    def _is_valid_player_candidate(
        self,
        box: List[float],
        conf: float,
        court_detector: CourtDetector,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Rejects spectators, bench sitters, gallery crowd, and background people.
        Only players within the active court playing envelope are admitted.
        """
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        # 1. Height & aspect ratio filter (reject tiny spectators and noise)
        if bh < 75 or bw < 25:
            return False

        # 2. Reject far background gallery (people sitting/standing behind far court)
        if foot_y < 155.0:
            return False

        # 3. Reject side benches / stands (spectators on far left or far right)
        if foot_x < 170.0 or foot_x > 1115.0:
            return False

        # 4. Court Boundary Distance Filter
        if court_detector.corners_px and len(court_detector.corners_px) == 4:
            court_poly = np.array(court_detector.corners_px, dtype=np.int32)
            dist = cv2.pointPolygonTest(court_poly, (foot_x, foot_y), measureDist=True)
            # Allow up to 45px outside lines for athletic wide returns, but reject crowd
            if dist < -45.0:
                return False

        return True

    def track_frame(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        court_detector: CourtDetector,
    ) -> List[PlayerTelemetry]:
        h, w = frame.shape[:2]
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],  # 0 = person / Player
            conf=self.conf_thresh,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        if results.boxes.id is None:
            return []

        boxes = results.boxes.xyxy.tolist()
        track_ids = results.boxes.id.tolist()
        confs = results.boxes.conf.tolist()

        # Step 1: Filter raw detections to genuine court players (reject spectators)
        valid_candidates = []
        for box, raw_tid, conf in zip(boxes, track_ids, confs):
            if not self._is_valid_player_candidate(box, conf, court_detector, w, h):
                continue
            raw_id = int(raw_tid)
            valid_candidates.append({
                "box": box,
                "raw_id": raw_id,
                "conf": conf,
                "foot_px": ((box[0] + box[2]) / 2.0, float(box[3])),
                "area": (box[2] - box[0]) * (box[3] - box[1]),
            })

        # Step 2: Doubles Partitioning — Select top 2 Near and top 2 Far players
        net_y = 302.0  # Physical net horizontal line
        near_candidates = [c for c in valid_candidates if c["foot_px"][1] >= net_y]
        far_candidates = [c for c in valid_candidates if c["foot_px"][1] < net_y]

        # Sort by confidence * area score to select the 2 active tournament players on each side
        near_candidates.sort(key=lambda c: c["conf"] * (c["area"] ** 0.5), reverse=True)
        far_candidates.sort(key=lambda c: c["conf"] * (c["area"] ** 0.5), reverse=True)

        selected_near = near_candidates[:2]
        selected_far = far_candidates[:2]

        # Order left-to-right to maintain stable P1/P2 (Near) and P3/P4 (Far) identity
        selected_near.sort(key=lambda c: c["foot_px"][0])
        selected_far.sort(key=lambda c: c["foot_px"][0])

        active_players = []
        for idx, c in enumerate(selected_near, start=1):
            c["assigned_id"] = idx
            active_players.append(c)
        for idx, c in enumerate(selected_far, start=3):
            c["assigned_id"] = idx
            active_players.append(c)

        # Step 3: Compute telemetry for the 4 active tournament players
        telemetry_list: List[PlayerTelemetry] = []
        for c in active_players:
            canonical_id = c["assigned_id"]
            box = c["box"]
            foot_px = c["foot_px"]
            conf = c["conf"]

            court_pos = court_detector.pixel_to_court(foot_px[0], foot_px[1])
            speed_mph = 0.0
            court_zone = "Unknown"

            kitchen_intrusion = False
            kitchen_margin_ft = 999.0
            if court_pos:
                court_zone = court_detector.classify_court_zone(court_pos[0], court_pos[1])
                is_intrusion, margin_px, _ = court_detector.check_foot_kitchen_intrusion(foot_px, court_pos)
                kitchen_intrusion = is_intrusion
                kitchen_margin_ft = margin_px / 30.0  # approximate scale to ft

                # Compute instantaneous player velocity
                if canonical_id in self.prev_positions:
                    px_prev, py_prev, t_prev = self.prev_positions[canonical_id]
                    dt = timestamp_s - t_prev
                    if 0.001 < dt < 0.5:
                        dist_ft = float(np.hypot(court_pos[0] - px_prev, court_pos[1] - py_prev))
                        speed_ft_s = dist_ft / dt
                        speed_mph = float(np.clip(speed_ft_s * 0.681818, 0.0, 22.0))
                self.prev_positions[canonical_id] = (court_pos[0], court_pos[1], timestamp_s)

            telemetry_list.append(PlayerTelemetry(
                id=canonical_id,
                bbox_xyxy=box,
                confidence=float(conf),
                foot_pixel=foot_px,
                court_pos_ft=court_pos,
                speed_mph=speed_mph,
                court_zone=court_zone,
                kitchen_intrusion=kitchen_intrusion,
                kitchen_margin_ft=kitchen_margin_ft,
            ))

        return sorted(telemetry_list, key=lambda p: p.id)
