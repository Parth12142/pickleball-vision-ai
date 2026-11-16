"""
detector.py - Neural Model Inference Wrappers for Players and Ball.
Provides pure spatial perception without scoring or foul rules.
"""
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class PlayerDetector:
    """Detects human players on court using pretrained or fine-tuned YOLO26."""
    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = 0.25):
        candidates = [
            model_path,
            os.path.join(REPO_ROOT, model_path) if model_path and not os.path.isabs(model_path) else None,
            "weights/player_yolo26_best.pt",
            os.path.join(REPO_ROOT, "weights/player_yolo26_best.pt"),
            "weights/yolo26n.pt",
            os.path.join(REPO_ROOT, "weights/yolo26n.pt"),
        ]
        resolved_path = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                resolved_path = cand
                break

        if not resolved_path:
            raise FileNotFoundError(f"Player model weights not found at: {model_path or 'weights/player_yolo26_best.pt'}")

        self.model_path = resolved_path
        self.model = YOLO(resolved_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray, imgsz: int = 640) -> List[Dict[str, Any]]:
        """
        Runs person detection (class 0).
        Returns list of dicts: {"box": [x1, y1, x2, y2], "conf": float, "foot_px": (cx, y2)}
        """
        results = self.model.predict(
            frame,
            classes=[0],  # class 0 = person in COCO / player
            conf=self.conf_thresh,
            imgsz=imgsz,
            verbose=False,
        )[0]

        detections = []
        for box, conf in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            detections.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "foot_px": (float((x1 + x2) / 2.0), float(y2)),
            })
        return detections


def _box_iou_px(box_a: List[float], box_b: List[float]) -> float:
    """Computes standard Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_ball_candidates(
    primary_cands: List[Dict[str, Any]],
    patch_cands: List[Dict[str, Any]],
    iou_thresh: float = 0.25,
    dist_thresh_px: float = 24.0,
) -> List[Dict[str, Any]]:
    """
    Fuses candidates detected from full frame and cropped patches.
    Corroborated candidates receive confidence boost and confidence-weighted bounding boxes.
    """
    if not patch_cands:
        return primary_cands
    if not primary_cands:
        return patch_cands

    merged: List[Dict[str, Any]] = []
    matched_patch_indices = set()

    for p in primary_cands:
        best_match_idx = None
        best_match_score = 0.0

        for idx, pc in enumerate(patch_cands):
            if idx in matched_patch_indices:
                continue
            iou = _box_iou_px(p["box"], pc["box"])
            dx = p["center_px"][0] - pc["center_px"][0]
            dy = p["center_px"][1] - pc["center_px"][1]
            dist = (dx * dx + dy * dy) ** 0.5

            if iou >= iou_thresh or dist <= dist_thresh_px:
                score = iou + max(0.0, 1.0 - dist / max(1.0, dist_thresh_px))
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = idx

        if best_match_idx is not None:
            matched_patch_indices.add(best_match_idx)
            pc = patch_cands[best_match_idx]
            # Weighted average coordinates based on confidence
            w1 = p["conf"]
            w2 = pc["conf"]
            w_total = max(1e-6, w1 + w2)
            box = [
                (p["box"][0] * w1 + pc["box"][0] * w2) / w_total,
                (p["box"][1] * w1 + pc["box"][1] * w2) / w_total,
                (p["box"][2] * w1 + pc["box"][2] * w2) / w_total,
                (p["box"][3] * w1 + pc["box"][3] * w2) / w_total,
            ]
            center = (
                (p["center_px"][0] * w1 + pc["center_px"][0] * w2) / w_total,
                (p["center_px"][1] * w1 + pc["center_px"][1] * w2) / w_total,
            )
            # Boost confidence for multi-scale consensus (up to 0.99)
            conf = min(0.99, max(w1, w2) * 1.08)
            merged.append({
                "box": box,
                "conf": round(conf, 4),
                "center_px": center,
                "consensus": True,
            })
        else:
            merged.append(p)

    # Add remaining unmatched patch candidates
    for idx, pc in enumerate(patch_cands):
        if idx not in matched_patch_indices:
            merged.append(pc)

    # Sort descending by confidence
    merged.sort(key=lambda c: c["conf"], reverse=True)
    return merged


class BallDetector:
    """
    Detects optic pickleball at high input resolution with micro-object gating.
    Supports multi-scale inference and cropped patch sliding for far-court micro-balls.
    """
    DEFAULT_FAR_COURT_SLICES = [
        (0.12, 0.08, 0.65, 0.92)  # Relative (ymin, xmin, ymax, xmax) far-court zone
    ]

    def __init__(
        self,
        model_path: str = "weights/ball_colab_dukcv.pt",
        conf_thresh: float = 0.15,
        imgsz: int = 1280,
        multiscale: bool = False,
        patch_slices: Optional[List[Tuple[float, float, float, float]]] = None,
        patch_conf_thresh: float = 0.12,
        patch_imgsz: int = 1280,
        iou_thresh: float = 0.25,
        dist_thresh_px: float = 24.0,
    ):
        candidates = [
            model_path,
            os.path.join(REPO_ROOT, model_path) if model_path and not os.path.isabs(model_path) else None,
            "weights/ball_yolo26_best.pt",
            os.path.join(REPO_ROOT, "weights/ball_yolo26_best.pt"),
            "weights/ball_colab_dukcv.pt",
            os.path.join(REPO_ROOT, "weights/ball_colab_dukcv.pt"),
        ]
        resolved_path = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                resolved_path = cand
                break

        if not resolved_path:
            raise FileNotFoundError(f"Ball model weights not found at: {model_path}")
        self.model = YOLO(resolved_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.multiscale = multiscale
        self.patch_slices = patch_slices if patch_slices is not None else list(self.DEFAULT_FAR_COURT_SLICES)
        self.patch_conf_thresh = patch_conf_thresh
        self.patch_imgsz = patch_imgsz
        self.iou_thresh = iou_thresh
        self.dist_thresh_px = dist_thresh_px

        # Dynamically determine ball class ID
        ball_ids = [k for k, v in self.model.names.items() if str(v).lower() == "ball"]
        self.ball_class_id = ball_ids[0] if ball_ids else 0

    def _predict_frame(self, img: np.ndarray, imgsz: int, conf_thresh: float) -> List[Dict[str, Any]]:
        """Internal runner for single frame/patch prediction with physical dimension filtering."""
        results = self.model.predict(
            img,
            classes=[self.ball_class_id],
            conf=conf_thresh,
            imgsz=imgsz,
            verbose=False,
        )[0]

        candidates = []
        for box, conf in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            # Strict physical dimension filtering: pickleball is compact (< 65px even in foreground)
            if bw > 65 or bh > 65 or bw < 2 or bh < 2:
                continue

            candidates.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "center_px": (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
            })
        return candidates

    def detect(self, frame: np.ndarray, multiscale: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Runs ball detection with optional multi-scale / cropped-patch sliding inference.
        Returns list of candidate dicts: {"box": [x1, y1, x2, y2], "conf": float, "center_px": (cx, cy)}
        """
        use_multiscale = self.multiscale if multiscale is None else multiscale
        full_cands = self._predict_frame(frame, imgsz=self.imgsz, conf_thresh=self.conf_thresh)

        if not use_multiscale or not self.patch_slices:
            return full_cands

        h, w = frame.shape[:2]
        patch_cands = []

        for (ymin, xmin, ymax, xmax) in self.patch_slices:
            py1 = max(0, min(h - 1, int(round(ymin * h))))
            py2 = max(py1 + 10, min(h, int(round(ymax * h))))
            px1 = max(0, min(w - 1, int(round(xmin * w))))
            px2 = max(px1 + 10, min(w, int(round(xmax * w))))

            patch_crop = frame[py1:py2, px1:px2]
            if patch_crop.size == 0 or patch_crop.shape[0] < 10 or patch_crop.shape[1] < 10:
                continue

            raw_patch = self._predict_frame(
                patch_crop,
                imgsz=self.patch_imgsz,
                conf_thresh=min(self.patch_conf_thresh, self.conf_thresh),
            )
            for c in raw_patch:
                bx1, by1, bx2, by2 = c["box"]
                gx1 = bx1 + px1
                gy1 = by1 + py1
                gx2 = bx2 + px1
                gy2 = by2 + py1
                patch_cands.append({
                    "box": [float(gx1), float(gy1), float(gx2), float(gy2)],
                    "conf": float(c["conf"]),
                    "center_px": (float((gx1 + gx2) / 2.0), float((gy1 + gy2) / 2.0)),
                    "patch_origin": True,
                })

        return _merge_ball_candidates(
            full_cands,
            patch_cands,
            iou_thresh=self.iou_thresh,
            dist_thresh_px=self.dist_thresh_px,
        )
