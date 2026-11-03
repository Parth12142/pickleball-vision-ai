"""
detector.py - Neural Model Inference Wrappers for Players and Ball.
Provides pure spatial perception without scoring or foul rules.
"""
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO


class PlayerDetector:
    """Detects human players on court using pretrained or fine-tuned YOLO."""
    def __init__(self, model_path: str = "weights/yolo11n.pt", conf_thresh: float = 0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Player model weights not found at: {model_path}")
        self.model = YOLO(model_path)
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


class BallDetector:
    """Detects optic pickleball at high input resolution with micro-object gating."""
    def __init__(self, model_path: str = "weights/ball_colab_dukcv.pt", conf_thresh: float = 0.15, imgsz: int = 1280):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ball model weights not found at: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Dynamically determine ball class ID
        ball_ids = [k for k, v in self.model.names.items() if str(v).lower() == "ball"]
        self.ball_class_id = ball_ids[0] if ball_ids else 0

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs ball detection at native high resolution.
        Returns list of candidate dicts: {"box": [x1, y1, x2, y2], "conf": float, "center_px": (cx, cy)}
        """
        results = self.model.predict(
            frame,
            classes=[self.ball_class_id],
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        candidates = []
        for box, conf in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            # Strict physical dimension filtering: pickleball is compact (< 60px even in foreground)
            if bw > 65 or bh > 65 or bw < 2 or bh < 2:
                continue

            candidates.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "center_px": (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
            })
        return candidates
