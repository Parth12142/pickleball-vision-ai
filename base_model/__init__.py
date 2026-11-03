"""
base_model - Unified Base Pickleball Vision Perception Architecture.

Provides pristine, high-accuracy computer vision perception (Court calibration,
Player tracking with persistent IDs, and Ball trajectory estimation) strictly
decoupled from downstream scoring rules, foul detection, and referee logic.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class CourtGeometryResult:
    """Court spatial calibration and homography representation."""
    homography_matrix: Optional[np.ndarray] = None
    corners_pixel: List[Tuple[float, float]] = field(default_factory=list)
    court_width_ft: float = 20.0
    court_length_ft: float = 44.0
    nvz_depth_ft: float = 7.0
    calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "corners_pixel": self.corners_pixel,
            "court_width_ft": self.court_width_ft,
            "court_length_ft": self.court_length_ft,
            "nvz_depth_ft": self.nvz_depth_ft,
            "homography": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
        }


@dataclass
class PlayerTelemetry:
    """Individual player perception data."""
    id: int
    bbox_xyxy: List[float]
    confidence: float
    foot_pixel: Tuple[float, float]
    court_pos_ft: Optional[Tuple[float, float]] = None
    speed_mph: float = 0.0
    court_zone: str = "Unknown"  # "Kitchen (NVZ)", "Transition", "Baseline"
    kitchen_intrusion: bool = False
    kitchen_margin_ft: float = 999.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "confidence": round(self.confidence, 3),
            "foot_pixel": (round(self.foot_pixel[0], 1), round(self.foot_pixel[1], 1)),
            "court_pos_ft": (round(self.court_pos_ft[0], 2), round(self.court_pos_ft[1], 2)) if self.court_pos_ft else None,
            "speed_mph": round(self.speed_mph, 1),
            "court_zone": self.court_zone,
            "kitchen_intrusion": self.kitchen_intrusion,
            "kitchen_margin_ft": round(self.kitchen_margin_ft, 2),
        }


@dataclass
class BallTelemetry:
    """Small optic pickleball perception and trajectory data."""
    detected: bool
    center_pixel: Optional[Tuple[float, float]] = None
    bbox_xyxy: Optional[List[float]] = None
    confidence: float = 0.0
    court_pos_ft: Optional[Tuple[float, float]] = None
    speed_mph: float = 0.0
    interpolated: bool = False
    is_bounce_candidate: bool = False
    is_kitchen_bounce: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "center_pixel": (round(self.center_pixel[0], 1), round(self.center_pixel[1], 1)) if self.center_pixel else None,
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy] if self.bbox_xyxy else None,
            "confidence": round(self.confidence, 3),
            "court_pos_ft": (round(self.court_pos_ft[0], 2), round(self.court_pos_ft[1], 2)) if self.court_pos_ft else None,
            "speed_mph": round(self.speed_mph, 1),
            "interpolated": self.interpolated,
            "is_bounce_candidate": self.is_bounce_candidate,
            "is_kitchen_bounce": self.is_kitchen_bounce,
        }


@dataclass
class FramePerceptionResult:
    """Comprehensive perception frame telemetry (Clean output stream)."""
    frame_idx: int
    timestamp_s: float
    court: CourtGeometryResult
    players: List[PlayerTelemetry] = field(default_factory=list)
    ball: Optional[BallTelemetry] = None
    bounces_detected: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "court": self.court.to_dict(),
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict() if self.ball else None,
            "bounces_detected": self.bounces_detected,
        }


from .unified_engine import PickleballBaseModel

__all__ = [
    "CourtGeometryResult",
    "PlayerTelemetry",
    "BallTelemetry",
    "FramePerceptionResult",
    "PickleballBaseModel",
]
