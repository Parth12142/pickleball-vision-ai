"""
court_reference.py - Standard regulation pickleball court reference dimensions and geometry.

Dimensions (USA Pickleball official):
- Court dimensions: 20 ft wide x 44 ft long (6.10 m x 13.41 m)
- Non-Volley Zone (Kitchen): 7 ft from net on each side (2.13 m)
- Service courts: 15 ft deep x 10 ft wide (4.57 m x 3.05 m)
- Net height: 36 in at sidelines, 34 in at center
"""
import numpy as np


class CourtGeometry:
    WIDTH_FT = 20.0
    LENGTH_FT = 44.0
    NVZ_DEPTH_FT = 7.0

    @classmethod
    def get_canonical_court_corners(cls):
        """Returns canonical 2D top-down coordinates in meters."""
        return np.array([
            [0.0, 0.0],
            [cls.WIDTH_FT * 0.3048, 0.0],
            [cls.WIDTH_FT * 0.3048, cls.LENGTH_FT * 0.3048],
            [0.0, cls.LENGTH_FT * 0.3048],
        ], dtype=np.float32)


def run(frames, fps, size, cfg):
    """Placeholder module runner for court tracking."""
    return {
        "data": {},
        "timing": {"load_s": 0.0, "process_s": 0.0},
        "stats": {"court_detected": False},
    }


def draw(frame, i, data):
    """Placeholder module drawer for court overlay."""
    pass
