"""
bounce_detector.py - Ball bounce event detection via vertical trajectory inflection.
"""
import numpy as np


def detect_bounces(ball_points, min_vertical_inflection=3.0):
    """Detects bounce events where dy changes sign from downward to upward."""
    bounces = []
    for i in range(1, len(ball_points) - 1):
        p_prev = ball_points[i - 1]
        p_curr = ball_points[i]
        p_next = ball_points[i + 1]

        if p_prev is None or p_curr is None or p_next is None:
            continue

        dy1 = p_curr["y"] - p_prev["y"]
        dy2 = p_next["y"] - p_curr["y"]

        # Downward trajectory followed by upward bounce
        if dy1 > min_vertical_inflection and dy2 < -min_vertical_inflection:
            bounces.append({
                "frame": i,
                "x": p_curr["x"],
                "y": p_curr["y"],
                "speed": p_curr.get("speed", 0.0),
            })
    return bounces


def run(frames, fps, size, cfg):
    return {
        "data": {"bounces": []},
        "timing": {"load_s": 0.0, "process_s": 0.0},
        "stats": {"bounces_detected": 0},
    }


def draw(frame, i, data):
    pass
