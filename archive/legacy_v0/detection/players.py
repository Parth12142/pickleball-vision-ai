import argparse
import collections
import json
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from ultralytics import YOLO
from video_io import read_video, write_video
from render.overlay import draw_player

DEFAULT_CLIP = "data/clips/match1.mp4"
CACHE_PATH = "cache/players.json"

# Normalized ROI coordinates: (x_frac, y_frac)
DEFAULT_ROI_FRAC = [(0.120, 0.120), (0.880, 0.120), (0.980, 0.995), (0.020, 0.995)]


def detect_players(frames, conf=0.28):
    model = YOLO("yolo11n.pt")
    cache = []
    for frame in frames:
        r = model.track(
            frame,
            persist=True,
            classes=[0],  # class 0 = person
            conf=conf,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        dets = []
        if r.boxes.id is not None:
            for box, tid, c in zip(
                r.boxes.xyxy.tolist(),
                r.boxes.id.tolist(),
                r.boxes.conf.tolist(),
            ):
                dets.append({"id": int(tid), "box": box, "conf": float(c)})
        cache.append(dets)
    return cache


def get_roi_polygon(w, h, frac_pts=DEFAULT_ROI_FRAC):
    return np.array([(int(x * w), int(y * h)) for x, y in frac_pts], dtype=np.int32)


def in_roi(box, roi_poly):
    if roi_poly is None:
        return True
    x1, y1, x2, y2 = box
    foot = ((x1 + x2) / 2.0, float(y2))
    return cv2.pointPolygonTest(roi_poly, foot, False) >= 0


def keep_real_tracks(cache, roi_poly, min_frames=20):
    counts = collections.Counter(d["id"] for dets in cache for d in dets if in_roi(d["box"], roi_poly))
    keep = {i for i, n in counts.items() if n >= min_frames}
    return [[d for d in dets if d["id"] in keep and in_roi(d["box"], roi_poly)] for dets in cache]


def draw(frames, cache, roi_poly=None):
    out = []
    for frame, dets in zip(frames, cache):
        f = frame.copy()
        for d in dets:
            if roi_poly is None or in_roi(d["box"], roi_poly):
                draw_player(f, d["box"], d["id"])
        out.append(f)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone ByteTrack player detection & cache generator")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="input video clip path")
    parser.add_argument("--max-frames", type=int, default=30, help="max frames to process")
    parser.add_argument("--no-cache", action="store_true", help="force re-detection")
    args = parser.parse_args()

    clip_path = args.clip if os.path.exists(args.clip) else "data/clips/rally_sample.mp4"
    frames, fps, (w, h) = read_video(clip_path, max_frames=args.max_frames)
    roi_poly = get_roi_polygon(w, h)

    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if os.path.exists(CACHE_PATH) and not args.no_cache:
        print(f"Loading cached detections from {CACHE_PATH} ...")
        with open(CACHE_PATH) as fh:
            cache = json.load(fh)
    else:
        print(f"Running YOLO detection on {len(frames)} frames ...")
        cache = detect_players(frames)
        with open(CACHE_PATH, "w") as fh:
            json.dump(cache, fh, indent=2)
        print(f"Saved cache to {CACHE_PATH}")

    # Draw preview
    if frames and cache:
        preview = draw(frames[:1], cache[:1], roi_poly)[0]
        cv2.imwrite("outputs/roi_check.png", preview)
        write_video(draw(frames, keep_real_tracks(cache, roi_poly)), "outputs/players.mp4", fps, (w, h))
        print("Saved outputs/players.mp4 and outputs/roi_check.png")
