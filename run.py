"""
run.py - runs every module that is switched on in config.yaml, times each one,
draws the results on one video and saves the timing to files.

    python run.py                                   # uses config.yaml
    python run.py --clip data/clips/match1.mp4      # same config, different clip
    python run.py --config other_config.yaml
    python run.py --clip data/clips/match1.mp4 --no-roi

Saved every run:
    results/runs/<clip>_<date-time>.json   full details of this run
    results/timing_log.csv                 one row per module per run (history)
    outputs/<clip>_<modules>.mp4           annotated video
"""
import argparse
import csv
import datetime
import json
import os
import time

import cv2
import numpy as np
import yaml

from video_io import read_video, write_video

# ---- module registry -------------------------------------------------------
from tracking import player_tracker
from detection import ball_detector
import court as court_module
import events as bounce_module

MODULES = {
    "players": player_tracker,
    "ball": ball_detector,
    "court": court_module,
    "bounces": bounce_module,
}
# -----------------------------------------------------------------------------


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return f"CPU: {cpu_name()}"


def draw_hud(frame, i, total_frames, fps, active, results):
    h, w = frame.shape[:2]

    # Responsive scaling for broadcast badge based on resolution
    scale = max(0.80, min(1.6, w / 1280.0))
    card_w = int(290 * scale)
    card_h = int(62 * scale)
    cx1 = int(18 * scale)
    cy1 = int(16 * scale)
    cx2 = cx1 + card_w
    cy2 = cy1 + card_h

    # Frosted glass background card
    overlay = frame.copy()
    r = int(7 * scale)
    cv2.rectangle(overlay, (cx1 + r, cy1), (cx2 - r, cy2), (18, 18, 18), -1)
    cv2.rectangle(overlay, (cx1, cy1 + r), (cx2, cy2 - r), (18, 18, 18), -1)
    cv2.circle(overlay, (cx1 + r, cy1 + r), r, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx2 - r, cy1 + r), r, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx1 + r, cy2 - r), r, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx2 - r, cy2 - r), r, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Sleek outline border
    cv2.rectangle(frame, (cx1 + r, cy1), (cx2 - r, cy2), (65, 65, 65), 1, cv2.LINE_AA)

    # Pulsing Red Broadcast Pip
    pulse = 0.5 + 0.5 * np.sin(i * 0.2)
    pip_color = (0, int(40 * pulse), int(220 + 35 * pulse))
    pip_x = cx1 + int(14 * scale)
    pip_y = cy1 + int(15 * scale)
    cv2.circle(frame, (pip_x, pip_y), max(3, int(4 * scale)), pip_color, -1, cv2.LINE_AA)

    # Badge Header
    font = cv2.FONT_HERSHEY_DUPLEX
    header_scale = 0.38 * scale
    cv2.putText(
        frame,
        "COURT-VISION AR",
        (cx1 + int(24 * scale), cy1 + int(18 * scale)),
        font,
        header_scale,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # Match timecode and frame counter
    sec_total = i / max(1.0, fps)
    mins = int(sec_total // 60)
    secs = int(sec_total % 60)
    millis = int((sec_total - int(sec_total)) * 100)
    timecode_str = f"{mins:02d}:{secs:02d}.{millis:02d}   [F:{i + 1}/{total_frames}]"
    body_scale = 0.34 * scale
    cv2.putText(
        frame,
        timecode_str,
        (cx1 + int(14 * scale), cy1 + int(34 * scale)),
        font,
        body_scale,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    # Active Players Count & Ball Tracking Status
    players_count = 0
    if "players" in results and i < len(results["players"]["data"]):
        players_count = len(results["players"]["data"][i])

    ball_status = "SEARCHING"
    speed_mph = 0
    ball_color = (130, 130, 130)

    if "ball" in results and i < len(results["ball"]["data"]["points"]):
        b_pt = results["ball"]["data"]["points"][i]
        if b_pt is not None:
            px_per_frame = b_pt.get("speed", 0.0)
            # Approximate mph: pixels/frame conversion estimate
            speed_mph = int(px_per_frame * 1.6)
            ball_status = f"{speed_mph} MPH"
            # Color code ball status
            if px_per_frame >= 22.0:
                ball_color = (20, 140, 255)  # Orange for smash
            elif px_per_frame >= 10.0:
                ball_color = (30, 240, 200)  # Chartreuse
            else:
                ball_color = (255, 230, 50)  # Ice Cyan

    stat_line = f"PLAYERS: {players_count}/4   |   BALL: {ball_status}"
    cv2.putText(
        frame,
        stat_line,
        (cx1 + int(14 * scale), cy1 + int(51 * scale)),
        font,
        body_scale,
        ball_color if ball_status != "SEARCHING" else (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clip", default=None, help="overrides 'clip' in the config")
    ap.add_argument("--no-roi", action="store_true", help="disables court ROI filtering")
    ap.add_argument("--max-frames", type=int, default=None, help="maximum frames to read and process")
    ap.add_argument("--conf", type=float, default=None, help="override player detection confidence")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    clip = a.clip or cfg.get("clip", "data/clips/match1.mp4")
    out_dir = cfg.get("output_dir", "outputs")
    res_dir = cfg.get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{res_dir}/runs", exist_ok=True)

    # Apply CLI overrides to module config
    mod_cfg = cfg.get("modules", {})
    if a.no_roi and "players" in mod_cfg:
        mod_cfg["players"]["use_roi"] = False
    if a.conf is not None and "players" in mod_cfg:
        mod_cfg["players"]["conf"] = a.conf

    max_frames = a.max_frames or cfg.get("max_frames", None)

    active = []
    for name, mcfg in mod_cfg.items():
        if not (mcfg or {}).get("enabled", False):
            print(f"[ off ] {name}")
            continue
        if name not in MODULES:
            print(f"[ skip] {name}: enabled in config but not implemented yet")
            continue
        print(f"[ ON  ] {name}")
        active.append(name)

    # read video
    t = time.perf_counter()
    frames, fps, (w, h) = read_video(clip, max_frames=max_frames)
    read_s = time.perf_counter() - t
    if not frames:
        raise SystemExit(f"No frames read from {clip}. Check the path and that the file exists.")
    clip_s = len(frames) / fps
    print(f"\nclip: {clip}  {len(frames)} frames, {fps:.1f} FPS, {w}x{h}, {clip_s:.2f} s\n")

    # run every enabled module and time it
    results, module_rows = {}, []
    wall_start = time.perf_counter()
    for name in active:
        print(f"running {name} ...")
        t = time.perf_counter()
        res = MODULES[name].run(frames, fps, (w, h), mod_cfg[name])
        wall_s = time.perf_counter() - t
        p = res["timing"]["process_s"]
        results[name] = res
        module_rows.append({
            "module": name,
            "load_s": round(res["timing"]["load_s"], 2),
            "process_s": round(p, 2),
            "ms_per_frame": round(1000 * p / len(frames), 1),
            "fps": round(len(frames) / p, 1),
            "realtime_factor": round(p / clip_s, 2),
            "wall_s": round(wall_s, 2),
            "stats": res["stats"],
        })
    compute_s = time.perf_counter() - wall_start

    # draw + write video
    render_s, out_path = 0.0, None
    if cfg.get("render", {}).get("enabled", True) and active:
        print("rendering overlays and HUD ...")
        t = time.perf_counter()
        for i, f in enumerate(frames):
            for name in active:
                MODULES[name].draw(f, i, results[name]["data"])
            draw_hud(f, i, len(frames), fps, active, results)
        stem = os.path.splitext(os.path.basename(clip))[0]
        out_path = f"{out_dir}/{stem}_{'-'.join(active)}.mp4"
        write_video(frames, out_path, fps, (w, h))
        render_s = time.perf_counter() - t

    # save results
    now = datetime.datetime.now()
    stem = os.path.splitext(os.path.basename(clip))[0]
    summary = {
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "clip": clip,
        "frames": len(frames),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "clip_seconds": round(clip_s, 2),
        "device": device_name(),
        "modules_on": active,
        "modules_off": [n for n in mod_cfg if n not in active],
        "read_s": round(read_s, 2),
        "compute_s": round(compute_s, 2),
        "render_s": round(render_s, 2),
        "total_s": round(read_s + compute_s + render_s, 2),
        "modules": module_rows,
        "output_video": out_path,
        "config_used": cfg,
    }
    json_path = f"{res_dir}/runs/{stem}_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    log = f"{res_dir}/timing_log.csv"
    new = not os.path.exists(log)
    with open(log, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["date", "clip", "clip_seconds", "frames", "device", "module",
                         "process_s", "ms_per_frame", "fps", "realtime_factor",
                         "read_s", "render_s", "total_s"])
        for m in module_rows:
            wr.writerow([summary["date"], clip, summary["clip_seconds"], len(frames),
                         summary["device"], m["module"], m["process_s"], m["ms_per_frame"],
                         m["fps"], m["realtime_factor"], summary["read_s"],
                         summary["render_s"], summary["total_s"]])

    # print the report
    print("\n================ RESULT ================")
    print(f"clip length : {clip_s:.2f} s ({len(frames)} frames @ {fps:.1f} FPS)   device: {summary['device']}")
    for m in module_rows:
        print(f"\n[{m['module']}]  process time {m['process_s']} s  |  {m['ms_per_frame']} ms/frame  |  "
              f"{m['fps']} FPS  |  real-time factor {m['realtime_factor']}x")
        for k, v in m["stats"].items():
            print(f"    {k}: {v}")
    print(f"\nread {read_s:.2f} s | all modules {compute_s:.2f} s | draw+write video {render_s:.2f} s "
          f"| total {summary['total_s']} s")
    print("real-time factor: 1.0x = takes as long as the clip, below 1.0x = faster than real time")
    if out_path:
        print(f"video  : {out_path}")
    print(f"saved  : {json_path}")
    print(f"history: {log}")


if __name__ == "__main__":
    main()
