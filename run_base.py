#!/usr/bin/env python3
"""
run_base.py - CLI Runner for Pickleball Base Vision Model (Zero Rules).
Usage:
  python run_base.py --clip clips/match1.mp4
  python run_base.py --clip clips/MicrosoftTeams-video.mp4 --max-frames 120
"""
import argparse
import os
import sys

# Ensure pickleball-base is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pbbase.pipeline import BasePerceptionPipeline

DEFAULT_CLIP = "clips/match1.mp4"
DEFAULT_OUT_DIR = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pickleball Base Perception Model (Zero Rules)")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="Path to input video clip")
    parser.add_argument("--player-weights", default="weights/yolo26n.pt", help="Weights for player detection (YOLO26)")
    parser.add_argument("--ball-weights", default="weights/ball_colab_dukcv.pt", help="Weights for ball detection")
    parser.add_argument("--ball-imgsz", type=int, default=1280, help="Input resolution for ball model")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--no-preview", action="store_true", help="Disable generating diagnostic preview video")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.clip))[0]
    out_json = os.path.join(args.out_dir, f"{stem}_base_telemetry.json")
    out_preview = None if args.no_preview else os.path.join(args.out_dir, f"{stem}_base_preview.mp4")

    print("=" * 80)
    print(" 🏓 PICKLEBALL BASE VISION MODEL — RUNNER (ZERO RULES)")
    print("=" * 80)
    print(f"  • Input Video        : {args.clip}")
    print(f"  • Player Weights     : {args.player_weights}")
    print(f"  • Ball Weights       : {args.ball_weights} (imgsz={args.ball_imgsz})")
    print(f"  • Output Telemetry   : {out_json}")
    print(f"  • Output Video       : {out_preview or 'Disabled'}")
    print("=" * 80)

    pipeline = BasePerceptionPipeline(
        player_weights=args.player_weights,
        ball_weights=args.ball_weights,
        ball_imgsz=args.ball_imgsz,
    )

    summary = pipeline.process_video(
        video_path=args.clip,
        max_frames=args.max_frames,
        out_json=out_json,
        out_preview=out_preview,
    )

    print("\n" + "=" * 80)
    print(" ✅ EXECUTION COMPLETED SUCCESSFULLY:")
    print("=" * 80)
    print(f"  • Frames Processed   : {summary['total_frames_processed']} ({summary['resolution']} @ {summary['fps']} FPS)")
    print(f"  • Processing Speed   : {summary['processing_fps']} FPS")
    print(f"  • Court Calibrated   : {summary['court_calibrated']} (Reprojection RMSE: {summary['court_reprojection_rmse_ft']} ft)")
    print(f"  • Mean Players/Frame : {summary['mean_players_per_frame']} / 4.0")
    print(f"  • Ball Detection Rate: {summary['ball_detection_rate_pct']}% ({summary['frames_with_ball']} frames tracked)")
    print(f"  • Structured JSON    : {out_json}")
    if out_preview:
        print(f"  • Preview Video      : {out_preview}")
    print("=" * 80)


if __name__ == "__main__":
    main()
