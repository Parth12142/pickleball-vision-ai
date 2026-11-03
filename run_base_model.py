"""
run_base_model.py - Standalone Runner for Unified Pickleball Base Vision Model.
"""
import argparse
import os
import sys

from base_model import PickleballBaseModel

DEFAULT_CLIP = "data/clips/match1.mp4"
DEFAULT_OUT_DIR = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unified Pickleball Base Vision Model")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="Path to input video clip")
    parser.add_argument("--player-weights", default="yolo11n.pt", help="Weights for player detection & tracking")
    parser.add_argument("--ball-weights", default="weights/ball_best.pt", help="Weights for ball detection & tracking")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (None = all)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save telemetry and preview")
    parser.add_argument("--no-preview", action="store_true", help="Disable generating preview mp4 video")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clip_basename = os.path.splitext(os.path.basename(args.clip))[0]
    out_json = os.path.join(args.out_dir, f"{clip_basename}_base_telemetry.json")
    out_preview = None if args.no_preview else os.path.join(args.out_dir, f"{clip_basename}_base_preview.mp4")

    print("=" * 80)
    print(" 🏓 RUNNING UNIFIED PICKLEBALL BASE VISION MODEL")
    print("=" * 80)
    print(f"  • Input Clip         : {args.clip}")
    print(f"  • Player Weights     : {args.player_weights}")
    print(f"  • Ball Weights       : {args.ball_weights}")
    print(f"  • Max Frames         : {args.max_frames or 'Full Video'}")
    print(f"  • Output Telemetry   : {out_json}")
    print(f"  • Output Preview     : {out_preview or 'Disabled'}")
    print("=" * 80)

    # If base_model_best.pt exists in weights/, use it as primary weights
    player_w = args.player_weights
    ball_w = args.ball_weights

    model = PickleballBaseModel(
        player_model_path=player_w,
        ball_model_path=ball_w,
    )

    summary = model.process_video(
        video_path=args.clip,
        max_frames=args.max_frames,
        out_json=out_json,
        out_preview=out_preview,
    )

    print("\n" + "=" * 80)
    print(" ✅ BASE MODEL PROCESSING COMPLETED SUCCESSFULLY:")
    print("=" * 80)
    print(f"  • Frames Processed   : {summary['total_frames_processed']}")
    print(f"  • Processing Speed   : {summary['processing_fps']} FPS")
    print(f"  • Court Calibrated   : {summary['court_calibrated']}")
    print(f"  • Bounces Detected   : {summary['total_bounces_detected']}")
    print(f"  • Clean JSON Stream  : {out_json}")
    if out_preview:
        print(f"  • Diagnostic Video   : {out_preview}")
    print("=" * 80)


if __name__ == "__main__":
    main()
