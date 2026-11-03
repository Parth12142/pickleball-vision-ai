"""
train_ball.py - Fine-Tuning Pipeline for Pickleball Ball Detector on Clean Dataset.
Uses datasets/ball/data.yaml (leakage-free v13 + v1i combined dataset).
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import pbbase.patch_env  # noqa: F401
from ultralytics import YOLO
DEFAULT_DATA_YAML = os.path.join(ROOT, "datasets", "ball", "data.yaml")
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "ball_colab_dukcv.pt")
OUTPUT_WEIGHTS = os.path.join(ROOT, "weights", "ball_best_finetuned.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO Pickleball Ball Detector")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Base weights to start from")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=960, help="Input resolution (960 or 1280)")
    parser.add_argument("--device", default="cpu", help="Device to train on ('cpu' or '0')")
    parser.add_argument("--name", default="ball_base_finetune", help="Training run experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(" 🏓 FINE-TUNING PICKLEBALL BALL DETECTOR")
    print("=" * 80)
    print(f"  • Dataset Config   : {args.data}")
    print(f"  • Base Weights     : {args.weights}")
    print(f"  • Target Epochs    : {args.epochs}")
    print(f"  • Batch Size       : {args.batch}")
    print(f"  • Resolution       : {args.imgsz}x{args.imgsz}")
    print(f"  • Compute Device   : {args.device}")
    print(f"  • Output Checkpoint: {OUTPUT_WEIGHTS}")
    print("=" * 80)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml not found at: {args.data}")

    model = YOLO(args.weights)
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        workers=0,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    trained_best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(trained_best):
        shutil.copy2(trained_best, OUTPUT_WEIGHTS)
        print(f"\n[DONE] Saved fine-tuned checkpoint to: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
