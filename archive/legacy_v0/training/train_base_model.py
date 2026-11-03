"""
train_base_model.py - End-to-End Fine-Tuning Pipeline for Unified Base Vision Model.
"""
import os
import sys

# Ensure root dir is in sys.path and apply multiprocessing patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import base_model.patch_env  # noqa: F401

import argparse
import shutil
from ultralytics import YOLO

DEFAULT_DATASET_YAML = "/home/pce118/Downloads/pick/pickleball.v13-bickleball_player.yolo26/data.yaml"
DEFAULT_BASE_WEIGHTS = "yolo11n.pt"
OUTPUT_WEIGHTS_DIR = "weights"
FINAL_BEST_WEIGHTS = os.path.join(OUTPUT_WEIGHTS_DIR, "base_model_best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-Tune Unified Pickleball Base Vision Model")
    parser.add_argument("--data", default=DEFAULT_DATASET_YAML, help="Path to Roboflow dataset data.yaml")
    parser.add_argument("--weights", default=DEFAULT_BASE_WEIGHTS, help="Base model weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--device", default="cpu", help="Device to train on ('cpu' or '0')")
    parser.add_argument("--name", default="pickleball_base_finetune", help="Training run name")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch dry-run to verify pipeline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)

    print("=" * 80)
    print(" 🏓 UNIFIED PICKLEBALL BASE VISION MODEL — TRAINING & FINE-TUNING PIPELINE")
    print("=" * 80)
    print(f"  • Dataset YAML       : {args.data}")
    print(f"  • Base Weights       : {args.weights}")
    print(f"  • Target Epochs      : {1 if args.dry_run else args.epochs}")
    print(f"  • Batch Size         : {args.batch}")
    print(f"  • Image Resolution   : {args.imgsz}x{args.imgsz}")
    print(f"  • Compute Device     : {args.device}")
    print(f"  • Target Checkpoint  : {FINAL_BEST_WEIGHTS}")
    print("=" * 80)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset YAML not found at: {args.data}")

    # Load base YOLO architecture
    print(f"\n[1/3] Loading base model architecture from {args.weights}...")
    model = YOLO(args.weights)

    epochs = 1 if args.dry_run else args.epochs
    print(f"\n[2/3] Launching fine-tuning for {epochs} epochs...")
    train_results = model.train(
        data=args.data,
        epochs=epochs,
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

    # Locate saved best weights
    trained_best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(trained_best):
        shutil.copy2(trained_best, FINAL_BEST_WEIGHTS)
        print(f"\n[3/3] Successfully saved best fine-tuned model checkpoint to:")
        print(f"      ==> {FINAL_BEST_WEIGHTS}")
    else:
        # Fallback to last.pt if dry-run
        trained_last = os.path.join(model.trainer.save_dir, "weights", "last.pt")
        if os.path.exists(trained_last):
            shutil.copy2(trained_last, FINAL_BEST_WEIGHTS)
            print(f"\n[3/3] Saved checkpoint to ==> {FINAL_BEST_WEIGHTS}")

    print("\nTraining completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
