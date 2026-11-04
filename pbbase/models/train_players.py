#!/usr/bin/env python3
"""
CLI training script for fine-tuning YOLO26 on the player-only dataset.
Supports both local execution (GPU/CPU) and headless server runs.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26 on Pickleball Player Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/players/data.yaml",
        help="Path to data.yaml dataset definition",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo26n.pt",
        help="Pretrained weights or model checkpoint to initialize from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32 for A100 / 16GB VRAM, reduce for lower VRAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train_players",
        help="Directory to save training run logs and checkpoints",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="player_yolo26",
        help="Run name under project directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker count (default: 8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset config '{args.data}' not found!")
        sys.exit(1)

    print(f"==================================================")
    print(f"🚀 Training YOLO26 Player Detector")
    print(f"   • Dataset Config: {args.data}")
    print(f"   • Base Model    : {args.weights}")
    print(f"   • Epochs        : {args.epochs}")
    print(f"   • Batch Size    : {args.batch}")
    print(f"   • Image Size    : {args.imgsz}")
    print(f"   • Device        : {args.device}")
    print(f"==================================================")

    model = YOLO(args.weights)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        pretrained=True,
        save=True,
        exist_ok=True,
    )

    best_checkpoint = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n✅ Training complete!")
    if best_checkpoint.exists():
        target_dir = Path("weights")
        target_dir.mkdir(exist_ok=True)
        target_pt = target_dir / "player_yolo26_best.pt"
        import shutil
        shutil.copy2(best_checkpoint, target_pt)
        print(f"📦 Best model exported to: {target_pt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
