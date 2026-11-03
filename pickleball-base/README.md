# 🏓 Pickleball Base Vision AI — Core Perception Engine

A fresh, pure computer vision base perception system for pickleball. It accurately detects and tracks **players**, the **ball**, and **court planar geometry** with **ZERO hard-coded scoring rules, referee decisions, or scoreboard logic**.

---

## 🌟 Architecture & Core Capabilities

```
Video Clip ──► Court Homography (H) ──► Player Tracker (P1-P4) ──► Ball Trajectory ──► Structured Telemetry JSON
                                                                                      └──► Diagnostic Preview MP4
```

1. **Court Calibration & Homography (`pbbase/court/homography.py`)**:
   - 4-corner baseline projective transformation mapping camera pixels $(u, v)$ to official USA Pickleball (USAP) 2D regulation plane ($20 \times 44\text{ ft}$).
   - Sub-millimeter numerical precision (reprojection RMSE $< 10^{-4}\text{ ft}$).

2. **Canonical Multi-Player Tracking (`pbbase/tracking/player_tracker.py`)**:
   - YOLO11 person detection with dynamic court-polygon distance gating (filters spectators and referee).
   - Hungarian optimal bipartite association preventing partner ID swaps.
   - Ground-contact foot coordinate estimation $(x_{\text{foot}}, y_{\text{foot}})$ with metric velocity estimation (mph).

3. **High-Resolution Micro-Ball Trajectory (`pbbase/tracking/ball_trajectory.py`)**:
   - Specialist high-resolution ($1280\text{p}$) ball detection.
   - Kinematic velocity gating with physical momentum prediction.
   - Parabolic gravitational arc interpolation bridging 1–4 frame paddle/net occlusions.
   - Speed-responsive dynamic color trail (Electric Orange $> 22\text{ px/f}$, Laser Chartreuse $10\text{--}22\text{ px/f}$, Ice Cyan $< 10\text{ px/f}$).

---

## 📁 Repository Structure

```
pickleball-base/
├── pbbase/
│   ├── court/               # Court geometry & homography projection
│   │   ├── homography.py
│   │   └── __init__.py
│   ├── models/              # YOLO inference wrappers & training scripts
│   │   ├── detector.py
│   │   ├── train_ball.py
│   │   └── __init__.py
│   ├── tracking/            # Canonical player tracking & ball trajectory
│   │   ├── player_tracker.py
│   │   ├── ball_trajectory.py
│   │   └── __init__.py
│   ├── data/                # Dataset curation & leakage-free splitting
│   │   ├── build_datasets.py
│   │   └── __init__.py
│   ├── tests/               # Unit tests
│   │   └── test_base.py
│   ├── pipeline.py          # Unified streaming pipeline orchestrator
│   └── patch_env.py         # POSIX SemLock sandbox compatibility patch
├── datasets/                # Clean train/val/test splits (v13 + v1i)
│   ├── ball/
│   └── players/
├── clips/                   # Test match clips
│   ├── match1.mp4
│   ├── MicrosoftTeams-video.mp4
│   └── rally_sample.mp4
├── weights/                 # Model checkpoints
│   ├── ball_colab_dukcv.pt
│   └── yolo11n.pt
├── outputs/                 # Output preview MP4s & structured JSON telemetry
├── reports/                 # Empirical accuracy benchmark reports
├── run_base.py              # CLI entry point to run perception on clips
├── evaluate_base.py         # Master empirical accuracy benchmark suite
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Base Perception on Video Clip
```bash
python run_base.py --clip clips/match1.mp4
```

To run on a subset of frames:
```bash
python run_base.py --clip clips/match1.mp4 --max-frames 90
```

Outputs will be saved in `outputs/`:
- `outputs/match1_base_telemetry.json`: Clean per-frame spatial coordinates, speeds, and player tracks.
- `outputs/match1_base_preview.mp4`: Diagnostic H.264 video with ground halos, laser trajectory ribbon, and HUD.

### 2. Run Automated Unit Tests
```bash
python -m unittest pbbase/tests/test_base.py
```

### 3. Evaluate Model Accuracy on Held-Out Test Splits
```bash
python evaluate_base.py
```
Outputs verifiable evaluation metrics to `reports/BASE_MODEL_ACCURACY_REPORT.md`.

### 4. Fine-Tune Ball Model
```bash
python pbbase/models/train_ball.py --epochs 20 --batch 8 --imgsz 960
```
