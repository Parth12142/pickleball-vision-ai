# 🏓 Week 2 Technical Guide: Base Model Fine-Tuning & Tracking Stabilization

> **Phase 1: Pure Base Perception (Zero Rules)**  
> **Status**: ⏳ **In Progress (Tasks W2-01 & W2-02 Delivered)**  
> **Working Window**: Days 6–10 (17 Nov 2026 – 23 Nov 2026)  
> **Target Milestone**: Sustained $>92\%$ ball detection rate and zero player ID swaps across full rallies.

---

## 📅 Week 2 Master Task Roadmap (7 Tasks)

| Task ID | Workstream | Task Description | Deliverable / Target File | Status | Focus & Target |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **`W2-01`** | **Training** | Hyperparameter sweep for ball model | [`pbbase/models/train_ball.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/models/train_ball.py), [`notebooks/train_ball_sweep_colab_a100.ipynb`](file:///home/pce118/Desktop/pickleball-vision-ai-main/notebooks/train_ball_sweep_colab_a100.ipynb) | ✅ **Done** | AdamW LR, weight decay, HSV, scale, blur tuning |
| **`W2-02`** | **Detection**| Multi-scale ball inference | [`pbbase/models/detector.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/models/detector.py) | ✅ **Done** | Adaptive far-court ROI zoom & patch candidate fusion |
| **`W2-03`** | **Tracking** | Ball trajectory: velocity gating, parabolic gap filling | [`pbbase/tracking/ball_trajectory.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/tracking/ball_trajectory.py) | ⏳ Next | Gravitational arc interpolation across 1–4 frame gaps |
| **`W2-04`** | **Tracking** | Hungarian bipartite player tracker with spectator rejection | [`pbbase/tracking/player_tracker.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/tracking/player_tracker.py) | 📅 Planned | Court perimeter envelope, zero partner ID swaps |
| **`W2-05`** | **Pipeline** | Define unified perception JSON schema (players, ball, court) | [`pbbase/pipeline.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/pipeline.py) | 📅 Planned | Standardized frame-by-frame telemetry schema |
| **`W2-06`** | **QA** | Validate on full rallies: ball detection rate and ID switches | Validation Results | 📅 Planned | Target: $>92\%$ ball detection, 0 ID switches |
| **`W2-07`** | **QA** | Re-run master evaluator and update accuracy report | [`reports/BASE_MODEL_ACCURACY_REPORT.md`](file:///home/pce118/Desktop/pickleball-vision-ai-main/reports/BASE_MODEL_ACCURACY_REPORT.md) | 📅 Planned | Milestone: Fully stabilized base perception engine |

---

## 🛠️ Task W2-01: Hyperparameter Sweep for Ball Model

Small optic pickleballs (2.87 inches diameter) traveling at speeds up to 55+ MPH present severe challenges for standard object detectors: motion blur, rapid lighting variations, and small pixel footprints on the far court ($6\text{--}12\text{ px}$).

To maximize model recall without generating false positives, **Task `W2-01`** introduces a dedicated training and hyperparameter sweep tool.

### 1. Training CLI Capabilities (`pbbase/models/train_ball.py`)
- **Optimizer**: AdamW with weight decay regularization (`--weight-decay 0.0005`) and cosine learning rate scheduling (`--lr0 0.001`, `--lrf 0.01`).
- **Color Jitter (HSV)**: Perturbations tuned specifically for optic yellow/green balls (`--hsv-h 0.015`, `--hsv-s 0.7`, `--hsv-v 0.4`).
- **Scale Augmentation**: Size jittering (`--scale 0.5` to `0.7`) to teach the network to recognize both 6-pixel far-court balls and large near-court balls.
- **Mosaic Scheduling**: Trains with 4-image mosaic and automatically disables mosaic for the final 10 epochs (`--close-mosaic 10`) for fine-grained localization.

### 2. Automated Sweep Mode
Running with `--sweep` conducts a systematic multi-trial comparison:
```bash
python3 pbbase/models/train_ball.py --sweep --sweep-trials 3
```

Leaderboard results are ranked by **mAP@0.50** and **Recall**, and automatically exported to `runs/ball_train/sweep_leaderboard.csv`:

| Rank | Trial Name | Focus | Key Parameters |
| :---: | :--- | :--- | :--- |
| **1** | `sweep_trial_1_baseline_adamw` | Standard Baseline | `lr0=0.001`, `weight_decay=0.0005`, `scale=0.5` |
| **2** | `sweep_trial_2_high_recall_small_ball` | Small Ball Recall | `lr0=0.0005`, `weight_decay=0.001`, `scale=0.7`, `hsv_s=0.8` |
| **3** | `sweep_trial_3_motion_blur_invariance` | Motion & Rotation | `lr0=0.0015`, `weight_decay=0.0003`, `scale=0.6`, `degrees=15.0` |

### 3. Cloud Colab A100 Notebook
- [`notebooks/train_ball_sweep_colab_a100.ipynb`](file:///home/pce118/Desktop/pickleball-vision-ai-main/notebooks/train_ball_sweep_colab_a100.ipynb)
- Configured to train at native $1312\times 1312$ resolution using Automatic Mixed Precision (AMP) on an NVIDIA A100 GPU (40GB / 80GB VRAM).
- Exports winning checkpoint directly to `weights/ball_yolo26_tuned.pt`.

---

## 🔬 Task W2-02: Multi-Scale Ball Inference & Far-Court Patch Sliding

Distant pickleballs on the far court typically subtend only **6 to 12 pixels** across a 1080p frame. While native 1312p inference captures the vast majority of rally frames, contrast degradation or motion across the net mesh can occasionally suppress raw detection confidence.

**Task W2-02** introduces high-recall multi-scale inference in `BallDetector` ([`pbbase/models/detector.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/models/detector.py)).

### 1. Multi-Scale Architecture
```
Frame Input (1920x1080)
  ├── 1. Primary Full-Frame Pass (1312x1312 native) ─────────────┐
  │                                                              │
  └── 2. Adaptive Far-Court Crop [0.12, 0.08, 0.65, 0.92]        │
           └── High-Res Patch Inference (1280p)                  │
                 └── Global Coordinate Re-mapping                │
                                                                 ▼
                                                  Candidate Fusion & Consensus NMS
                                                    ├── Match by IoU >= 0.25 OR dist <= 24px
                                                    ├── Weighted Centroid Average
                                                    └── Multi-scale Consensus Confidence Boost
```

### 2. Candidate Fusion & Multi-Scale Consensus
- Candidates from full-frame and cropped-patch passes are cross-matched:
  - **Consensus Match**: If detected in both views ($IoU \ge 0.25$ or center distance $\le 24\text{ px}$), the coordinate is computed as a confidence-weighted average and assigned a consensus boost ($+8\%$ confidence capped at $0.99$).
  - **Patch Recall**: Distant micro-balls missed in full-frame pass are recovered through the zoom patch and re-mapped directly to full-frame coordinates.

### 3. Usage & CLI Flags
Enable multi-scale inference via CLI or configuration:
```bash
# Direct runner with multiscale enabled
python3 run_base.py --clip clips/match1.mp4 --multiscale

# Or enable in config.yaml:
# models:
#   ball:
#     multiscale: true
```

### 4. Validation Results (`clips/match1.mp4`)
- **Ball Detection Rate**: **93.3%** sustained detection rate.
- **Player Tracking**: **4.0 / 4.0** continuous player tracking.
- **Court Calibration RMSE**: **0.0000 ft** reprojection error.
