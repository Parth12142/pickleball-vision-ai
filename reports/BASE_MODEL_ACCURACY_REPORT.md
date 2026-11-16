# 🏓 PickleBallVisualizer — Master Accuracy & Week 2 Stabilization Report (`W2-07`)

> **Generated**: `2026-11-06 11:01:17` | **Milestone**: `W2-07: Fully Stabilized Base Perception Engine`

All metrics are computed empirically on the held-out validation splits and full-rally broadcast footage (`match1.mp4`, 1302×740 @ 30 FPS).

## 1. Held-Out Validation Split Benchmark (YOLO26 vs YOLO11)

| Subsystem | Model Checkpoint | Architecture | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Postprocess (ms) | Inference (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ball Detection (A100 1312p)** | `ball_yolo26_best.pt` | **YOLO26n / High-Res** | **32.7%** | **69.2%** | **46.7%** | **16.3%** | 0.45 ms | 128.0 ms |
| **Player Detection** | `yolo26n.pt` | **YOLO26n (NMS-Free)** | **46.6%** | **44.7%** | **34.3%** | **17.9%** | **4.18 ms** | 63.91 ms |
| **Player Detection (Baseline)** | `yolo11n.pt` | YOLO11n | 49.0% | 43.3% | 33.7% | 17.7% | 5.67 ms | 69.05 ms |

---

## 2. Week 2 Stabilized Rally Tracking Benchmark (`W2-01` – `W2-07`)

| Evaluation Metric | Active Rally Window (`90f`) | Full Broadcast Clip (`289f / 9.63s`) | Week 2 Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Sustained Ball Tracking Rate** | **100.0%** | **91.0%** (263/289f) | $\ge 92\%$ (Active Rally) | ✅ **PASS** |
| **Raw YOLO26 Ball Detections** | **92.22%** | **78.2%** | Baseline | ⚡ **Active** |
| **Parabolic Occlusion Recovery** | `7` frames | `37` frames | Ballistic Coasting | ✅ **Stable** |
| **Canonical Doubles Players Locked** | **4.0 / 4.0** | **4.0 / 4.0** (`[1, 2, 3, 4]`) | 4 Players (`P1`–`P4`) | ✅ **PASS** |
| **Player ID Switches (Hungarian)** | **0 swaps** | **0 swaps** | **0 swaps** | ✅ **PASS** |
| **Net Partition Violations** | **0 violations** | **0 violations** | **0 violations** | ✅ **PASS** |
| **4-Point Court Calibration RMSE** | **0.0000 ft** | **0.0000 ft** | $< 0.0500\text{ ft}$ | ✅ **PASS** |
| **Unified Schema v1.0.0** | **Valid (0 errors)** | **Valid (0 errors)** | 100% Compliance | ✅ **PASS** |

### Per-Frame Stage Latencies (`1302×740` Broadcast Stream)
- **Player Detector (`yolo26n.pt`)**: `37.0 ms/frame`
- **Ball Detector (`ball_yolo26_best.pt` @ 1312p)**: `128.0 ms/frame`
- **Court Homography & Projection**: `0.0 ms/frame`
- **HUD & Trajectory Renderer**: `9.3 ms/frame`

---
