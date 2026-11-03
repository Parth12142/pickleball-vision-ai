# 🏓 Pickleball Base Vision AI — Master Accuracy Report (YOLO26 vs YOLO11)

> Generated: `2026-11-03 10:33:04` | Total Benchmark Latency: `122.81s`

All metrics are computed on the clean, held-out validation split with zero data leakage.

## 1. Perception Benchmark Summary

| Subsystem | Model Checkpoint | Architecture | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Postprocess (ms) | Inference (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ball Detection** | `ball_colab_dukcv.pt` | YOLO11n | **4.7%** | **4.1%** | **0.2%** | **0.1%** | 1.16 ms | 150.17 ms |
| **Player Detection** | `yolo26n.pt` | **YOLO26n (NMS-Free)** | **46.6%** | **44.7%** | **34.3%** | **17.9%** | **4.18 ms** | 63.91 ms |
| **Player Detection** | `yolo11n.pt` | YOLO11n | 49.0% | 43.3% | 33.7% | 17.7% | 5.67 ms | 69.05 ms |

---
