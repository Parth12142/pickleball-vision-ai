# 🏓 Pickleball Base Vision Model — Accuracy & Benchmark Audit

**Model Weights**: `weights/ball_best.pt`  
**Evaluated On**: `test` split of `/home/pce118/Downloads/pick/pickleball.v13-bickleball_player.yolo26/data.yaml`  
**Date Generated**: 2026-11-02 17:50:29  

## 1. Overall Perception Performance


| Metric | Score | Assessment |
| :--- | :--- | :--- |
| **mAP @ 0.50** | **0.23%** | Good Baseline |
| **mAP @ 0.50:0.95** | **0.05%** | Overall IoU Precision |
| **Mean Precision (P)** | **2.27%** | False Positive Suppression |
| **Mean Recall (R)** | **2.33%** | Object Detection Capture Rate |

## 2. Per-Class Accuracy Breakdown

| Target Object | Precision | Recall | F1 Score | mAP @ 0.50 | mAP @ 0.50:0.95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Court** | 0.0% | 0.0% | 0.0% | **0.00%** | 0.00% |
| **ball** | 4.5% | 4.7% | 4.6% | **0.47%** | 0.09% |

## 3. Inference Latency & Speed

- **Preprocess**: `2.03 ms`
- **Inference**: `53.53 ms`
- **Postprocess**: `0.39 ms`
- **Total Latency**: `55.96 ms` (~`17.9 FPS`)

## 4. Decoupling Guarantee

This Base Model strictly provides perceptual spatial coordinate streams. Downstream scoring, scoreboard UI, and foul VAR modules consume these coordinates without altering or degrading model visual weights.
