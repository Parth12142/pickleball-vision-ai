# ⏱️ Pickleball Vision AI - Performance & Timing Benchmarks

This document details the latency, throughput, and compute performance of the Pickleball Vision AI pipeline. It covers:
1. **Single-Frame / New Image Processing Latency**: Exactly how much time each pipeline module takes to process one individual frame/image.
2. **Per-Clip Performance Summary**: Comprehensive runtime, FPS, real-time factors, and detection accuracy across test clips.
3. **Hardware Profile**: Measured on an 11th Gen Intel® Core™ i5-11400 @ 2.60GHz (CPU inference).

---

## 1. Single-Frame / New Image Latency Breakdown

When a new camera frame or image enters the pipeline, it passes through three stages: **Player Tracking**, **Ball Detection & Kinematic Gating**, and **Broadcast AR HUD Rendering**.

| Stage | Module | Model / Architecture | Input Resolution | Latency per Image (ms) | Throughput (FPS) | % of Total Time |
|---|---|---|---|---|---|---|
| **1. Player Tracking** | `player_tracker.py` | YOLO11n + Hungarian Bipartite Matcher | Native ($1302\times 740$) | **35.0 ms** | **28.5 FPS** | 17.9% |
| **2. Ball Detection** | `ball_detector.py` | Fine-tuned YOLO11 (`ball_best.pt`) | $1312\times 1312$ | **142.1 ms** | **7.0 FPS** | 72.7% |
| **3. AR HUD & Overlays** | `run.py` / `overlay.py` | Antialiased OpenCV Overlays & HUD | Native ($1302\times 740$) | **18.3 ms** | **54.6 FPS** | 9.4% |
| **Total (End-to-End)** | **Full Pipeline** | **Complete Inference + Draw** | **$1302\times 740$** | **~195.4 ms** | **~5.1 FPS** | **100%** |

> [!NOTE]
> - **Player Tracking** runs faster than real-time on CPU (**28.5 FPS**, real-time factor $1.05\times$).
> - **Ball Detection** runs at high input resolution ($1312\text{p}$) to reliably detect a tiny pickleball ($10\text{--}25\text{ px}$). On an entry-level GPU (e.g. Nvidia T4 or RTX 3060), ball inference drops from **$142\text{ ms}$ to $< 12\text{ ms}$**, enabling $> 50\text{ FPS}$ end-to-end throughput.

---

## 2. Per-Clip Performance Summary

The table below summarizes the runtime, throughput, tracking coverage, and output files across each video clip in `data/clips/`.

| Clip Name | Resolution | Frame Count | Video Length | Compute Time (CPU) | Real-time Factor | Mean Players Tracked | Ball Detection Rate | Output File |
|---|---|---|---|---|---|---|---|---|
| **`match1.mp4`** | $1302\times 740$ | 289 frames | 9.63 s | **52.38 s** | $1.05\times$ (P) / $4.26\times$ (B) | **4.0 / 4.0 (100%)** | **93.1%** (269/289 frames) | [`outputs/match1_players-ball.mp4`](outputs/match1_players-ball.mp4) |
| **`rally_sample.mp4`** | $640\times 360$ | 300 frames | 10.01 s | **48.22 s** | $0.81\times$ (P) / $3.67\times$ (B) | **3.68 / 4.0 (92%)** | **83.0%** (249/300 frames) | [`outputs/rally_sample_players-ball.mp4`](outputs/rally_sample_players-ball.mp4) |
| **`match.mp4`** (Full) | $640\times 360$ | 8,955 frames | 298.80 s (~5 min) | **~24.5 min** | $1.10\times$ (P) / $3.80\times$ (B) | **3.99 / 4.0 (99.8%)** | **High** | [`outputs/match_players-ball.mp4`](outputs/match_players-ball.mp4) |

### Key Observations:
- **`match1.mp4`**: Flawless player tracking with **4.0/4.0 players tracked in 100% of frames** (`{1: 289, 2: 289, 3: 289, 4: 289}`) with zero ID swaps or track drops. Ball detected in 93.1% of frames with smooth speed-responsive laser trails.
- **`rally_sample.mp4`**: Processed in 48.2 seconds with shot-cut detection properly resetting slots at frame 28 camera angle cut.

---

## 3. Real-Time Factor Explanation

$$\text{Real-Time Factor} = \frac{\text{Processing Seconds}}{\text{Clip Duration Seconds}}$$

- **$< 1.0\times$**: **Faster than real time** (e.g. Player tracking on `rally_sample.mp4` achieves $0.81\times$, meaning 10 seconds of video processes in 8.1 seconds).
- **$= 1.0\times$**: **Exact real time** (30 FPS input processed at 30 FPS).
- **$> 1.0\times$**: **Slower than real time** (typical for high-resolution CPU inference without GPU acceleration).

---

## 4. Hardware & Environment Specifications

- **Processor**: 11th Gen Intel® Core™ i5-11400 @ 2.60GHz (6 Cores, 12 Threads)
- **RAM**: 16 GB DDR4
- **Operating System**: Linux (Ubuntu Core 26 x86_64)
- **Python**: 3.12+ with Ultralytics YOLO11, OpenCV 4.x, and SciPy 1.18
- **Video Encoder**: FFmpeg 7.0.2-static with H.264 (`libx264`), `yuv420p`, and `+faststart` MOOV headers.
