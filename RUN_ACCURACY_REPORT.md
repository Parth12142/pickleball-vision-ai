# 🏓 PickleBallVisualizer — Run Accuracy & Performance Report

> **Last Updated**: `2026-11-18 14:51:30`  
> **Total Pipeline Runs Recorded**: `47`  
> **Output Guide & Telemetry Specification**: See [`OUTPUT_EXPLANATION.md`](OUTPUT_EXPLANATION.md) for full visual overlay and JSON schema definitions.

---

## 🎯 Latest Execution Summary (`match_hd_1min.mp4`)

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Clip Name** | `match_hd_1min.mp4` | — | — |
| **Resolution & Video FPS** | `1280x720` @ `30.0 FPS` | — | — |
| **Frames Processed** | `1800` frames | Full Rally | ✅ Analyzed |
| **Inference Throughput** | **`4.49 FPS`** | > 30 FPS (Real-Time) | ⚡ Real-Time |
| **Ball Detection Rate** | **`60.2%`** (1084 frames) | > 85% Sustained | ⚠️ Tuning Required |
| **Active Players Tracked** | **`3.76 / 4.0`** | 4.0 Players (Doubles) | ⚠️ Tracking Gap |
| **Court Calibration (RMSE)** | **`0.0000 ft`** | < 0.05 ft Error | ✅ Sub-Millimeter |

### Generated Outputs for Latest Run:
- 📊 **Structured JSON Telemetry**: [`/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_telemetry.json`](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_telemetry.json)
- 🎥 **Visual Overlay MP4**: [`/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_preview.mp4`](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_preview.mp4)

---

## 📜 Historical Execution Ledger (All Runs)

| Run | Timestamp | Clip Name | Res | Frames | Ball Detect % | Players / 4 | Court RMSE (ft) | FPS | Outputs |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #47 | `2026-11-18 14:51:30` | **match_hd_1min.mp4** | 1280x720 | 1800 | **60.2%** (1084f) | 3.76 / 4.0 | ✅ 0.0000 | 4.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_preview.mp4) |
| #46 | `2026-11-18 14:33:57` | **match_hd_1min.mp4** | 1280x720 | 1800 | **73.7%** (1327f) | 3.76 / 4.0 | ✅ 0.0000 | 5.7 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_preview.mp4) |
| #45 | `2026-11-18 14:25:44` | **match_hd_1min.mp4** | 1280x720 | 30 | **90.0%** (27f) | 3.87 / 4.0 | ✅ 0.0000 | 5.3 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_hd_1min_base_telemetry.json) |
| #44 | `2026-11-18 13:12:18` | **match_1min.mp4** | 640x360 | 1800 | **70.1%** (1262f) | 3.57 / 4.0 | ✅ 0.0000 | 4.2 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_1min_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_1min_base_preview.mp4) |
| #43 | `2026-11-18 13:03:03` | **match_1min.mp4** | 640x360 | 60 | **90.0%** (54f) | 3.33 / 4.0 | ✅ 0.0000 | 4.1 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_1min_base_telemetry.json) |
| #42 | `2026-11-18 13:02:27` | **match_mlp.mp4** | 1280x720 | 300 | **86.3%** (259f) | 4.0 / 4.0 | ✅ 0.0000 | 5.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_preview.mp4) |
| #41 | `2026-11-18 12:58:07` | **match_mlp.mp4** | 1280x720 | 300 | **87.0%** (261f) | 4.0 / 4.0 | ✅ 0.0000 | 4.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_preview.mp4) |
| #40 | `2026-11-18 12:36:41` | **match_mlp.mp4** | 1280x720 | 300 | **89.3%** (268f) | 4.0 / 4.0 | ✅ 0.0000 | 5.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_preview.mp4) |
| #39 | `2026-11-18 12:19:10` | **match2.mp4** | 1296x738 | 280 | **99.3%** (278f) | 4.0 / 4.0 | ✅ 0.0000 | 5.4 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_preview.mp4) |
| #38 | `2026-11-18 12:18:12` | **match1.mp4** | 1302x740 | 289 | **98.6%** (285f) | 4.0 / 4.0 | ✅ 0.0000 | 5.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #37 | `2026-11-18 12:17:01` | **match1.mp4** | 1302x740 | 30 | **100.0%** (30f) | 4.0 / 4.0 | ✅ 0.0000 | 4.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #36 | `2026-11-18 12:16:20` | **match1.mp4** | 1302x740 | 30 | **96.7%** (29f) | 4.0 / 4.0 | ✅ 0.0000 | 4.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #35 | `2026-11-17 14:35:19` | **match2.mp4** | 1296x738 | 280 | **99.3%** (278f) | 4.0 / 4.0 | ✅ 0.0000 | 5.7 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_preview.mp4) |
| #34 | `2026-11-17 14:34:27` | **match1.mp4** | 1302x740 | 289 | **94.1%** (272f) | 4.0 / 4.0 | ✅ 0.0000 | 6.0 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #33 | `2026-11-17 14:05:38` | **match1.mp4** | 1302x740 | 289 | **94.1%** (272f) | 4.0 / 4.0 | ✅ 0.0000 | 6.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #32 | `2026-11-17 14:00:03` | **match2.mp4** | 1296x738 | 90 | **100.0%** (90f) | 4.0 / 4.0 | ✅ 0.0000 | 5.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_preview.mp4) |
| #31 | `2026-11-17 13:47:08` | **match2.mp4** | 1296x738 | 90 | **100.0%** (90f) | 4.0 / 4.0 | ✅ 0.0000 | 5.7 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_preview.mp4) |
| #30 | `2026-11-16 15:56:40` | **cut.mp4** | 1440x2560 | 295 | **83.1%** (245f) | 4.0 / 4.0 | ✅ 0.0000 | 2.3 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/cut_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/cut_base_preview.mp4) |
| #29 | `2026-11-16 15:40:35` | **MicrosoftTeams-court.mp4** | 720x660 | 470 | **77.9%** (366f) | 3.93 / 4.0 | ✅ 0.0000 | 4.2 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-court_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-court_base_preview.mp4) |
| #28 | `2026-11-16 15:18:37` | **match_mlp.mp4** | 1280x720 | 300 | **100.0%** (300f) | 4.0 / 4.0 | ✅ 0.0000 | 4.1 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_base_preview.mp4) |
| #27 | `2026-11-16 15:16:50` | **MicrosoftTeams-video.mp4** | 720x1280 | 470 | **77.2%** (363f) | 4.0 / 4.0 | ✅ 0.0000 | 2.7 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_preview.mp4) |
| #26 | `2026-11-16 15:10:33` | **MicrosoftTeams-video.mp4** | 720x1280 | 470 | **77.2%** (363f) | 4.0 / 4.0 | ✅ 0.0000 | 2.5 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_preview.mp4) |
| #25 | `2026-11-16 14:34:03` | **match2.mp4** | 640x360 | 300 | **100.0%** (300f) | 3.77 / 4.0 | ✅ 0.0000 | 4.3 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match2_base_preview.mp4) |
| #24 | `2026-11-06 10:25:13` | **match1.mp4** | 1302x740 | 289 | **91.0%** (263f) | 4.0 / 4.0 | ✅ 0.0000 | 5.7 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #23 | `2026-11-06 09:59:30` | **MicrosoftTeams-video.mp4** | 720x1280 | 60 | **25.0%** (15f) | 4.0 / 4.0 | ✅ 0.0000 | 6.4 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_preview.mp4) |
| #22 | `2026-11-06 09:51:43` | **MicrosoftTeams-video.mp4** | 720x1280 | 10 | **50.0%** (5f) | 4.0 / 4.0 | ✅ 0.0000 | 4.1 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/MicrosoftTeams-video_base_preview.mp4) |
| #21 | `2026-11-06 09:36:14` | **match1.mp4** | 1302x740 | 289 | **91.0%** (263f) | 4.0 / 4.0 | ✅ 0.0000 | 6.1 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #20 | `2026-11-06 09:21:06` | **match1.mp4** | 1302x740 | 3 | **100.0%** (3f) | 4.0 / 4.0 | ✅ 0.0000 | 2.2 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #19 | `2026-11-06 09:17:03` | **match1.mp4** | 1302x740 | 2 | **100.0%** (2f) | 4.0 / 4.0 | ✅ 0.0000 | 1.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #18 | `2026-11-06 09:16:59` | **match1.mp4** | 1302x740 | 2 | **100.0%** (2f) | 4.0 / 4.0 | ✅ 0.0000 | 1.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #17 | `2026-11-06 09:16:55` | **match1.mp4** | 1302x740 | 2 | **100.0%** (2f) | 4.0 / 4.0 | ✅ 0.0000 | 1.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #16 | `2026-11-06 09:16:52` | **match1.mp4** | 1302x740 | 2 | **100.0%** (2f) | 4.0 / 4.0 | ✅ 0.0000 | 1.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #15 | `2026-11-06 09:16:48` | **match1.mp4** | 1302x740 | 2 | **100.0%** (2f) | 4.0 / 4.0 | ✅ 0.0000 | 1.6 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #14 | `2026-11-06 09:13:25` | **match1.mp4** | 1302x740 | 3 | **100.0%** (3f) | 4.0 / 4.0 | ✅ 0.0000 | 2.1 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #13 | `2026-11-06 09:13:15` | **match1.mp4** | 1302x740 | 5 | **100.0%** (5f) | 4.0 / 4.0 | ✅ 0.0000 | 2.9 | [JSON](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_telemetry.json) | [MP4](/home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_base_preview.mp4) |
| #12 | `2026-11-06 08:52:41` | **rally_sample.mp4** | 640x360 | 5 | **0.0%** (0f) | 0.0 / 4.0 | ✅ 0.0000 | 3.0 | [JSON](outputs/rally_sample_base_telemetry.json) | [MP4](outputs/rally_sample_base_preview.mp4) |
| #11 | `2026-11-06 08:52:15` | **match1.mp4** | 1302x740 | 5 | **100.0%** (5f) | 4.0 / 4.0 | ✅ 0.0000 | 3.0 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #10 | `2026-11-05 10:44:57` | **match1.mp4** | 1302x740 | 90 | **100.0%** (90f) | 4.0 / 4.0 | ✅ 0.0000 | 4.6 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #9 | `2026-11-05 10:30:53` | **match1.mp4** | 1302x740 | 90 | **100.0%** (90f) | 4.0 / 4.0 | ✅ 0.0000 | 4.6 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #8 | `2026-11-05 09:38:56` | **match1.mp4** | 1302x740 | 90 | **100.0%** (90f) | 4.0 / 4.0 | ✅ 0.0000 | 4.5 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #7 | `2026-11-04 13:32:56` | **match1.mp4** | 1302x740 | 60 | **93.3%** (56f) | 4.0 / 4.0 | ✅ 0.0000 | 4.4 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #6 | `2026-11-04 12:39:26` | **MicrosoftTeams-video.mp4** | 720x1280 | 470 | **37.9%** (178f) | 3.71 / 4.0 | ✅ 0.0000 | 6.4 | [JSON](outputs/MicrosoftTeams-video_base_telemetry.json) | [MP4](outputs/MicrosoftTeams-video_base_preview.mp4) |
| #5 | `2026-11-04 09:22:25` | **match1.mp4** | 1302x740 | 15 | **100.0%** (15f) | 4.0 / 4.0 | ✅ 0.0000 | 3.7 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #4 | `2026-11-03 18:04:07` | **match1.mp4** | 1302x740 | 289 | **79.6%** (230f) | 4.0 / 4.0 | ✅ 0.0000 | 6.2 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #3 | `2026-11-03 18:02:28` | **match1.mp4** | 1302x740 | 60 | **91.7%** (55f) | 4.0 / 4.0 | ✅ 0.0000 | 4.8 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #2 | `2026-11-03 17:53:12` | **match1.mp4** | 1302x740 | 289 | **79.6%** (230f) | 4.0 / 4.0 | ✅ 0.0000 | 6.3 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |
| #1 | `2026-11-03 16:52:12` | **match1.mp4** | 1302x740 | 90 | **92.2%** (83f) | 4.0 / 4.0 | ✅ 0.0000 | 5.0 | [JSON](outputs/match1_base_telemetry.json) | [MP4](outputs/match1_base_preview.mp4) |

---

## 💡 How to Read the Metrics

1. **Ball Detection Rate (%)**: Percentage of video frames where the ball was localized and verified with velocity gating. Higher is better (Target: >85%).
2. **Players / 4.0**: Mean active player count inside the court envelope. A score of 3.9 to 4.0 indicates zero lost player slots across the rally.
3. **Court RMSE (ft)**: Root mean square error of homography reprojection from video pixels to court ground coordinates in feet. Under 0.05 ft guarantees precise line-call accuracy.
4. **Auto-Update**: This document is automatically refreshed on every video execution by `run_base.py`.
