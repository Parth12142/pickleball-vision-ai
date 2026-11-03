# 🏓 Pickleball Vision AI — Output Guide & Schema Specification

This document explains exactly how the perception pipeline processes video, what each visual element in the diagnostic preview MP4 represents, and how the structured JSON telemetry is formatted.

---

## 1. Visual Preview Video (`outputs/<clip>_base_preview.mp4`)

When you run `run_base.py`, it produces an annotated MP4 video with real-time perception overlays.

```
+-----------------------------------------------------------------------------------------+
| [TOP BANNER HUD] F: 42/300 | Players: 4/4 | Ball: LOCKED | Time: 00:01.40 | Zero Rules  |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|       [P1 (Near Left)]                                 [P2 (Near Right)]                |
|       Box: Cyan (#00E5FF)                              Box: Magenta (#E040FB)           |
|       Label: P1 (X: 5.2 ft, Y: 12.1 ft)                Label: P2 (X: 14.8 ft, Y: 11.9 ft)
|                                                                                         |
|                                    [BALL TRAJECTORY]                                    |
|                                    (•) LOCKED 28.4 MPH                                  |
|                                     \  Trajectory ribbon:                               |
|                                      \ Yellow (Fast Rally)                              |
|                                       •                                                 |
|                                                                                         |
|       [P3 (Far Left)]                                  [P4 (Far Right)]                 |
|       Box: Orange (#FF9100)                            Box: Blue (#2979FF)              |
|       Label: P3 (X: 4.9 ft, Y: 32.4 ft)                Label: P4 (X: 15.1 ft, Y: 33.2 ft)
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

### Visual Element Reference

| Visual Element | Color & Style | What It Represents |
| :--- | :--- | :--- |
| **Player Bounding Box (P1)** | **Cyan** (`#00E5FF`) | Near-court left player slot. Stable across frames without ID swapping. |
| **Player Bounding Box (P2)** | **Magenta** (`#E040FB`) | Near-court right player slot. |
| **Player Bounding Box (P3)** | **Orange** (`#FF9100`) | Far-court left player slot. |
| **Player Bounding Box (P4)** | **Blue** (`#2979FF`) | Far-court right player slot. |
| **Foot Ground Contact** | Green Dot at Box Bottom | Exact point where the player's feet touch the court surface. Used for Non-Volley Zone (Kitchen) foot-fault detection. |
| **Court Coordinates Label** | Floating text above player box | Player's exact ground coordinates `(X, Y) ft` mapped to official court dimensions ($20\text{ ft} \times 44\text{ ft}$). |
| **Ball Detection Target** | Cyan Concentric Circles | The detected micro-ball in the current frame. Shows speed in MPH when moving. |
| **Ball Trajectory Ribbon** | Trailing Multi-Color Line | Last 12 frames of ball positions with speed-based heat mapping: <br/>• **Blue**: Slow movement ($< 15\text{ MPH}$) <br/>• **Cyan**: Normal rally flight ($15\text{--}30\text{ MPH}$) <br/>• **Yellow**: Fast drive ($30\text{--}45\text{ MPH}$) <br/>• **Red**: Hard smash ($> 45\text{ MPH}$) |
| **Top Banner HUD** | Dark Slate Banner (Top Left) | Frame count (`F: X/Total`), players detected (`X/4`), ball state (`LOCKED` or `SEARCH`), and video timestamp. |

---

## 2. Structured JSON Telemetry (`outputs/<clip>_base_telemetry.json`)

The telemetry file contains a master summary at the top, followed by a frame-by-frame stream (`telemetry_stream`).

### Example JSON Record
```json
{
  "video_path": "clips/match1.mp4",
  "total_frames_processed": 300,
  "fps": 29.97,
  "resolution": "1920x1080",
  "elapsed_s": 4.12,
  "processing_fps": 72.8,
  "court_calibrated": true,
  "court_reprojection_rmse_ft": 0.0001,
  "mean_players_per_frame": 4.0,
  "frames_with_ball": 278,
  "ball_detection_rate_pct": 92.7,
  "telemetry_stream": [
    {
      "frame_idx": 42,
      "timestamp_s": 1.401,
      "court_calibrated": true,
      "players": [
        {
          "id": "P1",
          "bbox": [320.5, 410.2, 450.1, 780.0],
          "confidence": 0.91,
          "foot_px": [385.3, 780.0],
          "court_ft": [5.21, 12.14]
        },
        {
          "id": "P2",
          "bbox": [710.0, 420.1, 840.4, 790.2],
          "confidence": 0.89,
          "foot_px": [775.2, 790.2],
          "court_ft": [14.82, 11.91]
        }
      ],
      "ball": {
        "pixel": [582.4, 310.1],
        "confidence": 0.86,
        "speed_px_per_frame": 18.4,
        "speed_mph": 28.4,
        "court_ft": [9.85, 21.42]
      }
    }
  ]
}
```

### Telemetry Field Definitions

#### Top-Level Summary
- `video_path`: Input video file processed.
- `total_frames_processed`: Total frame count analyzed.
- `resolution` & `fps`: Video dimensions and capture frame rate.
- `processing_fps`: Real inference throughput achieved on your hardware.
- `court_calibrated`: `true` if camera planar homography succeeded.
- `court_reprojection_rmse_ft`: Calibration error in feet (lower is better, typically $< 0.01\text{ ft}$).
- `mean_players_per_frame`: Average players tracked per frame (target is $4.0$ for doubles).
- `ball_detection_rate_pct`: Percentage of video frames where the ball was detected or tracked.

#### Per-Frame Stream (`telemetry_stream[i]`)
- `frame_idx`: Integer 0-based frame index.
- `timestamp_s`: Exact video elapsed time in seconds.
- `players`: Array of active players detected on court:
  - `id`: Canonical slot (`"P1"`, `"P2"`, `"P3"`, or `"P4"`).
  - `bbox`: `[x1, y1, x2, y2]` in video pixel coordinates.
  - `confidence`: Detection score ($0.0\text{--}1.0$).
  - `foot_px`: `[x, y]` pixel location of the player's ground contact.
  - `court_ft`: `[X, Y]` in feet mapped to official court dimensions:
    - Width ($X$): $0.0\text{ ft}$ (left sideline) to $20.0\text{ ft}$ (right sideline).
    - Length ($Y$): $0.0\text{ ft}$ (near baseline) to $44.0\text{ ft}$ (far baseline).
    - Net location: $Y = 22.0\text{ ft}$.
    - Non-Volley Zones (Kitchen): $Y \in [15.0, 29.0]\text{ ft}$.
- `ball`: Ball detection data (`null` if ball was occluded or out of view):
  - `pixel`: `[x, y]` pixel coordinates of the ball center.
  - `speed_px_per_frame`: Motion displacement between consecutive frames.
  - `speed_mph`: Real-world calibrated speed in miles per hour.
  - `court_ft`: `[X, Y]` projected ball ground location in feet.

---

## 3. How Downstream Systems Use This Output

This output acts as the single source of truth for all subsequent phases:
1. **Week 3 (Court Minimap & Bounces)**:
   - Uses `court_ft` to render a 2D top-down bird's-eye broadcast view.
   - Uses sudden reversals in vertical velocity to pinpoint bounce timestamps and landing coordinates.
2. **Week 4 (USAP 0-0-2 Scoring & Rules)**:
   - Checks `players.foot_ft` against the Kitchen boundaries ($Y \in [15.0, 22.0]$ and $[22.0, 29.0]$) to flag foot faults.
   - Checks `ball.court_ft` against sideline and baseline boundaries to make automatic **IN** or **OUT** line calls.
3. **Week 5 (AI Coaching & Broadcast HUD)**:
   - Reads player speeds, shot speeds in MPH, and rally durations to generate player performance reports.
