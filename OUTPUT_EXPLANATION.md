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

The telemetry output strictly adheres to **Unified Perception Schema v1.0.0** ([`pbbase/data/schema.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/data/schema.py)). It exposes master court geometry, benchmark metrics, and a synchronized frame-by-frame stream (`telemetry_stream`).

### Example Telemetry v1.0.0 Record
```json
{
  "schema_version": "1.0.0",
  "producer": "Pickleball Base Perception Engine (Zero-Rule Core)",
  "timestamp": "2026-11-05 10:44:50",
  "video_path": "clips/match1.mp4",
  "total_frames_processed": 90,
  "fps": 30.0,
  "resolution": "1302x740",
  "elapsed_s": 19.59,
  "processing_fps": 4.59,
  "court": {
    "calibrated": true,
    "reprojection_rmse_ft": 0.0,
    "dimensions_ft": {
      "width": 20.0,
      "length": 44.0,
      "net_y": 22.0,
      "nvz_depth": 7.0
    },
    "corners_px": [
      [362.0, 168.0],
      [738.0, 168.0],
      [1126.0, 638.0],
      [14.0, 638.0]
    ]
  },
  "benchmark_summary": {
    "ball_detection_rate_pct": 100.0,
    "frames_with_ball": 90,
    "mean_players_per_frame": 4.0,
    "multiscale_inference": true,
    "stage_latencies_ms": {
      "player_ms": 37.6,
      "ball_ms": 169.9,
      "court_ms": 0.0,
      "render_ms": 7.8
    }
  },
  "telemetry_stream": [
    {
      "frame_idx": 42,
      "timestamp_s": 1.401,
      "court_calibrated": true,
      "players": [
        {
          "id": 1,
          "slot": "P1",
          "box": [768.4, 340.0, 889.5, 604.3],
          "foot_px": [829.0, 604.3],
          "court_ft": [14.22, 44.12],
          "speed_mph": 3.4,
          "is_coasting": false,
          "side": "near"
        },
        {
          "id": 2,
          "slot": "P2",
          "box": [374.2, 341.2, 438.5, 549.7],
          "foot_px": [406.4, 549.7],
          "court_ft": [4.26, 41.17],
          "speed_mph": 2.8,
          "is_coasting": false,
          "side": "near"
        }
      ],
      "ball": {
        "pixel": [582.4, 310.1],
        "box": [576.4, 304.1, 588.4, 316.1],
        "conf": 0.86,
        "speed_px_per_frame": 18.4,
        "speed_mph": 28.4,
        "court_ft": [9.85, 21.42],
        "interpolated": false
      }
    }
  ]
}
```

### Telemetry Field Definitions

#### Top-Level Metadata & Court Geometry
- `schema_version`: Semantic schema version string (`"1.0.0"`).
- `producer`: Pipeline identification string.
- `court`: Full geometric court specifications:
  - `calibrated`: Boolean indicating active camera calibration.
  - `reprojection_rmse_ft`: Calibration reprojection error in feet.
  - `dimensions_ft`: Official USAP specifications (width: $20'$, length: $44'$, net: $22'$, NVZ: $7'$).
  - `corners_px`: The 4 physical baseline court corners in image coordinates.
- `benchmark_summary`: Aggregate perception metrics across the rally:
  - `ball_detection_rate_pct`: Percentage of frames with confirmed ball track.
  - `mean_players_per_frame`: Average active players tracked (target: 4.0).
  - `stage_latencies_ms`: Microsecond/millisecond breakdown for Player, Ball, Court, and Rendering.

#### Per-Frame Stream (`telemetry_stream[i]`)
- `frame_idx`: Zero-based integer frame counter.
- `timestamp_s`: Exact video elapsed time in seconds.
- `court_calibrated`: Boolean verification for frame $i$.
- `players`: Array of active players detected within the athletic envelope:
  - `id`: Integer canonical slot ($1, 2, 3, 4$).
  - `slot`: Canonical label (`"P1"`, `"P2"` for near court; `"P3"`, `"P4"` for far court).
  - `box`: `[x1, y1, x2, y2]` in video pixel coordinates.
  - `foot_px`: `[x, y]` pixel location of the player's ground contact.
  - `court_ft`: `[X, Y]` coordinates in feet mapped to official court dimensions:
    - $X \in [0.0, 20.0]$: Left to right court width.
    - $Y \in [0.0, 44.0]$: Far baseline ($Y=0$) to near baseline ($Y=44$). Kitchen net line is at $Y=22.0$.
  - `speed_mph`: Instantaneous physical foot ground speed in MPH.
  - `is_coasting`: Boolean flag indicating brief occlusion coasting.
  - `side`: Court partition (`"near"` or `"far"`).
- `ball`: Object containing localized pickleball metrics (or `null` if lost):
  - `pixel`: `[x, y]` centroid in video coordinates.
  - `box`: `[x1, y1, x2, y2]` bounding coordinates.
  - `conf`: Confidence score ($0.0\text{--}1.0$).
  - `speed_px_per_frame`: Euclidean pixel velocity.
  - `speed_mph`: Metric 2D velocity calculated via planar homography.
  - `court_ft`: `[X, Y]` ground projection in feet.
  - `interpolated`: Boolean flag indicating ballistic occlusion coasting.
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
