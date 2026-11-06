# 🏓 Rally Validation Report (Task W2-06)

> **Generated**: `2026-11-06 09:25:57`  
> **Clip Tested**: `match1.mp4`  
> **Frames Analyzed**: `90`  
> **Status**: ✅ **ALL TARGETS PASSED**

---

## 1. Executive Metric Summary

| Criterion | Measured Result | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Ball Tracking Continuity** | **100.0%** (90/90 frames) | $\ge 92.0\%$ | ✅ PASS |
| **Raw YOLO26 Detections** | **92.22%** (83 frames) | Baseline Info | ⚡ Active |
| **Occlusion Interpolation** | **7 frames** | $\le 8\%$ of rally | ✅ Stable |
| **Player ID Switches** | **0 switches** | **0 switches** | ✅ PASS |
| **Net Barrier Crossings** | **0 violations** | **0 violations** | ✅ PASS |
| **Active Player Slots** | **4 canonical slots** | **4 players** | ✅ Stable doubles |
| **Court Reprojection RMSE** | **0.0000 ft** | $< 0.0500\text{ ft}$ | ✅ PASS |
| **JSON Schema Compliance** | **Valid (0 errors)** | **Valid v1.0.0** | ✅ PASS |

---

## 2. Ball Tracking Analysis
- **Total Analyzed Frames**: 90
- **Sustained Localized Frames**: 90 (100.0%)
- **Kinematic Occlusion Fills**: 7 frames recovered via ballistic physics coasting ($g_y=0.55$, $c_d=0.015$).

---

## 3. Player ID Persistence & Bipartite Tracking
- **Tracked Player IDs**: `[1, 2, 3, 4]`
- **ID Switch Events**: **0**
  - No ID switches, partner cross-over flips, or barrier leaps detected across rally.

---

## 4. Contract Schema Verification
- **Unified Perception Schema v1.0.0**: Passed all type constraints and required keys.
