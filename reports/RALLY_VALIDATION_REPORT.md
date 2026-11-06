# 🏓 Rally Validation Report (Task W2-06)

> **Generated**: `2026-11-06 09:36:14`  
> **Clip Tested**: `match1.mp4`  
> **Frames Analyzed**: `289`  
> **Status**: ⚠️ **ACTION REQUIRED**

---

## 1. Executive Metric Summary

| Criterion | Measured Result | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Ball Tracking Continuity** | **91.0%** (263/289 frames) | $\ge 92.0\%$ | ❌ FAIL |
| **Raw YOLO26 Detections** | **78.2%** (226 frames) | Baseline Info | ⚡ Active |
| **Occlusion Interpolation** | **37 frames** | $\le 8\%$ of rally | ✅ Stable |
| **Player ID Switches** | **0 switches** | **0 switches** | ✅ PASS |
| **Net Barrier Crossings** | **0 violations** | **0 violations** | ✅ PASS |
| **Active Player Slots** | **4 canonical slots** | **4 players** | ✅ Stable doubles |
| **Court Reprojection RMSE** | **0.0000 ft** | $< 0.0500\text{ ft}$ | ✅ PASS |
| **JSON Schema Compliance** | **Valid (0 errors)** | **Valid v1.0.0** | ✅ PASS |

---

## 2. Ball Tracking Analysis
- **Total Analyzed Frames**: 289
- **Sustained Localized Frames**: 263 (91.0%)
- **Kinematic Occlusion Fills**: 37 frames recovered via ballistic physics coasting ($g_y=0.55$, $c_d=0.015$).

---

## 3. Player ID Persistence & Bipartite Tracking
- **Tracked Player IDs**: `[1, 2, 3, 4]`
- **ID Switch Events**: **0**
  - No ID switches, partner cross-over flips, or barrier leaps detected across rally.

---

## 4. Contract Schema Verification
- **Unified Perception Schema v1.0.0**: Passed all type constraints and required keys.
