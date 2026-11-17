# Week 4 — Day 3 (Day 16): Two-Bounce Rule Enforcer & Kitchen (NVZ) Foot-Fault Detector (`W4-03` & `W4-04`)

**Schedule**: Day 16 (Dec 01, 2026) | **Tasks Completed**: `W4-03` (Two-Bounce Rule Enforcer) & `W4-04` (Kitchen / NVZ Foot-Fault Detector)

---

## Architecture Delivered on Day 3

1. **[`pbbase/rules/two_bounce.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/two_bounce.py)** (`W4-03` — USAP Rule 7.A / 4.A.2):
   - Implemented [`TwoBounceEnforcer`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/two_bounce.py) with state progression: `AWAITING_SERVE_BOUNCE` $\rightarrow$ `SERVE_BOUNCED_AWAITING_RETURN` $\rightarrow$ `AWAITING_RETURN_BOUNCE` $\rightarrow$ `TWO_BOUNCE_SATISFIED` (or `MID_RALLY_OPEN` via [`infer_serve_start_from_telemetry()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/two_bounce.py) for mid-rally clips).
   - Catches premature unbounced volleys as **`SERVE_VOLLEY_FAULT`** (receiving half) or **`RETURN_VOLLEY_FAULT`** (serving half), while distinguishing low groundstroke bounce pickups (`height_ft <= 1.95 ft`) from genuine airborne volleys.

2. **[`pbbase/rules/nvz_detector.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/nvz_detector.py)** (`W4-04` — USAP Rule 9 Non-Volley Zone):
   - Implemented [`NVZFootFaultDetector`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/nvz_detector.py) and [`compute_nvz_clearance_ft()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/nvz_detector.py) evaluating player ground foot projections against `Y = 15.0 ft` (Far NVZ) and `Y = 29.0 ft` (Near NVZ), including exact line-touch checks (`clearance <= 0.0 ft`).
   - Enforces all three Rule 9 violation classes:
     - **`NVZ_DIRECT_VOLLEY_FAULT`** (Rule 9.A): Volleying while touching or standing inside the NVZ.
     - **`NVZ_MOMENTUM_FAULT`** (Rule 9.B): Stepping into/onto the NVZ due to momentum within `momentum_window_frames` after executing an airborne volley.
     - **`NVZ_NO_RESET_FAULT`** (Rule 9.D): Re-entering from the NVZ without establishing `reset_frames_required` consecutive frames cleanly outside before volleying.
   - Separates legal bounced-ball kitchen entries (`IN_KITCHEN_LEGAL`) from volley infractions.

3. **[`pbbase/rules/match_tracker.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/match_tracker.py)** & **[`pbbase/rules/__init__.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/__init__.py)**:
   - Integrated `TwoBounceEnforcer` + `NVZFootFaultDetector` + `ScoringFSM` in [`MatchScoreTracker.analyze_clip_telemetry()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/match_tracker.py), picking the earliest rule violation or OOB event to advance USAP scoring automatically, and scanning forward when frame 0 has scoreboard occlusion so partial-frame clips seed cleanly.

4. **[`scripts/render_week4_day3.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/scripts/render_week4_day3.py)** (Output Videos in `outputs/`, No Minimap Overlay):
   - [`outputs/match1_week4_day3.mp4`](file:///home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_week4_day3.mp4) (`289` frames): Tracks `0-0-2` serve (`f.26` serve bounce IN $\rightarrow$ `f.62` return bounce / pickup clears 2-Bounce `2/2` $\rightarrow$ `P3` NVZ edge watch $\rightarrow$ `f.263` OOB `SIDE-OUT` to `0-0-1`).
   - [`outputs/match_mlp_week4_day3.mp4`](file:///home/pce118/Desktop/pickleball-vision-ai-main/outputs/match_mlp_week4_day3.mp4) (`300` frames): Tracks `f.83` serve bounce IN (`1/2`) $\rightarrow$ `f.177` Far NVZ-line ground bounce (`2/2` Volleys Open) and `P3`/`P4` legal kitchen occupancy (`IN KITCHEN LEGAL`).
   - [`outputs/cut_week4_day3.mp4`](file:///home/pce118/Desktop/pickleball-vision-ai-main/outputs/cut_week4_day3.mp4) (`295` frames, portrait `1440x2560`): Identifies mid-rally clip start (`MID_RALLY_OPEN`) and highlights `P1` sprinting into Near Kitchen (`f.264–294`) as `IN_KITCHEN_LEGAL` directly over the court floor zone.

---

## Remaining Week 4 Schedule (Day-Wise)

| Day | Tasks | Planned Deliverable |
| :--- | :--- | :--- |
| **Day 17** *(Next)* | `W4-05` | Precision IN / OUT line caller with boundary clearances & millimeter tolerances |
| **Day 18** | `W4-06` | Unified autonomous referee validation (`0-0-2` to win-by-2 match completion) & final report |
