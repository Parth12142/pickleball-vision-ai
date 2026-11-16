# Week 4 — Day 2 (Day 15): Side-Out Tracking & Score Progression (`W4-01` Complete + `W4-02`)

**Schedule**: Day 15 (Nov 30, 2026) | **Tasks Completed**: `W4-01` (Finished) & `W4-02` (Side-Out Tracking & Score Progression)

---

## Architecture Delivered on Day 2

1. **[`pbbase/rules/scoring_fsm.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/scoring_fsm.py)** (`W4-01` complete + `W4-02` core progression):
   - Added win-by-2 game completion logic (`target_score` = 11/15/21, `win_by` = 2) with explicit `GAME_WON` state and post-match guardrails.
   - Added regulation midpoint side-switch flag (`side_switch_score = (target_score // 2) + 1`, i.e., `6` pts in an 11-pt game).
   - Added progression metrics (`rally_count`, `side_out_count`, `half_inning_index`, `current_server_streak`), batch rally replay [`play_rallies()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/scoring_fsm.py), and full checkpoint rollback via [`undo_last_rally()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/scoring_fsm.py).
   - Enhanced [`seed_from_telemetry_players()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/scoring_fsm.py) with `initial_ball_court_ft` origin disambiguation so real match clips accurately lock onto the side physically serving on Frame 0.

2. **[`pbbase/rules/match_tracker.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/match_tracker.py)** (`W4-02` telemetry stream integration):
   - Implemented [`MatchScoreTracker`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/match_tracker.py) and [`FrameRuleSnapshot`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/rules/match_tracker.py) to emit synchronized frame-by-frame USAP scorebug state (`0..N` frames) directly from Weeks 1–3 telemetry JSON.

3. **[`scripts/render_week4_day2.py`](file:///home/pce118/Desktop/pickleball-vision-ai-main/scripts/render_week4_day2.py)** & [`encode_h264()`](file:///home/pce118/Desktop/pickleball-vision-ai-main/pbbase/pipeline.py#L484-L510):
   - Half-way visual verification output rendered on `clips/match1.mp4` (`289` frames) -> [`outputs/match1_week4_day2.mp4`](file:///home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_week4_day2.mp4) and [`outputs/match1_week4_day2_telemetry.json`](file:///home/pce118/Desktop/pickleball-vision-ai-main/outputs/match1_week4_day2_telemetry.json).
   - **Verified progression on `match1.mp4`**:
     - `f.0` (`t=0.00s`): **Team A (`P1` RIGHT, `[14.36, 43.83]ft`)** serves `0-0-2` diagonally toward **Team B (`P4` RIGHT, target `[0..10, 0..15]ft`)**.
     - `f.26` (`t=0.87s`): Diagonal serve bounces IN at **`[2.13, 8.33]ft`** inside `B RIGHT`.
     - `f.27–f.262`: Active rally across 7 contacts (`0-0-2`).
     - `f.263` (`t=8.76s`): Ball exits out-of-bounds wide right at **`[23.89, 0.20]ft`**, triggering **`SIDE-OUT`** -> **`0-0-1` (Team B `P4` serving next)**.

---

## Remaining Week 4 Schedule (Day-Wise)

| Day | Tasks | Planned Deliverable |
| :--- | :--- | :--- |
| **Day 16** *(Next)* | `W4-03`, `W4-04` | Two-bounce rule enforcer + Kitchen (NVZ) volley foot-fault detector |
| **Day 17** | `W4-05` | Precision IN / OUT line caller with boundary clearances & millimeter tolerances |
| **Day 18** | `W4-06` | Full unified autonomous referee validation (`0-0-2` to win-by-2 completion) & final report |
