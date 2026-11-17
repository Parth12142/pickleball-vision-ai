"""
test_week4_day3.py - Automated QA for Week 4 Day 3 (Day 16 / W4-03 & W4-04):
  - W4-03: Official USAP Two-Bounce Rule Enforcer (Serve Bounce + Return Bounce + Violations)
  - W4-04: Kitchen (NVZ) Foot-Fault Detector (Direct, Momentum, Re-entry Reset & Legal Entry)
  - Integration with MatchScoreTracker & Real Match Telemetry (match1, match_mlp, cut)
"""
import json
import os
import unittest
from typing import Any, Dict, List, Optional

from pbbase.rules import (
    MatchScoreTracker,
    NVZFootFaultDetector,
    TwoBounceEnforcer,
    compute_nvz_clearance_ft,
)


def _make_players(
    p1=(15.0, 40.0),
    p2=(5.0, 34.0),
    p3=(14.0, 12.0),
    p4=(4.0, 4.0),
) -> List[Dict[str, Any]]:
    return [
        {"id": 1, "slot": "P1", "side": "near", "court_ft": list(p1), "box": [700, 350, 780, 520]},
        {"id": 2, "slot": "P2", "side": "near", "court_ft": list(p2), "box": [350, 310, 430, 480]},
        {"id": 3, "slot": "P3", "side": "far", "court_ft": list(p3), "box": [650, 120, 710, 240]},
        {"id": 4, "slot": "P4", "side": "far", "court_ft": list(p4), "box": [420, 80, 480, 190]},
    ]


def _make_stream_template(
    n_frames: int = 120,
    ball_y_by_frame: Optional[Dict[int, float]] = None,
    ball_h_by_frame: Optional[Dict[int, float]] = None,
    players_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    bounces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    stream = []
    b_map = {b["frame_idx"]: b for b in (bounces or [])}
    last_y = 40.0
    for fi in range(n_frames):
        if ball_y_by_frame and fi in ball_y_by_frame:
            last_y = ball_y_by_frame[fi]
        h_ft = ball_h_by_frame.get(fi, 3.0) if ball_h_by_frame else 3.0
        pls = players_by_frame.get(fi) if (players_by_frame and fi in players_by_frame) else _make_players()
        rec: Dict[str, Any] = {
            "frame_idx": fi,
            "timestamp_s": fi / 30.0,
            "players": pls,
            "ball": {
                "frame_idx": fi,
                "pixel": [600.0, 300.0],
                "court_ft": [5.0, last_y],
                "height_ft": h_ft,
                "speed_px_per_frame": 12.0,
            },
        }
        if fi in b_map:
            rec["bounce"] = b_map[fi]
        stream.append(rec)
    return {
        "video_path": "synthetic.mp4",
        "fps": 30.0,
        "telemetry_stream": stream,
        "bounces": bounces or [],
    }


class TestWeek4Day3TwoBounceAndNVZ(unittest.TestCase):
    # ------------------------------------------------------------------
    # W4-03: Two-Bounce Rule Enforcer Tests
    # ------------------------------------------------------------------
    def test_01_two_bounce_legal_progression_and_unlock_volley(self):
        b_map_y = {0: 42.0, 15: 18.0, 26: 8.0, 45: 25.0, 62: 33.0, 80: 14.0, 95: 25.0}
        bounces = [
            {"frame_idx": 26, "court_ft": [4.0, 8.0], "is_ground_bounce": True},   # Serve bounce (Team B)
            {"frame_idx": 62, "court_ft": [5.0, 33.0], "is_ground_bounce": True},  # Return bounce (Team A)
            {"frame_idx": 95, "court_ft": [5.0, 25.0], "is_ground_bounce": False}, # 4th shot volley OK
        ]
        telem = _make_stream_template(105, ball_y_by_frame=b_map_y, bounces=bounces)
        res = TwoBounceEnforcer.analyze_telemetry(telem, serving_team="A", assume_serve_start=True)

        self.assertEqual(res["serve_bounce_frame"], 26)
        self.assertEqual(res["return_bounce_frame"], 62)
        self.assertEqual(res["satisfied_frame"], 62)
        self.assertEqual(res["bounces_completed"], 2)
        self.assertEqual(res["final_phase"], "TWO_BOUNCE_SATISFIED")
        self.assertIsNone(res["fault_event"])

    def test_02_two_bounce_serve_volley_fault_updates_score(self):
        # Team A serves 0-0-2; Team B P4 volleys in mid-air at f.22 before serve bounce!
        b_map_y = {0: 42.0, 14: 18.0, 22: 7.0}
        b_h = {22: 3.4}
        bounces = [
            {"frame_idx": 22, "court_ft": [4.5, 7.0], "is_ground_bounce": False},
        ]
        telem = _make_stream_template(50, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, bounces=bounces)
        res = MatchScoreTracker().analyze_clip_telemetry(telem, assume_serve_start=True)

        self.assertEqual(res["terminal_event_frame"], 22)
        self.assertEqual(res["terminal_fault_type"], "SERVE_VOLLEY_FAULT")
        self.assertIsNotNone(res["two_bounce_summary"]["fault_event"])
        self.assertEqual(res["two_bounce_summary"]["fault_event"]["faulting_team"], "B")
        self.assertEqual(res["two_bounce_summary"]["fault_event"]["rally_winner_team"], "A")
        # Team A was serving at 0-0-2 -> fault on B awards 1 pt to Team A -> 1-0-2
        self.assertEqual(res["fsm_summary"]["serve_context"]["score_call"], "1-0-2")

    def test_03_two_bounce_return_volley_fault_triggers_side_out(self):
        # Serve bounces at f.26 on Team B; return crosses at f.44 to Team A; Team A P2 volleys high at f.58 before bounce!
        b_map_y = {0: 42.0, 14: 18.0, 26: 8.0, 44: 26.0, 58: 31.0}
        b_h = {26: 0.1, 58: 3.2}
        bounces = [
            {"frame_idx": 26, "court_ft": [4.0, 8.0], "is_ground_bounce": True},
            {"frame_idx": 58, "court_ft": [5.0, 31.0], "is_ground_bounce": False},
        ]
        telem = _make_stream_template(80, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, bounces=bounces)
        res = MatchScoreTracker().analyze_clip_telemetry(telem, assume_serve_start=True)

        self.assertEqual(res["terminal_event_frame"], 58)
        self.assertEqual(res["terminal_fault_type"], "RETURN_VOLLEY_FAULT")
        self.assertEqual(res["two_bounce_summary"]["fault_event"]["faulting_team"], "A")
        self.assertEqual(res["two_bounce_summary"]["fault_event"]["rally_winner_team"], "B")
        # Opening 0-0-2 fault on serving Team A triggers immediate SIDE-OUT -> 0-0-1
        self.assertEqual(res["fsm_summary"]["serve_context"]["score_call"], "0-0-1")

    def test_04_low_ground_pickup_satisfies_two_bounce_without_false_fault(self):
        # Simulate occluded low pickup near floor (is_ground_bounce=False, height_ft=1.7) on return half
        b_map_y = {0: 42.0, 14: 18.0, 26: 8.0, 45: 25.0, 62: 33.5}
        b_h = {26: 0.1, 62: 1.7}
        bounces = [
            {"frame_idx": 26, "court_ft": [4.0, 8.0], "is_ground_bounce": True},
            {"frame_idx": 62, "court_ft": [5.0, 33.5], "is_ground_bounce": False},
        ]
        telem = _make_stream_template(80, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, bounces=bounces)
        res = TwoBounceEnforcer.analyze_telemetry(telem, serving_team="A", assume_serve_start=True)

        self.assertEqual(res["final_phase"], "TWO_BOUNCE_SATISFIED")
        self.assertEqual(res["bounces_completed"], 2)
        self.assertIsNone(res["fault_event"])

    # ------------------------------------------------------------------
    # W4-04: Kitchen (NVZ) Foot-Fault Detector Tests
    # ------------------------------------------------------------------
    def test_05_nvz_boundary_and_line_touch_precision(self):
        # Touching Y=29.0 or Y=15.0 IS inside NVZ per USAP Rule 9.A
        half_a_line, clr_a_line, in_a_line = compute_nvz_clearance_ft((10.0, 29.0))
        self.assertEqual(half_a_line, "A")
        self.assertAlmostEqual(clr_a_line, 0.0)
        self.assertTrue(in_a_line)

        _, clr_a_out, in_a_out = compute_nvz_clearance_ft((10.0, 29.4))
        self.assertAlmostEqual(clr_a_out, 0.4)
        self.assertFalse(in_a_out)

        half_b_line, clr_b_line, in_b_line = compute_nvz_clearance_ft((10.0, 15.0))
        self.assertEqual(half_b_line, "B")
        self.assertAlmostEqual(clr_b_line, 0.0)
        self.assertTrue(in_b_line)

    def test_06_nvz_direct_volley_foot_fault_triggers_side_out(self):
        # After 2 bounces (f.15, f.30), Team A P2 steps onto Kitchen (Y=27.8) and volleys at f.45!
        pls_by_f = {fi: _make_players(p2=(5.0, 27.8)) for fi in range(40, 55)}
        b_map_y = {0: 42.0, 10: 18.0, 15: 8.0, 25: 25.0, 30: 33.0, 45: 27.0}
        b_h = {15: 0.1, 30: 0.1, 45: 3.1}
        bounces = [
            {"frame_idx": 15, "court_ft": [4.0, 8.0], "is_ground_bounce": True},
            {"frame_idx": 30, "court_ft": [5.0, 33.0], "is_ground_bounce": True},
            {"frame_idx": 45, "court_ft": [5.0, 27.0], "is_ground_bounce": False},
        ]
        telem = _make_stream_template(65, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, players_by_frame=pls_by_f, bounces=bounces)
        res = MatchScoreTracker().analyze_clip_telemetry(telem, assume_serve_start=True)

        self.assertTrue(res["nvz_summary"]["fault_detected"])
        self.assertEqual(res["terminal_fault_type"], "NVZ_DIRECT_VOLLEY_FAULT")
        self.assertEqual(res["terminal_event_frame"], 45)
        self.assertEqual(res["nvz_summary"]["fault_event"]["offending_player_id"], "P2")
        self.assertGreater(res["nvz_summary"]["fault_event"]["nvz_penetration_ft"], 1.0)
        # Serving Team A foot-faulted at 0-0-2 -> SIDE-OUT -> 0-0-1
        self.assertEqual(res["fsm_summary"]["serve_context"]["score_call"], "0-0-1")

    def test_07_nvz_momentum_foot_fault_rule_9b(self):
        # P2 stands outside at Y=30.0 during volley at f.45, then momentum carries foot to Y=28.2 at f.53!
        pls_by_f = {}
        for fi in range(50, 62):
            pls_by_f[fi] = _make_players(p2=(5.0, 28.2))
        b_map_y = {0: 42.0, 10: 18.0, 15: 8.0, 25: 25.0, 30: 33.0, 45: 30.2}
        b_h = {15: 0.1, 30: 0.1, 45: 3.4}
        bounces = [
            {"frame_idx": 15, "court_ft": [4.0, 8.0], "is_ground_bounce": True},
            {"frame_idx": 30, "court_ft": [5.0, 33.0], "is_ground_bounce": True},
            {"frame_idx": 45, "court_ft": [5.0, 30.2], "is_ground_bounce": False},
        ]
        telem = _make_stream_template(70, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, players_by_frame=pls_by_f, bounces=bounces)
        nvz_res = NVZFootFaultDetector.analyze_telemetry(telem)

        self.assertTrue(nvz_res["fault_detected"])
        fe = nvz_res["fault_event"]
        self.assertEqual(fe["fault_type"], "NVZ_MOMENTUM_FAULT")
        self.assertEqual(fe["volley_frame_idx"], 45)
        self.assertEqual(fe["frame_idx"], 50)
        self.assertEqual(fe["offending_player_id"], "P2")

    def test_08_nvz_no_reset_reentry_fault_rule_9d(self):
        # P2 is in NVZ at f.40..43, steps out for ONLY 1 frame at f.44 (< 2 required!), volleys at f.45 -> FAULT
        pls_by_f = {}
        for fi in range(40, 44):
            pls_by_f[fi] = _make_players(p2=(5.0, 27.5))
        pls_by_f[44] = _make_players(p2=(5.0, 29.8))  # Only 1 outside frame (both feet not reset cleanly)
        pls_by_f[45] = _make_players(p2=(5.0, 29.8))  # Still outside physically at contact frame
        b_map_y = {45: 29.5}
        b_h = {45: 3.3}
        bounces = [{"frame_idx": 45, "court_ft": [5.0, 29.5], "is_ground_bounce": False}]

        telem = _make_stream_template(55, ball_y_by_frame=b_map_y, ball_h_by_frame=b_h, players_by_frame=pls_by_f, bounces=bounces)
        nvz_res = NVZFootFaultDetector.analyze_telemetry(telem, reset_frames_required=2)

        self.assertTrue(nvz_res["fault_detected"])
        self.assertEqual(nvz_res["fault_event"]["fault_type"], "NVZ_NO_RESET_FAULT")
        self.assertEqual(nvz_res["fault_event"]["offending_player_id"], "P2")

    def test_09_legal_kitchen_entry_after_bounce_no_fault(self):
        # P2 steps inside NVZ (Y=27.5) after ball bounced at f.45 -> legal groundstroke!
        pls_by_f = {fi: _make_players(p2=(5.0, 27.5)) for fi in range(42, 55)}
        bounces = [{"frame_idx": 45, "court_ft": [5.0, 26.5], "is_ground_bounce": True}]
        telem = _make_stream_template(60, players_by_frame=pls_by_f, bounces=bounces)
        nvz_res = NVZFootFaultDetector.analyze_telemetry(telem)

        self.assertFalse(nvz_res["fault_detected"])
        self.assertGreaterEqual(nvz_res["max_concurrent_nvz_players"], 1)
        self.assertIn("P2", nvz_res["frame_states"][45]["players_in_nvz"])

    def test_10_real_clips_day3_rules_integration(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        checks = [
            ("data/match1_full_telemetry.json", "TWO_BOUNCE_SATISFIED", 263, "0-0-1"),
            ("data/match2_telemetry.json", "TWO_BOUNCE_SATISFIED", None, "0-0-2"),
            ("data/match_mlp_telemetry.json", "TWO_BOUNCE_SATISFIED", 234, "1-0-2"),
            ("data/cut_telemetry.json", "TWO_BOUNCE_SATISFIED", None, "0-0-2"),
            ("data/screencast_telemetry.json", "TWO_BOUNCE_SATISFIED", 593, "0-0-1"),
        ]
        for rel_p, exp_tb_phase, exp_term_f, exp_score in checks:
            full_p = os.path.join(repo_root, rel_p)
            if not os.path.isfile(full_p):
                continue
            with open(full_p, "r") as f:
                telem = json.load(f)
            res = MatchScoreTracker().analyze_clip_telemetry(telem)
            self.assertEqual(res["two_bounce_summary"]["final_phase"], exp_tb_phase, msg=rel_p)
            self.assertEqual(res["terminal_event_frame"], exp_term_f, msg=rel_p)
            self.assertEqual(res["fsm_summary"]["serve_context"]["score_call"], exp_score, msg=rel_p)
            self.assertFalse(res["nvz_summary"]["fault_detected"], msg=rel_p)

    def test_11_six_part_court_and_savgol_exact_timing(self):
        from pbbase.court.broadcast_fx import get_court_6part_name, get_court_subzone_rect
        from pbbase.court.homography import CourtCalibrator
        from pbbase.tracking.bounce_detector import SavitzkyGolayBounceAnalyzer

        # Verify all 6 canonical USAP court parts
        self.assertEqual(get_court_6part_name(5.0, 8.0), "FAR_RIGHT_SERVICE")
        self.assertEqual(get_court_subzone_rect(5.0, 8.0), (0.0, 10.0, 0.0, 15.0))
        self.assertEqual(get_court_6part_name(15.0, 8.0), "FAR_LEFT_SERVICE")
        self.assertEqual(get_court_subzone_rect(15.0, 8.0), (10.0, 20.0, 0.0, 15.0))
        # Far Kitchen and Near Kitchen span full 20ft width (no centerline)
        self.assertEqual(get_court_6part_name(4.0, 18.0), "FAR_KITCHEN_NVZ")
        self.assertEqual(get_court_6part_name(16.0, 18.0), "FAR_KITCHEN_NVZ")
        self.assertEqual(get_court_subzone_rect(4.0, 18.0), (0.0, 20.0, 15.0, 22.0))
        self.assertEqual(get_court_6part_name(4.0, 25.0), "NEAR_KITCHEN_NVZ")
        self.assertEqual(get_court_6part_name(16.0, 25.0), "NEAR_KITCHEN_NVZ")
        self.assertEqual(get_court_subzone_rect(16.0, 25.0), (0.0, 20.0, 22.0, 29.0))
        self.assertEqual(get_court_6part_name(5.0, 36.0), "NEAR_LEFT_SERVICE")
        self.assertEqual(get_court_subzone_rect(5.0, 36.0), (0.0, 10.0, 29.0, 44.0))
        self.assertEqual(get_court_6part_name(15.0, 36.0), "NEAR_RIGHT_SERVICE")
        self.assertEqual(get_court_subzone_rect(15.0, 36.0), (10.0, 20.0, 29.0, 44.0))
        self.assertIsNone(get_court_6part_name(23.5, 1.0))

        # Verify SavitzkyGolayBounceAnalyzer zero-lag observed pixel preservation
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        m1_path = os.path.join(repo_root, "data/match1_full_telemetry.json")
        if os.path.isfile(m1_path):
            with open(m1_path, "r") as f:
                telem = json.load(f)
            corners = [tuple(pt) for pt in telem["court"]["corners_px"]]
            cal = CourtCalibrator(corners_px=corners)
            sg = SavitzkyGolayBounceAnalyzer().analyze_stream(telem["telemetry_stream"], cal, fps=30.0)
            b_frames = [b["frame_idx"] for b in sg["bounces"]]
            self.assertIn(b_frames, ([25, 91, 173, 201, 263], [25, 91, 146, 173, 201, 263]))


if __name__ == "__main__":
    unittest.main()
