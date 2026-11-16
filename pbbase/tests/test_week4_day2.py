"""
test_week4_day2.py - Automated QA for Week 4 Day 2 (Day 15 / W4-01 completion + W4-02):
Side-Out Tracking, Score Progression, Win-by-2 Completion & Match1 Clip Analysis.
"""
import json
import os
import unittest

from pbbase.rules import MatchScoreTracker, ScoringFSM


class TestWeek4Day2SideOutAndProgression(unittest.TestCase):
    def test_01_multi_inning_side_out_progression_and_counters(self):
        fsm = ScoringFSM(
            initial_serving_team="A",
            team_a_players=("P1", "P2"),
            team_b_players=("P4", "P3"),
        )

        # Opening 0-0-2 -> fault -> Side-Out #1 to Team B (0-0-1)
        e1 = fsm.advance_rally("B")
        self.assertEqual(e1.event_type, "SIDE_OUT")
        self.assertEqual(fsm.score_call, "0-0-1")
        self.assertEqual(fsm.side_out_count, 1)
        self.assertEqual(fsm.half_inning_index, 2)

        # Team B: P4 scores 1 -> 1-0-1 (B swaps: right=P3, left=P4)
        fsm.advance_rally("B")
        self.assertEqual(fsm.score_call, "1-0-1")

        # Team B Server #1 faults -> Server #2 (P3 in RIGHT) -> 1-0-2
        e3 = fsm.advance_rally("A")
        self.assertEqual(e3.event_type, "SECOND_SERVER")
        self.assertEqual(fsm.score_call, "1-0-2")
        self.assertEqual(fsm.active_server_id, "P3")

        # Team B Server #2 faults -> Side-Out #2 to Team A (0-1-1)
        e4 = fsm.advance_rally("A")
        self.assertEqual(e4.event_type, "SIDE_OUT")
        self.assertEqual(fsm.score_call, "0-1-1")
        self.assertEqual(fsm.side_out_count, 2)
        self.assertEqual(fsm.half_inning_index, 3)
        self.assertTrue(fsm.verify_all_invariants())

    def test_02_win_by_two_and_deuce_tiebreaker(self):
        fsm = ScoringFSM(target_score=11, win_by=2)
        # Clear opening 0-0-2 quickly -> Team B serves 0-0-1, push Team A to 10-10 after tiebreak
        # Let's drive Team A directly during opening 0-0-2 up to 10-0-2
        fsm.play_rallies(["A"] * 10)
        self.assertEqual(fsm.score_call, "10-0-2")
        self.assertFalse(fsm.game_over)
        self.assertTrue(fsm.side_switch_triggered)  # Crossed midpoint 6

        # Team A loses -> Side-out to B (0-10-1), B scores 10 straight to reach 10-10-1
        fsm.advance_rally("B")
        self.assertEqual(fsm.score_call, "0-10-1")
        fsm.play_rallies(["B"] * 10)
        self.assertEqual(fsm.score_call, "10-10-1")
        self.assertFalse(fsm.game_over)

        # Team B reaches 11-10 -> NOT game over because margin is 1 (< win_by=2)!
        ev_11_10 = fsm.advance_rally("B")
        self.assertEqual(ev_11_10.event_type, "POINT_SCORED")
        self.assertEqual(fsm.score_call, "11-10-1")
        self.assertFalse(fsm.game_over)

        # Team B scores 12-10 -> margin is 2 -> GAME_WON!
        ev_win = fsm.advance_rally("B")
        self.assertEqual(ev_win.event_type, "GAME_WON")
        self.assertTrue(fsm.game_over)
        self.assertEqual(fsm.winner_team, "B")
        self.assertEqual(fsm.score_call, "12-10-1")
        self.assertTrue(fsm.verify_all_invariants())

        # Advancing past match completion is guarded cleanly
        with self.assertRaises(RuntimeError):
            fsm.advance_rally("B")

    def test_03_undo_last_rally_restores_exact_state(self):
        fsm = ScoringFSM(target_score=11, win_by=2)
        fsm.play_rallies(["A", "A", "B", "B", "A"])  # 2-1-2 for Team B
        snap_before = fsm.to_dict()

        fsm.advance_rally("A")  # Side-out to A -> 2-1-1
        self.assertNotEqual(fsm.to_dict()["serve_context"], snap_before["serve_context"])

        undone = fsm.undo_last_rally()
        self.assertIsNotNone(undone)
        snap_after = fsm.to_dict()
        self.assertEqual(snap_after["serve_context"], snap_before["serve_context"])
        self.assertEqual(snap_after["teams"], snap_before["teams"])
        self.assertEqual(snap_after["progression"], snap_before["progression"])

    def test_04_match1_telemetry_end_to_end_day2_progression(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        telem_path = os.path.join(repo_root, "data", "match1_full_telemetry.json")
        if not os.path.isfile(telem_path):
            self.skipTest("data/match1_full_telemetry.json not available locally")

        with open(telem_path, "r") as f:
            telem = json.load(f)

        tracker = MatchScoreTracker()
        res = tracker.analyze_clip_telemetry(telem)

        self.assertEqual(res["total_frames"], 289)
        self.assertEqual(res["serve_bounce_frame"], 26)
        self.assertTrue(res["serve_valid_diagonal"])
        self.assertEqual(res["terminal_event_frame"], 263)

        frames = res["frame_rules"]
        # Frame 0..262: Team A serving 0-0-2 (P1 RIGHT -> P4 RIGHT)
        self.assertEqual(frames[0]["score_call"], "0-0-2")
        self.assertEqual(frames[0]["serving_team"], "A")
        self.assertEqual(frames[0]["active_server_id"], "P1")
        self.assertEqual(frames[0]["rally_state"], "SERVE_IN_FLIGHT")
        self.assertEqual(frames[26]["rally_state"], "RALLY_LIVE")

        # Frame 263: Out-of-bounds ground bounce [23.89, 0.2] triggers SIDE-OUT -> 0-0-1 (Team B P4)
        self.assertEqual(frames[263]["score_call"], "0-0-1")
        self.assertEqual(frames[263]["serving_team"], "B")
        self.assertEqual(frames[263]["active_server_id"], "P4")
        self.assertEqual(frames[263]["rally_state"], "RALLY_ENDED_SIDE_OUT")
        self.assertEqual(frames[263]["side_out_count"], 1)
        self.assertIsNotNone(frames[263]["event_emitted"])
        self.assertEqual(frames[263]["event_emitted"]["event_type"], "SIDE_OUT")


if __name__ == "__main__":
    unittest.main()
