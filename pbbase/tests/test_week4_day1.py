"""
test_week4_day1.py - Automated QA for Week 4 Day 1 (Day 14 / W4-01 Core):
USAP 0-0-2 Scoring Finite State Machine & Serving Sequence.
"""
import json
import os
import unittest

from pbbase.rules import ScoringFSM, SERVICE_QUADRANT_BOUNDS_FT


class TestWeek4Day1ServingFSM(unittest.TestCase):
    def test_01_initial_0_0_2_opening_state(self):
        fsm = ScoringFSM(
            initial_serving_team="A",
            team_a_players=("P1", "P2"),
            team_b_players=("P4", "P3"),
        )
        ctx = fsm.get_serve_context()

        self.assertEqual(ctx.score_call, "0-0-2")
        self.assertEqual(ctx.serving_team, "A")
        self.assertEqual(ctx.receiving_team, "B")
        self.assertEqual(ctx.server_number, 2)
        self.assertTrue(ctx.is_opening_half_inning)
        self.assertEqual(ctx.active_server_id, "P1")
        self.assertEqual(ctx.active_receiver_id, "P4")
        self.assertEqual(ctx.server_box, "RIGHT")
        self.assertEqual(ctx.target_receiver_box, "RIGHT")
        self.assertEqual(ctx.target_quadrant_ft, SERVICE_QUADRANT_BOUNDS_FT[("B", "RIGHT")])
        self.assertTrue(fsm.verify_all_invariants())

    def test_02_opening_0_0_2_streak_swaps_courts_without_changing_server_num(self):
        fsm = ScoringFSM()

        e1 = fsm.advance_rally("A")
        ctx1 = fsm.get_serve_context()
        self.assertEqual(e1.event_type, "POINT_SCORED")
        self.assertEqual(ctx1.score_call, "1-0-2")
        self.assertEqual(ctx1.active_server_id, "P1")
        self.assertEqual(ctx1.server_box, "LEFT")
        self.assertEqual(ctx1.active_receiver_id, "P3")  # B's Left player diagonally across from A's Left
        self.assertTrue(fsm.verify_all_invariants())

        e2 = fsm.advance_rally("A")
        ctx2 = fsm.get_serve_context()
        self.assertEqual(e2.event_type, "POINT_SCORED")
        self.assertEqual(ctx2.score_call, "2-0-2")
        self.assertEqual(ctx2.active_server_id, "P1")
        self.assertEqual(ctx2.server_box, "RIGHT")
        self.assertEqual(ctx2.active_receiver_id, "P4")
        self.assertTrue(fsm.verify_all_invariants())

    def test_03_opening_0_0_2_fault_triggers_side_out_to_0_0_1(self):
        fsm = ScoringFSM()
        ev = fsm.advance_rally("B")
        ctx = fsm.get_serve_context()

        self.assertEqual(ev.event_type, "SIDE_OUT")
        self.assertEqual(ev.prev_score_call, "0-0-2")
        self.assertEqual(ctx.score_call, "0-0-1")
        self.assertFalse(ctx.is_opening_half_inning)
        self.assertEqual(ctx.serving_team, "B")
        self.assertEqual(ctx.server_number, 1)
        self.assertEqual(ctx.active_server_id, "P4")  # Team B Right player
        self.assertEqual(ctx.server_box, "RIGHT")
        self.assertEqual(ctx.active_receiver_id, "P1")  # Diagonal Team A Right player
        self.assertTrue(fsm.verify_all_invariants())

    def test_04_first_server_to_second_server_sequence(self):
        fsm = ScoringFSM()
        # End 0-0-2 -> Team B serves 0-0-1
        fsm.advance_rally("B")
        # Team B scores once -> 1-0-1 (P4 moves to LEFT)
        fsm.advance_rally("B")
        self.assertEqual(fsm.score_call, "1-0-1")
        ctx_srv1 = fsm.get_serve_context()
        self.assertEqual(ctx_srv1.active_server_id, "P4")
        self.assertEqual(ctx_srv1.server_box, "LEFT")

        # Team B Server #1 loses rally -> advances to Server #2 (P3 in RIGHT, no score/swap)
        ev2 = fsm.advance_rally("A")
        ctx_srv2 = fsm.get_serve_context()
        self.assertEqual(ev2.event_type, "SECOND_SERVER")
        self.assertEqual(ctx_srv2.score_call, "1-0-2")
        self.assertEqual(ctx_srv2.serving_team, "B")
        self.assertEqual(ctx_srv2.server_number, 2)
        self.assertEqual(ctx_srv2.active_server_id, "P3")
        self.assertEqual(ctx_srv2.server_box, "RIGHT")
        self.assertEqual(ctx_srv2.target_quadrant_ft, SERVICE_QUADRANT_BOUNDS_FT[("A", "RIGHT")])
        self.assertTrue(fsm.verify_all_invariants())

        # Team B Server #2 scores -> 2-0-2 (P3 swaps to LEFT)
        fsm.advance_rally("B")
        ctx_after = fsm.get_serve_context()
        self.assertEqual(ctx_after.score_call, "2-0-2")
        self.assertEqual(ctx_after.active_server_id, "P3")
        self.assertEqual(ctx_after.server_box, "LEFT")
        self.assertTrue(fsm.verify_all_invariants())

    def test_05_odd_score_side_out_right_court_serves_first(self):
        fsm = ScoringFSM()
        # Team A scores 1 point at 0-0-2 -> 1-0-2 (A's right=P2, left=P1), then faults -> 0-1-1
        fsm.advance_rally("A")
        fsm.advance_rally("B")
        # Team B loses both servers (1 -> 2 -> side-out)
        fsm.advance_rally("A")  # 0-1-2
        fsm.advance_rally("A")  # side-out to A at 1-0-1

        ctx = fsm.get_serve_context()
        self.assertEqual(ctx.score_call, "1-0-1")
        self.assertEqual(ctx.serving_team, "A")
        self.assertEqual(ctx.server_number, 1)
        # Because Team A's score is odd (1), P2 is in Right court and MUST serve Server #1 from RIGHT
        self.assertEqual(ctx.active_server_id, "P2")
        self.assertEqual(ctx.server_box, "RIGHT")
        self.assertTrue(fsm.verify_all_invariants())

    def test_06_diagonal_target_quadrant_bounds(self):
        self.assertEqual(SERVICE_QUADRANT_BOUNDS_FT[("A", "RIGHT")], (10.0, 20.0, 29.0, 44.0))
        self.assertEqual(SERVICE_QUADRANT_BOUNDS_FT[("A", "LEFT")], (0.0, 10.0, 29.0, 44.0))
        self.assertEqual(SERVICE_QUADRANT_BOUNDS_FT[("B", "RIGHT")], (0.0, 10.0, 0.0, 15.0))
        self.assertEqual(SERVICE_QUADRANT_BOUNDS_FT[("B", "LEFT")], (10.0, 20.0, 0.0, 15.0))

    def test_07_seed_and_validate_on_match1_real_telemetry(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        telem_path = os.path.join(repo_root, "data", "match1_full_telemetry.json")
        if not os.path.isfile(telem_path):
            self.skipTest("data/match1_full_telemetry.json not present locally")

        with open(telem_path, "r") as f:
            data = json.load(f)

        frame0_players = data["telemetry_stream"][0]["players"]
        fsm = ScoringFSM.seed_from_telemetry_players(frame0_players, infer_initial_server=True)
        ctx = fsm.get_serve_context()

        # In match1 frame 0: P4 is behind far baseline [-0.92 ft] in Far RIGHT box;
        # P1 is diagonally across at [14.36, 43.83] in Near RIGHT box.
        self.assertEqual(ctx.serving_team, "B")
        self.assertEqual(ctx.score_call, "0-0-2")
        self.assertEqual(ctx.active_server_id, "P4")
        self.assertEqual(ctx.active_receiver_id, "P1")
        self.assertEqual(fsm.teams["A"].right_player, "P1")
        self.assertEqual(fsm.teams["A"].left_player, "P2")
        self.assertEqual(fsm.teams["B"].right_player, "P4")
        self.assertEqual(fsm.teams["B"].left_player, "P3")

        alignment = fsm.validate_player_alignment(frame0_players)
        self.assertTrue(alignment["both_aligned"])

        exported = fsm.to_dict()
        self.assertTrue(exported["invariants_valid"])
        self.assertEqual(exported["serve_context"]["score_call"], "0-0-2")

    def test_08_validation_guards(self):
        with self.assertRaises(ValueError):
            ScoringFSM(initial_serving_team="C")
        fsm = ScoringFSM()
        with self.assertRaises(ValueError):
            fsm.advance_rally("Z")
        with self.assertRaises(ValueError):
            ScoringFSM.seed_from_telemetry_players([{"slot": "P1", "side": "near", "court_ft": [12, 35]}])


if __name__ == "__main__":
    unittest.main()
