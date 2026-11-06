"""
test_base.py - Unit test suite for homography projection and tracking modules.
"""
import unittest
import numpy as np

from pbbase.court.homography import CourtGeometry, CourtCalibrator
from pbbase.tracking.player_tracker import CanonicalPlayerTracker, _box_iou
from pbbase.tracking.ball_trajectory import BallTrajectoryTracker, get_speed_color
from pbbase.models.detector import BallDetector, _merge_ball_candidates, _box_iou_px
from pbbase.data.schema import SCHEMA_VERSION, validate_telemetry_schema


class TestCourtHomography(unittest.TestCase):
    def test_known_court_projection(self):
        calibrator = CourtCalibrator()
        # 4 known camera corners
        corners = [
            (420.0, 180.0),   # Far-Left
            (860.0, 180.0),   # Far-Right
            (1140.0, 640.0),  # Near-Right
            (140.0, 640.0),   # Near-Left
        ]
        calibrated = calibrator.calibrate(corners)
        self.assertTrue(calibrated)
        self.assertLess(calibrator.result.reprojection_rmse_ft, 1e-4)

        # Check far-left corner maps to (0, 0) ft
        pos_fl = calibrator.pixel_to_court_ft(420.0, 180.0)
        self.assertIsNotNone(pos_fl)
        self.assertAlmostEqual(pos_fl[0], 0.0, places=2)
        self.assertAlmostEqual(pos_fl[1], 0.0, places=2)

        # Check near-right corner maps to (20, 44) ft
        pos_nr = calibrator.pixel_to_court_ft(1140.0, 640.0)
        self.assertIsNotNone(pos_nr)
        self.assertAlmostEqual(pos_nr[0], 20.0, places=2)
        self.assertAlmostEqual(pos_nr[1], 44.0, places=2)


class TestPlayerTracker(unittest.TestCase):
    def test_iou_calculation(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 0.0, 15.0, 10.0]
        iou = _box_iou(b1, b2)
        # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 1/3 = 0.3333
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=3)

    def test_canonical_slots(self):
        tracker = CanonicalPlayerTracker(max_players=4)
        calibrator = CourtCalibrator([(100, 100), (900, 100), (1000, 900), (50, 900)])

        fake_dets = [
            {"box": [200, 600, 250, 750], "conf": 0.9, "foot_px": (225, 750)},
            {"box": [700, 600, 750, 750], "conf": 0.88, "foot_px": (725, 750)},
            {"box": [300, 200, 350, 320], "conf": 0.85, "foot_px": (325, 320)},
            {"box": [600, 200, 650, 320], "conf": 0.82, "foot_px": (625, 320)},
        ]
        active = tracker.update(fake_dets, (1080, 1920), 0.0, calibrator)
        self.assertEqual(len(active), 4)
        assigned_ids = {p["id"] for p in active}
        self.assertEqual(assigned_ids, {1, 2, 3, 4})

    def test_spectator_rejection(self):
        tracker = CanonicalPlayerTracker(max_players=4)
        # Standard 4 court corners
        calibrator = CourtCalibrator([(420, 180), (860, 180), (1140, 640), (140, 640)])

        # Player 1 is inside court
        # Person 2 is 250px below baseline in spectator stands
        dets = [
            {"box": [500, 500, 560, 620], "conf": 0.90, "foot_px": (530, 620)},
            {"box": [500, 950, 560, 1050], "conf": 0.88, "foot_px": (530, 1050)},  # Spectator far outside
        ]
        active = tracker.update(dets, (1080, 1920), 0.0, calibrator)
        # Spectator should be rejected by athletic envelope
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], 1)

    def test_net_partition_barrier(self):
        tracker = CanonicalPlayerTracker(max_players=4)
        calibrator = CourtCalibrator([(420, 180), (860, 180), (1140, 640), (140, 640)])

        # Frame 0: P1 initialized on near court
        det_near = [{"box": [500, 500, 560, 620], "conf": 0.90, "foot_px": (530, 620)}]
        active0 = tracker.update(det_near, (1080, 1920), 0.0, calibrator)
        self.assertEqual(len(active0), 1)
        self.assertEqual(active0[0]["id"], 1)
        self.assertEqual(active0[0]["side"], "near")

        # Frame 1: Candidate on far court appears
        det_far = [{"box": [500, 200, 560, 300], "conf": 0.88, "foot_px": (530, 300)}]
        active1 = tracker.update(det_far, (1080, 1920), 0.033, calibrator)
        # P1 cannot be assigned to far candidate! Instead far candidate gets assigned a far slot (P3)
        assigned_ids = {p["id"]: p["side"] for p in active1}
        self.assertIn(3, assigned_ids)
        self.assertEqual(assigned_ids[3], "far")

    def test_partner_cross_over_with_appearance(self):
        tracker = CanonicalPlayerTracker(max_players=4)
        calibrator = CourtCalibrator([(420, 180), (860, 180), (1140, 640), (140, 640)])

        # Synthetic frame: Player A (Blue: Hue ~110), Player B (Red: Hue ~5)
        frame0 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Blue box at x=300
        frame0[450:600, 280:360] = [220, 50, 30]
        # Red box at x=700
        frame0[450:600, 680:760] = [30, 40, 220]

        dets0 = [
            {"box": [280, 450, 360, 600], "conf": 0.9, "foot_px": (320, 600)},  # P1 Left
            {"box": [680, 450, 760, 600], "conf": 0.9, "foot_px": (720, 600)},  # P2 Right
        ]
        active0 = tracker.update(dets0, (1080, 1920), 0.0, calibrator, frame=frame0)
        self.assertEqual(len(active0), 2)
        p1_id = active0[0]["id"]
        p2_id = active0[1]["id"]
        self.assertNotEqual(p1_id, p2_id)

        # Cross-over: Blue player moved to right (x=680), Red player moved to left (x=280)
        frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame1[450:600, 280:360] = [30, 40, 220]  # Red at left
        frame1[450:600, 680:760] = [220, 50, 30]  # Blue at right

        dets1 = [
            {"box": [280, 450, 360, 600], "conf": 0.9, "foot_px": (320, 600)},  # Red
            {"box": [680, 450, 760, 600], "conf": 0.9, "foot_px": (720, 600)},  # Blue
        ]
        active1 = tracker.update(dets1, (1080, 1920), 0.033, calibrator, frame=frame1)
        self.assertEqual(len(active1), 2)

        # Verify Blue player still has p1_id and Red player still has p2_id (zero swap!)
        blue_player = next(p for p in active1 if p["foot_px"][0] > 500)
        red_player = next(p for p in active1 if p["foot_px"][0] < 500)
        self.assertEqual(blue_player["id"], p1_id)
        self.assertEqual(red_player["id"], p2_id)


class TestBallTrajectory(unittest.TestCase):
    def test_speed_color_spectrum(self):
        col_smash = get_speed_color(25.0)
        col_rally = get_speed_color(15.0)
        col_dink = get_speed_color(5.0)
        self.assertEqual(col_smash, (20, 120, 255))
        self.assertNotEqual(col_smash, col_rally)
        self.assertNotEqual(col_rally, col_dink)


class TestMultiScaleBallInference(unittest.TestCase):
    def test_box_iou_calculation(self):
        b1 = [100.0, 100.0, 120.0, 120.0]
        b2 = [110.0, 100.0, 130.0, 120.0]
        iou = _box_iou_px(b1, b2)
        # Inter: 10 * 20 = 200. Area1: 400, Area2: 400. Union: 400 + 400 - 200 = 600. IoU = 1/3
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=3)

    def test_candidate_merging_consensus(self):
        primary = [
            {"box": [500.0, 200.0, 515.0, 215.0], "conf": 0.60, "center_px": (507.5, 207.5)}
        ]
        patch = [
            {"box": [502.0, 201.0, 516.0, 215.0], "conf": 0.85, "center_px": (509.0, 208.0), "patch_origin": True}
        ]
        merged = _merge_ball_candidates(primary, patch, iou_thresh=0.25, dist_thresh_px=24.0)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].get("consensus", False))
        # Confidence should be boosted from multi-scale consensus
        self.assertGreater(merged[0]["conf"], 0.85)

    def test_candidate_merging_disjoint(self):
        primary = [
            {"box": [100.0, 800.0, 120.0, 820.0], "conf": 0.70, "center_px": (110.0, 810.0)}
        ]
        patch = [
            {"box": [600.0, 200.0, 615.0, 215.0], "conf": 0.80, "center_px": (607.5, 207.5), "patch_origin": True}
        ]
        merged = _merge_ball_candidates(primary, patch, iou_thresh=0.25, dist_thresh_px=24.0)
        # Both distinct candidates should be preserved
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["conf"], 0.80)  # Highest conf first
        self.assertEqual(merged[1]["conf"], 0.70)


class TestBallTrajectoryKinematics(unittest.TestCase):
    def setUp(self):
        self.calibrator = CourtCalibrator()
        self.calibrator.calibrate([
            (420.0, 180.0), (860.0, 180.0), (1140.0, 640.0), (140.0, 640.0)
        ])
        self.tracker = BallTrajectoryTracker(
            max_jump_px=80.0,
            max_coast_frames=3,
            gravity_px=0.5,
            drag_coeff=0.01,
        )

    def test_velocity_gating_rejects_teleport(self):
        # Frame 0: Initial detection
        cand0 = [{"box": [500, 300, 512, 312], "conf": 0.8, "center_px": (506.0, 306.0)}]
        p0 = self.tracker.process_candidates(cand0, 0, 0.0, self.calibrator)
        self.assertIsNotNone(p0)
        self.assertFalse(p0["interpolated"])

        # Frame 1: Normal movement
        cand1 = [{"box": [515, 308, 527, 320], "conf": 0.82, "center_px": (521.0, 314.0)}]
        p1 = self.tracker.process_candidates(cand1, 1, 0.033, self.calibrator)
        self.assertIsNotNone(p1)
        self.assertFalse(p1["interpolated"])

        # Frame 2: Teleport noise candidate 300px away (should be gated out and coasted)
        teleport_cand = [{"box": [850, 700, 862, 712], "conf": 0.9, "center_px": (856.0, 706.0)}]
        p2 = self.tracker.process_candidates(teleport_cand, 2, 0.066, self.calibrator)
        self.assertIsNotNone(p2)
        self.assertTrue(p2["interpolated"])  # Coasted using velocity instead of jumping
        self.assertLess(abs(p2["pixel"][0] - 536.0), 5.0)

    def test_ballistic_occlusion_coasting(self):
        # Frame 0 & 1: Establish steady trajectory
        self.tracker.process_candidates([{"box": [100, 200, 110, 210], "conf": 0.8, "center_px": (105.0, 205.0)}], 0, 0.0, self.calibrator)
        self.tracker.process_candidates([{"box": [115, 210, 125, 220], "conf": 0.8, "center_px": (120.0, 215.0)}], 1, 0.033, self.calibrator)

        # Frame 2: Occlusion (no candidates) -> Coast 1
        p2 = self.tracker.process_candidates([], 2, 0.066, self.calibrator)
        self.assertIsNotNone(p2)
        self.assertTrue(p2["interpolated"])
        self.assertEqual(p2["coast_frame"], 1)

        # Frame 3: Occlusion -> Coast 2
        p3 = self.tracker.process_candidates([], 3, 0.099, self.calibrator)
        self.assertIsNotNone(p3)
        self.assertTrue(p3["interpolated"])
        self.assertEqual(p3["coast_frame"], 2)

        # Frame 4: Occlusion -> Coast 3
        p4 = self.tracker.process_candidates([], 4, 0.132, self.calibrator)
        self.assertIsNotNone(p4)
        self.assertTrue(p4["interpolated"])
        self.assertEqual(p4["coast_frame"], 3)

        # Frame 5: Exceeded max_coast_frames (3) -> Clean track termination
        p5 = self.tracker.process_candidates([], 5, 0.165, self.calibrator)
        self.assertIsNone(p5)

    def test_retroactive_parabolic_gap_filling(self):
        # Frames 0 & 1: Steady flight
        self.tracker.process_candidates([{"box": [100, 200, 110, 210], "conf": 0.8, "center_px": (105.0, 205.0)}], 0, 0.0, self.calibrator)
        self.tracker.process_candidates([{"box": [120, 210, 130, 220], "conf": 0.8, "center_px": (125.0, 215.0)}], 1, 0.033, self.calibrator)

        # Frames 2 & 3: Occluded behind net
        self.tracker.process_candidates([], 2, 0.066, self.calibrator)
        self.tracker.process_candidates([], 3, 0.099, self.calibrator)

        # Frame 4: Re-acquired!
        cand4 = [{"box": [178, 232, 188, 242], "conf": 0.85, "center_px": (183.0, 237.0)}]
        p4 = self.tracker.process_candidates(cand4, 4, 0.132, self.calibrator)
        self.assertIsNotNone(p4)
        self.assertFalse(p4["interpolated"])

        # Check retroactive parabolic smoothing on intermediate frames
        gap_frame_2 = self.tracker.raw_history[2]
        gap_frame_3 = self.tracker.raw_history[3]
        self.assertTrue(gap_frame_2.get("retro_parabolic_refined", False))
        self.assertTrue(gap_frame_3.get("retro_parabolic_refined", False))
        # Ensure smooth monotonic progression along X
        self.assertGreater(gap_frame_2["pixel"][0], 125.0)
        self.assertGreater(gap_frame_3["pixel"][0], gap_frame_2["pixel"][0])
        self.assertLess(gap_frame_3["pixel"][0], 183.0)


class TestPerceptionSchema(unittest.TestCase):
    def setUp(self):
        self.valid_telemetry = {
            "schema_version": SCHEMA_VERSION,
            "producer": "Pickleball Base Perception Engine",
            "timestamp": "2026-11-05 10:00:00",
            "video_path": "clips/match1.mp4",
            "total_frames_processed": 1,
            "fps": 30.0,
            "resolution": "1920x1080",
            "elapsed_s": 0.2,
            "processing_fps": 5.0,
            "court": {
                "calibrated": True,
                "reprojection_rmse_ft": 0.0001,
                "dimensions_ft": {"width": 20.0, "length": 44.0, "net_y": 22.0, "nvz_depth": 7.0},
                "corners_px": [[420.0, 180.0], [860.0, 180.0], [1140.0, 640.0], [140.0, 640.0]],
            },
            "benchmark_summary": {
                "ball_detection_rate_pct": 100.0,
                "frames_with_ball": 1,
                "mean_players_per_frame": 4.0,
                "multiscale_inference": True,
                "stage_latencies_ms": {"player_ms": 35.0, "ball_ms": 150.0, "court_ms": 0.0, "render_ms": 7.0},
            },
            "telemetry_stream": [
                {
                    "frame_idx": 0,
                    "timestamp_s": 0.0,
                    "court_calibrated": True,
                    "players": [
                        {
                            "id": 1,
                            "slot": "P1",
                            "box": [200.0, 400.0, 260.0, 600.0],
                            "foot_px": [230.0, 600.0],
                            "court_ft": [4.5, 38.0],
                            "speed_mph": 2.1,
                            "side": "near",
                            "is_coasting": False,
                        }
                    ],
                    "ball": {
                        "pixel": [500.0, 300.0],
                        "box": [495.0, 295.0, 505.0, 305.0],
                        "conf": 0.85,
                        "speed_mph": 24.5,
                        "speed_px_per_frame": 12.0,
                        "court_ft": [10.0, 22.0],
                        "interpolated": False,
                    },
                }
            ],
        }

    def test_valid_schema_passes(self):
        valid, errors = validate_telemetry_schema(self.valid_telemetry)
        self.assertTrue(valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_missing_top_level_key_fails(self):
        invalid_data = dict(self.valid_telemetry)
        del invalid_data["court"]
        valid, errors = validate_telemetry_schema(invalid_data)
        self.assertFalse(valid)
        self.assertTrue(any("Missing top-level key: 'court'" in e for e in errors))

    def test_missing_player_key_fails(self):
        import copy
        invalid_data = copy.deepcopy(self.valid_telemetry)
        del invalid_data["telemetry_stream"][0]["players"][0]["slot"]
        valid, errors = validate_telemetry_schema(invalid_data)
        self.assertFalse(valid)
        self.assertTrue(any("missing key 'slot'" in e for e in errors))


class TestRallyValidation(unittest.TestCase):
    def test_audit_zero_swaps_on_normal_trajectory(self):
        from scripts.validate_rally import audit_player_id_swaps
        stream = [
            {
                "frame_index": 0,
                "players": [
                    {"id": 1, "slot": "P1", "court_ft": [4.0, 36.0], "side": "near"},
                    {"id": 3, "slot": "P3", "court_ft": [5.0, 10.0], "side": "far"},
                ]
            },
            {
                "frame_index": 1,
                "players": [
                    {"id": 1, "slot": "P1", "court_ft": [4.2, 36.1], "side": "near"},
                    {"id": 3, "slot": "P3", "court_ft": [5.1, 10.2], "side": "far"},
                ]
            },
            {
                "frame_index": 2,
                "players": [
                    {"id": 1, "slot": "P1", "court_ft": [4.4, 36.3], "side": "near"},
                    {"id": 3, "slot": "P3", "court_ft": [5.3, 10.4], "side": "far"},
                ]
            },
        ]
        audit = audit_player_id_swaps(stream)
        self.assertEqual(audit["total_switch_events"], 0)
        self.assertEqual(audit["id_swaps"], 0)
        self.assertEqual(audit["net_partition_violations"], 0)

    def test_audit_detects_net_partition_violation(self):
        from scripts.validate_rally import audit_player_id_swaps
        stream = [
            {"frame_index": 0, "players": [{"id": 1, "slot": "P1", "court_ft": [4.0, 36.0], "side": "near"}]},
            {"frame_index": 1, "players": [{"id": 1, "slot": "P1", "court_ft": [4.0, 10.0], "side": "far"}]},
        ]
        audit = audit_player_id_swaps(stream)
        self.assertGreaterEqual(audit["net_partition_violations"], 1)
        self.assertGreaterEqual(audit["total_switch_events"], 1)

    def test_audit_detects_unnatural_displacement_leap(self):
        from scripts.validate_rally import audit_player_id_swaps
        stream = [
            {"frame_index": 0, "players": [{"id": 1, "slot": "P1", "court_ft": [2.0, 35.0], "side": "near"}]},
            {"frame_index": 1, "players": [{"id": 1, "slot": "P1", "court_ft": [18.0, 35.0], "side": "near"}]},  # 16 ft leap
        ]
        audit = audit_player_id_swaps(stream, max_step_ft_per_frame=6.0)
        self.assertGreaterEqual(audit["id_swaps"], 1)

    def test_ball_detection_audit_rate(self):
        from scripts.validate_rally import audit_ball_detection
        stream_failing = [{"frame_index": i, "ball": {"pixel": [100.0, 200.0]} if i < 9 else None} for i in range(10)]
        audit_f = audit_ball_detection(stream_failing)
        self.assertEqual(audit_f["tracking_rate_pct"], 90.0)
        self.assertFalse(audit_f["target_met"])

        stream_passing = [{"frame_index": i, "ball": {"pixel": [100.0, 200.0]}} for i in range(10)]
        audit_p = audit_ball_detection(stream_passing)
        self.assertEqual(audit_p["tracking_rate_pct"], 100.0)
        self.assertTrue(audit_p["target_met"])


if __name__ == "__main__":
    unittest.main()
