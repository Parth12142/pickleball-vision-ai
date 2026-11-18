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


class TestCourtCalibrationStore(unittest.TestCase):
    def test_load_stored_calibration(self):
        calib = CourtCalibrator.load_calibration_for_clip("MicrosoftTeams-video.mp4")
        self.assertIsNotNone(calib)
        self.assertEqual(len(calib), 4)
        # Verify corner coordinates
        self.assertEqual(calib[0], (245.0, 520.0))
        self.assertEqual(calib[1], (475.0, 520.0))
        self.assertEqual(calib[2], (680.0, 1040.0))
        self.assertEqual(calib[3], (40.0, 1040.0))

    def test_portrait_vs_landscape_recommendation(self):
        # Portrait (720x1280)
        portrait_corners = CourtCalibrator.get_recommended_corners(720, 1280)
        self.assertEqual(len(portrait_corners), 4)

        # Landscape (1302x740)
        landscape_corners = CourtCalibrator.get_recommended_corners(1302, 740)
        self.assertEqual(len(landscape_corners), 4)

        # Portrait far baseline y should be lower (higher y value in pixels, near frame center)
        self.assertGreater(portrait_corners[0][1], 1280 * 0.35)

    def test_custom_pinned_calibration_accuracy(self):
        custom_corners = [
            (245.0, 520.0),
            (475.0, 520.0),
            (680.0, 1040.0),
            (40.0, 1040.0),
        ]
        calibrator = CourtCalibrator(custom_corners)
        self.assertTrue(calibrator.result.calibrated)
        self.assertLess(calibrator.result.reprojection_rmse_ft, 1e-4)

        # Test feet to pixel and pixel to feet round-trip
        pt_ft = calibrator.pixel_to_court_ft(245.0, 520.0)
        self.assertIsNotNone(pt_ft)
        self.assertAlmostEqual(pt_ft[0], 0.0, places=2)
        self.assertAlmostEqual(pt_ft[1], 0.0, places=2)

        px_back = calibrator.court_to_pixel(0.0, 0.0)
        self.assertIsNotNone(px_back)
        self.assertAlmostEqual(px_back[0], 245.0, places=1)
        self.assertAlmostEqual(px_back[1], 520.0, places=1)

    def test_resolution_scaled_calibration_for_match2(self):
        # match2.mp4 is stored at 1296x738 in court_calibrations.json but video is 640x360
        scaled_corners = CourtCalibrator.get_recommended_corners(640, 360, clip_name="match2.mp4")
        self.assertEqual(len(scaled_corners), 4)
        for x, y in scaled_corners:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 640.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 360.0)


class TestCrossClipInvarianceUpgrades(unittest.TestCase):
    def test_static_decoy_eviction_and_motion_selection(self):
        calibrator = CourtCalibrator([(209.0, 86.0), (420.0, 86.0), (531.0, 297.0), (100.0, 293.0)], image_size=(640, 360))
        tracker = BallTrajectoryTracker(max_jump_px=85.0, max_coast_frames=3)
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Stationary high-conf decoy at (300, 150), real moving ball starting at (250, 120)
        for f_idx in range(7):
            cands = [
                {"box": [296, 146, 304, 154], "conf": 0.90, "center_px": (300.0, 150.0)},  # Static decoy
                {"box": [246 + f_idx * 8, 116 + f_idx * 5, 254 + f_idx * 8, 124 + f_idx * 5], "conf": 0.65, "center_px": (250.0 + f_idx * 8, 120.0 + f_idx * 5)},
            ]
            res = tracker.process_candidates(cands, f_idx, f_idx / 30.0, calibrator, frame=dummy_frame)

        # By frame 6, static decoy at (300, 150) must be blacklisted and tracker locked onto moving ball
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["pixel"][0], 250.0 + 6 * 8, delta=1.5)

    def test_fps_and_resolution_invariant_bounce_detection(self):
        from pbbase.tracking.bounce_detector import BallBounceDetector, SavitzkyGolayBounceAnalyzer
        # 640x360 half-scale court at 60 FPS
        calibrator = CourtCalibrator([(209.0, 86.0), (420.0, 86.0), (531.0, 297.0), (100.0, 293.0)], image_size=(640, 360))
        detector = BallBounceDetector()

        # Synthetic 60fps bounce trajectory in 360p pixel scale (nadir at frame 6)
        ys = [160.0, 163.5, 167.2, 171.2, 175.5, 180.0, 184.8, 181.0, 176.8, 172.2, 167.5, 162.5, 157.5]
        stream = []
        bounces_5pt = []
        for idx, y in enumerate(ys):
            ts = idx / 60.0
            ball = {"pixel": [310.0 + idx * 2.0, y], "conf": 0.85, "speed_mph": 22.0, "interpolated": False}
            ev = detector.update(ball, idx, ts, calibrator, players=[])
            if ev is not None:
                bounces_5pt.append(ev)
            stream.append({"frame_idx": idx, "timestamp_s": ts, "ball": ball, "players": [], "bounce": ev.to_dict() if ev else None})

        self.assertGreaterEqual(len(bounces_5pt), 1)
        sg = SavitzkyGolayBounceAnalyzer()
        sg_out = sg.analyze_stream(stream, calibrator, fps=60.0)
        self.assertGreaterEqual(len(sg_out["bounces"]), 1)
        self.assertEqual(sg_out["bounces"][0]["frame_idx"], 6)

    def test_3d_reconstructor_does_not_hallucinate_long_gaps(self):
        from pbbase.tracking.ball_3d_tracker import Ball3DReconstructor
        calibrator = CourtCalibrator([(425.0, 177.0), (855.0, 177.0), (1080.0, 610.0), (204.0, 602.0)], image_size=(1302, 740))
        reconstructor = Ball3DReconstructor()

        stream = []
        for k in range(60):
            # Observed only at frames 0..4 and 50..54 (45-frame dead gap in between)
            if k <= 4 or k >= 50:
                ball = {
                    "frame_idx": k,
                    "timestamp_s": k / 30.0,
                    "pixel": [500.0 + k * 2.0, 300.0 + k * 1.5],
                    "box": [494.0, 294.0, 506.0, 306.0],
                    "conf": 0.8,
                    "speed_mph": 15.0,
                    "speed_px_per_frame": 2.5,
                    "court_ft": [10.0, 20.0],
                    "interpolated": False,
                }
            else:
                ball = None
            stream.append({"frame_idx": k, "timestamp_s": k / 30.0, "players": [], "ball": ball, "bounce": None})

        out_stream, _ = reconstructor.reconstruct_stream(stream, [], calibrator)
        # Frame 25 is in the middle of a 45-frame untracked gap -> must remain None!
        self.assertIsNone(out_stream[25]["ball"])

    def test_25d_minimap_native_scaling_and_shadow(self):
        from pbbase.court.minimap import CourtMinimap
        minimap = CourtMinimap()
        minimap.configure_for_frame(640, 360)
        self.assertLess(minimap.canvas_h, 250)
        canvas = minimap.render(
            players=[{"id": 1, "court_ft": [5.0, 38.0]}],
            ball={"minimap_court_ft": [10.0, 22.0], "height_ft": 4.5},
            frame_idx=0,
        )
        self.assertEqual(canvas.shape[:2], (minimap.canvas_h, minimap.canvas_w))

    def test_slow_lob_apex_not_evicted_as_static_decoy(self):
        calibrator = CourtCalibrator([(486.0, 116.5), (789.0, 120.0), (1144.3, 578.8), (113.0, 581.0)], image_size=(1280, 720))
        tracker = BallTrajectoryTracker(max_jump_px=85.0, max_coast_frames=3)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Simulate lob apex slowing to ~1.8 px/frame before accelerating down-left
        pts = [
            (710.7, 109.4), (713.1, 107.0), (715.3, 105.4), (717.1, 104.5),
            (718.7, 105.0), (719.8, 106.3), (718.5, 105.1), (714.8, 103.2), (711.2, 102.4),
        ]
        results = []
        for f_idx, (px, py) in enumerate(pts):
            cands = [{"box": [px - 4, py - 4, px + 4, py + 4], "conf": 0.75, "center_px": (px, py)}]
            res = tracker.process_candidates(cands, f_idx, f_idx / 30.0, calibrator, frame=dummy_frame)
            results.append(res)

        self.assertTrue(all(r is not None and not r["interpolated"] for r in results))
        self.assertEqual(results[-1]["pixel"], (711.2, 102.4))
        self.assertEqual(results[0]["track_id"], results[-1]["track_id"])

    def test_3d_reconstructor_prevents_depth_collapse_and_chord_spikes(self):
        from pbbase.tracking.ball_3d_tracker import Ball3DReconstructor, ShotAnchor
        calibrator = CourtCalibrator([(486.0, 116.5), (789.0, 120.0), (1144.3, 578.8), (113.0, 581.0)], image_size=(1280, 720))
        reconstructor = Ball3DReconstructor()

        # Construct realistic rally segment with 3 missing frames mid-flight; pixel gap outputs must stay near chord
        stream = []
        for k in range(30):
            tau = k / 29.0
            u = 600.0 + tau * 120.0
            v = 450.0 - tau * 345.0
            if k in (14, 15, 16):
                ball = None
            else:
                ball = {
                    "frame_idx": k,
                    "timestamp_s": k / 30.0,
                    "pixel": [round(u, 1), round(v, 1)],
                    "box": [u - 5, v - 5, u + 5, v + 5],
                    "conf": 0.75,
                    "speed_mph": 30.0,
                    "speed_px_per_frame": 12.0,
                    "court_ft": [10.0, 44.0 * (1.0 - tau)],
                    "interpolated": False,
                    "track_id": 0,
                }
            stream.append({"frame_idx": k, "timestamp_s": k / 30.0, "players": [], "ball": ball, "bounce": None})

        out_stream, _ = reconstructor.reconstruct_stream(stream, [], calibrator)
        for k in (14, 15, 16):
            b = out_stream[k]["ball"]
            self.assertIsNotNone(b)
            tau = k / 29.0
            exp_u, exp_v = 600.0 + tau * 120.0, 450.0 - tau * 345.0
            self.assertLess(float(np.hypot(b["pixel"][0] - exp_u, b["pixel"][1] - exp_v)), 45.0)
            self.assertGreaterEqual(b["pixel"][1], 0.0)

    def test_scrub_unconfirmed_coast_tails_preserves_bounded_and_drops_open_tails(self):
        from pbbase.pipeline import _scrub_unconfirmed_coast_tails

        def _det(tid, px, interp=False, conf=0.65):
            return {
                "track_id": tid,
                "pixel": [float(px[0]), float(px[1])],
                "interpolated": interp,
                "conf": conf,
            }

        stream = [
            {"frame_idx": 0, "ball": _det(1, (500, 300), interp=False)},
            {"frame_idx": 1, "ball": _det(1, (515, 290), interp=False)},
            {"frame_idx": 2, "ball": _det(1, (530, 282), interp=True, conf=0.20)},
            {"frame_idx": 3, "ball": _det(1, (545, 276), interp=False)},
            {"frame_idx": 4, "ball": _det(1, (560, 255), interp=True, conf=0.18)},
            {"frame_idx": 5, "ball": _det(1, (575, 230), interp=True, conf=0.16)},
            {"frame_idx": 6, "ball": None},
            {"frame_idx": 7, "ball": _det(2, (700, 260), interp=False)},
            {"frame_idx": 8, "ball": _det(2, (712, 240), interp=True, conf=0.19)},
        ]

        _scrub_unconfirmed_coast_tails(stream)

        self.assertIsNotNone(stream[2]["ball"], "Bounded occlusion coast should remain")
        self.assertIsNone(stream[4]["ball"], "Unconfirmed open-loop coast tail must be scrubbed")
        self.assertIsNone(stream[5]["ball"], "Unconfirmed open-loop coast tail must be scrubbed")
        self.assertIsNotNone(stream[7]["ball"], "Observed detection should remain")
        self.assertIsNone(stream[8]["ball"], "End-of-stream coast tail must be scrubbed")

    def test_prune_ghost_ball_tracks_drops_short_ghost_and_keeps_real_rally(self):
        from pbbase.pipeline import _prune_ghost_ball_tracks

        stream = [{"frame_idx": i, "ball": None} for i in range(25)]
        # Short 2-frame low-confidence ghost fragment
        for idx in (0, 1):
            stream[idx]["ball"] = {
                "track_id": 10,
                "pixel": [843.0 + idx * 0.5, 83.0],
                "interpolated": False,
                "conf": 0.24,
            }

        # 10-frame moving real rally track
        for k in range(10):
            stream[10 + k]["ball"] = {
                "track_id": 11,
                "pixel": [400.0 + 20.0 * k, 350.0 - 15.0 * k + 1.2 * k * k],
                "interpolated": False,
                "conf": 0.58,
            }

        _prune_ghost_ball_tracks(stream, scale_px=1.0)

        self.assertIsNone(stream[0]["ball"])
        self.assertIsNone(stream[1]["ball"])
        for k in range(10):
            self.assertIsNotNone(stream[10 + k]["ball"], f"Real track frame {10 + k} should survive")

    def test_low_conf_noise_does_not_initialize_new_track(self):
        calibrator = CourtCalibrator([(420, 180), (860, 180), (1140, 640), (140, 640)])
        tracker = BallTrajectoryTracker()
        noisy = [{"center_px": (640.0, 360.0), "conf": 0.18, "box": [635.0, 355.0, 645.0, 365.0]}]
        self.assertIsNone(
            tracker.process_candidates(noisy, frame_idx=0, timestamp_s=0.0, court_calibrator=calibrator)
        )

        solid = [{"center_px": (640.0, 360.0), "conf": 0.45, "box": [635.0, 355.0, 645.0, 365.0]}]
        self.assertIsNotNone(
            tracker.process_candidates(solid, frame_idx=1, timestamp_s=1.0 / 30.0, court_calibrator=calibrator)
        )


if __name__ == "__main__":
    unittest.main()
