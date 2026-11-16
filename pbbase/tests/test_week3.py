"""
test_week3.py - Comprehensive Unit Tests for Week 3 Minimap & Bounce Kinematics.
Tests:
  - USAP 2D Orthographic Court Minimap projection and canvas coordinates
  - Quadratic least-squares nadir fitting and sub-pixel localization
  - Kinematic vertical derivative filter and acceleration inflection
  - Ground bounce vs paddle volley disambiguation
  - Minimap HUD video frame blending and ripple decay
  - Telemetry schema validation with Week 3 bounce structures
"""
import unittest
import numpy as np

from pbbase.court.homography import CourtCalibrator, CourtGeometry
from pbbase.court.minimap import CourtMinimap, MinimapBounceMarker
from pbbase.tracking.bounce_detector import BallBounceDetector, BounceEvent
from pbbase.data.schema import validate_telemetry_schema, SCHEMA_VERSION, SCHEMA_PRODUCER


class TestWeek3MinimapAndBounce(unittest.TestCase):
    def setUp(self):
        # Canonical court calibrator
        self.calibrator = CourtCalibrator()
        corners = [(100.0, 100.0), (1100.0, 100.0), (1200.0, 650.0), (50.0, 650.0)]
        self.calibrator.calibrate(corners)

    def test_minimap_projection_and_geometry(self):
        """W3-01: Verify 20x44 ft court orthographic canvas dimensions and projection."""
        minimap = CourtMinimap(court_scale=10.0, padding=20)
        self.assertEqual(minimap.canvas_w, 240)
        self.assertEqual(minimap.canvas_h, 480)

        # Baseline origin (0, 0) ft should map to (padding, padding) = (20, 20)
        cx0, cy0 = minimap.court_to_canvas(0.0, 0.0)
        self.assertEqual((cx0, cy0), (20, 20))

        # Far-right baseline (20, 44) ft should map to (20 + 200, 20 + 440) = (220, 460)
        cx1, cy1 = minimap.court_to_canvas(20.0, 44.0)
        self.assertEqual((cx1, cy1), (220, 460))

        # Net center (10, 22) ft
        cx_net, cy_net = minimap.court_to_canvas(10.0, 22.0)
        self.assertEqual((cx_net, cy_net), (120, 240))

        # Inverse mapping: canvas -> court ft
        x0, y0 = minimap.canvas_to_court(20, 20)
        self.assertAlmostEqual(x0, 0.0, places=2)
        self.assertAlmostEqual(y0, 0.0, places=2)

        x1, y1 = minimap.canvas_to_court(220, 460)
        self.assertAlmostEqual(x1, 20.0, places=2)
        self.assertAlmostEqual(y1, 44.0, places=2)

        # Boundary checks
        self.assertTrue(minimap.is_in_court_bounds(5.0, 15.0))
        self.assertTrue(minimap.is_in_court_bounds(0.0, 0.0))
        self.assertTrue(minimap.is_in_court_bounds(20.0, 44.0))
        self.assertFalse(minimap.is_in_court_bounds(-1.0, 20.0))
        self.assertFalse(minimap.is_in_court_bounds(10.0, 45.0))

    def test_minimap_rendering_and_composition(self):
        """W3-01 & W3-04: Test minimap rendering, player badges, and video HUD blending."""
        minimap = CourtMinimap(court_scale=10.0, padding=20)

        mock_players = [
            {"id": 1, "slot": 0, "court_ft": (5.0, 35.0), "box": [100, 200, 150, 300]},
            {"id": 2, "slot": 1, "court_ft": (15.0, 36.0), "box": [300, 200, 350, 300]},
            {"id": 3, "slot": 2, "court_ft": (6.0, 10.0), "box": [100, 50, 140, 120]},
            {"id": 4, "slot": 3, "court_ft": (14.0, 9.0), "box": [250, 50, 290, 120]},
        ]
        mock_ball = {"court_ft": (10.0, 24.0), "pixel": (600, 400), "speed_mph": 28.5}

        canvas = minimap.render(players=mock_players, ball=mock_ball, frame_idx=10)
        self.assertEqual(canvas.shape, (480, 240, 3))
        self.assertEqual(canvas.dtype, np.uint8)

        # Verify video composition
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        composed = minimap.compose_onto_frame(frame, canvas, position="bottom-right")
        self.assertEqual(composed.shape, (720, 1280, 3))

        # Check that canvas was drawn into bottom-right ROI
        # Bottom-right margin=18, canvas=240x480 -> y from 720-480-18=222 to 702, x from 1280-240-18=1022 to 1262
        roi = composed[230:300, 1030:1100]
        self.assertTrue(np.any(roi > 0))

    def test_parabolic_nadir_fitting_math(self):
        """W3-03: Verify closed-form 5-point quadratic fit and sub-pixel nadir math."""
        # Exact parabola: y(t) = -2*t^2 + 1*t + 50
        # Peak vertex: t* = -1 / (2 * -2) = 0.25
        # y* = -2*(0.25)^2 + 1*(0.25) + 50 = 50.125
        ts = [-2, -1, 0, 1, 2]
        ys = [-2 * (t ** 2) + 1 * t + 50 for t in ts]  # [40, 47, 50, 49, 44]

        a, b, c = BallBounceDetector.fit_parabola_5pt(ys)
        self.assertAlmostEqual(a, -2.0, places=4)
        self.assertAlmostEqual(b, 1.0, places=4)
        self.assertAlmostEqual(c, 50.0, places=4)

        t_star = -b / (2.0 * a)
        self.assertAlmostEqual(t_star, 0.25, places=4)

        sub_y = a * (t_star ** 2) + b * t_star + c
        self.assertAlmostEqual(sub_y, 50.125, places=4)

        # Linear fit for horizontal trajectory
        xs = [100.0, 110.0, 120.0, 130.0, 140.0]
        m, c_x = BallBounceDetector.fit_linear_5pt(xs)
        self.assertAlmostEqual(m, 10.0, places=4)
        self.assertAlmostEqual(c_x, 120.0, places=4)
        sub_x = m * t_star + c_x
        self.assertAlmostEqual(sub_x, 122.5, places=4)

    def test_ground_bounce_derivative_detection(self):
        """W3-02 & W3-03: End-to-end ground bounce detection via vertical acceleration filter."""
        detector = BallBounceDetector(min_accel_thresh=2.0)

        # Simulated downward arc bouncing off court at frame 4:
        # Frames 2, 3, 4, 5, 6
        # Downward motion (increasing y): 300 -> 340 -> 370 (impact at 4) -> 345 -> 310
        stream = [
            {"frame_idx": 2, "timestamp_s": 0.066, "pixel": (400.0, 300.0), "speed_mph": 30.0, "conf": 0.9},
            {"frame_idx": 3, "timestamp_s": 0.100, "pixel": (420.0, 340.0), "speed_mph": 30.0, "conf": 0.9},
            {"frame_idx": 4, "timestamp_s": 0.133, "pixel": (440.0, 370.0), "speed_mph": 29.0, "conf": 0.9},
            {"frame_idx": 5, "timestamp_s": 0.166, "pixel": (460.0, 345.0), "speed_mph": 28.0, "conf": 0.9},
            {"frame_idx": 6, "timestamp_s": 0.200, "pixel": (480.0, 310.0), "speed_mph": 27.0, "conf": 0.9},
        ]

        bounces = []
        for pt in stream:
            ev = detector.update(pt, pt["frame_idx"], pt["timestamp_s"], self.calibrator)
            if ev is not None:
                bounces.append(ev)

        self.assertEqual(len(bounces), 1)
        ev = bounces[0]
        self.assertEqual(ev.frame_idx, 4)
        self.assertTrue(ev.is_ground_bounce)
        self.assertGreater(ev.vertical_accel, 2.0)
        self.assertIsNotNone(ev.court_xy_ft)
        self.assertAlmostEqual(ev.pixel_xy[0], 440.0, delta=10.0)
        self.assertGreaterEqual(ev.pixel_xy[1], 360.0)

    def test_paddle_volley_disambiguation(self):
        """W3-02: Verify that ball reversal occurring at player torso height is flagged as paddle volley."""
        detector = BallBounceDetector(min_accel_thresh=2.0)

        # Player at x in [400, 480], y in [200, 400]
        player = [{"id": 1, "box": [400, 200, 480, 400]}]

        stream = [
            {"frame_idx": 10, "timestamp_s": 0.333, "pixel": (440.0, 240.0), "speed_mph": 25.0},
            {"frame_idx": 11, "timestamp_s": 0.366, "pixel": (440.0, 270.0), "speed_mph": 25.0},
            {"frame_idx": 12, "timestamp_s": 0.400, "pixel": (440.0, 290.0), "speed_mph": 25.0},  # Torso hit at y=290
            {"frame_idx": 13, "timestamp_s": 0.433, "pixel": (440.0, 270.0), "speed_mph": 25.0},
            {"frame_idx": 14, "timestamp_s": 0.466, "pixel": (440.0, 240.0), "speed_mph": 25.0},
        ]

        bounces = []
        for pt in stream:
            ev = detector.update(pt, pt["frame_idx"], pt["timestamp_s"], self.calibrator, players=player)
            if ev is not None:
                bounces.append(ev)

        self.assertEqual(len(bounces), 1)
        ev = bounces[0]
        self.assertEqual(ev.frame_idx, 12)
        # Should be identified as paddle volley (not ground bounce)
        self.assertFalse(ev.is_ground_bounce)

    def test_minimap_bounce_ripples_and_decay(self):
        """W3-04: Test minimap bounce ripple addition and frame lifetime decay."""
        minimap = CourtMinimap(court_scale=10.0, padding=20, bounce_duration=5)
        self.assertEqual(len(minimap.active_bounces), 0)

        minimap.add_bounce((10.0, 30.0), frame_idx=50)
        self.assertEqual(len(minimap.active_bounces), 1)
        self.assertTrue(minimap.active_bounces[0].is_in_bounds)

        # Render 3 frames -> remaining should decrement
        minimap.render(frame_idx=51)
        self.assertEqual(minimap.active_bounces[0].remaining_frames, 4)

        minimap.render(frame_idx=52)
        minimap.render(frame_idx=53)
        minimap.render(frame_idx=54)
        minimap.render(frame_idx=55)

        # After 5 renders, remaining frames drop to 0 and ripple is removed
        self.assertEqual(len(minimap.active_bounces), 0)

    def test_schema_validation_week3_bounces(self):
        """W3-04 & W3-05: Ensure telemetry schema validates Week 3 bounce telemetry structure."""
        valid_doc = {
            "schema_version": SCHEMA_VERSION,
            "producer": SCHEMA_PRODUCER,
            "timestamp": "2026-11-26 12:00:00",
            "video_path": "clips/match1.mp4",
            "total_frames_processed": 5,
            "fps": 30.0,
            "resolution": "1920x1080",
            "elapsed_s": 0.15,
            "processing_fps": 33.3,
            "court": {
                "calibrated": True,
                "reprojection_rmse_ft": 0.05,
                "dimensions_ft": {"width": 20.0, "length": 44.0, "net_y": 22.0, "nvz_depth": 7.0},
                "corners_px": [[100, 100], [1100, 100], [1200, 650], [50, 650]],
            },
            "benchmark_summary": {
                "ball_detection_rate_pct": 100.0,
                "mean_players_per_frame": 4.0,
                "multiscale_inference": False,
                "stage_latencies_ms": {"player_ms": 10.0, "ball_ms": 12.0, "court_ms": 0.1, "render_ms": 2.0},
            },
            "bounces": [
                {
                    "frame_idx": 2,
                    "timestamp_s": 0.0667,
                    "sub_frame_offset": 0.12,
                    "pixel": [450.2, 380.5],
                    "court_ft": [10.5, 32.1],
                    "is_ground_bounce": True,
                    "confidence": 0.95,
                    "vertical_accel": 5.4,
                    "speed_mph": 26.4,
                }
            ],
            "telemetry_stream": [
                {
                    "frame_idx": 2,
                    "timestamp_s": 0.0667,
                    "court_calibrated": True,
                    "players": [
                        {
                            "id": 1,
                            "slot": 0,
                            "box": [100, 200, 150, 300],
                            "foot_px": [125, 300],
                            "court_ft": [5.0, 35.0],
                            "speed_mph": 2.1,
                            "side": "near",
                            "is_coasting": False,
                        }
                    ],
                    "ball": {
                        "pixel": [450.0, 380.0],
                        "box": [444, 374, 456, 386],
                        "conf": 0.92,
                        "speed_mph": 26.4,
                        "speed_px_per_frame": 12.5,
                        "court_ft": [10.5, 32.1],
                        "interpolated": False,
                    },
                    "bounce": {
                        "frame_idx": 2,
                        "timestamp_s": 0.0667,
                        "pixel": [450.2, 380.5],
                        "court_ft": [10.5, 32.1],
                        "is_ground_bounce": True,
                    },
                }
            ],
        }

        is_valid, errors = validate_telemetry_schema(valid_doc)
        self.assertTrue(is_valid, f"Schema validation errors: {errors}")


if __name__ == "__main__":
    unittest.main()
