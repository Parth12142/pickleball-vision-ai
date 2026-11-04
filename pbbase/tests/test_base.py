"""
test_base.py - Unit test suite for homography projection and tracking modules.
"""
import unittest
import numpy as np

from pbbase.court.homography import CourtGeometry, CourtCalibrator
from pbbase.tracking.player_tracker import CanonicalPlayerTracker, _box_iou
from pbbase.tracking.ball_trajectory import BallTrajectoryTracker, get_speed_color
from pbbase.models.detector import BallDetector, _merge_ball_candidates, _box_iou_px


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


if __name__ == "__main__":
    unittest.main()
