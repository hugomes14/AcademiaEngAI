import unittest
import numpy as np

from src.posturaai.temporal import PoseTracker, scene_difference, serialise_points, person_nms


class TrackingTests(unittest.TestCase):
    def test_nms_removes_duplicate_boxes_but_keeps_separate_people(self):
        boxes = [[0,0,20,40], [1,1,18,38], [100,0,120,40]]
        self.assertEqual(person_nms(boxes, [0.8,0.9,0.7]).tolist(), [1,2])
        self.assertEqual(person_nms([], []).tolist(), [])

    def test_detection_order_does_not_change_ids(self):
        tracker = PoseTracker()
        a, b = [0, 0, 10, 20], [100, 0, 110, 20]
        self.assertEqual(tracker.update([a, b], 0), [1, 2])
        self.assertEqual(tracker.update([b, a], 1), [2, 1])

    def test_ema_and_missing_point_recovery(self):
        tracker = PoseTracker(alpha=0.5)
        points, valid = np.zeros((30, 2)), np.ones(30, dtype=bool)
        tracker.update([[0, 0, 10, 20]], 0)
        tracker.smooth(1, points, valid)
        tracker.update([[0, 0, 10, 20]], 1)
        valid[0] = False
        result = tracker.smooth(1, points+10, valid)
        self.assertTrue(np.isnan(result[0]).all())
        np.testing.assert_allclose(result[1], [5, 5])
        tracker.update([[0, 0, 10, 20]], 2)
        valid[0] = True
        result = tracker.smooth(1, points+20, valid)
        np.testing.assert_allclose(result[0], [20, 20])
        self.assertIsNone(serialise_points([[np.nan, 0]], [0.1])[0][0])

    def test_gap_restarts_segment_and_smoothing(self):
        tracker = PoseTracker(alpha=0.1, max_gap=2)
        box = [[0, 0, 10, 20]]
        tracker.update(box, 0)
        tracker.smooth(1, np.zeros((30, 2)), np.ones(30, bool))
        tracker.update([], 1)
        self.assertEqual(tracker.update(box, 2), [1])
        self.assertEqual(tracker.tracks[1].segment_id, 1)
        np.testing.assert_allclose(tracker.smooth(1, np.ones((30, 2))*100, np.ones(30, bool)), 100)
        tracker.update([], 3)
        self.assertEqual(tracker.update(box, 6), [2])

    def test_scene_cut_never_reuses_id(self):
        tracker = PoseTracker()
        box = [[0, 0, 10, 20]]
        tracker.update(box, 0)
        tracker.reset()
        self.assertEqual(tracker.update(box, 1), [2])
        self.assertEqual(scene_difference(np.zeros((2,2,3)), np.ones((2,2,3))*255), 1)

    def test_gated_assignment_maximizes_feasible_pairs(self):
        # Ungated assignment could choose one high-IoU and one forbidden pair.
        from unittest.mock import patch
        tracker = PoseTracker(iou_threshold=0.3)
        tracker.update([[0,0,10,20], [50,0,60,20]], 0)
        with patch('src.posturaai.temporal.box_iou', side_effect=[0.9, 0.31, 0.31, 0.0]):
            self.assertEqual(tracker.update([[0,0,10,20], [50,0,60,20]], 1), [2,1])


if __name__ == '__main__':
    unittest.main()
