"""Casos pequenos com resultados conhecidos para as métricas de M1."""

import unittest

import numpy as np

from src.posturaai.metrics import evaluate_keypoints


class MetricsTests(unittest.TestCase):
    def test_normalization_pck_and_foot_group(self):
        targets = np.zeros((2, 30, 2))
        predictions = targets.copy()
        predictions[0, 0, 0] = 5.0
        predictions[1, 15, 0] = 10.0
        visibility = np.zeros((2, 30), dtype=np.uint8)
        visibility[0, 0] = 2
        visibility[1, 15] = 2
        boxes = np.array([[0, 0, 20, 100], [0, 0, 20, 100]])

        result = evaluate_keypoints(predictions, targets, boxes, visibility)

        self.assertEqual(result["global"]["count"], 2)
        self.assertAlmostEqual(result["global"]["nme"], 0.075)
        self.assertAlmostEqual(result["global"]["pck_005"], 0.5)
        self.assertAlmostEqual(result["feet"]["nme"], 0.1)
        self.assertEqual(result["feet"]["count"], 1)

    def test_invisible_points_do_not_affect_result(self):
        gt = np.zeros((1, 30, 2))
        pred = gt.copy()
        pred[0, 24] = [1000, 1000]
        visibility = np.zeros((1, 30), dtype=np.uint8)
        visibility[0, 0] = 2
        result = evaluate_keypoints(pred, gt, np.array([[0, 0, 5, 10]]), visibility)
        self.assertEqual(result["global"]["nme"], 0.0)
        self.assertIsNone(result["feet"]["nme"])

    def test_invalid_shapes_and_boxes_are_rejected(self):
        points = np.zeros((1, 30, 2))
        visibility = np.ones((1, 30))
        with self.assertRaises(ValueError):
            evaluate_keypoints(points, points, np.array([[0, 0, 1, 0]]), visibility)
        with self.assertRaises(ValueError):
            evaluate_keypoints(points[:, :29], points, np.array([[0, 0, 1, 1]]), visibility)
        with self.assertRaises(ValueError):
            evaluate_keypoints(points, points, np.array([[0, 0, 1, 1]]), np.zeros((1, 30)))


if __name__ == "__main__":
    unittest.main()
