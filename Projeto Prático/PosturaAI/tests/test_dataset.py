"""Focused tests for source validation and non-destructive boundary fixes."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.posturaai.dataset import (
    EXPECTED_KEYPOINT_NAMES,
    derive_annotation,
    image_relative_path,
    source_path,
    validate_source,
)


class DerivationTests(unittest.TestCase):
    def test_clips_bbox_and_hides_outside_points_without_mutating_source(self) -> None:
        image = {"id": 1, "file_name": "000001.jpeg", "width": 640, "height": 360}
        points = [10, 10, 2] * 30
        points[15 * 3 : 15 * 3 + 3] = [100, 360, 2]
        annotation = {
            "id": 0,
            "image_id": 1,
            "bbox": [5, 300, 20, 66],
            "area": 1320,
            "keypoints": points,
            "num_keypoints": 30,
        }
        original = json.loads(json.dumps(annotation))

        derived, corrections = derive_annotation(annotation, image)

        self.assertEqual(annotation, original)
        self.assertEqual(derived["bbox"], [5, 300, 20, 60])
        self.assertEqual(derived["area"], 1200)
        self.assertEqual(derived["keypoints"][45:48], [0, 0, 0])
        self.assertEqual(derived["num_keypoints"], 29)
        self.assertEqual([row["field"] for row in corrections], ["bbox", "keypoint"])
        self.assertEqual(corrections[1]["keypoint_index"], 15)

    def test_clips_all_bbox_sides_and_rejects_empty_intersection(self) -> None:
        image = {"width": 640, "height": 360}
        annotation = {
            "id": 4,
            "image_id": 8,
            "bbox": [-5, -2, 650, 370],
            "area": 240500,
            "keypoints": [1, 1, 2] * 30,
            "num_keypoints": 30,
        }
        fixed, _ = derive_annotation(annotation, image)
        self.assertEqual(fixed["bbox"], [0, 0, 640, 360])
        self.assertEqual(fixed["area"], 640 * 360)
        annotation["bbox"] = [700, 10, 20, 20]
        with self.assertRaises(ValueError):
            derive_annotation(annotation, image)

    def test_maps_both_existing_image_trees(self) -> None:
        first = image_relative_path({"file_name": "001219.jpeg"})
        second = image_relative_path({"file_name": "001220.jpeg"})
        self.assertIn("/Images/001219.jpeg", first)
        self.assertIn("/Images/001220.jpeg", second)
        self.assertNotEqual(first.split("/Images/")[0], second.split("/Images/")[0])
        with self.assertRaises(ValueError):
            image_relative_path({"file_name": "../000001.jpeg"})


class ValidationTests(unittest.TestCase):
    def test_fixture_validation_reports_known_overflow_and_missing_file(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not available")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = {"id": 1, "file_name": "000001.jpeg", "width": 640, "height": 360}
            image_file = root / image_relative_path(image)
            image_file.parent.mkdir(parents=True)
            Image.new("RGB", (640, 360)).save(image_file)
            points = [1, 1, 2] * 30
            points[15 * 3 : 15 * 3 + 3] = [10, 360, 2]
            document = {
                "images": [image],
                "annotations": [{
                    "id": 0,
                    "image_id": 1,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [5, 300, 20, 66],
                    "area": 1320,
                    "keypoints": points,
                    "num_keypoints": 30,
                }],
                "categories": [{
                    "id": 1,
                    "name": "person",
                    "keypoints": list(EXPECTED_KEYPOINT_NAMES),
                    "skeleton": [[1, 2]],
                }],
            }
            json_file = source_path(root)
            json_file.write_text(json.dumps(document), encoding="utf-8")
            report = validate_source(root, strict_baseline=False)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["summary"]["present_images"], 1)
            self.assertEqual(report["known_anomalies"]["bbox_overflows"], 1)
            self.assertEqual(report["known_anomalies"]["visible_points_outside"], 1)
            image_file.unlink()
            report = validate_source(root, strict_baseline=False)
            self.assertFalse(report["ok"])
            self.assertIn("Missing image", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
