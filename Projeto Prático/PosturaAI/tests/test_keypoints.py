"""Focused SRKD keypoint and annotation visualizer checks."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.visualize_annotations import RAW_IMAGE_DIRS, run, select_samples
from src.posturaai.keypoints import (
    FLIP_PAIRS,
    KEYPOINT_NAMES,
    SKELETON_EDGES,
    flip_indices,
    srkd_metainfo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ANNOTATIONS = PROJECT_ROOT / "Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json"


def fake_annotation(image_id: int, foot_y: int = 15) -> dict:
    values = [10, 15, 2] * 30
    values[15 * 3 + 1] = foot_y
    return dict(id=image_id, image_id=image_id, category_id=1, bbox=[5, 5, 15, 20], keypoints=values, num_keypoints=30)


class KeypointsTests(unittest.TestCase):
    def test_constants_match_raw_category(self) -> None:
        with RAW_ANNOTATIONS.open(encoding="utf-8") as stream:
            source = json.load(stream)
        category = source["categories"][0]
        self.assertEqual(tuple(category["keypoints"]), KEYPOINT_NAMES)
        source_edges = tuple(dict.fromkeys(tuple(sorted((a - 1, b - 1))) for a, b in category["skeleton"]))
        self.assertEqual(source_edges, SKELETON_EDGES)
        self.assertEqual(len(KEYPOINT_NAMES), 30)
        self.assertEqual(len(SKELETON_EDGES), 33)
        self.assertEqual(len(category["skeleton"]), 35)

    def test_flip_is_involution_and_covers_expected_pairs(self) -> None:
        self.assertEqual(len(FLIP_PAIRS), 12)
        permutation = flip_indices()
        self.assertEqual(tuple(permutation[i] for i in permutation), tuple(range(30)))
        self.assertEqual({i for pair in FLIP_PAIRS for i in pair}, set(range(1, 25)))
        self.assertEqual([permutation[i] for i in (0, 25, 26, 27, 28, 29)], [0, 25, 26, 27, 28, 29])

    def test_mmpose_metainfo_has_30_channels_and_only_sentinel_sigmas(self) -> None:
        meta = srkd_metainfo()
        self.assertEqual(len(meta["keypoint_info"]), 30)
        self.assertEqual(len(meta["skeleton_info"]), 33)
        self.assertEqual(len(meta["joint_weights"]), 30)
        self.assertEqual(meta["sigmas"], [0.0] * 30)
        self.assertFalse(meta["sigmas_calibrated"])
        for left, right in FLIP_PAIRS:
            self.assertEqual(meta["keypoint_info"][left]["swap"], KEYPOINT_NAMES[right])
            self.assertEqual(meta["keypoint_info"][right]["swap"], KEYPOINT_NAMES[left])

    def test_stratified_sample_is_deterministic_and_prioritizes_low_feet(self) -> None:
        images = [dict(id=i, file_name=f"{i:06d}.jpeg", width=32, height=32) for i in range(1, 101)]
        annotations = {i: fake_annotation(i, 30 if i % 10 == 0 else 15) for i in range(1, 101)}
        first = select_samples(images, annotations, 20, 42)
        second = select_samples(list(reversed(images)), annotations, 20, 42)
        self.assertEqual(first, second)
        self.assertEqual(len({image["id"] for image, *_ in first}), 20)
        self.assertEqual({interval for _, _, interval, _ in first}, set(range(10)))
        self.assertEqual(sum(reason == "foot_near_lower_edge" for *_, reason in first), 10)

    def test_render_finds_both_raw_image_directories_and_records_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            annotations = []
            for image_id, image_dir in enumerate(RAW_IMAGE_DIRS, start=1):
                path = root / image_dir / f"{image_id:06d}.jpeg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (20, 20, 20)).save(path)
                images.append(dict(id=image_id, file_name=path.name, width=32, height=32))
                annotations.append(fake_annotation(image_id, 30))
            annotation_path = root / "raw.json"
            annotation_path.write_text(json.dumps(dict(
                categories=[dict(id=1, keypoints=list(KEYPOINT_NAMES))],
                images=images,
                annotations=annotations,
            )), encoding="utf-8")
            manifest_path = run(root, Path("raw.json"), 2, Path("output"), 42)
            with manifest_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({int(row["image_id"]) for row in rows}, {1, 2})
            self.assertTrue(all((root / "output" / row["overlay"]).is_file() for row in rows))


if __name__ == "__main__":
    unittest.main()
