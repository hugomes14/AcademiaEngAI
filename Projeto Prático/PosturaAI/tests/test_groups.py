"""Focused integration tests for the M0 group and split gates."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.posturaai.dataset import EXPECTED_KEYPOINT_NAMES
from src.posturaai.groups import (
    RAW_JSON, apply_same_decisions, audit_groups, build_groups, final_split,
    provisional_split, sha256_file,
)


def fixture(root: Path) -> None:
    images = []
    annotations = []
    image_dir = root / RAW_JSON.parent / "Images"
    image_dir.mkdir(parents=True)
    for group in range(10):
        for frame in range(4):
            image_id = group * 4 + frame + 1
            view = frame % 2
            image = np.zeros((36, 64, 3), dtype=np.uint8)
            image[:] = [25 + group * 20, 20 + group * 5, 160 - group * 10]
            image[:, :32] = np.clip(image[:, :32].astype(int) + (45 if view else 0), 0, 255)
            name = f"{image_id:06d}.jpeg"
            Image.fromarray(image).save(image_dir / name)
            images.append({"id": image_id, "width": 64, "height": 36, "file_name": name, "licence": 4})
            points = [20, 20, 2] * 30
            bbox = [10, 10, 20, 20]
            if image_id == 1:
                bbox = [10, 10, 20, 30]
                points[15 * 3:15 * 3 + 3] = [15, 38, 2]
            annotations.append({"id": image_id - 1, "image_id": image_id, "category_id": 1,
                                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                                "keypoints": points, "num_keypoints": 30})
    source = {"dataset": "fixture", "info": {}, "licences": [{"id": 4, "name": "test"}],
              "images": images, "annotations": annotations,
              "categories": [{"id": 1, "name": "person", "keypoints": list(EXPECTED_KEYPOINT_NAMES), "skeleton": []}]}
    (root / RAW_JSON).write_text(json.dumps(source), encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class GroupSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.manifest = self.root / "group_manifest.csv"
        self.cache = self.root / "group_features.npy"
        self.candidates = self.root / "group_candidates.csv"
        self.assignment = self.root / "provisional_assignment.csv"
        self.decisions = self.root / "audit_decisions.csv"
        self.approval = self.root / "audit_approval.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _build(self) -> None:
        summary = build_groups(self.root, self.manifest, self.cache, self.candidates)
        self.assertTrue(summary["alternating_cameras"])
        self.assertEqual(summary["images"], 40)

    def test_build_and_provisional_are_deterministic(self) -> None:
        self._build()
        rows = read_csv(self.manifest)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["group_id"] for row in rows}), 10)
        self.assertEqual({row["audit_status"] for row in rows}, {"pending"})
        provisional_split(self.root, self.manifest, self.assignment, 42)
        first = self.assignment.read_bytes()
        provisional_split(self.root, self.manifest, self.assignment, 42)
        self.assertEqual(first, self.assignment.read_bytes())
        self.assertEqual({row["split"] for row in read_csv(self.assignment)}, {"train", "val", "test"})
        with self.assertRaises(ValueError):
            final_split(self.root, self.manifest, self.assignment, self.decisions, self.approval, self.root / "derived")

    def test_review_queue_preserves_known_same_and_merge_invalidates_split(self) -> None:
        self._build()
        provisional_split(self.root, self.manifest, self.assignment, 42)
        report = self.root / "split_audit.md"
        audit_groups(self.root, self.manifest, self.assignment, self.candidates, self.decisions,
                     report, self.approval, self.cache, self.root / "sheets")
        decisions = read_csv(self.decisions)
        self.assertTrue(decisions)
        pair = next(row for row in decisions if row["group_id_a"] != row["group_id_b"])
        pair["decision"] = "same"
        pair["reviewer"] = "human"
        write_csv(self.decisions, decisions)
        # A reassignment may remove the pair from the top 100; it must persist.
        provisional_split(self.root, self.manifest, self.assignment, 17)
        audit_groups(self.root, self.manifest, self.assignment, self.candidates, self.decisions,
                     report, self.approval, self.cache, self.root / "sheets")
        self.assertTrue(any(row["decision"] == "same" for row in read_csv(self.decisions)))
        result = apply_same_decisions(self.root, self.manifest, self.decisions, self.approval)
        self.assertGreaterEqual(result["merged_groups"], 1)
        self.assertTrue(result["provisional_split_invalidated"])
        with self.assertRaises(ValueError):
            final_split(self.root, self.manifest, self.assignment, self.decisions, self.approval, self.root / "derived")

    def test_final_split_repairs_derived_only_after_approval(self) -> None:
        self._build()
        provisional_split(self.root, self.manifest, self.assignment, 42)
        rows = read_csv(self.manifest)
        for row in rows:
            row["audit_status"] = "approved"
        write_csv(self.manifest, rows)
        self.decisions.write_text("candidate_id,kind,image_id_a,image_id_b,group_id_a,group_id_b,distance,decision,reviewer,note\n", encoding="utf-8")
        self.approval.write_text(json.dumps({
            "approved": True, "reviewer": "fixture",
            "manifest_sha256": sha256_file(self.manifest),
            "assignment_sha256": sha256_file(self.assignment),
            "decisions_sha256": sha256_file(self.decisions),
            "source_sha256": sha256_file(self.root / RAW_JSON),
        }), encoding="utf-8")
        report = final_split(self.root, self.manifest, self.assignment, self.decisions,
                             self.approval, self.root / "derived")
        self.assertEqual(report["corrections"], 2)
        correction_rows = read_csv(self.root / "derived/corrections.csv")
        self.assertEqual({row["field"] for row in correction_rows}, {"bbox", "keypoint"})
        self.assertTrue(any(row["keypoint_index"] == "15" for row in correction_rows))
        all_images = []
        all_annotations = []
        for split in ("train", "val", "test"):
            document = json.loads((self.root / "derived" / f"{split}.json").read_text())
            all_images.extend(document["images"])
            all_annotations.extend(document["annotations"])
            self.assertIn("licenses", document)
            self.assertTrue(all(image["file_name"].startswith(str(RAW_JSON.parent)) for image in document["images"]))
        self.assertEqual(len(all_images), 40)
        repaired = next(annotation for annotation in all_annotations if annotation["image_id"] == 1)
        self.assertEqual(repaired["bbox"], [10, 10, 20, 26])
        self.assertEqual(repaired["keypoints"][45:48], [0, 0, 0])
        self.assertEqual(repaired["num_keypoints"], 29)
        raw = json.loads((self.root / RAW_JSON).read_text())
        self.assertEqual(raw["annotations"][0]["bbox"], [10, 10, 20, 30])
        self.assertEqual(raw["annotations"][0]["keypoints"][45:48], [15, 38, 2])


if __name__ == "__main__":
    unittest.main()
