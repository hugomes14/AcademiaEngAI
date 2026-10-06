"""Training provenance for deliberately unaudited experiments, without GPU imports."""

import json
from pathlib import Path
import tempfile
import unittest

from src.posturaai.dataset import SOURCE_SHA256
from src.posturaai.training import assert_training_gate, sha256_file


class ExperimentalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        srkd = self.root / "data/srkd"
        self.derived = srkd / "derived"
        self.derived.mkdir(parents=True)
        manifest = srkd / "group_manifest.csv"
        assignment = srkd / "provisional_assignment.csv"
        manifest.write_text("image_id,group_id,audit_status\n1,g1,pending\n")
        assignment.write_text("group_id,split\ng1,train\n")
        report = {
            "audit_status": "not_reviewed", "experimental": True,
            "audit_reviewer": None, "source_sha256": SOURCE_SHA256,
            "audit_bypass_reason": "explicit_unaudited_mode",
            "manifest_sha256": sha256_file(manifest),
            "assignment_sha256": sha256_file(assignment), "splits": {},
        }
        for split in ("train", "val"):
            path = self.derived / f"{split}.json"
            path.write_text('{"images": [], "annotations": []}')
            report["splits"][split] = {"sha256": sha256_file(path)}
        (self.derived / "preparation_report.json").write_text(json.dumps(report))

    def test_full_training_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "--allow-unaudited"):
            assert_training_gate(self.root, "1")
        metadata = assert_training_gate(self.root, "1", allow_unaudited=True)
        self.assertTrue(metadata["experimental"])
        self.assertEqual(metadata["audit_status"], "not_reviewed")
        self.assertFalse((self.root / "data/srkd/audit_approval.json").exists())

    def test_changed_annotations_or_assignment_cannot_reuse_preparation(self):
        for relative in ("derived/train.json", "derived/val.json", "provisional_assignment.csv", "group_manifest.csv"):
            with self.subTest(relative=relative):
                path = self.root / "data/srkd" / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(RuntimeError, "changed after preparation"):
                    assert_training_gate(self.root, "1", allow_unaudited=True)
                path.write_bytes(original)

    def test_smoke_retains_experimental_provenance(self):
        metadata = assert_training_gate(self.root, "smoke")
        self.assertTrue(metadata["smoke_only"])
        self.assertTrue(metadata["experimental"])


if __name__ == "__main__":
    unittest.main()
