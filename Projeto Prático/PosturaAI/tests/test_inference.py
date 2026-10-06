"""Reference-box inference must not silently associate an unrelated image."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.posturaai.dataset import SOURCE_JSON
from src.posturaai.inference import reference_bbox


class InferenceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / SOURCE_JSON
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "000001.jpeg", "width": 64, "height": 36}],
            "annotations": [{"image_id": 1, "bbox": [10, 30, 20, 12]}],
        }))
        self.image = source.parent / "Images/000001.jpeg"
        self.image.parent.mkdir()
        Image.new("RGB", (64, 36)).save(self.image)

    def test_reference_bbox_uses_exact_source_and_clips_known_overflow(self):
        self.assertEqual(reference_bbox(self.root, self.image), [10, 30, 20, 6])

    def test_same_basename_elsewhere_requires_explicit_box(self):
        duplicate = self.root / "other/000001.jpeg"
        duplicate.parent.mkdir()
        Image.new("RGB", (64, 36)).save(duplicate)
        with self.assertRaisesRegex(ValueError, "does not match"):
            reference_bbox(self.root, duplicate)
        with self.assertRaisesRegex(ValueError, "non-SRKD"):
            reference_bbox(self.root, self.root / "runner.jpeg")


if __name__ == "__main__":
    unittest.main()
