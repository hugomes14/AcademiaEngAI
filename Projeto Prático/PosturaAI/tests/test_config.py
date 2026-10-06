"""A configuração de dataset mantém o contrato declarado no JSON bruto."""

import unittest
import importlib.util
from pathlib import Path

from configs.datasets.srkd import dataset_info


class ConfigTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mmpose") is not None, "MMPose runtime not installed")
    def test_mmpose_parser_loads_eager_srkd_config(self):
        from mmpose.datasets.datasets.utils import parse_pose_metainfo

        config = Path(__file__).resolve().parents[1] / "configs/datasets/srkd.py"
        parsed = parse_pose_metainfo(dict(from_file=str(config)))
        self.assertEqual(parsed["num_keypoints"], 30)
        self.assertEqual(len(parsed["flip_indices"]), 30)
        self.assertEqual(len(parsed["skeleton_links"]), 33)

    def test_srkd_metainfo_has_thirty_channels_and_uncalibrated_sentinels(self):
        self.assertEqual(len(dataset_info["keypoint_info"]), 30)
        self.assertEqual(len(dataset_info["skeleton_info"]), 33)
        self.assertEqual(len(dataset_info["joint_weights"]), 30)
        self.assertEqual(dataset_info["sigmas"], [0.0] * 30)
        self.assertFalse(dataset_info["sigmas_calibrated"])


if __name__ == "__main__":
    unittest.main()
