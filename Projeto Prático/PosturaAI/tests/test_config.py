"""A configuração de dataset mantém o contrato declarado no JSON bruto."""

import unittest

from configs.datasets.srkd import dataset_info


class ConfigTests(unittest.TestCase):
    def test_srkd_metainfo_has_thirty_channels_and_uncalibrated_sentinels(self):
        self.assertEqual(len(dataset_info["keypoint_info"]), 30)
        self.assertEqual(len(dataset_info["skeleton_info"]), 33)
        self.assertEqual(len(dataset_info["joint_weights"]), 30)
        self.assertEqual(dataset_info["sigmas"], [0.0] * 30)
        self.assertFalse(dataset_info["sigmas_calibrated"])


if __name__ == "__main__":
    unittest.main()
