import unittest
import numpy as np

from src.posturaai.biomechanics import INDEX, angle_degrees, measure_pose, FeatureExtractor


def fixture():
    p = np.zeros((30, 2), dtype=float)
    for name, xy in {'Pelvis': [10,10], 'UpTrunk': [10,0], 'LeftShoulder': [10,0],
                     'LeftHip': [10,10], 'LeftKnee': [10,20], 'LeftAnkle': [20,20],
                     'LeftFirstMetatarsal': [25,20], 'LeftElbow': [15,5], 'LeftWrist': [20,0],
                     'RightHip': [15,10]}.items():
        p[INDEX[name]] = xy
    return p, np.ones(30, bool)


class GeometryTests(unittest.TestCase):
    def test_known_angles_and_degenerate_vectors(self):
        self.assertAlmostEqual(angle_degrees([0,1], np.array([0,0]), [1,0]), 90)
        self.assertAlmostEqual(angle_degrees([0,1], np.array([0,0]), [0,-1]), 180)
        self.assertIsNone(angle_degrees([0,0], np.array([0,0]), [1,0]))

    def test_unknown_and_mixed_views_preserve_geometry_without_interpretation(self):
        points, valid = fixture()
        for view in ('unknown', 'mixed', 'front', 'oblique'):
            result = measure_pose(points, valid, view=view)
            self.assertEqual(result['features']['left_knee_projected_angle_deg'], 90)
            self.assertIsNone(result['features']['trunk_forward_lean_deg'])
            self.assertFalse(result['interpretation']['sagittal_interpretation_eligible'])
            self.assertIsNone(result['posture_label'])

    def test_normalization_and_angles_are_translation_scale_invariant(self):
        p, v = fixture()
        first = measure_pose(p, v, view='side', direction='right')
        second = measure_pose(p*3+[100,200], v, view='side', direction='right')
        self.assertEqual(first['features']['left_knee_projected_angle_deg'], second['features']['left_knee_projected_angle_deg'])
        np.testing.assert_allclose(first['normalized_keypoints'], second['normalized_keypoints'])
        self.assertAlmostEqual(first['features']['left_ankle_forward_offset'], 1)
        third = measure_pose(p, v, view='side', direction='left')
        self.assertAlmostEqual(third['features']['left_ankle_forward_offset'], -1)

    def test_low_confidence_excludes_measures_and_fixed_camera_is_explicit(self):
        p, v = fixture()
        v[INDEX['LeftKnee']] = False
        result = measure_pose(p, v, view='side')
        self.assertIsNone(result['features']['left_knee_projected_angle_deg'])
        self.assertIsNone(result['features']['pelvis_image_y_px'])
        result = measure_pose(p, v, view='side', camera_motion='fixed')
        self.assertEqual(result['features']['pelvis_image_y_px'], 10)

    def test_temporal_derivatives_restart_after_gap_and_segment_change(self):
        p, v = fixture()
        extractor = FeatureExtractor(view='side')
        person = dict(track_id=1, segment_id=0, keypoints=np.column_stack([p, np.ones(30)]).tolist(), valid=v.tolist())
        feature = 'left_knee_projected_angle_velocity_deg_s'
        self.assertIsNone(extractor.process(0, 0, [person])[0]['features'][feature])
        self.assertEqual(extractor.process(1, 0.1, [person])[0]['features'][feature], 0)
        extractor.process(2, 0.2, [])
        self.assertIsNone(extractor.process(3, 0.3, [person])[0]['features'][feature])
        person['segment_id'] = 1
        self.assertIsNone(extractor.process(4, 0.4, [person])[0]['features'][feature])


if __name__ == '__main__':
    unittest.main()
