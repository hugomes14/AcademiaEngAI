import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.posturaai.video_features import export_features
from scripts.build_sequences import build_sequences
from tests.test_biomechanics import fixture


class ExportTests(unittest.TestCase):
    def test_export_missing_frames_and_unlabelled_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'runner.poses.jsonl'
            points, valid = fixture()
            person = dict(track_id=1, segment_id=0, valid=valid.tolist(),
                          keypoints=np.column_stack([points, np.ones(30)]).tolist(),
                          keypoints_raw=np.column_stack([points, np.ones(30)]).tolist())
            header = dict(schema='posturaai.poses.v1', video_id='video', fps=10,
                          athlete_id='runner1', source_id='recording', view='mixed')
            with source.open('w') as target:
                target.write(json.dumps(header)+'\n')
                for frame in range(45):
                    target.write(json.dumps(dict(type='frame', frame=frame, timestamp_s=frame/10,
                                                 people=[] if frame == 20 else [person]))+'\n')
            result = export_features(source, plot=False)
            features = [json.loads(line) for line in Path(result['jsonl']).read_text().splitlines()]
            self.assertEqual(features[0]['labels']['posture'], None)
            self.assertEqual(features[21]['people'], [])
            self.assertIsNone(features[22]['people'][0]['features']['left_knee_projected_angle_velocity_deg_s'])
            sequence_path = Path(directory) / 'sequences.jsonl'
            report = build_sequences(result['jsonl'], sequence_path, window_seconds=2, stride_seconds=1)
            self.assertGreater(report['sequences'], 0)
            self.assertGreater(report['rejected_windows'], 0)
            self.assertFalse(report['classification_ready'])
            windows = [json.loads(line) for line in sequence_path.read_text().splitlines()][1:]
            for window in windows:
                self.assertEqual(window['end_frame']-window['start_frame'], 19)
                self.assertFalse(window['start_frame'] <= 20 <= window['end_frame'])
                self.assertEqual(window['grouping_key'], 'runner1')
                self.assertIsNone(window['labels'])
                self.assertIsNone(window['split'])
                self.assertEqual(window['view'], 'mixed')
                self.assertEqual(len(window['features']), 20)
                self.assertEqual(len(window['normalized_keypoints'][0]), 30)
            with self.assertRaisesRegex(ValueError, 'differ'):
                export_features(source, source, plot=False)

    def test_recompute_view_context_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.poses.jsonl'
            path.write_text(json.dumps(dict(schema='posturaai.poses.v1', video_id='v', fps=25))+'\n')
            result = export_features(path, context_overrides={'view': 'side', 'direction': 'left'}, plot=False)
            header = json.loads(Path(result['jsonl']).read_text().splitlines()[0])
            self.assertEqual(header['context']['view'], 'side')
            self.assertEqual(header['context']['direction'], 'left')
            self.assertEqual(json.loads(Path(result['summary']).read_text())['person_observations'], 0)


if __name__ == '__main__':
    unittest.main()
