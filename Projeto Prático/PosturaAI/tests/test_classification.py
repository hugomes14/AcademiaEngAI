import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.posturaai.classification import DESCRIPTOR_NAMES, fit_classifier, grouped_validation, load_windows, predict_scores, window_descriptor, classification_metrics


class ClassifierTests(unittest.TestCase):
    def fixture(self):
        x=np.zeros((12,len(DESCRIPTOR_NAMES)))
        x[:6,0]=np.arange(6)*0.01-2
        x[6:,0]=np.arange(6)*0.01+2
        x[:,1]=np.nan
        y=np.r_[np.zeros(6,int),np.ones(6,int)]
        groups=np.array(['bad1']*3+['bad2']*3+['good1']*3+['good2']*3)
        return x,y,groups

    def test_grouped_validation_and_json_roundtrip(self):
        x,y,groups=self.fixture()
        result=grouped_validation(x,y,groups)
        self.assertEqual(result['metrics']['accuracy'],1)
        self.assertEqual(result['metrics']['per_class']['ma_postura']['average_precision'],1)
        for fold in result['folds']:
            self.assertNotIn(fold['held_out_group'],fold['training_groups'])
        model=fit_classifier(x,y,groups)
        restored=json.loads(json.dumps(model,allow_nan=False))
        np.testing.assert_allclose(predict_scores(model,x),predict_scores(restored,x))

    def test_preprocessing_fits_only_given_training_samples(self):
        x,y,groups=self.fixture()
        model=fit_classifier(x[:9],y[:9],groups[:9])
        self.assertAlmostEqual(model['mean'][0],np.mean(x[:9,0]))
        self.assertEqual(model['impute'][1],0)
        before=json.dumps(model,sort_keys=True)
        predict_scores(model,np.ones((1,len(DESCRIPTOR_NAMES)))*1e6)
        self.assertEqual(before,json.dumps(model,sort_keys=True))

    def test_groups_with_only_one_bad_recording_are_rejected(self):
        x,y,groups=self.fixture()
        groups[:6]='bad1'
        with self.assertRaisesRegex(ValueError,'two independent'):
            grouped_validation(x,y,groups)

    def test_descriptor_is_left_right_symmetric_and_ignores_velocity(self):
        f={side+'_'+joint+'_projected_angle_deg':90+index*5+(10 if side=='left' else 0)
           for index,joint in enumerate(('knee','hip','ankle','elbow','shoulder')) for side in ('left','right')}
        f.update(trunk_image_lean_deg=-15,head_trunk_projected_angle_deg=170)
        before=window_descriptor([{'features':f}])
        flipped={k.replace('left_','temporary_').replace('right_','left_').replace('temporary_','right_'):v for k,v in f.items()}
        flipped['trunk_image_lean_deg']=15
        flipped['left_knee_projected_angle_velocity_deg_s']=10000
        np.testing.assert_allclose(before,window_descriptor([{'features':flipped}]))

    def test_windows_do_not_cross_labels_or_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);poses=root/'p.jsonl';features=root/'f.jsonl'
            pose_rows=[dict(schema='posturaai.poses.v1',video_id='v',fps=10)]
            feature_rows=[dict(schema='posturaai.features.v1',video_id='v',fps=10)]
            values={side+'_'+joint+'_projected_angle_deg':90 for joint in ('knee','hip','ankle','elbow','shoulder') for side in ('left','right')}
            values.update(trunk_image_lean_deg=15,head_trunk_projected_angle_deg=160)
            for i in range(30):
                person=dict(track_id=1,segment_id=0,valid=[True]*30,features=values)
                feature_rows.append(dict(frame=i,timestamp_s=i/10,people=[] if i==15 else [person]))
                pose_rows.append(dict(frame=i,people=[] if i==15 else [dict(track_id=1,bbox_xyxy=[0,0,10,20])]))
            poses.write_text(''.join(json.dumps(r)+'\n' for r in pose_rows))
            features.write_text(''.join(json.dumps(r)+'\n' for r in feature_rows))
            intervals=[dict(start_s=0,end_s=1.2,label=0,evidence='bad'),dict(start_s=1.3,end_s=3,label=1,evidence='good')]
            _,windows=load_windows(features,poses,intervals=intervals)
            self.assertTrue(windows)
            for w in windows:
                self.assertFalse(w['start_frame']<=15<=w['end_frame'])
                self.assertTrue(w['end_s']<=1.2 or w['start_s']>=1.3)

    def test_average_precision_groups_tied_scores(self):
        metrics=classification_metrics([0,1],[0.5,0.5])
        self.assertEqual(metrics['per_class']['boa_postura']['average_precision'],0.5)


if __name__=='__main__':unittest.main()
