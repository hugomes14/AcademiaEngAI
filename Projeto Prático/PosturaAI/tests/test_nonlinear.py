import json
import unittest
import numpy as np
from src.posturaai.classification import DESCRIPTOR_NAMES, fit_classifier, predict_scores
from src.posturaai.model_selection import nested_selection


class NonlinearTests(unittest.TestCase):
    def fixture(self):
        # XOR cannot be separated by the existing linear decision boundary.
        corners = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])
        x = np.zeros((64, len(DESCRIPTOR_NAMES)))
        x[:, :2] = np.tile(corners, (16, 1))
        x[:, 2] = np.nan
        y = np.tile([0, 1, 1, 0], 16)
        groups = np.repeat(['g'+str(i) for i in range(8)], 8)
        return x, y, groups

    def test_nonlinear_families_learn_xor_and_survive_json_roundtrip(self):
        x, y, groups = self.fixture()
        linear = predict_scores(fit_classifier(x, y, groups), x)
        self.assertLessEqual(np.mean((linear >= .5) == y), .5)
        for algorithm in ('random_forest', 'extra_trees', 'svm_rbf'):
            with self.subTest(algorithm=algorithm):
                model = fit_classifier(x, y, groups, algorithm=algorithm)
                before = json.dumps(model, allow_nan=False, sort_keys=True)
                restored = json.loads(before)
                scores = predict_scores(restored, x)
                self.assertGreater(np.mean((scores >= .5) == y), .95)
                np.testing.assert_allclose(scores, predict_scores(model, x))
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(before, json.dumps(model, allow_nan=False, sort_keys=True))
                self.assertEqual(predict_scores(model, x[:0]).shape, (0,))

    def test_nested_selection_keeps_outer_group_out_of_every_inner_fit(self):
        x, y, groups = self.fixture()
        result = nested_selection(x, y, groups, candidates=('logistic', 'svm_rbf'))
        self.assertEqual(result['metrics']['accuracy'], 1)
        for fold in result['folds']:
            held_out = fold['held_out_group']
            self.assertNotIn(held_out, fold['training_groups'])
            for inner in fold['inner_validation'].values():
                for inner_fold in inner['folds']:
                    self.assertNotIn(held_out, inner_fold['training_groups'])
                    self.assertNotEqual(held_out, inner_fold['held_out_group'])
                    self.assertNotIn(inner_fold['held_out_group'], inner_fold['training_groups'])

    def test_preprocessing_fits_training_data_only_for_all_families(self):
        x, y, groups = self.fixture()
        for algorithm in ('random_forest', 'extra_trees', 'svm_rbf'):
            model = fit_classifier(x[:20], y[:20], groups[:20], algorithm=algorithm)
            self.assertEqual(model['impute'][2], 0)
            self.assertAlmostEqual(model['mean'][0], np.mean(x[:20, 0]))
            before = json.dumps(model, sort_keys=True)
            predict_scores(model, np.full((2, len(DESCRIPTOR_NAMES)), 10000.))
            self.assertEqual(before, json.dumps(model, sort_keys=True))
