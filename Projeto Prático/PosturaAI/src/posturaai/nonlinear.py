"""Fixed nonlinear candidates, exported as JSON for dependency-light inference."""
from collections import Counter
import numpy as np
from scipy.special import expit

CANDIDATES = ('logistic', 'random_forest', 'extra_trees', 'svm_rbf')


def fit_nonlinear(x, y, groups, algorithm):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from .classification import AGGREGATION, CLASSES, DESCRIPTOR_NAMES
    x, y, groups = np.asarray(x, float), np.asarray(y, int), np.asarray(groups)
    if x.ndim != 2 or x.shape[1] != len(DESCRIPTOR_NAMES) or len(x) != len(y) or len(y) != len(groups):
        raise ValueError('Invalid training arrays')
    if set(y.tolist()) != {0, 1}:
        raise ValueError('Training requires both posture classes')
    fill = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in x.T])
    imputed = np.where(np.isfinite(x), x, fill)
    mean, scale = imputed.mean(axis=0), imputed.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (imputed - mean) / scale
    class_groups = {label: set(groups[y == label]) for label in (0, 1)}
    counts = Counter(zip(y.tolist(), groups.tolist()))
    weights = np.array([1/(2*len(class_groups[label])*counts[(label, group)]) for label, group in zip(y, groups)])
    weights *= len(y) / weights.sum()
    if algorithm in ('random_forest', 'extra_trees'):
        estimator_class = RandomForestClassifier if algorithm == 'random_forest' else ExtraTreesClassifier
        parameters = dict(n_estimators=200, max_depth=6, min_samples_leaf=3,
                          max_features=0.7, random_state=42, n_jobs=1)
        estimator = estimator_class(**parameters)
    elif algorithm == 'svm_rbf':
        parameters = dict(C=1.0, kernel='rbf', gamma='scale', probability=False)
        estimator = SVC(**parameters)
    else:
        raise ValueError('Unknown nonlinear classifier: '+algorithm)
    estimator.fit(z, y, sample_weight=weights)
    model = dict(schema='posturaai.classifier.v2', algorithm=algorithm,
                 aggregation=AGGREGATION, classes=list(CLASSES), descriptor_names=list(DESCRIPTOR_NAMES),
                 impute=fill.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                 training_groups=sorted(set(groups.tolist())), experimental=True,
                 parameters=parameters, score_note='Uncalibrated model score, not a validated posture probability')
    if algorithm == 'svm_rbf':
        model.update(support_vectors=estimator.support_vectors_.tolist(),
                     dual_coef=estimator.dual_coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
                     gamma=float(estimator._gamma), score_transform='sigmoid of decision margin')
        expected = expit(estimator.decision_function(z))
    else:
        model['trees'] = []
        for tree_estimator in estimator.estimators_:
            tree = tree_estimator.tree_
            values = tree.value[:, 0, :]
            model['trees'].append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                                       feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                                       score_boa=(values[:, 1]/values.sum(axis=1)).tolist()))
        expected = estimator.predict_proba(z)[:, 1]
    # Catch serialization discrepancies before a model can be written or deployed.
    np.testing.assert_allclose(predict_nonlinear(model, z), expected, rtol=1e-7, atol=1e-9)
    return model


def predict_nonlinear(model, z):
    algorithm = model['algorithm']
    if algorithm == 'svm_rbf':
        support = np.asarray(model['support_vectors'])
        distance = np.maximum(0, (z*z).sum(axis=1)[:, None] + (support*support).sum(axis=1)[None, :] - 2*z@support.T)
        return expit(np.exp(-model['gamma']*distance)@np.asarray(model['dual_coef']) + model['intercept'])
    if algorithm not in ('random_forest', 'extra_trees'):
        raise ValueError('Unknown nonlinear model algorithm')
    # sklearn CART uses float32 input for comparisons against stored thresholds.
    z = z.astype(np.float32)
    result = np.zeros(len(z))
    rows = np.arange(len(z))
    for tree in model['trees']:
        nodes = np.zeros(len(z), dtype=int)
        left, right, feature = (np.asarray(tree[key]) for key in ('left', 'right', 'feature'))
        threshold = np.asarray(tree['threshold'])
        for _ in range(len(left)):
            active = left[nodes] != -1
            if not active.any():
                break
            at = nodes[active]
            nodes[active] = np.where(z[rows[active], feature[at]] <= threshold[at], left[at], right[at])
        else:
            raise ValueError('Invalid tree topology')
        result += np.asarray(tree['score_boa'])[nodes]
    return result / len(model['trees'])
