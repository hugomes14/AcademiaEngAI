"""Select a fixed classifier family with group isolation at both CV levels."""
import numpy as np
from .classification import classification_metrics, fit_classifier, grouped_validation, predict_scores
from .nonlinear import CANDIDATES


def compare_candidates(x, y, groups, candidates=CANDIDATES):
    reports = {name: grouped_validation(x, y, groups, algorithm=name) for name in candidates}
    selected = max(candidates, key=lambda name: (reports[name]['metrics']['balanced_accuracy'],
                                               reports[name]['metrics']['macro_f1']))
    return selected, reports


def nested_selection(x, y, groups, candidates=CANDIDATES):
    x, y, groups = np.asarray(x, float), np.asarray(y, int), np.asarray(groups)
    scores = np.full(len(y), np.nan)
    folds = []
    for held_out in sorted(set(groups.tolist())):
        test = groups == held_out
        train = ~test
        selected, inner = compare_candidates(x[train], y[train], groups[train], candidates)
        model = fit_classifier(x[train], y[train], groups[train], algorithm=selected)
        scores[test] = predict_scores(model, x[test])
        folds.append(dict(held_out_group=held_out, training_groups=sorted(set(groups[train].tolist())),
                          selected_algorithm=selected, train_samples=int(train.sum()), validation_samples=int(test.sum()),
                          inner_validation={name: {k: v for k, v in report.items() if k != 'scores'} for name, report in inner.items()},
                          metrics=classification_metrics(y[test], scores[test])))
    return dict(protocol='Nested leave-one-group-out; classifier family selected only within outer training groups',
                scores=scores.tolist(), folds=folds, metrics=classification_metrics(y, scores))
