"""Compare fixed nonlinear families on cached training windows; never use the visual test for selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from src.posturaai.classification import DESCRIPTOR_NAMES, fit_classifier
from src.posturaai.model_selection import compare_candidates, nested_selection
from src.posturaai.nonlinear import CANDIDATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('outputs/classification/posture_baseline_02'))
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/classification/posture_nonlinear_03'))
    args = parser.parse_args()
    if args.source_run.resolve() == args.run_dir.resolve():
        parser.error('Comparison must preserve the previous run')
    source = args.source_run.resolve()
    baseline = json.loads((source/'classifier.json').read_text())
    annotations = source/'annotations.json'
    if hashlib.sha256(annotations.read_bytes()).hexdigest() != baseline['annotations_sha256']:
        raise ValueError('Source annotations changed after training')
    rows = [json.loads(line) for line in (source/'labelled_windows.jsonl').read_text().splitlines()[1:]]
    if set(row['video_sha256'] for row in rows) != set(baseline['trained_video_sha256']):
        raise ValueError('Training source hashes differ from cached model')
    x = np.array([row['descriptor'] for row in rows], float)
    y = np.array([row['label'] for row in rows], int)
    groups = np.array([row['group'] for row in rows])
    print(f'Comparing {CANDIDATES}: {len(y)} windows, {len(set(groups))} groups', flush=True)
    selected, comparison = compare_candidates(x, y, groups)
    for name, report in comparison.items():
        print(name, report['metrics'], flush=True)
    print('Evaluating nested model selection without outer-group leakage…', flush=True)
    nested = nested_selection(x, y, groups)
    model = fit_classifier(x, y, groups, algorithm=selected)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotations, run/'annotations.json')
    model.update(algorithm=selected, pose_checkpoint_sha256=baseline['pose_checkpoint_sha256'],
                 window_seconds=baseline['window_seconds'], stride_seconds=baseline['stride_seconds'],
                 min_valid_fraction=baseline['min_valid_fraction'], trained_video_sha256=baseline['trained_video_sha256'],
                 annotations_sha256=baseline['annotations_sha256'], annotation_file=str(run/'annotations.json'),
                 label_source=baseline['label_source'])
    report = dict(status='completed', selected_algorithm=selected, candidates=list(CANDIDATES),
                  samples=len(y), source_recordings=len(baseline['trained_video_sha256']), groups=len(set(groups)),
                  classes={'ma_postura': int((y == 0).sum()), 'boa_postura': int((y == 1).sum())},
                  comparison={name: {k: v for k, v in value.items() if k != 'scores'} for name, value in comparison.items()},
                  nested_validation={k: v for k, v in nested.items() if k != 'scores'},
                  selection_rule='Highest grouped balanced accuracy; macro F1 breaks ties. Fixed candidates and hyperparameters.',
                  test_video_used_for_selection=False, source_run=str(source),
                  windows_sha256=hashlib.sha256((source/'labelled_windows.jsonl').read_bytes()).hexdigest(),
                  limitations=['Only eight recordings in six groups; overlapping windows are correlated.',
                               'Winner grouped metrics are selection scores; nested metrics evaluate the selection procedure.',
                               '2D geometry and tutorial labels do not establish clinical correctness.'])
    (run/'classifier.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    (run/'comparison_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (run/'labelled_windows.jsonl').open('w') as target:
        target.write(json.dumps(dict(type='metadata', schema='posturaai.labelled_windows.v1',
                                     descriptor_names=list(DESCRIPTOR_NAMES), annotations_sha256=model['annotations_sha256']))+'\n')
        for index, row in enumerate(rows):
            row.update(selected_algorithm_oof_score=comparison[selected]['scores'][index],
                       nested_selection_oof_score=nested['scores'][index])
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(selected=selected, nested_metrics=nested['metrics'], run_dir=str(run)), indent=2))


if __name__ == '__main__':
    main()
