"""Extract standalone features from saved poses, without Torch/MMPose imports."""

import csv
import json
from pathlib import Path
import numpy as np

from src.posturaai.biomechanics import FEATURE_SPEC, FeatureExtractor


def export_features(input_path, output_path=None, *, context_overrides=None, plot=True):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(input_path.stem.replace('.poses', '') + '.features.jsonl')
    if output_path.resolve() == input_path.resolve():
        raise ValueError('Feature output must differ from pose input')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, all_values = [], {}
    raw_motion, smooth_motion, score_changes = [], [], []
    previous = {}
    with input_path.open() as source:
        header = json.loads(next(source))
        if header.get('schema') != 'posturaai.poses.v1':
            raise ValueError('Expected posturaai.poses.v1 input')
        context = dict(view=header.get('view', 'unknown'), direction=header.get('running_direction', 'unknown'),
                       visible_side=header.get('visible_side', 'unknown'), camera_motion=header.get('camera_motion', 'unknown'))
        context.update(context_overrides or {})
        extractor = FeatureExtractor(**context)
        metadata = dict(type='metadata', schema='posturaai.features.v1', source_poses=str(input_path.resolve()),
                        video_id=header['video_id'], athlete_id=header.get('athlete_id'), source_id=header.get('source_id'),
                        fps=header['fps'], context=context, feature_spec=FEATURE_SPEC,
                        labels={'posture': None, 'review_status': 'unlabelled'},
                        model_checkpoint_sha256=header.get('checkpoint_sha256'))
        with output_path.open('w') as output:
            output.write(json.dumps(metadata, allow_nan=False)+'\n')
            for line in source:
                frame = json.loads(line)
                if frame.get('type') != 'frame':
                    continue
                people = extractor.process(frame['frame'], frame['timestamp_s'], frame['people'])
                output.write(json.dumps(dict(type='frame', frame=frame['frame'], timestamp_s=frame['timestamp_s'], people=people), allow_nan=False)+'\n')
                for person in people:
                    row = dict(video_id=header['video_id'], athlete_id=header.get('athlete_id'), source_id=header.get('source_id'),
                               frame=frame['frame'], timestamp_s=frame['timestamp_s'], track_id=person['track_id'], segment_id=person['segment_id'],
                               view=context['view'], valid_points=sum(person['valid']), **person['features'])
                    rows.append(row)
                    for name, value in person['features'].items():
                        if value is not None:
                            all_values.setdefault(name, []).append(value)
                current = {}
                for person in frame['people']:
                    key = (person['track_id'], person['segment_id'])
                    raw = np.asarray(person['keypoints_raw'], dtype=float)
                    smooth = np.asarray(person['keypoints'], dtype=float)
                    valid = np.asarray(person['valid'], dtype=bool)
                    before = previous.get(key)
                    if before and before['frame'] == frame['frame']-1:
                        both = valid & before['valid']
                        if both.any():
                            raw_motion.extend(np.linalg.norm(raw[both, :2]-before['raw'][both, :2], axis=1).tolist())
                            smooth_motion.extend(np.linalg.norm(smooth[both, :2]-before['smooth'][both, :2], axis=1).tolist())
                            score_changes.extend(np.abs(raw[both, 2]-before['raw'][both, 2]).tolist())
                    current[key] = dict(frame=frame['frame'], raw=raw, smooth=smooth, valid=valid)
                previous = current
    csv_path = output_path.with_suffix('.csv')
    with csv_path.open('w', newline='') as target:
        names = list(rows[0]) if rows else ['video_id', 'frame', 'track_id']
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(person_observations=len(rows),
                   features={name: dict(count=len(values), mean=float(np.mean(values)),
                                        median=float(np.median(values)), p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)))
                             for name, values in all_values.items()},
                   temporal_quality=dict(raw_mean_displacement_px=float(np.mean(raw_motion)) if raw_motion else None,
                                         smoothed_mean_displacement_px=float(np.mean(smooth_motion)) if smooth_motion else None,
                                         mean_absolute_score_change=float(np.mean(score_changes)) if score_changes else None,
                                         note='Frame displacement includes real motion; reduction is not evidence of accuracy or jitter removal.'))
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    plot_path = None
    if plot and rows:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        series = [('left_knee_projected_angle_deg', 'right_knee_projected_angle_deg'),
                  ('trunk_image_lean_deg',), ('valid_points',)]
        for ax, names in zip(axes, series):
            labelled = set()
            for key in sorted({(r['track_id'], r['segment_id']) for r in rows}):
                group = [r for r in rows if (r['track_id'], r['segment_id']) == key]
                for name in names:
                    label_key = (key[0], name)
                    label = f'{name} / track {key[0]}' if label_key not in labelled else '_nolegend_'
                    labelled.add(label_key)
                    # Fixed colour per feature so segment resets do not change it.
                    ax.plot([r['timestamp_s'] for r in group], [np.nan if r.get(name) is None else r[name] for r in group],
                            label=label, color=f'C{names.index(name)}', linewidth=1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
        axes[0].set_ylabel('Projected angle (degrees)')
        axes[1].set_ylabel('Screen lean (degrees)')
        axes[2].set_ylabel('Valid points / 30')
        axes[2].set_xlabel('Time (seconds)')
        fig.suptitle('PosturaAI — 2D image measurements; no posture labels')
        fig.tight_layout()
        plot_path = output_path.with_suffix('.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return dict(jsonl=str(output_path.resolve()), csv=str(csv_path.resolve()), summary=str(summary_path.resolve()),
                plot=str(plot_path.resolve()) if plot_path else None)
