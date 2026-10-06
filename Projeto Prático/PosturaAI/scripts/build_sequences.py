"""Package unlabelled contiguous pose/feature windows for future classification."""

import argparse
import json
from pathlib import Path


def build_sequences(input_path, output_path, *, window_seconds=2.0, stride_seconds=1.0, min_valid_fraction=0.6):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output cannot overwrite feature input')
    if not 1 <= window_seconds <= 3 or stride_seconds <= 0 or not 0 <= min_valid_fraction <= 1:
        raise ValueError('Windows must be 1–3 seconds; positive stride; validity fraction in [0,1]')
    segments = {}
    with input_path.open() as source:
        header = json.loads(next(source))
        if header.get('schema') != 'posturaai.features.v1':
            raise ValueError('Expected posturaai.features.v1 input')
        fps = header['fps']
        for line in source:
            frame = json.loads(line)
            for person in frame.get('people', []):
                key = (person['track_id'], person['segment_id'])
                segments.setdefault(key, []).append(dict(frame=frame['frame'], timestamp_s=frame['timestamp_s'], **person))
    length, stride = round(window_seconds * fps), max(1, round(stride_seconds * fps))
    if length < 2:
        raise ValueError('Window requires at least two frames')
    count = rejected = 0
    feature_names = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as target:
        metadata = dict(type='metadata', schema='posturaai.sequences.v1', source_features=str(input_path.resolve()),
                        fps=fps, window_seconds=window_seconds, stride_seconds=stride_seconds,
                        min_valid_fraction=min_valid_fraction, context=header['context'],
                        athlete_id=header.get('athlete_id'), source_id=header.get('source_id'),
                        video_id=header['video_id'], model_checkpoint_sha256=header.get('model_checkpoint_sha256'),
                        feature_spec=header['feature_spec'], classification_ready=False,
                        label_status='unlabelled; manual labels and grouped splits required')
        target.write(json.dumps(metadata, allow_nan=False)+'\n')
        for (track, segment), observations in sorted(segments.items()):
            for start in range(0, len(observations)-length+1, stride):
                window = observations[start:start+length]
                contiguous = all(b['frame'] == a['frame']+1 and abs((b['timestamp_s']-a['timestamp_s']) - 1/fps) < 1e-5
                                 for a, b in zip(window, window[1:]))
                valid_fraction = sum(sum(p['valid']) for p in window) / (length*30)
                if not contiguous or valid_fraction < min_valid_fraction:
                    rejected += 1
                    continue
                feature_names = feature_names or list(window[0]['features'])
                values = [[p['features'].get(n) for n in feature_names] for p in window]
                record = dict(type='sequence', sequence_id=f"{header['video_id']}:t{track}:s{segment}:f{window[0]['frame']}",
                              video_id=header['video_id'], source_id=header.get('source_id'), athlete_id=header.get('athlete_id'),
                              grouping_key=header.get('athlete_id') or header.get('source_id') or header['video_id'],
                              track_id=track, segment_id=segment, start_frame=window[0]['frame'],
                              end_frame=window[-1]['frame'], timestamps_s=[p['timestamp_s'] for p in window],
                              view=header['context']['view'], valid_fraction=valid_fraction,
                              feature_names=feature_names, features=values,
                              feature_valid=[[v is not None for v in row] for row in values],
                              normalized_keypoints=[p['normalized_keypoints'] for p in window],
                              keypoint_valid=[p['valid'] for p in window], labels=None, split=None)
                target.write(json.dumps(record, allow_nan=False)+'\n')
                count += 1
    return dict(output=str(output_path.resolve()), sequences=count, rejected_windows=rejected,
                classification_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-seconds', type=float, default=2)
    parser.add_argument('--stride-seconds', type=float, default=1)
    parser.add_argument('--min-valid-fraction', type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(build_sequences(args.input, args.output, window_seconds=args.window_seconds,
                                    stride_seconds=args.stride_seconds, min_valid_fraction=args.min_valid_fraction), indent=2))


if __name__ == '__main__':
    main()
