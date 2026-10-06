"""Recompute measures from exported keypoints without rerunning pose inference."""

import argparse
import json
from pathlib import Path

from src.posturaai.biomechanics import VIEWS, DIRECTIONS, SIDES, CAMERA_MOTIONS
from src.posturaai.video_features import export_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--view', choices=VIEWS)
    parser.add_argument('--running-direction', choices=DIRECTIONS)
    parser.add_argument('--visible-side', choices=SIDES)
    parser.add_argument('--camera-motion', choices=CAMERA_MOTIONS)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    overrides = {k: v for k, v in dict(view=args.view, direction=args.running_direction,
                                    visible_side=args.visible_side, camera_motion=args.camera_motion).items() if v is not None}
    print(json.dumps(export_features(args.input, args.output, context_overrides=overrides, plot=not args.no_plot), indent=2))


if __name__ == '__main__':
    main()
