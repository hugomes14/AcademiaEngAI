"""Infer 30 points in one image using an explicit box or an original SRKD reference box."""

import argparse
import json
from pathlib import Path

from src.posturaai.inference import infer_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/inference"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    path = lambda value: value if value.is_absolute() else root / value
    try:
        result = infer_image(root, path(args.checkpoint), path(args.input), path(args.output), bbox=args.bbox, device=args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"infer_image: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
