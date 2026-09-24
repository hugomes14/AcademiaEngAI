"""Build an initial SRKD group manifest and visual evidence."""

import argparse
import json
from pathlib import Path

from src.posturaai.groups import build_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("data/srkd/group_manifest.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/srkd/derived/group_features.npy"))
    parser.add_argument("--candidates", type=Path, default=Path("data/srkd/group_candidates.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    path = lambda value: value if value.is_absolute() else root / value
    try:
        summary = build_groups(root, path(args.output), path(args.cache), path(args.candidates))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"build_groups: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
