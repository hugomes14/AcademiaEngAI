"""Create a provisional group assignment or approved derived COCO splits."""

import argparse
import json
from pathlib import Path

from src.posturaai.groups import final_split, provisional_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/srkd/group_manifest.csv"))
    parser.add_argument("--assignment", type=Path, default=Path("data/srkd/provisional_assignment.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("data/srkd/audit_decisions.csv"))
    parser.add_argument("--approval", type=Path, default=Path("data/srkd/audit_approval.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/srkd/derived"))
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--provisional", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    path = lambda value: value if value.is_absolute() else root / value
    try:
        if args.provisional:
            summary = provisional_split(root, path(args.manifest), path(args.assignment), args.seed)
        else:
            summary = final_split(root, path(args.manifest), path(args.assignment), path(args.decisions), path(args.approval), path(args.output_dir))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"prepare_splits: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
