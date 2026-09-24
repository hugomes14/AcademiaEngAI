"""Generate review sheets, record manual decisions, and approve audited groups."""

import argparse
import json
from pathlib import Path

from src.posturaai.groups import apply_same_decisions, audit_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("data/srkd/group_manifest.csv"))
    parser.add_argument("--assignment", type=Path, default=Path("data/srkd/provisional_assignment.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("data/srkd/group_candidates.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("data/srkd/audit_decisions.csv"))
    parser.add_argument("--approval", type=Path, default=Path("data/srkd/audit_approval.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/srkd/derived/group_features.npy"))
    parser.add_argument("--sheets", type=Path, default=Path("outputs/visualizations/group_audit"))
    parser.add_argument("--output", type=Path, default=Path("docs/split_audit.md"))
    parser.add_argument("--approve", action="store_true", help="Approve only after all review decisions are recorded")
    parser.add_argument("--reviewer", default="", help="Name of the human reviewer")
    parser.add_argument("--apply-decisions", action="store_true", help="Merge groups marked same; then regenerate provisional split")
    args = parser.parse_args()
    root = args.root.resolve()
    path = lambda value: value if value.is_absolute() else root / value
    try:
        if args.apply_decisions:
            summary = apply_same_decisions(root, path(args.manifest), path(args.decisions), path(args.approval))
        else:
            summary = audit_groups(root, path(args.manifest), path(args.assignment), path(args.candidates), path(args.decisions), path(args.output), path(args.approval), path(args.cache), path(args.sheets), args.approve, args.reviewer)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"audit_groups: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
