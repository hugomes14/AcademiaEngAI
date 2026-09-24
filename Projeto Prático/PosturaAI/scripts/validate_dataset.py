"""Validate the immutable SRKD source annotations and image files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.posturaai.dataset import validate_source

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="PosturaAI project root (default: directory containing scripts/)",
    )
    parser.add_argument(
        "--skip-image-dimensions",
        action="store_true",
        help="Skip opening image headers; useful only for a quick metadata check",
    )
    args = parser.parse_args(argv)
    report = validate_source(args.root, check_image_dimensions=not args.skip_image_dimensions)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
