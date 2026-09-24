"""Run one RTMPose-S training stage after the M0 audit and GPU checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from src.posturaai.training import CONFIG_FILE, STAGES, build_stage_config, train_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resume_checkpoint(value: Path | None) -> Path | None:
    if value is None:
        return None
    if value.is_dir():
        record = value / "last_checkpoint"
        if not record.is_file():
            raise ValueError(f"Run directory has no last_checkpoint record: {value}")
        value = Path(record.read_text(encoding="utf-8").strip())
    if not value.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {value}")
    return value.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES), help="smoke, 1, 2 or 3")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="PosturaAI project root")
    parser.add_argument("--batch-size", type=int, default=16, choices=(16, 8, 4, 2))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-id", help="Output directory name under outputs/")
    parser.add_argument("--init-checkpoint", type=Path, help="Best checkpoint from the preceding stage")
    parser.add_argument("--resume-run", type=Path, help="Previous run directory or its exact latest checkpoint")
    parser.add_argument("--backbone-checkpoint", type=Path, help="Local official CSPNeXt checkpoint for smoke/stage 1")
    parser.add_argument("--show-config", action="store_true", help="Show resolved stage settings without training or GPU imports")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        resume = _resume_checkpoint(args.resume_run)
        for path in (args.init_checkpoint, args.backbone_checkpoint):
            if path is not None and not path.is_file():
                raise ValueError(f"Checkpoint does not exist: {path}")
        if args.show_config:
            raw = runpy.run_path(str(root / CONFIG_FILE))
            config = build_stage_config(
                raw,
                stage=args.stage,
                root=root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                init_checkpoint=args.init_checkpoint,
                resume_checkpoint=resume,
                backbone_checkpoint=args.backbone_checkpoint,
            )
            summary = {
                "stage": args.stage,
                "max_epochs": config["train_cfg"]["max_epochs"],
                "input_size": config["input_size"],
                "out_channels": config["model"]["head"]["out_channels"],
                "frozen_stages": config["model"]["backbone"]["frozen_stages"],
                "optimizer": config["optim_wrapper"]["optimizer"],
                "accumulative_counts": config["optim_wrapper"]["accumulative_counts"],
                "checkpoint_metric": config["default_hooks"]["checkpoint"]["save_best"],
                "val_evaluator": config["val_evaluator"],
                "train_ann_file": config["train_dataloader"]["dataset"]["ann_file"],
                "val_ann_file": config["val_dataloader"]["dataset"]["ann_file"],
                "init_checkpoint": config["load_from"],
                "resume": config["resume"],
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        metadata = train_stage(
            root=root,
            stage=args.stage,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            run_id=args.run_id,
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=resume,
            backbone_checkpoint=args.backbone_checkpoint,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Training could not start: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
