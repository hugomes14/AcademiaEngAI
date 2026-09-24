"""M1 RTMPose training configuration, gates, and an SRKD validation metric.

This module remains importable without PyTorch/OpenMMLab so that static config
checks and CLI help work before the dedicated GPU environment is installed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.posturaai.dataset import SOURCE_SHA256


OFFICIAL_BACKBONE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth"
)
CONFIG_FILE = "configs/rtmpose/rtmpose_s_srkd.py"
EXPECTED_METRIC = "srkd/NME"


@dataclass(frozen=True)
class StageSpec:
    name: str
    max_epochs: int
    frozen_stages: int
    head_lr: float
    backbone_lr: float
    patience: int | None


STAGES = {
    "smoke": StageSpec("smoke", 3, -1, 2e-5, 2e-5, None),
    "1": StageSpec("1", 10, 4, 1e-4, 0.0, 4),
    "2": StageSpec("2", 25, 2, 1e-4, 1e-5, 8),
    "3": StageSpec("3", 25, -1, 2e-5, 5e-6, 10),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_training_gate(root: Path, stage: str) -> dict[str, Any]:
    """Reject full training until the signed final split exists."""
    root = Path(root).resolve()
    derived = root / "data/srkd/derived"
    train_file, val_file = derived / "train.json", derived / "val.json"
    if not train_file.is_file():
        raise RuntimeError(f"Missing training annotations: {train_file}. Complete M0 preparation first.")
    if stage == "smoke":
        return {"train_sha256": sha256_file(train_file), "smoke_only": True}
    if not val_file.is_file():
        raise RuntimeError(f"Missing validation annotations: {val_file}. Complete M0 preparation first.")
    approval_path = root / "data/srkd/audit_approval.json"
    report_path = derived / "preparation_report.json"
    for path in (approval_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"M0 gate is incomplete: missing {path}")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"M0 gate metadata cannot be read: {exc}") from exc
    if approval.get("approved") is not True or not approval.get("reviewer"):
        raise RuntimeError("M0 group audit is not approved by a named reviewer")
    if approval.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("M0 approval references a different SRKD source hash")
    if not report.get("audit_reviewer"):
        raise RuntimeError("Final split preparation report lacks the audit reviewer")
    if report.get("audit_reviewer") != approval.get("reviewer"):
        raise RuntimeError("Final split audit reviewer disagrees with approval")
    return {
        "train_sha256": sha256_file(train_file),
        "val_sha256": sha256_file(val_file),
        "approval_sha256": sha256_file(approval_path),
        "preparation_report_sha256": sha256_file(report_path),
        "audit_reviewer": approval["reviewer"],
    }


def build_stage_config(
    base: dict[str, Any],
    *,
    stage: str,
    root: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    work_dir: Path | None = None,
    init_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
    backbone_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Apply stage overrides to a MMPose/MMEngine config or plain dict."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    if batch_size not in (16, 8, 4, 2):
        raise ValueError("Physical batch size must be one of 16, 8, 4, 2")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("--init-checkpoint and --resume-run are mutually exclusive")
    if stage in ("2", "3") and init_checkpoint is None and resume_checkpoint is None:
        raise ValueError(f"Stage {stage} requires --init-checkpoint from the previous stage")
    if stage in ("smoke", "1") and init_checkpoint is not None:
        raise ValueError(f"Stage {stage} initializes from the pretrained backbone, not a stage checkpoint")
    if backbone_checkpoint is not None and stage in ("2", "3"):
        raise ValueError("Backbone checkpoint is only used for smoke and stage 1")

    spec = STAGES[stage]
    cfg = deepcopy(base)
    root = Path(root).resolve()
    cfg["data_root"] = str(root)
    for loader_name in ("train_dataloader", "val_dataloader"):
        loader = cfg[loader_name]
        loader["batch_size"] = batch_size
        loader["num_workers"] = num_workers
        loader["persistent_workers"] = num_workers > 0
        loader["dataset"]["data_root"] = str(root)
        loader["dataset"]["metainfo"] = dict(from_file=str(root / "configs/datasets/srkd.py"))
    if stage == "smoke":
        cfg["train_dataloader"]["dataset"]["indices"] = 64
        cfg["val_dataloader"]["dataset"]["ann_file"] = "data/srkd/derived/train.json"
        cfg["val_dataloader"]["dataset"]["indices"] = 64
        cfg["train_dataloader"]["dataset"]["pipeline"] = deepcopy(cfg["val_pipeline"][:-1]) + [
            dict(type="GenerateTarget", encoder=cfg["codec"]),
            dict(type="PackPoseInputs"),
        ]
        # The same 64 training images are evaluated solely for an overfit check.
        cfg["default_hooks"]["checkpoint"]["save_best"] = None
        cfg["custom_hooks"] = []
    elif stage == "2":
        cfg["train_dataloader"]["dataset"]["pipeline"] = deepcopy(cfg["stage2_train_pipeline"])

    cfg["model"]["backbone"]["frozen_stages"] = spec.frozen_stages
    if stage in ("2", "3") or resume_checkpoint is not None:
        cfg["model"]["backbone"]["init_cfg"] = None
    elif backbone_checkpoint is not None:
        cfg["model"]["backbone"]["init_cfg"]["checkpoint"] = str(Path(backbone_checkpoint).resolve())
    if init_checkpoint is not None:
        cfg["load_from"] = str(Path(init_checkpoint).resolve())
        cfg["resume"] = False
    elif resume_checkpoint is not None:
        cfg["load_from"] = str(Path(resume_checkpoint).resolve())
        cfg["resume"] = True
    else:
        cfg["load_from"] = None
        cfg["resume"] = False

    cfg["train_cfg"]["max_epochs"] = spec.max_epochs
    wrapper = cfg["optim_wrapper"]
    wrapper["accumulative_counts"] = 64 // batch_size
    wrapper["optimizer"]["lr"] = spec.head_lr
    wrapper["paramwise_cfg"].pop("custom_keys", None)
    if spec.backbone_lr and spec.backbone_lr != spec.head_lr:
        wrapper["paramwise_cfg"]["custom_keys"] = {
            "backbone": dict(lr_mult=spec.backbone_lr / spec.head_lr)
        }
    if stage == "smoke":
        cfg["param_scheduler"] = [
            dict(type="CosineAnnealingLR", eta_min=1e-6, begin=0, end=spec.max_epochs, by_epoch=True)
        ]
    else:
        cfg["param_scheduler"][1]["end"] = spec.max_epochs
        cfg["param_scheduler"][1]["eta_min"] = min(spec.head_lr, spec.backbone_lr or spec.head_lr) * 0.05
        cfg["custom_hooks"][0]["patience"] = spec.patience
    if work_dir is not None:
        cfg["work_dir"] = str(Path(work_dir).resolve())
    return cfg


def preflight_runtime(stage: str) -> dict[str, Any]:
    """Import dependencies and verify a usable CUDA device before epochs."""
    missing: list[str] = []
    versions: dict[str, str] = {}
    for package in ("torch", "mmcv", "mmengine", "mmdet", "mmpose"):
        try:
            module = importlib.import_module(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except (ImportError, RuntimeError) as exc:
            missing.append(f"{package}: {exc}")
    if stage == "2":
        try:
            module = importlib.import_module("albumentations")
            versions["albumentations"] = str(getattr(module, "__version__", "unknown"))
            if int(versions["albumentations"].split(".")[0]) < 2:
                missing.append("albumentations>=2 is required for stage 2 transforms")
        except (ImportError, RuntimeError, ValueError) as exc:
            missing.append(f"albumentations>=2: {exc}")
    if missing:
        raise RuntimeError(
            "M1 runtime dependencies are unavailable or incompatible. Install a PyTorch/CUDA, "
            "MMCV 2.x, MMEngine, MMDetection 3.x and MMPose 1.x combination in the dedicated "
            "Python 3.11 environment before training. Details: " + "; ".join(missing)
        )
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. M1 training requires a verified local GPU runtime.")
    versions["cuda_runtime"] = str(torch.version.cuda)
    versions["gpu"] = torch.cuda.get_device_name(0)
    try:
        query = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
        versions["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        versions["nvidia_smi"] = str(exc)
    return versions


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def train_stage(
    *,
    root: Path,
    stage: str,
    batch_size: int = 16,
    num_workers: int = 4,
    run_id: str | None = None,
    init_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
    backbone_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run one stage after all gates and dependency checks pass."""
    root = Path(root).resolve()
    gate = assert_training_gate(root, stage)
    versions = preflight_runtime(stage)
    from mmengine.config import Config
    from mmengine.runner import Runner

    run_id = run_id or ("smoke" if stage == "smoke" else f"stage{stage}")
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    work_dir = root / "outputs" / run_id
    if resume_checkpoint is None and work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"Run directory already contains files: {work_dir}. Choose another --run-id.")
    base = Config.fromfile(str(root / CONFIG_FILE))
    cfg = build_stage_config(
        base,
        stage=stage,
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        work_dir=work_dir,
        init_checkpoint=init_checkpoint,
        resume_checkpoint=resume_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    effective_path = work_dir / "effective_config.py"
    cfg.dump(str(effective_path))
    metadata_path = work_dir / "run_metadata.json"
    metadata: dict[str, Any] = {
        "status": "starting",
        "stage": stage,
        "seed": 42,
        "max_epochs": STAGES[stage].max_epochs,
        "batch_size": batch_size,
        "accumulation": 64 // batch_size,
        "effective_batch_size": 64,
        "num_workers": num_workers,
        "frozen_stages": STAGES[stage].frozen_stages,
        "head_lr": STAGES[stage].head_lr,
        "backbone_lr": STAGES[stage].backbone_lr,
        "versions": versions,
        "git_commit": _git_commit(root),
        "plan_sha256": sha256_file(root / "rtmpose_srk_implementation_plan.md"),
        "effective_config_sha256": sha256_file(effective_path),
        "gate": gate,
        "initial_checkpoint": (
            str(init_checkpoint or resume_checkpoint or backbone_checkpoint or OFFICIAL_BACKBONE_URL)
        ),
        "initial_checkpoint_sha256": (
            sha256_file(init_checkpoint or resume_checkpoint or backbone_checkpoint)
            if (init_checkpoint or resume_checkpoint or backbone_checkpoint) else None
        ),
        "initialization": "resume" if resume_checkpoint else (
            "stage_weights_new_optimizer" if init_checkpoint else "pretrained_backbone_new_head"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        runner = Runner.from_cfg(cfg)
        # Catch operator/CUDA incompatibility before the first training epoch.
        import torch
        dummy = torch.zeros((1, 3, 256, 192), device="cuda")
        runner.model.cuda().eval()
        with torch.no_grad():
            output = runner.model(inputs=dummy, data_samples=None, mode="tensor")
        if output is None:
            raise RuntimeError("RTMPose forward preflight returned no output")
        runner.train()
        metadata["status"] = "completed"
        best = runner.message_hub.get_info("best_ckpt")
        if best and Path(best).is_file():
            shutil.copy2(best, work_dir / "best.pth")
            metadata["best_checkpoint"] = str(work_dir / "best.pth")
        last_record = work_dir / "last_checkpoint"
        if last_record.is_file():
            last_path = Path(last_record.read_text(encoding="utf-8").strip())
            if last_path.is_file():
                shutil.copy2(last_path, work_dir / "latest.pth")
                metadata["latest_checkpoint"] = str(work_dir / "latest.pth")
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _metric_from_data_samples(samples: list[Any]) -> dict[str, float]:
    """Use the project's exact bbox-height metric on MMPose pose samples."""
    import numpy as np
    from src.posturaai.metrics import evaluate_keypoints

    predictions, targets, bboxes, visibility = [], [], [], []
    for sample in samples:
        pred = np.asarray(sample["pred_instances"]["keypoints"])
        gt = sample["gt_instances"]
        target = np.asarray(gt["keypoints"])
        visible = np.asarray(gt["keypoints_visible"])
        xyxy = np.asarray(gt["bboxes"])
        if pred.shape != (1, 30, 2) or target.shape != (1, 30, 2) or xyxy.shape != (1, 4):
            raise ValueError("MMPose prediction/target must contain one person and 30 points")
        predictions.append(pred[0])
        targets.append(target[0])
        bboxes.append([xyxy[0, 0], xyxy[0, 1], xyxy[0, 2] - xyxy[0, 0], xyxy[0, 3] - xyxy[0, 1]])
        visibility.append(visible.reshape(1, 30)[0])
    metrics = evaluate_keypoints(
        np.asarray(predictions), np.asarray(targets), np.asarray(bboxes), np.asarray(visibility)
    )
    return {
        "NME": metrics["global"]["nme"],
        "PCK@0.05": metrics["global"]["pck_005"],
        "FeetNME": metrics["feet"]["nme"],
        "FeetPCK@0.05": metrics["feet"]["pck_005"],
        "VisiblePoints": metrics["global"]["count"],
        "VisibleFootPoints": metrics["feet"]["count"],
    }


try:
    from mmengine.evaluator import BaseMetric
    from mmpose.registry import METRICS
except ImportError:
    BaseMetric = object
    METRICS = None


if METRICS is not None:

    @METRICS.register_module()
    class SRKDBboxHeightMetric(BaseMetric):
        """NME/PCK from visible GT points divided by the derived GT box height."""

        def process(self, data_batch: Any, data_samples: list[Any]) -> None:
            self.results.extend(data_samples)

        def compute_metrics(self, results: list[Any]) -> dict[str, float]:
            return _metric_from_data_samples(results)
