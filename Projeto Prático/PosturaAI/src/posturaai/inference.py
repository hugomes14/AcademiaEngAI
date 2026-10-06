"""Single-person top-down pose inference; person detection belongs to M2."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.posturaai.dataset import SOURCE_JSON, resolve_image_path
from src.posturaai.keypoints import KEYPOINT_NAMES, SKELETON_EDGES
from src.posturaai.training import CONFIG_FILE, checkpoint_load_context, sha256_file


def reference_bbox(root: Path, image_path: Path) -> list[float]:
    """Find the SRKD reference box only if the exact original image matches."""
    try:
        image_id = int(image_path.stem)
    except ValueError as exc:
        raise ValueError("For a non-SRKD image, provide --bbox X Y WIDTH HEIGHT") from exc
    source = json.loads((root / SOURCE_JSON).read_text(encoding="utf-8"))
    image = next((row for row in source["images"] if row["id"] == image_id), None)
    if image is None or resolve_image_path(root, image) != image_path.resolve():
        raise ValueError("Input does not match an original SRKD image; provide --bbox")
    annotation = next((row for row in source["annotations"] if row["image_id"] == image_id), None)
    if annotation is None:
        raise ValueError("SRKD input has no reference annotation")
    x, y, w, h = annotation["bbox"]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image["width"], x + w), min(image["height"], y + h)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def infer_image(root: Path, checkpoint: Path, image_path: Path, output_dir: Path,
                *, bbox: list[float] | None = None, device: str = "cuda:0") -> dict:
    from mmengine.config import Config
    from mmpose.apis import inference_topdown, init_model

    root, checkpoint, image_path = root.resolve(), checkpoint.resolve(), image_path.resolve()
    if not checkpoint.is_file() or not image_path.is_file():
        raise ValueError("Checkpoint and input image must exist")
    bbox_source = "srkd_reference" if bbox is None else "supplied"
    bbox = reference_bbox(root, image_path) if bbox is None else list(bbox)
    if len(bbox) != 4 or not np.isfinite(bbox).all() or min(bbox[2:]) <= 0:
        raise ValueError("Bounding box must contain finite X Y and positive WIDTH HEIGHT")
    config_path = checkpoint.parent / "effective_config.py"
    if not config_path.exists():
        config_path = root / CONFIG_FILE
    cfg = Config.fromfile(str(config_path))
    cfg.model.backbone.init_cfg = None
    cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset.metainfo = dict(from_file=str(root / "configs/datasets/srkd.py"))
    with checkpoint_load_context():
        model = init_model(cfg, str(checkpoint), device=device)
        results = inference_topdown(model, str(image_path), bboxes=np.asarray([bbox], dtype=np.float32), bbox_format="xywh")
    if len(results) != 1:
        raise RuntimeError("Expected one pose for the supplied bounding box")
    points = np.asarray(results[0].pred_instances.keypoints)[0]
    scores = np.asarray(results[0].pred_instances.keypoint_scores)[0]
    if points.shape != (30, 2) or scores.shape != (30,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise RuntimeError("Pose prediction must contain 30 finite points and confidence scores")
    run_metadata = checkpoint.parent / "run_metadata.json"
    provenance = json.loads(run_metadata.read_text()) if run_metadata.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{checkpoint.parent.name}_{checkpoint.stem}"
    overlay_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"
    if overlay_path.resolve() == image_path:
        raise ValueError("Output overlay cannot overwrite the input image")
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for a, b in SKELETON_EDGES:
        draw.line([tuple(points[a]), tuple(points[b])], fill="#00c9a7", width=2)
    for index, (x, y) in enumerate(points):
        draw.ellipse((float(x - 3), float(y - 3), float(x + 3), float(y + 3)), fill="#ff9933")
        draw.text((float(x + 4), float(y - 4)), str(index), fill="white")
    canvas.save(overlay_path, quality=92)
    payload = {
        "image": str(image_path), "bbox_xywh": bbox,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "audit_status": provenance.get("audit_status", "unknown"),
        "experimental": provenance.get("experimental", True),
        "keypoint_names": list(KEYPOINT_NAMES),
        "keypoints": np.column_stack((points, scores)).tolist(),
        "overlay": str(overlay_path.resolve()),
        "bbox_source": bbox_source, "skeleton_status": "not_reviewed",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"overlay": str(overlay_path.resolve()), "prediction": str(json_path.resolve()), "keypoints": 30}
