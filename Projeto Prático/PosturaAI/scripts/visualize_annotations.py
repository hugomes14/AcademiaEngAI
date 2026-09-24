"""Render a reproducible, stratified sample of raw SRKD COCO annotations.

Run from the PosturaAI root with ``python -m scripts.visualize_annotations``.
The source images and annotation JSON are read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw
from src.posturaai.keypoints import (
    FOOT_INDICES,
    KEYPOINT_NAMES,
    SKELETON_EDGES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ANNOTATIONS = Path(
    "Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json"
)
RAW_IMAGE_DIRS = (
    Path("Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/Images"),
    Path(
        "Synthetic-Runner-Keypoint-Dataset-2026-06-07/"
        "Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/Images"
    ),
)


def _inside_root(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def resolve_image(root: Path, file_name: str) -> Path:
    """Resolve a raw bare filename or a derived root-relative image path."""

    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {file_name}")
    if len(relative.parts) > 1:
        candidate = _inside_root(root, root / relative)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    matches = [
        _inside_root(root, root / image_dir / relative)
        for image_dir in RAW_IMAGE_DIRS
        if (root / image_dir / relative).is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one raw image for {file_name}; found {len(matches)}"
        )
    return matches[0]


def feet_near_lower_edge(image: dict, annotation: dict, margin: int = 12) -> bool:
    values = annotation["keypoints"]
    height = image["height"]
    return any(
        values[index * 3 + 2] > 0
        and values[index * 3 + 1] >= height - margin
        for index in FOOT_INDICES
    )


def select_samples(
    images: list[dict], annotations_by_image: dict[int, dict], count: int, seed: int
) -> list[tuple[dict, dict, int, str]]:
    """Sample across ten ID intervals and force one low-foot case per interval.

    Returns (image, annotation, interval, selection reason) sorted by image ID.
    """

    if count < 1 or count > len(images):
        raise ValueError(f"--samples must be between 1 and {len(images)}")
    ordered = sorted(images, key=lambda image: image["id"])
    strata = min(10, count)
    selected: list[tuple[dict, dict, int, str]] = []
    for interval in range(strata):
        candidates = ordered[
            interval * len(ordered) // strata : (interval + 1) * len(ordered) // strata
        ]
        quota = (interval + 1) * count // strata - interval * count // strata
        boundary = [
            image
            for image in candidates
            if feet_near_lower_edge(image, annotations_by_image[image["id"]])
        ]
        chosen: list[tuple[dict, str]] = []
        if boundary:
            # Prefer the most extreme lower-foot case, including out-of-frame
            # points that the derived annotations must later correct.
            edge_image = max(
                boundary,
                key=lambda image: (
                    max(
                        annotations_by_image[image["id"]]["keypoints"][i * 3 + 1]
                        for i in FOOT_INDICES
                        if annotations_by_image[image["id"]]["keypoints"][i * 3 + 2] > 0
                    ) - image["height"],
                    -image["id"],
                ),
            )
            chosen.append((edge_image, "foot_near_lower_edge"))
        chosen_ids = {image["id"] for image, _ in chosen}
        remaining = [image for image in candidates if image["id"] not in chosen_ids]
        rng = random.Random(seed + interval)
        chosen.extend(
            (image, "stratified_random")
            for image in rng.sample(remaining, quota - len(chosen))
        )
        selected.extend(
            (image, annotations_by_image[image["id"]], interval, reason)
            for image, reason in chosen
        )
    return sorted(selected, key=lambda item: item[0]["id"])


def render_overlay(image_path: Path, image_info: dict, annotation: dict, output: Path) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    if image.size != (image_info["width"], image_info["height"]):
        raise ValueError(f"Image dimensions disagree with JSON: {image_path}")
    values = annotation["keypoints"]
    if len(values) != 3 * len(KEYPOINT_NAMES):
        raise ValueError(f"Expected 30 keypoints in image {image_info['id']}")

    draw = ImageDraw.Draw(image)
    x, y, width, height = annotation["bbox"]
    draw.rectangle((x, y, x + width, y + height), outline=(255, 0, 255), width=2)
    for start, end in SKELETON_EDGES:
        if values[start * 3 + 2] > 0 and values[end * 3 + 2] > 0:
            draw.line(
                (
                    values[start * 3], values[start * 3 + 1],
                    values[end * 3], values[end * 3 + 1]
                ),
                fill=(80, 255, 80),
                width=2,
            )
    for index, name in enumerate(KEYPOINT_NAMES):
        px, py, visibility = values[index * 3 : index * 3 + 3]
        if visibility <= 0:
            continue
        color = (0, 204, 255) if name.startswith("Left") or index == 1 else (
            (255, 170, 0) if name.startswith("Right") or index == 2 else (255, 255, 255)
        )
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=(0, 0, 0))
        draw.text((px + 4, py - 6), str(index), fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    image.save(output, quality=92)


def run(root: Path, annotations: Path, count: int, output: Path, seed: int) -> Path:
    root = root.resolve()
    annotation_path = _inside_root(root, annotations if annotations.is_absolute() else root / annotations)
    output_dir = _inside_root(root, output if output.is_absolute() else root / output)
    with annotation_path.open(encoding="utf-8") as stream:
        dataset = json.load(stream)
    if len(dataset["categories"]) != 1 or tuple(dataset["categories"][0]["keypoints"]) != KEYPOINT_NAMES:
        raise ValueError("Category keypoint order differs from src/posturaai/keypoints.py")
    images = dataset["images"]
    annotations_by_image = {annotation["image_id"]: annotation for annotation in dataset["annotations"]}
    if len(annotations_by_image) != len(dataset["annotations"]):
        raise ValueError("Expected exactly one annotation per image")
    if {image["id"] for image in images} != set(annotations_by_image):
        raise ValueError("Image and annotation IDs differ")
    selected = select_samples(images, annotations_by_image, count, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_list.csv"
    rows = []
    for image, annotation, interval, reason in selected:
        image_path = resolve_image(root, image["file_name"])
        overlay_name = f"image_{image['id']:06d}.jpg"
        render_overlay(image_path, image, annotation, output_dir / overlay_name)
        x, y, width, height = annotation["bbox"]
        values = annotation["keypoints"]
        outside = sum(
            values[i * 3 + 2] > 0
            and not (0 <= values[i * 3] < image["width"] and 0 <= values[i * 3 + 1] < image["height"])
            for i in range(len(KEYPOINT_NAMES))
        )
        rows.append(dict(
            image_id=image["id"],
            annotation_id=annotation["id"],
            file_name=image["file_name"],
            source_path=image_path.relative_to(root).as_posix(),
            overlay=overlay_name,
            stratum=interval,
            selection_reason=reason,
            bbox_outside_image=int(x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]),
            visible_keypoints_outside_image=outside,
        ))
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="PosturaAI project root")
    parser.add_argument("--annotations", type=Path, default=RAW_ANNOTATIONS, help="COCO JSON path relative to root")
    parser.add_argument("--samples", type=int, default=100, help="Number of overlays to render")
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations/ground_truth"), help="Output directory relative to root")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed")
    args = parser.parse_args()
    try:
        manifest_path = run(args.root, args.annotations, args.samples, args.output, args.seed)
    except (KeyError, ValueError, FileNotFoundError, OSError) as error:
        parser.exit(1, f"visualize_annotations: {error}\n")
    print(f"Rendered {args.samples} overlays; sample list: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
