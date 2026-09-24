"""Validation and non-destructive annotation corrections for the SRKD source."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE_DIRECTORY = "Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07"
SECOND_DIRECTORY = (
    "Synthetic-Runner-Keypoint-Dataset-2026-06-07/"
    "Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07"
)
SOURCE_JSON = f"{SOURCE_DIRECTORY}/keypoints_srkd.json"
SOURCE_SHA256 = "66c8b3927c238ffd0efe878d7854b494a5afc95e6acf2b1a5e73d0cbb9b14da3"
EXPECTED_IMAGES = 92_824
EXPECTED_BBOX_OVERFLOWS = 18
EXPECTED_VISIBLE_POINTS_OUTSIDE = 84
FOOT_KEYPOINTS = frozenset(range(15, 25))
EXPECTED_KEYPOINT_NAMES = (
    "Nose", "LEye", "REye", "LeftEar", "RightEar", "LeftShoulder",
    "RightShoulder", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
    "LeftHip", "RightHip", "LeftKnee", "RightKnee", "LeftAnkle",
    "RightAnkle", "LeftHeel", "RightHeel", "LeftFirstMetatarsal",
    "RightFirstMetatarsal", "LeftFifthMetatarsal", "RightFifthMetatarsal",
    "LeftToe", "RightToe", "TopHead", "BackHead", "UpTrunk",
    "MiddleTrunk", "Pelvis",
)


def source_path(root: Path) -> Path:
    """Return the immutable annotation source under a PosturaAI root."""
    return Path(root) / SOURCE_JSON


def image_relative_path(image: dict[str, Any]) -> str:
    """Map a source image to its existing tree without duplicating image bytes."""
    file_name = image["file_name"]
    if not isinstance(file_name, str) or len(file_name) != 11 or not file_name.endswith(".jpeg"):
        raise ValueError(f"Invalid SRKD image name: {file_name!r}")
    number = file_name[:6]
    if not number.isdecimal() or not 1 <= int(number) <= EXPECTED_IMAGES:
        raise ValueError(f"Invalid SRKD image number: {file_name!r}")
    directory = SOURCE_DIRECTORY if int(number) <= 1_219 else SECOND_DIRECTORY
    return f"{directory}/Images/{file_name}"


def resolve_image_path(root: Path, image: dict[str, Any]) -> Path:
    """Resolve an image and reject a path escaping the project root."""
    root = Path(root).resolve()
    path = (root / image_relative_path(image)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes project root: {path}")
    return path


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bbox_outside(bbox: list[Any], width: int, height: int) -> bool:
    x, y, w, h = bbox
    return x < 0 or y < 0 or x + w > width or y + h > height


def _point_outside(x: float, y: float, width: int, height: int) -> bool:
    # Pixel coordinates are half-open. x==width or y==height is already outside.
    return x < 0 or y < 0 or x >= width or y >= height


def derive_annotation(
    annotation: dict[str, Any], image: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy and repair known boundary errors, retaining an audit row per change.

    This function intentionally never modifies its inputs. It expects an annotation
    that has already passed the schema checks in :func:`validate_source`.
    """
    derived = deepcopy(annotation)
    corrections: list[dict[str, Any]] = []
    width, height = image["width"], image["height"]
    image_id, annotation_id = annotation["image_id"], annotation["id"]

    bbox = annotation["bbox"]
    if _bbox_outside(bbox, width, height):
        x, y, w, h = bbox
        x0, y0 = max(0, min(width, x)), max(0, min(height, y))
        x1, y1 = max(0, min(width, x + w)), max(0, min(height, y + h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Bounding box has no visible area: annotation {annotation_id}")
        new_bbox = [x0, y0, x1 - x0, y1 - y0]
        new_area = (x1 - x0) * (y1 - y0)
        derived["bbox"] = new_bbox
        derived["area"] = new_area
        corrections.append(
            {
                "image_id": image_id,
                "annotation_id": annotation_id,
                "field": "bbox",
                "keypoint_index": None,
                "original": {"bbox": list(bbox), "area": annotation["area"]},
                "derived": {"bbox": new_bbox, "area": new_area},
                "reason": "bbox_outside_image",
            }
        )

    points = derived["keypoints"]
    for offset in range(0, len(points), 3):
        x, y, visibility = points[offset : offset + 3]
        if visibility > 0 and _point_outside(x, y, width, height):
            original = [x, y, visibility]
            points[offset : offset + 3] = [0, 0, 0]
            corrections.append(
                {
                    "image_id": image_id,
                    "annotation_id": annotation_id,
                    "field": "keypoint",
                    "keypoint_index": offset // 3,
                    "original": original,
                    "derived": [0, 0, 0],
                    "reason": "visible_keypoint_outside_image",
                }
            )
    derived["num_keypoints"] = sum(points[offset] > 0 for offset in range(2, len(points), 3))
    return derived, corrections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(
    root: Path,
    *,
    strict_baseline: bool = True,
    check_image_dimensions: bool = True,
) -> dict[str, Any]:
    """Validate the raw SRKD JSON and referenced images.

    `strict_baseline=False` allows tiny fixture datasets in tests; production CLI
    always uses the pinned source hash, cardinality, and known anomaly counts.
    Unexpected errors are bounded in the returned list while `error_count` retains
    the total, so corrupt data cannot create an unbounded terminal report.
    """
    root = Path(root).resolve()
    errors: list[str] = []
    error_count = 0

    def error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(message)

    path = source_path(root)
    if not path.is_file():
        return {
            "ok": False,
            "source": str(path),
            "source_sha256": None,
            "summary": {},
            "known_anomalies": {},
            "error_count": 1,
            "errors": [f"Source JSON not found: {path}"],
        }
    source_hash = _sha256(path)
    if strict_baseline and source_hash != SOURCE_SHA256:
        error(f"Source SHA-256 changed: {source_hash} (expected {SOURCE_SHA256})")
    try:
        with path.open(encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        error(f"Could not parse source JSON: {exc}")
        return {
            "ok": False,
            "source": str(path),
            "source_sha256": source_hash,
            "summary": {},
            "known_anomalies": {},
            "error_count": error_count,
            "errors": errors,
        }

    images = document.get("images", [])
    annotations = document.get("annotations", [])
    categories = document.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        error("images, annotations and categories must be lists")
        images = images if isinstance(images, list) else []
        annotations = annotations if isinstance(annotations, list) else []
        categories = categories if isinstance(categories, list) else []
    if strict_baseline and (len(images) != EXPECTED_IMAGES or len(annotations) != EXPECTED_IMAGES):
        error(f"Expected {EXPECTED_IMAGES} images and annotations; got {len(images)} and {len(annotations)}")

    if len(categories) != 1 or not isinstance(categories[0], dict):
        error("Expected exactly one person category")
    else:
        category = categories[0]
        if category.get("id") != 1 or category.get("name") != "person":
            error("Expected category id=1, name='person'")
        keypoint_names = category.get("keypoints")
        if not isinstance(keypoint_names, list) or tuple(keypoint_names) != EXPECTED_KEYPOINT_NAMES:
            error("Category keypoint names or order differ from the SRKD specification")
        skeleton = category.get("skeleton", [])
        if not isinstance(skeleton, list) or any(
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(index, int) or not 1 <= index <= 30 for index in edge)
            for edge in skeleton
        ):
            error("Category skeleton contains an invalid 1-based keypoint index")

    image_by_id: dict[int, dict[str, Any]] = {}
    seen_names: set[str] = set()
    present_images = 0
    dimension_mismatches = 0
    if check_image_dimensions:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError:
            Image = None
            UnidentifiedImageError = OSError
            error("Pillow is required to check image dimensions")
    else:
        Image = None
        UnidentifiedImageError = OSError

    for image in images:
        if not isinstance(image, dict):
            error("Image entry is not an object")
            continue
        image_id, file_name = image.get("id"), image.get("file_name")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in image_by_id:
            error(f"Invalid or duplicate image id: {image_id!r}")
            continue
        image_by_id[image_id] = image
        if not isinstance(file_name, str):
            error(f"Invalid image file name: {file_name!r}")
            continue
        if file_name in seen_names:
            error(f"Duplicate image file name: {file_name!r}")
        seen_names.add(file_name)
        if strict_baseline and file_name != f"{image_id:06d}.jpeg":
            error(f"Image {image_id} has unexpected file name {file_name!r}")
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            error(f"Invalid dimensions for image {image_id}: {width!r}x{height!r}")
            continue
        if strict_baseline and (width, height) != (640, 360):
            error(f"Unexpected dimensions for image {image_id}: {width}x{height}")
        try:
            image_path = resolve_image_path(root, image)
        except (KeyError, TypeError, ValueError) as exc:
            error(f"Invalid image path for {image_id}: {exc}")
            continue
        if not image_path.is_file():
            error(f"Missing image {image_id}: {image_path}")
            continue
        present_images += 1
        if Image is not None:
            try:
                with Image.open(image_path) as loaded:
                    if loaded.size != (width, height):
                        dimension_mismatches += 1
                        error(f"Image {image_id} dimensions {loaded.size} disagree with JSON {(width, height)}")
            except (OSError, UnidentifiedImageError) as exc:
                error(f"Image {image_id} cannot be read: {exc}")

    seen_annotation_ids: set[int] = set()
    annotations_per_image: Counter[int] = Counter()
    bbox_overflows = 0
    outside_points = 0
    outside_point_images: set[int] = set()
    outside_point_indices: Counter[int] = Counter()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            error("Annotation entry is not an object")
            continue
        annotation_id, image_id = annotation.get("id"), annotation.get("image_id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool) or annotation_id in seen_annotation_ids:
            error(f"Invalid or duplicate annotation id: {annotation_id!r}")
            continue
        seen_annotation_ids.add(annotation_id)
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id not in image_by_id:
            error(f"Annotation {annotation_id} references missing image id {image_id!r}")
            continue
        annotations_per_image[image_id] += 1
        image = image_by_id[image_id]
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            continue
        if annotation.get("category_id") != 1:
            error(f"Annotation {annotation_id} has unexpected category_id")
        if annotation.get("iscrowd") != 0:
            error(f"Annotation {annotation_id} has unexpected iscrowd")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_finite_number(value) for value in bbox):
            error(f"Annotation {annotation_id} has invalid bbox")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x >= width or y >= height or x + w <= 0 or y + h <= 0:
                error(f"Annotation {annotation_id} has nonpositive or nonintersecting bbox")
            elif _bbox_outside(bbox, width, height):
                bbox_overflows += 1
            area = annotation.get("area")
            if not _finite_number(area) or area <= 0 or not math.isclose(area, w * h, rel_tol=0, abs_tol=1e-6):
                error(f"Annotation {annotation_id} area disagrees with bbox")

        points = annotation.get("keypoints")
        if not isinstance(points, list) or len(points) != 90:
            error(f"Annotation {annotation_id} must have 90 keypoint values")
            continue
        if not all(_finite_number(value) for value in points):
            error(f"Annotation {annotation_id} has non-finite keypoint values")
            continue
        visible_count = 0
        for offset in range(0, 90, 3):
            x, y, visibility = points[offset : offset + 3]
            if visibility not in (0, 1, 2):
                error(f"Annotation {annotation_id} has invalid visibility at keypoint {offset // 3}")
                continue
            if visibility > 0:
                visible_count += 1
                if _point_outside(x, y, width, height):
                    outside_points += 1
                    outside_point_images.add(image_id)
                    outside_point_indices[offset // 3] += 1
                    if offset // 3 not in FOOT_KEYPOINTS:
                        error(f"Annotation {annotation_id} has non-foot visible point outside image")
        if annotation.get("num_keypoints") != visible_count:
            error(f"Annotation {annotation_id} num_keypoints disagrees with visibility")

    for image_id in image_by_id:
        if annotations_per_image[image_id] != 1:
            error(f"Image {image_id} has {annotations_per_image[image_id]} annotations; expected 1")
    if strict_baseline:
        if bbox_overflows != EXPECTED_BBOX_OVERFLOWS:
            error(f"Expected {EXPECTED_BBOX_OVERFLOWS} known bbox overflows; got {bbox_overflows}")
        if outside_points != EXPECTED_VISIBLE_POINTS_OUTSIDE:
            error(f"Expected {EXPECTED_VISIBLE_POINTS_OUTSIDE} known outside visible points; got {outside_points}")

    return {
        "ok": error_count == 0,
        "source": str(path),
        "source_sha256": source_hash,
        "summary": {
            "images": len(images),
            "annotations": len(annotations),
            "present_images": present_images,
            "dimension_mismatches": dimension_mismatches,
            "categories": len(categories),
        },
        "known_anomalies": {
            "bbox_overflows": bbox_overflows,
            "visible_points_outside": outside_points,
            "images_with_outside_points": len(outside_point_images),
            "outside_point_indices": dict(sorted(outside_point_indices.items())),
        },
        "error_count": error_count,
        "errors": errors,
    }


def validate_derived(json_path: Path, root: Path) -> dict[str, Any]:
    """Check a generated COCO split before accepting it as training input.

    Image names must be relative to ``root``. This checks existence, rather
    than reopening every JPEG header already checked by ``validate_source``.
    """
    json_path, root = Path(json_path), Path(root).resolve()
    errors: list[str] = []
    error_count = 0

    def error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(message)

    try:
        with json_path.open(encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        error(f"Could not read derived JSON {json_path}: {exc}")
        return {"ok": False, "summary": {}, "error_count": error_count, "errors": errors}

    images, annotations, categories = (
        document.get("images"), document.get("annotations"), document.get("categories")
    )
    if not isinstance(images, list) or not isinstance(annotations, list):
        error("Derived images and annotations must be lists")
        return {"ok": False, "summary": {}, "error_count": error_count, "errors": errors}
    if not isinstance(categories, list) or len(categories) != 1 or not isinstance(categories[0], dict):
        error("Derived data must have one person category")
    elif tuple(categories[0].get("keypoints", [])) != EXPECTED_KEYPOINT_NAMES:
        error("Derived keypoint names or order differ from SRKD")

    image_by_id: dict[int, dict[str, Any]] = {}
    image_names: set[str] = set()
    for image in images:
        if not isinstance(image, dict):
            error("Derived image entry is not an object")
            continue
        image_id, name = image.get("id"), image.get("file_name")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in image_by_id:
            error(f"Invalid or duplicate derived image id: {image_id!r}")
            continue
        image_by_id[image_id] = image
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            error(f"Invalid derived image path for {image_id}: {name!r}")
            continue
        if name in image_names:
            error(f"Duplicate derived image path: {name}")
        image_names.add(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            error(f"Derived image path escapes project root: {name}")
        elif not path.is_file():
            error(f"Missing derived image {image_id}: {name}")
        width, height = image.get("width"), image.get("height")
        if (
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            or not isinstance(height, int) or isinstance(height, bool) or height <= 0
        ):
            error(f"Invalid derived image dimensions for {image_id}")

    annotation_ids: set[int] = set()
    annotations_per_image: Counter[int] = Counter()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            error("Derived annotation entry is not an object")
            continue
        annotation_id, image_id = annotation.get("id"), annotation.get("image_id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool) or annotation_id in annotation_ids:
            error(f"Invalid or duplicate derived annotation id: {annotation_id!r}")
            continue
        annotation_ids.add(annotation_id)
        if not isinstance(image_id, int) or image_id not in image_by_id:
            error(f"Derived annotation {annotation_id} references missing image {image_id!r}")
            continue
        annotations_per_image[image_id] += 1
        image = image_by_id[image_id]
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            continue
        if annotation.get("category_id") != 1:
            error(f"Derived annotation {annotation_id} has unexpected category")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_finite_number(v) for v in bbox):
            error(f"Derived annotation {annotation_id} has invalid bbox")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or _bbox_outside(bbox, width, height):
                error(f"Derived annotation {annotation_id} bbox is outside image")
            if not _finite_number(annotation.get("area")) or not math.isclose(
                annotation["area"], w * h, rel_tol=0, abs_tol=1e-6
            ):
                error(f"Derived annotation {annotation_id} area disagrees with bbox")
        points = annotation.get("keypoints")
        if not isinstance(points, list) or len(points) != 90 or not all(_finite_number(v) for v in points):
            error(f"Derived annotation {annotation_id} has invalid keypoints")
            continue
        visible_count = 0
        for offset in range(0, 90, 3):
            x, y, visibility = points[offset : offset + 3]
            if visibility not in (0, 1, 2):
                error(f"Derived annotation {annotation_id} has invalid visibility")
            elif visibility > 0:
                visible_count += 1
                if _point_outside(x, y, width, height):
                    error(f"Derived annotation {annotation_id} has visible point outside image")
        if annotation.get("num_keypoints") != visible_count:
            error(f"Derived annotation {annotation_id} num_keypoints disagrees with visibility")

    for image_id in image_by_id:
        if annotations_per_image[image_id] != 1:
            error(f"Derived image {image_id} has {annotations_per_image[image_id]} annotations")
    return {
        "ok": error_count == 0,
        "summary": {"images": len(images), "annotations": len(annotations)},
        "error_count": error_count,
        "errors": errors,
    }
