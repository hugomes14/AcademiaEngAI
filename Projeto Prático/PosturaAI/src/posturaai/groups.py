"""Reproducible SRKD grouping, split preparation, and human audit gates.

The SRKD alternates two camera views.  Temporal comparisons therefore use
same-parity frames when that pattern is detected.  Visual matches are evidence
for conservative grouping, never proof that two groups are independent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.posturaai.dataset import derive_annotation, image_relative_path, resolve_image_path, validate_derived


RAW_JSON = Path("Dataset-Synthetic-Runner-Keypoint-Dataset-2026-06-07/keypoints_srkd.json")
MANIFEST_FIELDS = ("image_id", "file_name", "group_id", "group_method", "audit_status")
DECISION_FIELDS = (
    "candidate_id", "kind", "image_id_a", "image_id_b", "group_id_a",
    "group_id_b", "distance", "ambiguous", "decision", "reviewer", "note",
)
SPLITS = ("train", "val", "test")
TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
FEATURE_SIZE = (24, 14)
TEMPORAL_CUT = 5.0
AMBIGUOUS_LOW = 3.0
AMBIGUOUS_HIGH = 12.0
VISUAL_MERGE = 1.5
VISUAL_REVIEW = 6.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _csv_write(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_source(root: Path) -> dict:
    path = root / RAW_JSON
    with path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    source["images"].sort(key=lambda item: int(item["id"]))
    return source


def _image_ids(source: dict) -> np.ndarray:
    return np.asarray([int(image["id"]) for image in source["images"]], dtype=np.int64)


def _features(root: Path, source: dict, cache: Path) -> np.ndarray:
    """Decode once into a resumable memory mapped cache (~94 MB for SRKD)."""
    images = source["images"]
    ids = _image_ids(source)
    source_hash = sha256_file(root / RAW_JSON)
    meta_path = cache.with_suffix(".json")
    completed = 0
    if cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids_hash = hashlib.sha256(ids.tobytes()).hexdigest()
        if meta.get("source_sha256") == source_hash and meta.get("ids_sha256") == ids_hash and meta.get("size") == list(FEATURE_SIZE):
            data = np.lib.format.open_memmap(cache, mode="r+")
            if data.shape == (len(images), FEATURE_SIZE[1], FEATURE_SIZE[0], 3):
                completed = min(int(meta.get("completed", 0)), len(images))
            else:
                raise ValueError(f"Cache shape inválido: {cache}")
        else:
            raise ValueError(f"Cache obsoleto: {cache}. Remover explicitamente antes de reconstruir.")
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        data = np.lib.format.open_memmap(
            cache, mode="w+", dtype=np.uint8,
            shape=(len(images), FEATURE_SIZE[1], FEATURE_SIZE[0], 3),
        )
    for start in range(completed, len(images), 500):
        end = min(start + 500, len(images))
        for index in range(start, end):
            path = resolve_image_path(root, images[index])
            with Image.open(path) as image:
                data[index] = np.asarray(image.convert("RGB").resize(FEATURE_SIZE, Image.Resampling.BILINEAR))
        data.flush()
        _atomic_json(meta_path, {
            "source_sha256": source_hash,
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "size": list(FEATURE_SIZE),
            "completed": end,
        })
    return data


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


class _Union:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.parent[max(a, b)] = min(a, b)
        return True


def _visual_candidates(features: np.ndarray, segments: list[list[int]]) -> dict[tuple[int, int], tuple[float, int, int]]:
    """Index six representative views per segment, not all image pairs."""
    from scipy.spatial import cKDTree

    reps: list[tuple[int, int]] = []
    for segment_id, indices in enumerate(segments):
        positions = sorted({0, len(indices) // 2, len(indices) - 1})
        for pos in positions:
            reps.append((segment_id, indices[pos]))
    if len(reps) < 2:
        return {}
    # 4x4 RGB thumbnail retains coarse scene/camera appearance for indexing.
    vectors = np.asarray([
        np.asarray(Image.fromarray(features[index]).resize((4, 4), Image.Resampling.BILINEAR)).reshape(-1)
        for _, index in reps
    ], dtype=np.float32)
    tree = cKDTree(vectors)
    _, neighbors = tree.query(vectors, k=min(24, len(reps)))
    candidates: dict[tuple[int, int], tuple[float, int, int]] = {}
    for rep_index, nearest in enumerate(np.atleast_2d(neighbors)):
        segment_a, image_a = reps[rep_index]
        for other in np.atleast_1d(nearest)[1:]:
            segment_b, image_b = reps[int(other)]
            if segment_a == segment_b:
                continue
            pair = tuple(sorted((segment_a, segment_b)))
            distance = _distance(features[image_a], features[image_b])
            old = candidates.get(pair)
            if old is None or distance < old[0]:
                candidates[pair] = (distance, image_a, image_b)
    return candidates


def build_groups(root: Path, manifest: Path, cache: Path, candidates_path: Path) -> dict:
    source = load_source(root)
    images = source["images"]
    ids = _image_ids(source)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("IDs de imagens repetidos")
    features = _features(root, source, cache)
    if len(ids) >= 6:
        sample = np.linspace(0, len(ids) - 3, min(1000, len(ids) - 2), dtype=int)
        d1 = np.asarray([_distance(features[i], features[i + 1]) for i in sample])
        d2 = np.asarray([_distance(features[i], features[i + 2]) for i in sample])
        alternating = bool(np.median(d2) < np.median(d1) * 0.3)
    else:
        alternating = False
    units = [list(range(i, min(i + (2 if alternating else 1), len(ids)))) for i in range(0, len(ids), 2 if alternating else 1)]
    segments: list[list[int]] = []
    cuts: list[tuple[int, int, float]] = []
    current = units[0][:] if units else []
    for previous, next_unit in zip(units, units[1:]):
        d = min(_distance(features[a], features[b]) for a, b in zip(previous, next_unit))
        if d > TEMPORAL_CUT:
            segments.append(current)
            cuts.append((previous[-1], next_unit[0], d))
            current = next_unit[:]
        else:
            current.extend(next_unit)
    if current:
        segments.append(current)
    union = _Union(len(segments))
    visual = _visual_candidates(features, segments)
    ambiguous: list[tuple[int, int, float, str]] = []
    visual_merges = 0
    for (a, b), (distance, image_a, image_b) in sorted(visual.items()):
        if distance <= VISUAL_MERGE:
            visual_merges += int(union.join(a, b))
        elif distance <= VISUAL_REVIEW:
            ambiguous.append((image_a, image_b, distance, "visual"))
    segment_for_index = {}
    for segment_id, indices in enumerate(segments):
        for index in indices:
            segment_for_index[index] = segment_id
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ids)):
        groups[union.find(segment_for_index[index])].append(index)
    canonical = {root_id: f"g{int(ids[min(indices)]):06d}" for root_id, indices in groups.items()}
    merged_roots = Counter(union.find(i) for i in range(len(segments)))
    rows = []
    for index, image in enumerate(images):
        root_id = union.find(segment_for_index[index])
        rows.append({
            "image_id": int(image["id"]),
            "file_name": image_relative_path(image),
            "group_id": canonical[root_id],
            "group_method": "temporal+visual" if merged_roots[root_id] > 1 else "temporal",
            "audit_status": "pending",
        })
    _csv_write(manifest, MANIFEST_FIELDS, rows)
    evidence = []
    for a, b, distance in cuts:
        evidence.append({"kind": "boundary", "image_id_a": int(ids[a]), "image_id_b": int(ids[b]), "distance": round(distance, 4), "ambiguous": AMBIGUOUS_LOW <= distance <= AMBIGUOUS_HIGH})
    for a, b, distance, kind in ambiguous:
        evidence.append({"kind": kind, "image_id_a": int(ids[a]), "image_id_b": int(ids[b]), "distance": round(distance, 4), "ambiguous": True})
    _csv_write(candidates_path, ("kind", "image_id_a", "image_id_b", "distance", "ambiguous"), evidence)
    return {"images": len(images), "groups": len(groups), "temporal_segments": len(segments), "visual_merges": visual_merges, "alternating_cameras": alternating, "cut_candidates": len(cuts), "ambiguous_candidates": sum(item["ambiguous"] for item in evidence), "manifest": str(manifest), "candidates": str(candidates_path)}


def _manifest(root: Path, path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    rows = _csv_read(path)
    source = load_source(root)
    expected = {int(image["id"]): image_relative_path(image) for image in source["images"]}
    observed = {int(row["image_id"]): row for row in rows}
    if len(rows) != len(expected) or set(observed) != set(expected):
        raise ValueError("Manifesto incompleto ou com IDs repetidos")
    for image_id, row in observed.items():
        if row["file_name"] != expected[image_id] or not row["group_id"]:
            raise ValueError(f"Entrada inválida no manifesto: {image_id}")
    return rows, observed


def _assign_groups(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    sizes = Counter(row["group_id"] for row in rows)
    if len(sizes) < 3:
        raise ValueError("Pelo menos três grupos são necessários para train/val/test")
    rng = random.Random(seed)
    order = list(sizes)
    rng.shuffle(order)
    order.sort(key=lambda group: sizes[group], reverse=True)
    target = {split: len(rows) * ratio for split, ratio in TARGETS.items()}
    counts = Counter()
    result = {}
    for index, group in enumerate(order):
        empty = [name for name in SPLITS if counts[name] == 0]
        remaining = len(order) - index
        choices = empty if len(empty) == remaining else SPLITS
        split = max(choices, key=lambda name: target[name] - counts[name])
        result[group] = split
        counts[split] += sizes[group]
    if any(counts[name] == 0 for name in SPLITS):
        raise ValueError(f"Split vazio por distribuição dos grupos: {dict(counts)}")
    # A single group cannot be split; permit one largest-group share or 5 pp,
    # whichever is larger, then fail visibly rather than silently misbalance.
    tolerance = max(0.05, max(sizes.values()) / len(rows))
    fractions = {split: counts[split] / len(rows) for split in SPLITS}
    if any(abs(fractions[split] - TARGETS[split]) > tolerance + 1e-12 for split in SPLITS):
        raise ValueError(f"80/10/10 inviável com grupos atuais: {fractions}; tolerância ±{tolerance:.1%}")
    return result


def provisional_split(root: Path, manifest: Path, assignment_path: Path, seed: int = 42) -> dict:
    rows, _ = _manifest(root, manifest)
    assignment = _assign_groups(rows, seed)
    output = [{"group_id": group, "split": split} for group, split in sorted(assignment.items())]
    _csv_write(assignment_path, ("group_id", "split"), output)
    counts = Counter(assignment[row["group_id"]] for row in rows)
    return {"groups": len(assignment), "images": dict(counts),
            "fractions": {split: round(counts[split] / len(rows), 4) for split in SPLITS},
            "assignment": str(assignment_path), "seed": seed}


def _read_assignment(path: Path, groups: set[str]) -> dict[str, str]:
    rows = _csv_read(path)
    assignment = {row["group_id"]: row["split"] for row in rows}
    if len(rows) != len(groups) or set(assignment) != groups or set(assignment.values()) - set(SPLITS):
        raise ValueError("A atribuição provisória está obsoleta; refazer prepare_splits --provisional")
    return assignment


def _candidate_id(kind: str, a: int, b: int, group_a: str, group_b: str) -> str:
    raw = f"{kind}:{min(a,b)}:{max(a,b)}:{min(group_a,group_b)}:{max(group_a,group_b)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _candidate(kind: str, a: int, b: int, distance: float, by_id: dict[int, dict[str, str]]) -> dict:
    ga, gb = by_id[a]["group_id"], by_id[b]["group_id"]
    return {"candidate_id": _candidate_id(kind, a, b, ga, gb), "kind": kind, "image_id_a": a,
            "image_id_b": b, "group_id_a": ga, "group_id_b": gb,
            "distance": round(distance, 4), "ambiguous": False,
            "decision": "", "reviewer": "", "note": ""}


def _review_queue(root: Path, manifest: Path, assignment_path: Path, candidates_path: Path, cache: Path) -> tuple[list[dict], dict]:
    rows, by_id = _manifest(root, manifest)
    assignment = _read_assignment(assignment_path, {row["group_id"] for row in rows})
    evidence = _csv_read(candidates_path)
    boundaries = [item for item in evidence if item["kind"] == "boundary" and by_id[int(item["image_id_a"])]["group_id"] != by_id[int(item["image_id_b"])]["group_id"]]
    # Spread the 100 required boundary pairs over the whole numeric range.
    chosen = [boundaries[i] for i in np.linspace(0, len(boundaries) - 1, min(100, len(boundaries)), dtype=int)] if boundaries else []
    chosen.extend(item for item in evidence if item["ambiguous"] == "True" and by_id[int(item["image_id_a"])]["group_id"] != by_id[int(item["image_id_b"])]["group_id"])
    queue: dict[str, dict] = {}
    for item in chosen:
        candidate = _candidate(item["kind"], int(item["image_id_a"]), int(item["image_id_b"]), float(item["distance"]), by_id)
        candidate["ambiguous"] = item["ambiguous"] == "True"
        queue[candidate["candidate_id"]] = candidate
    source = load_source(root)
    ids = _image_ids(source)
    features = _features(root, source, cache)
    # Search *every* image.  A 12-D PCA index keeps this subquadratic; exact RGB
    # distance re-ranks the nearest coarse matches.  Three trees exclude each
    # query's own split, so long groups cannot hide duplicate interior frames.
    from scipy.spatial import cKDTree
    gray = features.astype(np.float32).mean(axis=3)
    coarse = gray[:, ::3, ::3].reshape(len(ids), -1)
    sample = coarse[np.linspace(0, len(ids) - 1, min(4096, len(ids)), dtype=int)]
    center = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - center, full_matrices=False)
    vectors = (coarse - center) @ vt[:min(12, vt.shape[0])].T
    split_by_index = np.asarray([assignment[by_id[int(image_id)]["group_id"]] for image_id in ids])
    split_indices = {split: np.flatnonzero(split_by_index == split) for split in SPLITS}
    nearest_pairs: list[tuple[float, int, int]] = []
    for target in SPLITS:
        target_indices = split_indices[target]
        if not len(target_indices):
            continue
        query_indices = np.flatnonzero(split_by_index != target)
        tree = cKDTree(vectors[target_indices])
        for start in range(0, len(query_indices), 4096):
            batch = query_indices[start:start + 4096]
            distances, neighbors = tree.query(vectors[batch], k=min(2, len(target_indices)))
            distances = np.atleast_2d(distances).reshape(len(batch), -1)
            neighbors = np.atleast_2d(neighbors).reshape(len(batch), -1)
            for row, image_index in enumerate(batch):
                for column in range(neighbors.shape[1]):
                    nearest_pairs.append((float(distances[row, column]), int(image_index), int(target_indices[neighbors[row, column]])))
    nearest_pairs.sort(key=lambda item: item[0])
    ranked = []
    seen_images = set()
    for _, ia, ib in nearest_pairs[:max(10000, min(len(nearest_pairs), 20000))]:
        pair = tuple(sorted((int(ids[ia]), int(ids[ib]))))
        if pair in seen_images:
            continue
        seen_images.add(pair)
        ranked.append((_distance(features[ia], features[ib]), pair[0], pair[1]))
    ranked.sort()
    group_pair_counts: Counter[tuple[str, str]] = Counter()
    selected = []
    for distance, a, b in ranked:
        group_pair = tuple(sorted((by_id[a]["group_id"], by_id[b]["group_id"])))
        if group_pair_counts[group_pair] >= 5:
            continue
        selected.append((distance, a, b))
        group_pair_counts[group_pair] += 1
        if len(selected) >= 100:
            break
    if len(selected) < 100:
        seen = {(a, b) for _, a, b in selected}
        selected.extend((d, a, b) for d, a, b in ranked if (a, b) not in seen)  # small fixtures may have fewer
    for distance, a, b in selected[:100]:
        candidate = _candidate("cross_split", a, b, distance, by_id)
        queue[candidate["candidate_id"]] = candidate
    return list(queue.values()), {"boundary_total": len(boundaries), "boundary_sample": len({item["candidate_id"] for item in queue.values() if item["kind"] == "boundary"}), "cross_split_sample": sum(item["kind"] == "cross_split" for item in queue.values()), "ambiguous_total": sum(bool(item["ambiguous"]) for item in queue.values())}


def _contact_sheets(root: Path, queue: list[dict], images_by_id: dict[int, dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(queue), 10), 1):
        sheet = Image.new("RGB", (1040, 1740), "white")
        draw = ImageDraw.Draw(sheet)
        for row, item in enumerate(queue[start:start + 10]):
            for side in ("a", "b"):
                image_id = int(item[f"image_id_{side}"])
                with Image.open(resolve_image_path(root, images_by_id[image_id])) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((500, 144))
                    sheet.paste(thumb, (0 if side == "a" else 520, row * 174 + 22))
            draw.text((0, row * 174), f"{item['candidate_id']} {item['kind']} distance={item['distance']}", fill="black")
        sheet.save(output / f"pairs_{page:03d}.jpg", quality=82)


def audit_groups(root: Path, manifest: Path, assignment_path: Path, candidates_path: Path,
                 decisions_path: Path, report_path: Path, approval_path: Path, cache: Path,
                 sheets_path: Path, approve: bool = False, reviewer: str = "") -> dict:
    queue, counts = _review_queue(root, manifest, assignment_path, candidates_path, cache)
    if len(queue) > 3000:
        raise ValueError(f"{len(queue)} pares exigem revisão; demasiados para gerar folhas automaticamente. Rever os limiares e candidatos de agrupamento antes de continuar.")
    prior = {row["candidate_id"]: row for row in _csv_read(decisions_path)} if decisions_path.exists() else {}
    prior_same_by_pair = {
        tuple(sorted((int(row["image_id_a"]), int(row["image_id_b"])))): row
        for row in prior.values() if row["decision"] == "same"
    }
    for item in queue:
        pair = tuple(sorted((int(item["image_id_a"]), int(item["image_id_b"]))))
        previous = prior_same_by_pair.get(pair) or prior.get(item["candidate_id"])
        if previous:
            for field in ("decision", "reviewer", "note"):
                item[field] = previous[field]
    # Keep every known same-sequence decision until the groups are actually
    # merged, even when a new provisional split removes it from the top 100.
    _, current_by_id = _manifest(root, manifest)
    queued_pairs = {tuple(sorted((int(item["image_id_a"]), int(item["image_id_b"])))) for item in queue}
    for previous in prior_same_by_pair.values():
        if previous["decision"] != "same":
            continue
        a, b = int(previous["image_id_a"]), int(previous["image_id_b"])
        if a not in current_by_id or b not in current_by_id:
            raise ValueError("Decisão 'same' refere uma imagem ausente do manifesto")
        if current_by_id[a]["group_id"] == current_by_id[b]["group_id"]:
            continue
        pair = tuple(sorted((a, b)))
        if pair not in queued_pairs:
            item = _candidate("known_same", a, b, float(previous["distance"]), current_by_id)
            item["decision"] = "same"
            item["reviewer"] = previous["reviewer"]
            item["note"] = previous["note"]
            queue.append(item)
            queued_pairs.add(pair)
    _csv_write(decisions_path, DECISION_FIELDS, queue)
    source = load_source(root)
    by_id = {int(image["id"]): image for image in source["images"]}
    _contact_sheets(root, queue, by_id, sheets_path)
    pending = [item for item in queue if item["decision"] not in {"same", "different"} or not item["reviewer"].strip()]
    same = [item for item in queue if item["decision"] == "same" and item["group_id_a"] != item["group_id_b"]]
    ambiguous_reviewed = sum(bool(item["ambiguous"]) and item["decision"] in {"same", "different"} and bool(item["reviewer"].strip()) for item in queue)
    enough = counts["boundary_sample"] >= 100 and counts["cross_split_sample"] >= 100
    rows, _ = _manifest(root, manifest)
    existing_approval = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.exists() else {}
    existing_valid = bool(
        existing_approval.get("approved")
        and all(row["audit_status"] == "approved" for row in rows)
        and existing_approval.get("manifest_sha256") == sha256_file(manifest)
        and existing_approval.get("assignment_sha256") == sha256_file(assignment_path)
        and existing_approval.get("decisions_sha256") == sha256_file(decisions_path)
        and existing_approval.get("source_sha256") == sha256_file(root / RAW_JSON)
    )
    requested_approval = bool(approve and reviewer.strip() and not pending and not same and enough)
    if requested_approval:
        for row in rows:
            row["audit_status"] = "approved"
        _csv_write(manifest, MANIFEST_FIELDS, rows)
        _atomic_json(approval_path, {
            "approved": True, "reviewer": reviewer, "approved_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(manifest), "assignment_sha256": sha256_file(assignment_path),
            "decisions_sha256": sha256_file(decisions_path), "source_sha256": sha256_file(root / RAW_JSON),
            "boundary_reviewed": counts["boundary_sample"], "cross_split_reviewed": counts["cross_split_sample"],
            "ambiguous_reviewed": ambiguous_reviewed,
        })
    approved = requested_approval or existing_valid
    if approved and not reviewer:
        reviewer = existing_approval.get("reviewer", "")
    status = "APPROVED" if approved else "PENDING"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Auditoria dos grupos SRKD\n\nEstado: **{status}**. Revisor: {reviewer or 'por definir'}.\n\n"
        f"- Limites amostrados: {counts['boundary_sample']} de {counts['boundary_total']}.\n"
        f"- Pares visualmente semelhantes entre splits: {counts['cross_split_sample']}.\n"
        f"- Casos ambíguos revistos: {ambiguous_reviewed} de {counts['ambiguous_total']}.\n"
        f"- Decisões pendentes: {len(pending)}; pares marcados `same` entre grupos: {len(same)}.\n"
        f"- Decisões: `{decisions_path.relative_to(root) if decisions_path.is_relative_to(root) else decisions_path}`. Evidência visual: `{sheets_path.relative_to(root) if sheets_path.is_relative_to(root) else sheets_path}/`.\n\n"
        + ("Conclusão: sem sobreposição conhecida após auditoria. Os grupos foram inferidos; permanece risco de leakage.\n" if approved else "Para aprovar: rever imagens, preencher `decision` (`same`/`different`) e `reviewer` para todos os pares; reunir grupos marcados `same`, refazer o split provisório e repetir a auditoria.\n"),
        encoding="utf-8",
    )
    if approve and not requested_approval:
        raise ValueError(f"Auditoria não aprovada: {len(pending)} pendentes, {len(same)} fusões, cobertura={enough}")
    return {"status": status, "candidates": len(queue), "pending": len(pending), "same_between_groups": len(same), "ambiguous_reviewed": ambiguous_reviewed, **counts, "report": str(report_path)}


def apply_same_decisions(root: Path, manifest: Path, decisions_path: Path, approval_path: Path) -> dict:
    """Merge reviewed same-sequence groups, invalidating the provisional split."""
    rows, _ = _manifest(root, manifest)
    if not decisions_path.exists():
        raise ValueError("Faltam decisões de auditoria")
    groups = sorted({row["group_id"] for row in rows})
    position = {group: index for index, group in enumerate(groups)}
    union = _Union(len(groups))
    merged = 0
    for item in _csv_read(decisions_path):
        if item["decision"] != "same":
            continue
        if not item["reviewer"].strip():
            raise ValueError(f"Decisão 'same' sem revisor: {item['candidate_id']}")
        ga, gb = item["group_id_a"], item["group_id_b"]
        if ga not in position or gb not in position:
            raise ValueError("Decisões obsoletas; reconstruir a fila de auditoria")
        merged += int(union.join(position[ga], position[gb]))
    if merged:
        canonical = {}
        for group in groups:
            canonical[group] = groups[union.find(position[group])]
        for row in rows:
            new_group = canonical[row["group_id"]]
            if new_group != row["group_id"]:
                row["group_id"] = new_group
                row["group_method"] = "manual_merge"
            row["audit_status"] = "pending"
        _csv_write(manifest, MANIFEST_FIELDS, rows)
        approval_path.unlink(missing_ok=True)
    return {"merged_groups": merged, "groups_remaining": len(groups) - merged,
            "provisional_split_invalidated": bool(merged), "manifest": str(manifest)}


def final_split(root: Path, manifest: Path, assignment_path: Path, decisions_path: Path,
                approval_path: Path, output_dir: Path) -> dict:
    rows, by_id = _manifest(root, manifest)
    groups = {row["group_id"] for row in rows}
    assignment = _read_assignment(assignment_path, groups)
    if any(row["audit_status"] != "approved" for row in rows):
        raise ValueError("Manifesto ainda tem auditoria pendente")
    if not approval_path.exists():
        raise ValueError("Falta aprovação de auditoria")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    actual = {
        "manifest_sha256": sha256_file(manifest), "assignment_sha256": sha256_file(assignment_path),
        "decisions_sha256": sha256_file(decisions_path), "source_sha256": sha256_file(root / RAW_JSON),
    }
    if not approval.get("approved") or any(approval.get(key) != value for key, value in actual.items()):
        raise ValueError("Manifesto, split, decisões ou JSON bruto mudou após aprovação")
    source = load_source(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_images = {split: [] for split in SPLITS}
    split_annotations = {split: [] for split in SPLITS}
    corrections = []
    image_map = {int(image["id"]): image for image in source["images"]}
    for image in source["images"]:
        image_id = int(image["id"])
        relative = image_relative_path(image)
        resolved = resolve_image_path(root, image)
        if not resolved.is_file() or not resolved.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Imagem ausente ou fora do projeto: {relative}")
        split = assignment[by_id[image_id]["group_id"]]
        derived_image = {**image, "file_name": relative}
        if "licence" in derived_image:
            derived_image["license"] = derived_image.pop("licence")
        split_images[split].append(derived_image)
    for annotation in source["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in image_map:
            raise ValueError(f"Anotação sem imagem: {image_id}")
        split = assignment[by_id[image_id]["group_id"]]
        derived, changes = derive_annotation(annotation, image_map[image_id])
        split_annotations[split].append(derived)
        for change in changes:
            corrections.append({
                "image_id": image_id, "annotation_id": annotation["id"],
                "field": change["field"], "keypoint_index": change.get("keypoint_index", ""),
                "original": json.dumps(change.get("original"), ensure_ascii=False),
                "derived": json.dumps(change.get("derived"), ensure_ascii=False),
                "reason": change.get("reason", "out_of_bounds"),
            })
    output = {}
    for split in SPLITS:
        path = output_dir / f"{split}.json"
        payload = {key: value for key, value in source.items() if key not in {"images", "annotations", "licences", "dataset"}}
        payload["licenses"] = source.get("licenses", source.get("licences", []))
        payload["images"] = split_images[split]
        payload["annotations"] = split_annotations[split]
        _atomic_json(path, payload)
        validation = validate_derived(path, root)
        if not validation["ok"]:
            path.unlink(missing_ok=True)
            raise ValueError(f"Validação derivada falhou em {split}: {validation['error_count']} erros; {validation['errors'][:3]}")
        output[split] = {"images": len(split_images[split]), "groups": sum(value == split for value in assignment.values()), "annotations": len(split_annotations[split]), "sha256": sha256_file(path)}
    if len(source["images"]) == 92824:
        field_counts = Counter(change["field"] for change in corrections)
        if field_counts != {"bbox": 18, "keypoint": 84}:
            raise ValueError(f"Contagem de correções inesperada: {dict(field_counts)}")
    corrections_path = output_dir / "corrections.csv"
    _csv_write(corrections_path, ("image_id", "annotation_id", "field", "keypoint_index", "original", "derived", "reason"), corrections)
    report = {"source_sha256": actual["source_sha256"], "manifest_sha256": actual["manifest_sha256"], "assignment_sha256": actual["assignment_sha256"], "splits": output, "corrections": len(corrections), "corrections_sha256": sha256_file(corrections_path), "audit_reviewer": approval["reviewer"]}
    _atomic_json(output_dir / "preparation_report.json", report)
    return report
