"""IoU association and per-track EMA; no inferred poses during missing frames."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    size = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(np.maximum(0, a[2:] - a[:2])) +
                  np.prod(np.maximum(0, b[2:] - b[:2])) - intersection)
    return intersection / union if union > 0 else 0.0


def person_nms(boxes, scores, threshold=0.3):
    """Keep highest-score boxes, suppressing overlapping person duplicates."""
    if not 0 < threshold <= 1:
        raise ValueError('NMS threshold must be in (0,1]')
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(boxes),):
        raise ValueError('Expected one score per box')
    order = np.argsort(-scores, kind='stable').tolist()
    keep = []
    while order:
        index = order.pop(0)
        keep.append(index)
        order = [j for j in order if box_iou(boxes[index], boxes[j]) <= threshold]
    return np.asarray(keep, dtype=int)


@dataclass
class Track:
    bbox: np.ndarray
    last_frame: int
    segment_id: int = 0
    points: np.ndarray | None = None
    valid: np.ndarray | None = None


class PoseTracker:
    """Hungarian IoU assignment with a gated cost; ambiguous IDs remain a risk.

    IDs may survive max_gap missing frames. The EMA and feature segment restart
    after *any* gap; scene cuts clear tracks without reusing old IDs.
    """

    def __init__(self, *, iou_threshold=0.3, max_gap=5, alpha=0.65):
        if not 0 < iou_threshold <= 1 or not 0 < alpha <= 1 or max_gap < 0:
            raise ValueError('Invalid tracking threshold, alpha or max_gap')
        self.iou_threshold, self.max_gap, self.alpha = iou_threshold, max_gap, alpha
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_frame = -1

    def reset(self):
        self.tracks.clear()

    def update(self, boxes, frame: int) -> list[int]:
        if frame <= self.last_frame:
            raise ValueError('Frame indices must increase')
        self.last_frame = frame
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.tracks = {i: t for i, t in self.tracks.items()
                       if frame - t.last_frame <= self.max_gap + 1}
        identifiers = list(self.tracks)
        assignments = {}
        if identifiers and len(boxes):
            overlaps = np.array([[box_iou(self.tracks[i].bbox, b) for b in boxes]
                                 for i in identifiers])
            # A prohibited match must not steal a viable pair from the assignment.
            cost = np.where(overlaps >= self.iou_threshold, 1 - overlaps, 1e6)
            rows, cols = linear_sum_assignment(cost)
            assignments = {int(c): identifiers[r] for r, c in zip(rows, cols)
                           if overlaps[r, c] >= self.iou_threshold}
        output = []
        for index, box in enumerate(boxes):
            identifier = assignments.get(index)
            if identifier is None:
                identifier = self.next_id
                self.next_id += 1
                self.tracks[identifier] = Track(box.copy(), frame)
            else:
                track = self.tracks[identifier]
                if frame != track.last_frame + 1:
                    track.points = track.valid = None
                    track.segment_id += 1
                track.bbox = box.copy()
                track.last_frame = frame
            output.append(identifier)
        return output

    def smooth(self, identifier, points, valid):
        track = self.tracks[identifier]
        points = np.asarray(points, dtype=float)
        valid = np.asarray(valid, dtype=bool)
        if points.shape != (30, 2) or valid.shape != (30,):
            raise ValueError('Expected 30 two-dimensional points and validity flags')
        valid = valid & np.isfinite(points).all(axis=1)
        result = np.full((30, 2), np.nan)
        result[valid] = points[valid]
        if track.points is not None:
            both = valid & track.valid
            result[both] = self.alpha * points[both] + (1 - self.alpha) * track.points[both]
        track.points, track.valid = result.copy(), valid.copy()
        return result


def scene_difference(previous, current) -> float:
    """Mean RGB thumbnail difference in [0,1], a simple cut heuristic."""
    return float(np.mean(np.abs(np.asarray(current, dtype=float) -
                                np.asarray(previous, dtype=float))) / 255)


def serialise_points(points, scores):
    """Unavailable x/y become JSON null, retaining finite model scores."""
    return [[float(x) if np.isfinite(x) else None,
             float(y) if np.isfinite(y) else None,
             float(s) if np.isfinite(s) else None]
            for (x, y), s in zip(points, scores)]
