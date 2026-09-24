"""Métricas de localização de keypoints sem pressupor sigmas de OKS."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


NUM_KEYPOINTS = 30
FOOT_INDICES = tuple(range(15, 25))


def _summarize(errors_px: np.ndarray, errors_norm: np.ndarray) -> dict[str, float | int | None]:
    """Resume apenas erros de pontos visíveis; ``None`` indica ausência de dados."""
    count = int(errors_px.size)
    if count == 0:
        return {
            "count": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "p90_error_px": None,
            "nme": None,
            "median_normalized_error": None,
            "p90_normalized_error": None,
            "pck_005": None,
        }
    return {
        "count": count,
        "mean_error_px": float(np.mean(errors_px)),
        "median_error_px": float(np.median(errors_px)),
        "p90_error_px": float(np.percentile(errors_px, 90)),
        "nme": float(np.mean(errors_norm)),
        "median_normalized_error": float(np.median(errors_norm)),
        "p90_normalized_error": float(np.percentile(errors_norm, 90)),
        "pck_005": float(np.mean(errors_norm <= 0.05)),
    }


def evaluate_keypoints(
    predictions: np.ndarray,
    targets: np.ndarray,
    bboxes_xywh: np.ndarray,
    visibility: np.ndarray,
    keypoint_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Calcula NME e PCK@0,05 em coordenadas de imagem.

    ``predictions`` e ``targets`` têm shape ``(N, 30, 2)``; as boxes GT
    derivadas têm shape ``(N, 4)`` e formato XYWH. O denominador de NME é
    ``max(altura_bbox_gt, 1)`` por pessoa. ``visibility > 0`` seleciona os
    pontos com ground truth utilizável. Esta função não usa confiança predita
    para filtrar resultados.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(targets, dtype=np.float64)
    boxes = np.asarray(bboxes_xywh, dtype=np.float64)
    visible = np.asarray(visibility) > 0

    if pred.ndim != 3 or pred.shape[1:] != (NUM_KEYPOINTS, 2):
        raise ValueError("predictions deve ter shape (N, 30, 2)")
    if gt.shape != pred.shape:
        raise ValueError("targets deve ter o mesmo shape de predictions")
    count = pred.shape[0]
    if boxes.shape != (count, 4):
        raise ValueError("bboxes_xywh deve ter shape (N, 4)")
    if visible.shape != (count, NUM_KEYPOINTS):
        raise ValueError("visibility deve ter shape (N, 30)")
    if count == 0 or not visible.any():
        raise ValueError("não há keypoints visíveis para avaliar")
    if keypoint_names is not None and len(keypoint_names) != NUM_KEYPOINTS:
        raise ValueError("keypoint_names deve conter 30 nomes")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
        raise ValueError("bounding boxes GT devem ser finitas e positivas")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("coordenadas devem ser finitas")

    errors_px = np.linalg.norm(pred - gt, axis=-1)
    errors_norm = errors_px / np.maximum(boxes[:, 3:4], 1.0)
    names = list(keypoint_names) if keypoint_names is not None else [str(index) for index in range(NUM_KEYPOINTS)]

    per_keypoint = {
        name: _summarize(errors_px[visible[:, index], index], errors_norm[visible[:, index], index])
        for index, name in enumerate(names)
    }
    foot_mask = visible[:, FOOT_INDICES]
    foot_px = errors_px[:, FOOT_INDICES][foot_mask]
    foot_norm = errors_norm[:, FOOT_INDICES][foot_mask]
    return {
        "global": _summarize(errors_px[visible], errors_norm[visible]),
        "feet": _summarize(foot_px, foot_norm),
        "per_keypoint": per_keypoint,
    }
