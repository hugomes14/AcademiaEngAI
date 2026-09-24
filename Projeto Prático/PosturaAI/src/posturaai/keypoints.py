"""SRKD keypoint order and provisional skeleton, as declared by the raw COCO file.

The skeleton is useful for inspection but has not yet passed the M0 visual audit.
Indices in this module are zero based; the source COCO skeleton is one based.
"""

from __future__ import annotations


KEYPOINT_NAMES: tuple[str, ...] = (
    "Nose",
    "LEye",
    "REye",
    "LeftEar",
    "RightEar",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
    "LeftAnkle",
    "RightAnkle",
    "LeftHeel",
    "RightHeel",
    "LeftFirstMetatarsal",
    "RightFirstMetatarsal",
    "LeftFifthMetatarsal",
    "RightFifthMetatarsal",
    "LeftToe",
    "RightToe",
    "TopHead",
    "BackHead",
    "UpTrunk",
    "MiddleTrunk",
    "Pelvis",
)

FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
)

# First occurrence of each undirected source edge, converted from one based
# COCO indices and normalized to ascending endpoints. The source repeats
# (2, 4) and (3, 5) once each.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (25, 26), (0, 26),
    (0, 27), (5, 27), (6, 27), (27, 28), (0, 3), (1, 2),
    (28, 29), (11, 29), (12, 29), (5, 7), (7, 9), (6, 8),
    (8, 10), (11, 13), (13, 15), (15, 17), (17, 19),
    (17, 21), (19, 23), (21, 23), (12, 14), (14, 16),
    (16, 18), (18, 20), (18, 22), (20, 24), (22, 24),
)

FOOT_INDICES: tuple[int, ...] = tuple(range(15, 25))


def flip_indices() -> tuple[int, ...]:
    """Return the permutation of keypoint channels for horizontal flipping."""

    indices = list(range(len(KEYPOINT_NAMES)))
    for left, right in FLIP_PAIRS:
        indices[left], indices[right] = indices[right], indices[left]
    return tuple(indices)


def srkd_metainfo() -> dict:
    """Return MMPose dataset metainfo with zero *parser sentinel* sigmas.

    MMPose's metainfo parser requires a ``sigmas`` field. These zeros satisfy
    the parser only: AP/OKS must stay disabled until 30 SRKD-specific sigmas
    have been justified and replace them.
    """

    swapped_names = {KEYPOINT_NAMES[i]: KEYPOINT_NAMES[j] for i, j in FLIP_PAIRS}
    swapped_names.update({KEYPOINT_NAMES[j]: KEYPOINT_NAMES[i] for i, j in FLIP_PAIRS})
    keypoint_info = {}
    for index, name in enumerate(KEYPOINT_NAMES):
        side = "left" if name.startswith("Left") or (index == 1) else (
            "right" if name.startswith("Right") or (index == 2) else "center"
        )
        color = {
            "left": [0, 204, 255],
            "right": [255, 170, 0],
            "center": [255, 255, 255],
        }[side]
        keypoint_info[index] = dict(
            name=name,
            id=index,
            color=color,
            type="lower" if index in (*range(11, 25), 29) else "upper",
            swap=swapped_names.get(name, ""),
        )

    skeleton_info = {
        index: dict(
            id=index,
            link=(KEYPOINT_NAMES[start], KEYPOINT_NAMES[end]),
            color=[0, 255, 0],
        )
        for index, (start, end) in enumerate(SKELETON_EDGES)
    }
    return dict(
        dataset_name="srkd",
        keypoint_info=keypoint_info,
        skeleton_info=skeleton_info,
        joint_weights=[1.0] * len(KEYPOINT_NAMES),
        sigmas=[0.0] * len(KEYPOINT_NAMES),
        sigmas_calibrated=False,
    )
