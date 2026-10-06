"""Metainfo dos 30 keypoints SRKD para MMPose.

Os sigmas zero são sentinelas exigidas pelo parser MMPose, não estimativas
para OKS. O treino usa NME/PCK até existir calibração.
"""

# Eager MMEngine config: execute the metainfo factory instead of a LazyObject.
_base_ = []

from src.posturaai.keypoints import srkd_metainfo


dataset_info = srkd_metainfo()
