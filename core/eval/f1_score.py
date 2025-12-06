"""
F1 score 评估工具模块。

用于计算点云分类质量指标，例如生成点云与参考点云之间的
F1 score 等。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score as sklearn_f1_score


def classify_points_by_distance(
    points_generated: np.ndarray,
    points_reference: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    基于距离阈值将生成点云的每个点分类为 1（有效）或 0（无效）。

    参数:
        points_generated: 形状为 (Na, 3) 的 float32 数组，生成点云。
        points_reference: 形状为 (Nb, 3) 的 float32 数组，参考点云。
        threshold: 距离阈值，超过此距离的点被标记为无效。

    返回:
        形状为 (Na,) 的 int32 标签数组，1 表示有效，0 表示无效。

    说明:
        计算生成点到参考点集的最近距离，若距离不超过阈值则标记为有效。
    """
    points_generated = np.asarray(points_generated, dtype=np.float32)
    points_reference = np.asarray(points_reference, dtype=np.float32)

    if points_generated.size == 0 or points_reference.size == 0:
        return np.array([], dtype=np.int32)

    # 计算生成点到参考点的最近距离
    distances = np.min(cdist(points_generated, points_reference), axis=1)
    labels = (distances <= threshold).astype(np.int32)
    return labels


def f1_score_from_distance(
    points_generated: np.ndarray,
    points_reference: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    基于几何距离计算 F1 score。

    将生成点云中的点分类为"有效"（接近参考点）和"无效"（远离参考点），
    以参考点云的完全有效性作为真值标签，计算 F1 score。

    参数:
        points_generated: 形状为 (Na, 3) 的 float32 数组，生成点云。
        points_reference: 形状为 (Nb, 3) 的 float32 数组，参考点云。
        threshold: 距离阈值。

    返回:
        标量 F1 score，范围为 [0, 1]。

    说明:
        为了公平比较，仅对两个点云的最小长度范围内的点进行评估。
    """
    points_generated = np.asarray(points_generated, dtype=np.float32)
    points_reference = np.asarray(points_reference, dtype=np.float32)

    if points_generated.size == 0 or points_reference.size == 0:
        return 0.0

    # 获取生成点的预测标签
    pred_labels = classify_points_by_distance(points_generated, points_reference, threshold)

    # 参考点云中的所有点都被认为是有效的
    ref_labels = np.ones(len(points_reference), dtype=np.int32)

    # 为了公平比较，取最小长度
    min_len = min(len(pred_labels), len(ref_labels))
    pred_labels = pred_labels[:min_len]
    ref_labels = ref_labels[:min_len]

    f1 = sklearn_f1_score(ref_labels, pred_labels, zero_division=0)
    return float(f1)


def f1_score_from_labels(
    labels_reference: np.ndarray,
    labels_generated: np.ndarray,
    average: str = "weighted",
) -> float:
    """
    基于标签数组计算 F1 score。

    参数:
        labels_reference: 形状为 (N,) 的 int32 数组，参考标签。
        labels_generated: 形状为 (N,) 的 int32 数组，生成标签。
        average: 多分类平均方式，可选 "weighted"、"macro"、"micro"。

    返回:
        标量 F1 score，范围为 [0, 1]。

    说明:
        两个标签数组的长度必须相同。
    """
    labels_reference = np.asarray(labels_reference, dtype=np.int32)
    labels_generated = np.asarray(labels_generated, dtype=np.int32)

    if labels_reference.size == 0 or labels_generated.size == 0:
        return 0.0

    if len(labels_reference) != len(labels_generated):
        raise ValueError(
            f"标签数组长度不匹配: {len(labels_reference)} vs {len(labels_generated)}"
        )

    f1 = sklearn_f1_score(labels_reference, labels_generated, average=average, zero_division=0)
    return float(f1)


__all__ = ["classify_points_by_distance", "f1_score_from_distance", "f1_score_from_labels"]