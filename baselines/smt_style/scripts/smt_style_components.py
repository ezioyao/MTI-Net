"""Portable components for the SMT-style Temporal Mean Teacher baseline.

These helpers intentionally do not import the official Stable Mean Teacher
repository. The official code is action-mask specific; this file only ports the
transferable EMA and ramp-up rules and implements a YOLO-box temporal
consistency filter without using DART.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np


@dataclass(frozen=True)
class PseudoLabelStats:
    candidate_count: int
    retained_count: int
    rejected_confidence_count: int
    rejected_temporal_count: int
    empty_frame_count: int
    frame_count: int

    @property
    def retained_per_frame(self) -> float:
        return self.retained_count / max(self.frame_count, 1)

    @property
    def empty_frame_ratio(self) -> float:
        return self.empty_frame_count / max(self.frame_count, 1)


def sigmoid_rampup(epoch: float, rampup_length: float) -> float:
    """Official SMT-style exponential sigmoid ramp-up."""

    if rampup_length <= 0:
        return 1.0
    if epoch < rampup_length:
        clipped = min(max(epoch, 0.0), rampup_length)
        phase = 1.0 - clipped / rampup_length
        return float(math.exp(-5.0 * phase * phase))
    return 1.0


def update_ema(student, teacher, global_step: int, ema_decay: float) -> float:
    """Update teacher parameters using the official Stable Mean Teacher EMA rule.

    The objects are expected to be PyTorch modules. The function avoids importing
    torch at module import time so the numpy-only unit tests can still run.
    """

    alpha = min(1.0 - 1.0 / float(global_step + 1), ema_decay)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
    return alpha


def box_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU for xyxy boxes."""

    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def valid_boxes_xyxy(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Return a boolean mask for non-empty boxes inside image bounds."""

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    x1, y1, x2, y2 = boxes.T
    return (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0) & (x2 <= width) & (y2 <= height)


def greedy_same_class_temporal_match(
    target_boxes: np.ndarray,
    target_classes: np.ndarray,
    neighbor_boxes: np.ndarray,
    neighbor_classes: np.ndarray,
    tau_temporal: float,
) -> np.ndarray:
    """Return mask of target boxes that can be matched to same-class neighbor boxes."""

    target_boxes = np.asarray(target_boxes, dtype=np.float64)
    neighbor_boxes = np.asarray(neighbor_boxes, dtype=np.float64)
    target_classes = np.asarray(target_classes)
    neighbor_classes = np.asarray(neighbor_classes)
    if len(target_boxes) == 0 or len(neighbor_boxes) == 0:
        return np.zeros((len(target_boxes),), dtype=bool)

    ious = box_iou_xyxy(target_boxes, neighbor_boxes)
    same_cls = target_classes[:, None] == neighbor_classes[None, :]
    ious = np.where(same_cls, ious, -1.0)

    retained = np.zeros((len(target_boxes),), dtype=bool)
    used_neighbors: set[int] = set()
    order = np.argsort(-ious.max(axis=1))
    for target_idx in order:
        neighbor_idx = int(np.argmax(ious[target_idx]))
        if neighbor_idx in used_neighbors:
            continue
        if ious[target_idx, neighbor_idx] >= tau_temporal:
            retained[target_idx] = True
            used_neighbors.add(neighbor_idx)
    return retained


def temporal_consistency_filter(
    target_boxes: np.ndarray,
    target_scores: np.ndarray,
    target_classes: np.ndarray,
    neighbor_predictions: Iterable[tuple[np.ndarray, np.ndarray]],
    tau_conf: float,
    tau_temporal: float,
    image_width: float,
    image_height: float,
) -> tuple[np.ndarray, PseudoLabelStats]:
    """Filter target-frame teacher boxes by confidence, validity, and temporal consistency.

    Parameters
    ----------
    target_boxes:
        Target-frame teacher boxes in xyxy pixel coordinates.
    target_scores:
        Target-frame confidence scores.
    target_classes:
        Target-frame integer class ids.
    neighbor_predictions:
        Iterable of `(boxes, classes)` for valid temporal neighbors. The target
        pseudo box is kept when it matches at least one neighbor.
    """

    target_boxes = np.asarray(target_boxes, dtype=np.float64)
    target_scores = np.asarray(target_scores, dtype=np.float64)
    target_classes = np.asarray(target_classes)
    candidate_count = len(target_boxes)

    conf_mask = target_scores >= tau_conf
    valid_mask = valid_boxes_xyxy(target_boxes, image_width, image_height)
    pre_temporal = conf_mask & valid_mask

    retained_temporal = np.zeros((candidate_count,), dtype=bool)
    if pre_temporal.any():
        for neighbor_boxes, neighbor_classes in neighbor_predictions:
            matched = greedy_same_class_temporal_match(
                target_boxes[pre_temporal],
                target_classes[pre_temporal],
                neighbor_boxes,
                neighbor_classes,
                tau_temporal,
            )
            retained_temporal[np.where(pre_temporal)[0][matched]] = True

    retained_mask = pre_temporal & retained_temporal
    stats = PseudoLabelStats(
        candidate_count=candidate_count,
        retained_count=int(retained_mask.sum()),
        rejected_confidence_count=int((~conf_mask).sum()),
        rejected_temporal_count=int((pre_temporal & ~retained_temporal).sum()),
        empty_frame_count=int(retained_mask.sum() == 0),
        frame_count=1,
    )
    return retained_mask, stats

