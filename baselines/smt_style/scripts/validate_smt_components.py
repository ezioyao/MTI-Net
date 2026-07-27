"""Unit checks for portable SMT-style components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smt_style_components import box_iou_xyxy, sigmoid_rampup, temporal_consistency_filter


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ramp_values = [sigmoid_rampup(e, 30) for e in (0, 5, 15, 30, 60)]
    assert 0.0 < ramp_values[0] < ramp_values[1] < ramp_values[2] < ramp_values[3] == ramp_values[4] == 1.0

    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=float)
    boxes_b = np.array([[0, 0, 10, 10], [22, 22, 42, 42]], dtype=float)
    iou = box_iou_xyxy(boxes_a, boxes_b)
    assert np.isclose(iou[0, 0], 1.0)
    assert 0.0 < iou[1, 1] < 1.0

    target_boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [80, 80, 90, 90]], dtype=float)
    target_scores = np.array([0.9, 0.8, 0.2], dtype=float)
    target_classes = np.array([0, 0, 0], dtype=int)
    neighbor_boxes = np.array([[1, 1, 11, 11], [60, 60, 70, 70]], dtype=float)
    neighbor_classes = np.array([0, 0], dtype=int)
    keep, stats = temporal_consistency_filter(
        target_boxes,
        target_scores,
        target_classes,
        [(neighbor_boxes, neighbor_classes)],
        tau_conf=0.5,
        tau_temporal=0.5,
        image_width=100,
        image_height=100,
    )
    assert keep.tolist() == [True, False, False]
    assert stats.candidate_count == 3
    assert stats.retained_count == 1
    assert stats.rejected_confidence_count == 1
    assert stats.rejected_temporal_count == 1

    report = {
        "status": "passed",
        "ramp_values": ramp_values,
        "iou": iou.tolist(),
        "temporal_keep": keep.tolist(),
        "stats": stats.__dict__,
    }
    (out_dir / "smt_component_unit_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

