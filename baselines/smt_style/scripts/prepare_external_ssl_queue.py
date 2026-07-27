"""Prepare a server-side queue template for the external video SSL comparison.

This script does not fabricate results. It only writes the intended experiment
matrix and command templates so the server runner can execute or resume them
after the 25% configs and run directories are verified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEEDS = [42, 43, 44, 45, 46]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in SEEDS:
        rows.append(
            {
                "experiment_id": f"external_ssl_y12_25pct_supervised_s{seed}_worker32",
                "method": "YOLO12-S supervised",
                "seed": str(seed),
                "dataset_config": "coco_25pct.yaml",
                "model_config": "ultralytics/cfg/models/12/yolo12.yaml",
                "epochs": "150",
                "batch": "8",
                "workers": "32",
                "status": "verify_existing_or_run",
                "command": (
                    "python tools/train_experiment.py "
                    f"--exp-id external_ssl_y12_25pct_supervised_s{seed}_worker32 "
                    "--engine yolo --variant base "
                    "--model-config ultralytics/cfg/models/12/yolo12.yaml "
                    "--data-config coco_25pct.yaml --project runs/detect "
                    "--epochs 150 --batch 8 --imgsz 640 "
                    f"--seed {seed} --workers 32"
                ),
            }
        )
        rows.append(
            {
                "experiment_id": f"external_ssl_y12_25pct_smt_style_s{seed}_worker32",
                "method": "SMT-style Temporal Mean Teacher",
                "seed": str(seed),
                "dataset_config": "coco_25pct.yaml",
                "model_config": "ultralytics/cfg/models/12/yolo12.yaml",
                "epochs": "150",
                "batch": "8",
                "workers": "32",
                "status": "pending_after_seed42_dry_run",
                "command": (
                    "python baselines/smt_style/scripts/train_smt_style_yolo12.py "
                    f"--exp-id external_ssl_y12_25pct_smt_style_s{seed}_worker32 "
                    "--config baselines/smt_style/configs/smt_style_yolo12_25pct.yaml "
                    f"--seed {seed}"
                ),
            }
        )
        rows.append(
            {
                "experiment_id": f"external_ssl_y12_25pct_dart_only_s{seed}_worker32",
                "method": "DART-only",
                "seed": str(seed),
                "dataset_config": "coco_dart_25pct.yaml",
                "model_config": "ultralytics/cfg/models/12/yolo12.yaml",
                "epochs": "150",
                "batch": "8",
                "workers": "32",
                "status": "verify_existing_or_run",
                "command": (
                    "python tools/train_experiment.py "
                    f"--exp-id external_ssl_y12_25pct_dart_only_s{seed}_worker32 "
                    "--engine yolo --variant base "
                    "--model-config ultralytics/cfg/models/12/yolo12.yaml "
                    "--data-config coco_dart_25pct.yaml --project runs/detect "
                    "--epochs 150 --batch 8 --imgsz 640 "
                    f"--seed {seed} --workers 32"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="baselines/smt_style/external_ssl_queue_template.csv")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
