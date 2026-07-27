# SMT-style Temporal Mean Teacher Baseline

This folder contains the code used for the task-adapted SMT-style baseline in the DART external video semi-supervised comparison.

It is not a full reproduction of the AAAI 2025 Stable Mean Teacher method. The original method targets video action detection and uses action-localization-mask modules that do not transfer cleanly to YOLO box detection. This implementation keeps the transferable parts:


- EMA teacher-student training.
- Weak-view teacher pseudo-label generation.
- Strong-view student supervision.
- Box-level temporal consistency filtering with neighboring video frames.
- Sigmoid ramp-up for the unsupervised branch.

The EoR module is not ported because the official implementation is tied to auxiliary action-localization masks rather than detector bounding boxes.

## Files

- `configs/smt_style_yolo12_25pct.yaml`: main experiment configuration for YOLO12-S with 25% sparse-anchor supervision.
- `scripts/smt_style_components.py`: portable EMA, ramp-up, IoU, and temporal consistency utilities.
- `scripts/train_smt_style_yolo12.py`: YOLO12-S training entry with the SMT-style unlabeled branch.
- `scripts/validate_smt_components.py`: small unit checks for the portable components.
- `scripts/visualize_smt_pseudo_labels.py`: offline pseudo-label visualization helper.
- `scripts/prepare_external_ssl_queue.py`: writes the intended multi-seed command table.

## Expected Project Layout

Run these scripts from the project root that contains `ultralytics/`.

The default configuration expects these dataset YAML files to be placed under `ultralytics/cfg/datasets/`:

- `ultralytics/cfg/datasets/coco_25pct.yaml`
- `ultralytics/cfg/datasets/coco_ori.yaml`
- `ultralytics/cfg/datasets/coco_dart_25pct.yaml` for the DART-only reference queue

The YOLO12-S model configuration is `ultralytics/cfg/models/12/yolo12.yaml`.

## Quick Checks

```bash
python baselines/smt_style/scripts/validate_smt_components.py
```

```bash
python baselines/smt_style/scripts/train_smt_style_yolo12.py \
  --mode preflight \
  --exp-id external_ssl_y12_25pct_smt_style_s42_worker32 \
  --config baselines/smt_style/configs/smt_style_yolo12_25pct.yaml \
  --seed 42
```

## Training

Example seed-42 run:

```bash
python baselines/smt_style/scripts/train_smt_style_yolo12.py \
  --exp-id external_ssl_y12_25pct_smt_style_s42_worker32 \
  --config baselines/smt_style/configs/smt_style_yolo12_25pct.yaml \
  --seed 42 \
  --device 0 \
  --project runs/detect
```

Generate the five-seed command table:

```bash
python baselines/smt_style/scripts/prepare_external_ssl_queue.py
```

## Pseudo-label Visualization

```bash
python baselines/smt_style/scripts/visualize_smt_pseudo_labels.py \
  --weights runs/detect/external_ssl_y12_25pct_smt_style_s42_worker32/weights/best.pt \
  --config baselines/smt_style/configs/smt_style_yolo12_25pct.yaml \
  --output baselines/smt_style/visualizations \
  --num 20 \
  --seed 42 \
  --device 0
```

## Notes

- MTI-Net is disabled in this comparison.
- DS-IM is not used.
- DART outputs are not used by the SMT-style baseline.
- The final inference model is the student YOLO12-S detector.
- The official Stable Mean Teacher repository and commit are recorded in the YAML config for traceability.
