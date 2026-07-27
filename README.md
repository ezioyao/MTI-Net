# Acknowledgements
This work is built upon the excellent [YOLO-Master](https://github.com/Tencent/YOLO-Master) framework and the [Ultralytics](https://github.com/ultralytics/ultralytics) ecosystem. We sincerely thank the authors and contributors for their outstanding open-source work.

# Environment Setup
This project requires `Python 3.11`.

Install dependencies:
```bash
pip install -r requirements.txt
```

# Project Structure (After Full Setup)
```
project_root/
|-- dataset/                # Auto-generated after dataset extraction
|   |-- history_frames/     # Historical frame data
|   `-- wheat_yolo_dataset/ # Test dataset
|-- runs/                   # Auto-generated after weight extraction
|   `-- detect/             # All model weights
|-- ultralytics/cfg/
|   |-- datasets/           # Dataset configuration files
|   `-- models/             # Model configuration files
|-- baselines/smt_style/    # SMT-style external video SSL baseline
|-- eval_ori.py             # Original baseline model evaluation
|-- eval_ours.py            # Our model with DART + MTI evaluation
|-- requirements.txt        # Dependencies
`-- README.md
```

# Dataset Preparation
1. Download the test dataset from: [UAV-WheatSeedling](https://www.scidb.cn/en/detail?dataSetId=d0870b6a49af4216921cd8efc11d8850)
2. Unzip it in the **project root directory**
3. The `dataset/` folder will be created automatically

# Model Weights Preparation
1. Download pretrained weights from: [UAV-WheatSeedling](https://www.scidb.cn/en/detail?dataSetId=d0870b6a49af4216921cd8efc11d8850)
2. Unzip them in the **project root directory**
3. The `runs/` folder will be created automatically

# Configuration Files
Dataset configuration files are in `ultralytics/cfg/datasets/`. The default evaluation scripts use `coco_ori.yaml`.

Model configuration files are in `ultralytics/cfg/models/`. The main YOLO12-S configurations are `ultralytics/cfg/models/12/yolo12.yaml` and `ultralytics/cfg/models/12/yolomti-12.yaml`.

# Evaluation Instructions
We provide two evaluation scripts:
- `eval_ori.py`: evaluate the original baseline model
- `eval_ours.py`: evaluate our improved model with DART and MTI modules

## Evaluate Original Model
```bash
python eval_ori.py
```
To switch models, modify the `MODEL_WEIGHTS` path inside `eval_ori.py`.

## Evaluate Our Improved Model (DART + MTI)
```bash
python eval_ours.py
```
To switch models, modify the `MODEL_WEIGHTS` path inside `eval_ours.py`.

# SMT-style Baseline
The external video semi-supervised baseline is provided in `baselines/smt_style/`.

This baseline should be named `SMT-style Temporal Mean Teacher`, not a full Stable Mean Teacher reproduction, because only the transferable teacher-student and temporal-consistency components are adapted to YOLO box detection.

Main configuration:
```text
baselines/smt_style/configs/smt_style_yolo12_25pct.yaml
```

Quick component check:
```bash
python baselines/smt_style/scripts/validate_smt_components.py
```

# Star Support
If you find this project useful, please ⭐ star this repository and the original YOLO-Master repository!