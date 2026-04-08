```markdown
# Acknowledgements
This work is built upon the excellent [YOLO-Master](https://github.com/your-yolormaster-repo) framework and the [Ultralytics](https://github.com/ultralytics/ultralytics) ecosystem.
We sincerely thank the authors and contributors for their outstanding open-source work.

# Environment Setup
This project requires `Python 3.11`.

Install dependencies:
```bash
pip install -r requirements.txt
```

# Project Structure (After Full Setup)
```
project_root/
├── dataset/                # Auto-generated after dataset extraction
│   ├── history_frames/     # Historical frame data
│   └── wheat_yolo_dataset/ # Test dataset
├── runs/                   # Auto-generated after weight extraction
│   └── ...                 # All model weights
├── eval_ori.py             # Original baseline model evaluation
├── eval_ours.py            # Our model with DART + MTI evaluation
├── requirements.txt        # Dependencies
└── README.md
```

# Dataset Preparation
1. Download the test dataset from: 
2. Unzip it in the **project root directory**
3. The `dataset/` folder will be created automatically

# Model Weights Preparation
1. Download pretrained weights from: 
2. Unzip them in the **project root directory**
3. The `runs/` folder will be created automatically

# Evaluation Instructions
We provide two evaluation scripts:
- `eval_ori.py`: evaluate the original YOLO-Master baseline model
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

# Star Support
If you find this project useful, please ⭐ star this repository and the original YOLO-Master repository!
```