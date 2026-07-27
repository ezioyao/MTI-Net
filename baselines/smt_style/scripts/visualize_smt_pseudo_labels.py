"""Generate pseudo-label visualizations for the SMT-style baseline.

The script uses the current student checkpoint as the teacher snapshot for
offline inspection. It does not affect training and does not use DART outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

def find_repo_root(start: Path) -> Path:
    """Find the YOLO project root from this script location."""

    for path in [start, *start.parents]:
        if (path / "ultralytics").exists() and (path / "cfg").exists():
            return path
        if (path / "ultralytics").exists() and (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Could not locate the project root containing ultralytics/")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from smt_style_components import temporal_consistency_filter
from train_smt_style_yolo12 import UnlabeledTripletPool, dataset_root_from_yaml, letterbox_image, load_yaml
from ultralytics import YOLO


def read_image_640(path: Path, imgsz: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return letterbox_image(img, imgsz)


def predict_boxes(model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float, device: str):
    result = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        max_det=300,
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    return boxes, scores, classes


def strong_view_preview(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    gain = rng.uniform(0.75, 1.25)
    bias = rng.uniform(-0.08, 0.08)
    noise = rng.normalvariate(0.0, 0.015)
    out = np.clip(img * gain + bias + noise, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return out


def make_grid(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    drawn = []
    for img, label in zip(images, labels):
        canvas = img.copy()
        cv2.putText(canvas, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        drawn.append(canvas)
    top = np.concatenate(drawn[:2], axis=1)
    bottom = np.concatenate(drawn[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baselines/smt_style/configs/smt_style_yolo12_25pct.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="baselines/smt_style/visualizations")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tau-conf", type=float, default=None)
    parser.add_argument("--tau-temporal", type=float, default=None)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    data_cfg = cfg["data"]
    smt_cfg = cfg["smt_style"]
    imgsz = int(cfg["detector"]["imgsz"])
    tau_conf = args.tau_conf if args.tau_conf is not None else float(smt_cfg["selected_thresholds"]["tau_conf"])
    tau_temporal = (
        args.tau_temporal if args.tau_temporal is not None else float(smt_cfg["selected_thresholds"]["tau_temporal"])
    )

    labeled_yaml = Path("ultralytics/cfg/datasets") / data_cfg["labeled_dataset_config"]
    full_yaml = Path("ultralytics/cfg/datasets") / data_cfg.get("full_dataset_config", "coco_ori.yaml")
    pool = UnlabeledTripletPool(
        full_root=dataset_root_from_yaml(full_yaml),
        labeled_root=dataset_root_from_yaml(labeled_yaml),
        delta_t=int(data_cfg.get("delta_t", 4)),
        seed=args.seed,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rng = random.Random(args.seed)
    metadata = []
    attempts = 0
    saved = 0
    candidates = pool.valid_targets.copy()
    rng.shuffle(candidates)

    for target_path in candidates:
        if saved >= args.num:
            break
        attempts += 1
        neighbor_paths = pool._neighbors_for(target_path)
        if not neighbor_paths:
            continue
        target_img = read_image_640(target_path, imgsz)
        target_boxes, target_scores, target_classes = predict_boxes(model, target_img, imgsz, tau_conf, args.device)

        neighbor_boxes = []
        neighbor_classes = []
        neighbor_imgs = []
        for n_path in neighbor_paths[:2]:
            n_img = read_image_640(n_path, imgsz)
            n_boxes, _, n_classes = predict_boxes(model, n_img, imgsz, tau_conf, args.device)
            neighbor_imgs.append((n_path, n_img, n_boxes))
            neighbor_boxes.append(n_boxes)
            neighbor_classes.append(n_classes)

        keep, stats = temporal_consistency_filter(
            target_boxes,
            target_scores,
            target_classes,
            list(zip(neighbor_boxes, neighbor_classes)),
            tau_conf=tau_conf,
            tau_temporal=tau_temporal,
            image_width=imgsz,
            image_height=imgsz,
        )
        if stats.retained_count == 0:
            continue

        retained = target_boxes[keep]
        rejected = target_boxes[~keep]
        weak = draw_boxes(target_img, rejected, (60, 60, 220), "weak teacher: rejected")
        weak = draw_boxes(weak, retained, (40, 170, 40), "weak teacher: retained")
        strong = draw_boxes(strong_view_preview(target_img, rng), retained, (40, 170, 40), "strong view pseudo labels")
        if neighbor_imgs:
            n0 = draw_boxes(neighbor_imgs[0][1], neighbor_imgs[0][2], (180, 110, 40), "temporal neighbor A")
        else:
            n0 = np.full_like(target_img, 114)
        if len(neighbor_imgs) > 1:
            n1 = draw_boxes(neighbor_imgs[1][1], neighbor_imgs[1][2], (180, 110, 40), "temporal neighbor B")
        else:
            n1 = np.full_like(target_img, 114)
            cv2.putText(n1, "no second neighbor", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        grid = make_grid([weak, strong, n0, n1], ["", "", "", ""])
        out_path = out_dir / f"smt_pseudo_{saved + 1:03d}_{target_path.stem}.jpg"
        cv2.imwrite(str(out_path), grid)
        metadata.append(
            {
                "visualization": out_path.name,
                "target": str(target_path),
                "neighbors": ";".join(str(p) for p in neighbor_paths),
                "candidate": stats.candidate_count,
                "retained": stats.retained_count,
                "rejected_temporal": stats.rejected_temporal_count,
            }
        )
        saved += 1

    meta_csv = out_dir / "smt_pseudo_visualization_index.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["visualization", "target", "neighbors", "candidate", "retained", "rejected_temporal"],
        )
        writer.writeheader()
        writer.writerows(metadata)
    summary = {
        "weights": str(args.weights),
        "num_requested": args.num,
        "num_saved": saved,
        "attempts": attempts,
        "tau_conf": tau_conf,
        "tau_temporal": tau_temporal,
        "output": str(out_dir),
    }
    (out_dir / "smt_pseudo_visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if saved < args.num:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
