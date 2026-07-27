"""SMT-style Temporal Mean Teacher training entry for YOLO12-S.

This is a task-adapted external video SSL baseline. It ports the transferable
Stable Mean Teacher mechanisms:

- EMA teacher.
- Weak-view teacher pseudo-label generation.
- Strong-view student pseudo-label supervision.
- Temporal consistency filtering from adjacent frames.
- Sigmoid ramp-up for the unsupervised branch.

It intentionally does not implement official EoR, because the official module is
an action-localization-mask auxiliary UNet rather than a YOLO box-detection
component.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from bisect import bisect_left

import cv2
import numpy as np
import torch

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for SMT-style training.") from exc

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

from smt_style_components import sigmoid_rampup, temporal_consistency_filter
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywh


FRAME_RE = re.compile(r"^(?P<video>DJI_\d+)_frame_(?P<frame>\d+)")


@dataclass
class RuntimePseudoStats:
    candidate: int = 0
    retained: int = 0
    rejected_confidence: int = 0
    rejected_temporal: int = 0
    frames: int = 0
    empty_frames: int = 0

    def update(self, stats) -> None:
        self.candidate += int(stats.candidate_count)
        self.retained += int(stats.retained_count)
        self.rejected_confidence += int(stats.rejected_confidence_count)
        self.rejected_temporal += int(stats.rejected_temporal_count)
        self.frames += int(stats.frame_count)
        self.empty_frames += int(stats.empty_frame_count)

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        out["retained_per_frame"] = self.retained / max(self.frames, 1)
        out["empty_frame_ratio"] = self.empty_frames / max(self.frames, 1)
        return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_root_from_yaml(data_yaml: Path) -> Path:
    data = load_yaml(data_yaml)
    path = Path(data["path"])
    if not path.is_absolute():
        path = data_yaml.parent / path
    return path


def parse_image_id(path: Path) -> tuple[str, int] | None:
    match = FRAME_RE.match(path.stem)
    if not match:
        return None
    return match.group("video"), int(match.group("frame"))


class UnlabeledTripletPool:
    """Enumerate full-training images and sample temporal neighbors."""

    def __init__(self, full_root: Path, labeled_root: Path, delta_t: int, seed: int):
        full_dir = full_root / "images" / "train"
        labeled_dir = labeled_root / "images" / "train"
        if not full_dir.exists():
            raise FileNotFoundError(f"missing full train image directory: {full_dir}")
        if not labeled_dir.exists():
            raise FileNotFoundError(f"missing labeled train image directory: {labeled_dir}")

        labeled_names = {p.name for p in labeled_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        all_images = sorted(p for p in full_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.images = [p for p in all_images if p.name not in labeled_names]
        self.delta_t = int(delta_t)
        self.rng = random.Random(seed)
        self.by_video: dict[str, list[tuple[int, Path]]] = {}
        for p in all_images:
            parsed = parse_image_id(p)
            if parsed is None:
                continue
            video, frame = parsed
            self.by_video.setdefault(video, []).append((frame, p))
        for items in self.by_video.values():
            items.sort(key=lambda x: x[0])

        self.valid_targets = [p for p in self.images if parse_image_id(p) and self._neighbors_for(p)]
        if not self.valid_targets:
            raise RuntimeError("no valid unlabeled target frames with temporal neighbors were found")

    def _neighbors_for(self, path: Path) -> list[Path]:
        parsed = parse_image_id(path)
        if parsed is None:
            return []
        video, frame = parsed
        frames = self.by_video.get(video, [])
        neighbors = []
        frame_ids = [f for f, _ in frames]
        for target in (frame - self.delta_t, frame + self.delta_t):
            idx = bisect_left(frame_ids, target)
            candidates = []
            if idx < len(frames):
                candidates.append(frames[idx])
            if idx > 0:
                candidates.append(frames[idx - 1])
            candidates = [(abs(f - target), f, p) for f, p in candidates if f != frame]
            if candidates:
                _, _, neighbor_path = min(candidates, key=lambda x: x[0])
                if neighbor_path not in neighbors:
                    neighbors.append(neighbor_path)
        return neighbors

    def sample(self, n: int) -> list[tuple[Path, list[Path]]]:
        return [(p, self._neighbors_for(p)) for p in self.rng.sample(self.valid_targets, k=min(n, len(self.valid_targets)))]


def letterbox_image(img_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def load_tensor(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    img = letterbox_image(img, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).to(device).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor


def photometric_strong_view(imgs: torch.Tensor) -> torch.Tensor:
    """Strong view without geometric transform, so pseudo-box coordinates remain valid."""

    if imgs.numel() == 0:
        return imgs
    gain = torch.empty((imgs.shape[0], 1, 1, 1), device=imgs.device).uniform_(0.75, 1.25)
    bias = torch.empty((imgs.shape[0], 1, 1, 1), device=imgs.device).uniform_(-0.08, 0.08)
    noise = torch.randn_like(imgs) * 0.015
    return (imgs * gain + bias + noise).clamp(0.0, 1.0)


class SMTStyleDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, smt_config: dict[str, Any], config_path: Path, **kwargs):
        self.smt_config = smt_config
        self.smt_config_path = config_path
        self.unlabeled_pool: UnlabeledTripletPool | None = None
        self.pseudo_stats = RuntimePseudoStats()
        self._smt_prepared = False
        super().__init__(*args, **kwargs)

    def _prepare_smt(self) -> None:
        if self._smt_prepared:
            return
        data_cfg = self.smt_config["data"]
        labeled_yaml = Path("ultralytics/cfg/datasets") / data_cfg["labeled_dataset_config"]
        full_yaml = Path("ultralytics/cfg/datasets") / data_cfg.get("full_dataset_config", "coco_ori.yaml")
        labeled_root = dataset_root_from_yaml(labeled_yaml)
        full_root = dataset_root_from_yaml(full_yaml)
        self.unlabeled_pool = UnlabeledTripletPool(
            full_root=full_root,
            labeled_root=labeled_root,
            delta_t=int(data_cfg.get("delta_t", 4)),
            seed=int(self.args.seed),
        )
        LOGGER.info(
            "SMT-style unlabeled pool: %d candidate target frames from %s",
            len(self.unlabeled_pool.valid_targets),
            full_root,
        )
        self._smt_prepared = True

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        self._prepare_smt()
        if self.ema is None or self.unlabeled_pool is None:
            return batch

        smt = self.smt_config["smt_style"]
        if self.epoch + 1 < int(smt.get("pseudo_start_epoch", 11)):
            return batch
        ramp_epoch = self.epoch + 1 - int(smt.get("pseudo_start_epoch", 11))
        ramp = sigmoid_rampup(ramp_epoch, float(smt.get("rampup_epochs", 30)))
        if random.random() > ramp:
            return batch
        tau_conf = float(smt["selected_thresholds"].get("tau_conf") or smt["threshold_search"]["tau_conf"][1])
        tau_temporal = float(
            smt["selected_thresholds"].get("tau_temporal") or smt["threshold_search"]["tau_temporal"][1]
        )
        unlabeled_batch = int(smt.get("unlabeled_batch", 2))
        imgsz = int(self.args.imgsz)
        samples = self.unlabeled_pool.sample(unlabeled_batch)
        if not samples:
            return batch

        target_imgs = torch.stack([load_tensor(path, imgsz, self.device) for path, _ in samples], 0)
        strong_imgs = photometric_strong_view(target_imgs)

        teacher = self.ema.ema
        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            target_preds = non_max_suppression(teacher(target_imgs), conf_thres=tau_conf, iou_thres=0.7, max_det=300)

        pseudo_bboxes = []
        pseudo_cls = []
        pseudo_batch_idx = []
        for j, ((target_path, neighbor_paths), det) in enumerate(zip(samples, target_preds)):
            neighbor_boxes = []
            neighbor_classes = []
            if neighbor_paths:
                neighbor_imgs = torch.stack([load_tensor(path, imgsz, self.device) for path in neighbor_paths], 0)
                with torch.no_grad():
                    neighbor_preds = non_max_suppression(
                        teacher(neighbor_imgs), conf_thres=tau_conf, iou_thres=0.7, max_det=300
                    )
                for n_det in neighbor_preds:
                    if len(n_det):
                        neighbor_boxes.append(n_det[:, :4].detach().cpu().numpy())
                        neighbor_classes.append(n_det[:, 5].detach().cpu().numpy().astype(int))

            if len(det):
                boxes = det[:, :4].detach().cpu().numpy()
                scores = det[:, 4].detach().cpu().numpy()
                classes = det[:, 5].detach().cpu().numpy().astype(int)
            else:
                boxes = np.zeros((0, 4), dtype=float)
                scores = np.zeros((0,), dtype=float)
                classes = np.zeros((0,), dtype=int)

            keep, stats = temporal_consistency_filter(
                boxes,
                scores,
                classes,
                list(zip(neighbor_boxes, neighbor_classes)),
                tau_conf=tau_conf,
                tau_temporal=tau_temporal,
                image_width=imgsz,
                image_height=imgsz,
            )
            self.pseudo_stats.update(stats)
            if keep.any():
                keep_boxes = torch.as_tensor(boxes[keep], device=self.device, dtype=batch["img"].dtype)
                keep_xywh = xyxy2xywh(keep_boxes)
                keep_xywh[:, [0, 2]] /= imgsz
                keep_xywh[:, [1, 3]] /= imgsz
                pseudo_bboxes.append(keep_xywh)
                pseudo_cls.append(torch.zeros((keep_xywh.shape[0], 1), device=self.device, dtype=batch["cls"].dtype))
                pseudo_batch_idx.append(
                    torch.full((keep_xywh.shape[0],), batch["img"].shape[0] + j, device=self.device)
                )

        if was_training:
            teacher.train()

        batch["img"] = torch.cat([batch["img"], strong_imgs], 0)
        batch["im_file"] = list(batch.get("im_file", [])) + [str(path) for path, _ in samples]
        if "ori_shape" in batch:
            batch["ori_shape"] = list(batch["ori_shape"]) + [(imgsz, imgsz)] * len(samples)
        if "resized_shape" in batch:
            batch["resized_shape"] = list(batch["resized_shape"]) + [(imgsz, imgsz)] * len(samples)

        if pseudo_bboxes:
            batch["bboxes"] = torch.cat([batch["bboxes"], *pseudo_bboxes], 0)
            batch["cls"] = torch.cat([batch["cls"], *pseudo_cls], 0)
            batch["batch_idx"] = torch.cat([batch["batch_idx"], *pseudo_batch_idx], 0)
        return batch

    def optimizer_step(self):
        super().optimizer_step()
        if RANK in {-1, 0} and self.epoch >= 0 and self.pseudo_stats.frames:
            stats_path = Path(self.save_dir) / "smt_pseudo_stats.json"
            stats_path.write_text(json.dumps(self.pseudo_stats.to_dict(), indent=2), encoding="utf-8")


def preflight(config: dict, exp_id: str, seed: int) -> dict:
    required = [
        ("data", "labeled_dataset_config"),
        ("detector", "model_config"),
        ("smt_style", "ema_decay"),
        ("smt_style", "consistency_weight"),
        ("smt_style", "rampup_epochs"),
    ]
    missing = []
    for section, key in required:
        if section not in config or key not in config[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")

    ramp = [sigmoid_rampup(e, float(config["smt_style"]["rampup_epochs"])) for e in (0, 5, 15, 30)]
    return {
        "exp_id": exp_id,
        "seed": seed,
        "method": config.get("method_name", "SMT-style Temporal Mean Teacher"),
        "labeled_dataset_config": config["data"]["labeled_dataset_config"],
        "reference_dart_dataset_config": config["data"].get("reference_dart_dataset_config"),
        "model_config": config["detector"]["model_config"],
        "ema_decay": config["smt_style"]["ema_decay"],
        "consistency_weight": config["smt_style"]["consistency_weight"],
        "rampup_check": ramp,
        "status": "preflight_passed",
    }


def build_train_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    data = config["data"]
    detector = config["detector"]
    return {
        "model": detector["model_config"],
        "data": data["labeled_dataset_config"],
        "epochs": args.epochs or int(training["epochs"]),
        "batch": args.batch or int(training["batch"]),
        "imgsz": int(detector["imgsz"]),
        "seed": args.seed,
        "device": args.device,
        "project": args.project,
        "name": args.exp_id,
        "exist_ok": args.allow_existing,
        "optimizer": training.get("optimizer", "AdamW"),
        "lr0": float(training.get("lr0", 0.001)),
        "momentum": float(training.get("momentum", 0.937)),
        "weight_decay": float(training.get("weight_decay", 0.0005)),
        "cos_lr": bool(training.get("cos_lr", True)),
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "rect": False,
        "cache": False,
        "plots": False,
        "workers": args.workers or int(training.get("workers", 32)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["preflight", "train"], default="train")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--pseudo-start-epoch", type=int, default=-1, help="Smoke-test override; -1 keeps config.")
    parser.add_argument("--tau-conf", type=float, default=-1.0, help="Smoke-test override; negative keeps config.")
    parser.add_argument("--tau-temporal", type=float, default=-1.0, help="Smoke-test override; negative keeps config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    if args.pseudo_start_epoch >= 0:
        config["smt_style"]["pseudo_start_epoch"] = int(args.pseudo_start_epoch)
    if args.tau_conf >= 0:
        config["smt_style"]["selected_thresholds"]["tau_conf"] = float(args.tau_conf)
    if args.tau_temporal >= 0:
        config["smt_style"]["selected_thresholds"]["tau_temporal"] = float(args.tau_temporal)
    report = preflight(config, args.exp_id, args.seed)
    out_dir = Path(config.get("outputs", {}).get("root", "baselines/smt_style")) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.exp_id}.preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.mode == "preflight":
        return

    train_kwargs = build_train_kwargs(args, config)
    run_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    if run_dir.exists() and not args.allow_existing:
        raise SystemExit(f"Refusing to overwrite existing run directory: {run_dir}")

    model = YOLO(train_kwargs["model"])
    trainer = SMTStyleDetectionTrainer(overrides=train_kwargs, _callbacks=model.callbacks, smt_config=config, config_path=config_path)
    trainer.model = model.model
    model.trainer = trainer
    trainer.train()


if __name__ == "__main__":
    main()
