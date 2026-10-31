"""Epoch metrics and validation artifacts; independent of the YOLO forward pass."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .common import discover_images, read_rgb, save_mask, save_overlay, write_json

MONITOR_ENV = "SKYSEG_TRAIN_MONITOR"
DEFAULTS = {"tensorboard": True, "val_save_limit": 0, "val_mask_conf": 0.25, "tensorboard_images": 4}


def monitoring_settings(args, config: dict) -> dict:
    supplied = config.get("skyseg", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError(f"skyseg settings must be a mapping with keys: {', '.join(DEFAULTS)}")
    settings = {**DEFAULTS, **supplied}
    for key in ("val_save_limit", "val_mask_conf"):
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    if getattr(args, "no_tensorboard", False):
        settings["tensorboard"] = False
    if not isinstance(settings["tensorboard"], bool):
        raise ValueError("skyseg.tensorboard must be true or false")
    for key in ("val_save_limit", "tensorboard_images"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"skyseg.{key} must be a non-negative integer (val_save_limit=0 saves all)")
    confidence = settings["val_mask_conf"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("skyseg.val_mask_conf must be in [0,1]")
    return settings


def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def restore_mask(mask: np.ndarray, original_size: tuple[int, int], input_shape, ratio_pad) -> np.ndarray:
    """Undo validation letterboxing, including masks on a smaller prototype grid.

    ratio_pad contains ((height_gain, width_gain), (left_pad, top_pad)).
    Resize to the network grid first so its padding remains in network pixels.
    """
    height, width = map(int, input_shape)
    gains, pad = ratio_pad
    gain_h, gain_w = (float(gains), float(gains)) if np.isscalar(gains) else map(float, gains)
    left, top = round(float(pad[0]) - 0.1), round(float(pad[1]) - 0.1)
    content_w, content_h = round(original_size[0] * gain_w), round(original_size[1] * gain_h)
    right, bottom = left + content_w, top + content_h
    if left < 0 or top < 0 or right > width or bottom > height or content_w <= 0 or content_h <= 0:
        raise ValueError(f"Invalid validation letterbox geometry: {original_size}, {input_shape}, {ratio_pad}")
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    image = image.crop((left, top, right, bottom)).resize(original_size, Image.Resampling.BILINEAR)
    return np.asarray(image) >= 128


class TrainingMonitor:
    def __init__(self, settings=None, *, main_process=True, writer_factory=None, rank=0, world_size=1):
        self.settings = {**DEFAULTS, **(settings or {})}
        self.main_process = main_process
        self.writer_factory = writer_factory
        self.writer = None
        self.rank, self.world_size = rank, world_size
        self.epoch_root = None
        self.allowed_ids = None
        self.pending_epoch = None
        self.epoch_dir = None
        self.manifest = None
        self.saved = 0
        self.completed = False
        self.saved_ids = set()

    def on_train_start(self, trainer):
        if not self.main_process:
            return
        if self.settings["tensorboard"]:
            factory = self.writer_factory
            if factory is None:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as exc:
                    raise RuntimeError("TensorBoard is required: pip install -e '.[yolo]'; "
                                       "or use --no-tensorboard") from exc
                factory = SummaryWriter
            log_dir = Path(trainer.save_dir) / "tensorboard"
            purge_step = int(trainer.start_epoch) + 1 if trainer.args.resume else None
            self.writer = factory(log_dir=str(log_dir), purge_step=purge_step)
            print(f"[train] TensorBoard: {log_dir}", flush=True)
        write_json(Path(trainer.save_dir) / "skyseg_monitor.json", self.settings)
        print("[train] Validation runs every epoch; masks/overlays are saved from the validation predictions "
              f"(limit={self.settings['val_save_limit']}, 0=all).", flush=True)

    def on_train_epoch_start(self, trainer):
        self.pending_epoch = int(trainer.epoch) + 1

    def start_validation(self, trainer):
        if not self.main_process and self.world_size == 1:
            return
        if self.manifest is not None:
            self.manifest.close()
        self.epoch_root = Path(trainer.save_dir) / "validation" / f"epoch_{int(trainer.epoch) + 1:04d}"
        self.epoch_dir = self.epoch_root / f"rank_{self.rank:04d}" if self.world_size > 1 else self.epoch_root
        if self.world_size > 1 and self.settings["val_save_limit"]:
            paths = discover_images(Path(trainer.data["val"]))[:self.settings["val_save_limit"]]
            self.allowed_ids = {path.stem for path in paths}
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self.saved, self.completed = 0, False
        self.saved_ids = set()
        self.manifest = (self.epoch_dir / "predictions.jsonl.tmp").open("w", encoding="utf-8")

    def wants_image(self, path):
        limit = self.settings["val_save_limit"]
        return (self.manifest is not None and Path(path).stem not in self.saved_ids
                and (self.allowed_ids is None or Path(path).stem in self.allowed_ids)
                and (limit == 0 or self.saved < limit))

    def save_prediction(self, path, masks, classes, scores, input_shape, ratio_pad):
        if not self.wants_image(path):
            return
        path = Path(path)
        image = read_rgb(path)
        classes, scores = numpy_value(classes).reshape(-1), numpy_value(scores).reshape(-1)
        keep = (classes == 0) & (scores >= self.settings["val_mask_conf"])
        if not keep.any():
            mask = np.zeros((image.height, image.width), dtype=bool)
        else:
            # Only transfer the union to CPU, not the entire instance-mask batch.
            if hasattr(masks, "detach"):
                import torch
                selected = torch.as_tensor(keep, device=masks.device)
                union = numpy_value(masks[selected].bool().any(dim=0)).astype(bool)
            else:
                union = np.any(np.asarray(masks)[keep] > 0.5, axis=0)
            mask = restore_mask(union, image.size, input_shape, ratio_pad)
        sid = path.stem
        save_mask(self.epoch_dir / "masks" / f"{sid}.png", mask)
        overlay = self.epoch_dir / "overlays" / f"{sid}.jpg"
        save_overlay(overlay, image, mask)
        self.manifest.write(json.dumps({
            "id": sid, "source": str(path), "width": image.width, "height": image.height,
            "mask": f"masks/{sid}.png", "overlay": f"overlays/{sid}.jpg",
            "instances": int(keep.sum()), "sky_fraction": float(mask.mean()),
            "confidence": self.settings["val_mask_conf"],
        }, ensure_ascii=False) + "\n")
        if self.writer is not None and self.saved < self.settings["tensorboard_images"]:
            with Image.open(overlay) as preview:
                self.writer.add_image(f"validation/{sid}", np.asarray(preview.convert("RGB")),
                                      self.pending_epoch, dataformats="HWC")
        self.saved += 1
        self.saved_ids.add(sid)

    def finish_validation(self):
        if self.manifest is not None:
            self.manifest.close()
            self.manifest = None
            (self.epoch_dir / "predictions.jsonl.tmp").replace(self.epoch_dir / "predictions.jsonl")
            self.completed = True

    def merge_distributed_predictions(self):
        """Called by rank zero after all validation shards finish writing."""
        if not self.main_process or self.world_size == 1:
            return
        seen = set()
        temporary = self.epoch_root / "predictions.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as destination:
            for rank in range(self.world_size):
                folder = self.epoch_root / f"rank_{rank:04d}"
                with (folder / "predictions.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        if record["id"] in seen:  # DistributedSampler can pad the last batch
                            continue
                        seen.add(record["id"])
                        for key in ("mask", "overlay"):
                            record[key] = f"{folder.name}/{record[key]}"
                        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(self.epoch_root / "predictions.jsonl")
        self.saved = len(seen)

    def on_fit_epoch_end(self, trainer):
        step = int(trainer.epoch) + 1
        # Ultralytics also calls this after final best.pt evaluation. That is
        # not another training epoch, and must not overwrite curves/artifacts.
        if not self.main_process or self.pending_epoch != step:
            return
        values = {**trainer.label_loss_items(trainer.tloss, prefix="train"),
                  **(trainer.metrics or {}), **(trainer.lr or {})}
        scalars = {key: float(value) for key, value in values.items() if math.isfinite(float(value))}
        for split in ("train", "val"):
            losses = [v for k, v in scalars.items() if k.startswith(f"{split}/") and k.endswith("loss")]
            if losses:
                scalars[f"{split}/total_loss"] = sum(losses)
        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)
            self.writer.flush()
        if self.completed:
            write_json(self.epoch_root / "metrics.json", {
                "epoch": step, "saved_images": self.saved, "mask_confidence": self.settings["val_mask_conf"],
                "metrics": scalars, "prediction_source": "Ultralytics training validation (EMA when available)",
            })
            print(f"[train] Epoch {step}: saved {self.saved} validation masks/overlays to {self.epoch_root}", flush=True)
        self.pending_epoch = None

    def close(self, trainer=None):
        if self.manifest is not None:
            self.manifest.close()  # .tmp marks an interrupted validation
            self.manifest = None
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def settings_from_environment():
    return json.loads(os.environ.get(MONITOR_ENV, "{}"))
