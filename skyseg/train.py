"""Thin, inspectable wrapper around the official Ultralytics training API."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import yaml

from .common import write_json
from .png_profiles import prepare_training_pngs
from .monitoring import MONITOR_ENV, monitoring_settings
from .validate import validate_dataset


def training_settings(args) -> tuple[str, dict]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    config.pop("skyseg", None)
    model_path = args.model or config.pop("model", "yolov8s-seg.pt")
    config.pop("model", None)
    if not model_path.lower().endswith(("-seg.pt", "-seg.yaml")) and args.model is None:
        raise ValueError("Use a segmentation model ending in -seg.pt or specify --model for custom weights")
    for key in ("epochs", "imgsz", "batch", "device", "workers", "project", "name"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    config.update(data=str(args.data.resolve()), task="segment", val=True, plots=True)
    return model_path, config


def run(args) -> int:
    print("[train] Validating the dataset before loading YOLO. Large images or slow storage can take time.", flush=True)
    validate_dataset(args.data)
    model_path, config = training_settings(args)
    monitoring = monitoring_settings(args, yaml.safe_load(args.config.read_text(encoding="utf-8")) or {})
    print(yaml.safe_dump({"model": model_path, **config, "skyseg": monitoring}, sort_keys=False), flush=True)
    print("[train] Preparing PNG ICC metadata in the actual dataset splits before YOLO loads images.", flush=True)
    prepare_training_pngs(args.data, in_place=not args.dry_run)
    if args.dry_run:
        print("Dry run finished; PNG metadata was inspected only. No model loaded or training started.", flush=True)
        return 0
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install YOLO dependencies: pip install -e ".[yolo]"') from exc
    if args.resume:
        if args.model is None or not Path(args.model).is_file():
            raise ValueError("--resume requires --model /path/to/weights/last.pt")
        # Ultralytics restores training arguments from the checkpoint itself.
        overrides = {key: config[key] for key in ("device", "workers") if config.get(key) is not None}
        print(f"[train] Loading resume checkpoint: {args.model}", flush=True)
        model = YOLO(args.model, task="segment")
        checkpoint_data = model.ckpt.get("train_args", {}).get("data")
        if checkpoint_data and Path(str(checkpoint_data)).resolve() != args.data.resolve():
            raise ValueError("Resume checkpoint uses a different dataset path; use the original --data, "
                             "or start a new fine-tuning run without --resume")
        print("[train] Resuming Ultralytics training.", flush=True)
    else:
        print(f"[train] Loading segmentation model: {model_path}", flush=True)
        model = YOLO(model_path, task="segment")
        print("[train] Starting Ultralytics training.", flush=True)
    run_training(model, monitoring, resume=args.resume, config=overrides if args.resume else config)
    save_dir = Path(model.trainer.save_dir)
    write_json(save_dir / "pipeline_environment.json", {
        name: importlib.metadata.version(name)
        for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "opencv-python")
    })
    print(f"Training completed. Best weights: {save_dir / 'weights' / 'best.pt'}", flush=True)
    return 0


def run_training(model, monitoring, *, resume, config):
    from .yolo_training import SkySegmentationTrainer

    previous = os.environ.get(MONITOR_ENV)
    os.environ[MONITOR_ENV] = json.dumps(monitoring)
    try:
        model.train(trainer=SkySegmentationTrainer, **({"resume": True} if resume else {}), **config)
    finally:
        monitor = getattr(getattr(model, "trainer", None), "skyseg_monitor", None)
        if monitor is not None:
            monitor.close()
        if previous is None:
            os.environ.pop(MONITOR_ENV, None)
        else:
            os.environ[MONITOR_ENV] = previous
