"""YOLO inference and pixel-level sky/ground evaluation at original resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (discover_images, read_mask, read_rgb, require_separate_output,
                     sample_id, save_mask, save_overlay, write_json, write_jsonl)
from .validate import dataset_paths, validate_dataset


def load_model(path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install YOLO dependencies: pip install -e ".[yolo]"') from exc
    model = YOLO(path, task="segment")
    if model.names not in ({0: "sky"}, ["sky"]):
        raise ValueError("Expected a trained single-class sky model; COCO pretrained weights are not a sky model")
    return model


def predict_mask(model, image, args) -> tuple[np.ndarray, dict]:
    results = model.predict(source=image, imgsz=args.imgsz, conf=args.confidence, iou=args.iou,
                            device=args.device, retina_masks=True, max_det=args.max_det, verbose=False)
    result = results[0]
    union = np.zeros((image.height, image.width), dtype=bool)
    scores = []
    if result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        if masks.ndim != 3 or masks.shape[1:] != union.shape or len(masks) != len(classes):
            raise ValueError("Expected retina_masks=True to produce masks at the original image size")
        keep = classes == 0
        if keep.any():
            union = np.any(masks[keep] > 0.5, axis=0)
            scores = confidence[keep].tolist()
    return union, {"status": "ok" if union.any() else "no_sky_detection", "scores": scores,
                   "sky_fraction": float(union.mean()), "speed_ms": result.speed}


def prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inference output must be empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)


def run_predict(args) -> int:
    source, output = args.images.resolve(), args.output.resolve()
    require_separate_output(source if source.is_dir() else source.parent, output)
    images = discover_images(source)
    if not images:
        raise ValueError("No supported images found")
    prepare_output(output)
    model = load_model(args.model)
    rows = []
    for index, path in enumerate(images, 1):
        relative = path.relative_to(source if source.is_dir() else source.parent).as_posix()
        sid = sample_id(relative)
        image = read_rgb(path)
        mask, stats = predict_mask(model, image, args)
        save_mask(output / "masks" / f"{sid}.png", mask)
        save_overlay(output / "overlays" / f"{sid}.jpg", image, mask)
        if args.save_ground:
            save_mask(output / "ground_masks" / f"{sid}.png", ~mask)
        rows.append({"id": sid, "source": str(path), "mask": f"masks/{sid}.png", **stats})
        print(f"[{index}/{len(images)}] {stats['status']}: {relative}", flush=True)
    write_jsonl(output / "predictions.jsonl", rows)
    return 0


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    return {"tp": int(np.count_nonzero(prediction & target)),
            "fp": int(np.count_nonzero(prediction & ~target)),
            "fn": int(np.count_nonzero(~prediction & target)),
            "tn": int(np.count_nonzero(~prediction & ~target))}


def semantic_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    sky_iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    ground_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    return {"sky_iou": sky_iou, "ground_iou": ground_iou,
            "miou": (sky_iou + ground_iou) / 2,
            "sky_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 1.0}


def run_evaluate(args) -> int:
    validate_dataset(args.data)
    config, root = dataset_paths(args.data)
    if args.split not in config:
        raise ValueError(f"Dataset has no {args.split} split")
    require_separate_output(root, args.output)
    prepare_output(args.output)
    model = load_model(args.model)
    totals = {k: 0 for k in ("tp", "fp", "fn", "tn")}
    rows = []
    for path in discover_images(root / config[args.split]):
        image = read_rgb(path)
        target = read_mask(root / "masks" / args.split / f"{path.stem}.png", image.size)
        prediction, stats = predict_mask(model, image, args)
        counts = confusion_counts(prediction, target)
        for key in totals:
            totals[key] += counts[key]
        rows.append({"id": path.stem, **stats, **semantic_metrics(counts)})
        save_mask(args.output / "masks" / f"{path.stem}.png", prediction)
        save_overlay(args.output / "overlays" / f"{path.stem}.jpg", image, prediction)
    metrics = {"model": args.model, "data": str(args.data.resolve()), "split": args.split,
               "images": len(rows), "reference": "reviewed masks if supplied; otherwise SAM3 pseudo-labels",
               "confusion_counts": totals, **semantic_metrics(totals),
               "mean_image_sky_iou": float(np.mean([r["sky_iou"] for r in rows])),
               "confidence": args.confidence, "imgsz": args.imgsz}
    write_json(args.output / "semantic_metrics.json", metrics)
    write_jsonl(args.output / "per_image.jsonl", rows)
    print(metrics)
    return 0
