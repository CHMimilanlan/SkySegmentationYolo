"""Fail early on missing, corrupt or leaked training data without loading torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .common import discover_images, mask_iou, pixel_hash, read_jsonl, read_mask, read_rgb
from .polygons import polygons_to_mask


def read_labels(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Missing label file (negative samples still need empty .txt): {path}")
    polygons = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = np.array([float(x) for x in line.split()], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Non-numeric label: {path}:{line_no}") from exc
        if len(fields) < 7 or len(fields) % 2 != 1 or fields[0] != 0:
            raise ValueError(f"Expected class 0 and >=3 coordinate pairs: {path}:{line_no}")
        xy = fields[1:].reshape(-1, 2)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
            raise ValueError(f"Coordinates outside [0,1]: {path}:{line_no}")
        area = abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2
        if area <= 0 or len(np.unique(xy, axis=0)) < 3:
            raise ValueError(f"Degenerate polygon: {path}:{line_no}")
        polygons.append(xy)
    return polygons


def dataset_paths(data: Path):
    data = data.resolve()
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = config.get("names")
    if names not in ({0: "sky"}, ["sky"]):
        raise ValueError("This pipeline expects one class: names: {0: sky}")
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    return config, root


def validate_dataset(data: Path) -> dict:
    config, root = dataset_paths(data)
    hashes, result, actual_ids = {}, {}, set()
    for split in ("train", "val", "test"):
        if split not in config:
            if split != "test":
                raise ValueError(f"Missing {split} split in YAML")
            continue
        expected = f"images/{split}"
        if config[split] != expected:
            raise ValueError(f"Expected {split}: {expected}; use a dataset built by skyseg build")
        images = discover_images(root / expected)
        if not images:
            raise ValueError(f"Empty {split} split")
        positive = 0
        for path in images:
            image = read_rgb(path)
            digest = pixel_hash(image)
            if digest in hashes:
                raise ValueError(f"Duplicate pixels in dataset: {path} and {hashes[digest]}")
            hashes[digest] = str(path)
            relative = path.relative_to(root / "images" / split)
            label = root / "labels" / split / relative.with_suffix(".txt")
            polygons = read_labels(label)
            positive += bool(polygons)
            actual_ids.add((split, path.stem))
            mask_path = root / "masks" / split / relative.with_suffix(".png")
            if not mask_path.is_file():
                raise ValueError(f"Missing semantic reference mask: {mask_path}")
            mask = read_mask(mask_path, image.size)
            if bool(mask.any()) != bool(polygons):
                raise ValueError(f"Mask/label foreground mismatch: {path}")
        if split in {"train", "val"} and not positive:
            raise ValueError(f"No positive labels in {split}")
        result[split] = {"images": len(images), "positive": positive, "negative": len(images) - positive}
    manifest_file = root / "dataset_manifest.jsonl"
    if not manifest_file.is_file():
        raise ValueError("Missing dataset_manifest.jsonl")
    manifest = read_jsonl(manifest_file)
    if len(manifest) != len(actual_ids) or {(r["split"], r["id"]) for r in manifest} != actual_ids:
        raise ValueError("Dataset manifest does not match actual images")
    groups = {}
    for record in manifest:
        group = record["split_group"]
        if group in groups and groups[group] != record["split"]:
            raise ValueError(f"Scene group leaks between splits: {group}")
        groups[group] = record["split"]
        image = read_rgb(root / record["image"])
        if pixel_hash(image) != record["pixel_sha256"]:
            raise ValueError(f"Image pixels differ from manifest: {record['id']}")
        mask = read_mask(root / record["mask"], image.size)
        import hashlib
        if hashlib.sha256(mask.tobytes()).hexdigest() != record["mask_sha256"]:
            raise ValueError(f"Reference mask changed; rebuild the dataset: {record['id']}")
        polygons = read_labels(root / record["label"])
        iou = mask_iou(mask, polygons_to_mask(polygons, mask.shape))
        if abs(iou - record["polygon_iou"]) > 1e-6 or len(polygons) != record["instances"]:
            raise ValueError(f"Labels differ from conversion manifest; rebuild: {record['id']}")
    print(f"Dataset validation passed: {result}")
    return result


def run(args) -> int:
    validate_dataset(args.data)
    return 0
