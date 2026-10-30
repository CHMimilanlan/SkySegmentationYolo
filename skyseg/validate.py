"""Fail early on missing, corrupt or leaked training data without loading torch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

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
    print(f"[validate] Loading dataset configuration: {data.resolve()}", flush=True)
    config, root = dataset_paths(data)
    manifest_file = root / "dataset_manifest.jsonl"
    print(f"[validate] Reading manifest: {manifest_file}", flush=True)
    if not manifest_file.is_file():
        raise ValueError("Missing dataset_manifest.jsonl")
    manifest = read_jsonl(manifest_file)
    split_images, actual_ids = {}, set()
    for split in ("train", "val", "test"):
        if split not in config:
            if split != "test":
                raise ValueError(f"Missing {split} split in YAML")
            continue
        expected = f"images/{split}"
        if config[split] != expected:
            raise ValueError(f"Expected {split}: {expected}; use a dataset built by skyseg build")
        print(f"[validate] Scanning {split}: {root / expected}", flush=True)
        images = discover_images(root / expected)
        if not images:
            raise ValueError(f"Empty {split} split")
        split_images[split] = images
        for path in images:
            key = (split, path.stem)
            if key in actual_ids:
                raise ValueError(f"Duplicate sample id in {split}: {path.stem}")
            actual_ids.add(key)
        print(f"[validate] Found {len(images)} images in {split}.", flush=True)
    manifest_ids = [(record["split"], record["id"]) for record in manifest]
    if len(manifest_ids) != len(actual_ids) or set(manifest_ids) != actual_ids:
        raise ValueError("Dataset manifest does not match actual images")
    groups = {}
    for record in manifest:
        group = record["split_group"]
        if group in groups and groups[group] != record["split"]:
            raise ValueError(f"Scene group leaks between splits: {group}")
        groups[group] = record["split"]
    records_by_id = {(record["split"], record["id"]): record for record in manifest}
    hashes, result = {}, {}
    print(f"[validate] Checking {len(manifest)} image/mask/label pairs; each image is decoded once.", flush=True)
    for split, images in split_images.items():
        positive = 0
        with tqdm(images, desc=f"Validate {split}", unit="image", dynamic_ncols=True) as progress:
            for path in progress:
                record = records_by_id[(split, path.stem)]
                relative = path.relative_to(root / "images" / split)
                label = root / "labels" / split / relative.with_suffix(".txt")
                mask_path = root / "masks" / split / relative.with_suffix(".png")
                # Refresh before decoding, so a slow image is visible even if its
                # decoder or the underlying storage takes a long time to return.
                progress.set_postfix(file=path.name, stage="image", refresh=True)
                current_path = path
                try:
                    for field, expected_path in (("image", path), ("label", label), ("mask", mask_path)):
                        if Path(record[field]) != expected_path.relative_to(root):
                            raise ValueError(f"Manifest {field} path does not match sample: {record['id']}")
                    image = read_rgb(path)
                    digest = pixel_hash(image)
                    if digest in hashes:
                        raise ValueError(f"Duplicate pixels in dataset: {path} and {hashes[digest]}")
                    hashes[digest] = str(path)
                    if digest != record["pixel_sha256"]:
                        raise ValueError(f"Image pixels differ from manifest: {record['id']}")
                    current_path = label
                    progress.set_postfix(file=path.name, stage="labels", refresh=False)
                    polygons = read_labels(label)
                    positive += bool(polygons)
                    current_path = mask_path
                    progress.set_postfix(file=path.name, stage="mask", refresh=False)
                    if not mask_path.is_file():
                        raise ValueError(f"Missing semantic reference mask: {mask_path}")
                    mask = read_mask(mask_path, image.size)
                    if bool(mask.any()) != bool(polygons):
                        raise ValueError(f"Mask/label foreground mismatch: {path}")
                    if hashlib.sha256(mask.tobytes()).hexdigest() != record["mask_sha256"]:
                        raise ValueError(f"Reference mask changed; rebuild the dataset: {record['id']}")
                    current_path = label
                    progress.set_postfix(file=path.name, stage="polygon IoU", refresh=False)
                    iou = mask_iou(mask, polygons_to_mask(polygons, mask.shape))
                    if abs(iou - record["polygon_iou"]) > 1e-6 or len(polygons) != record["instances"]:
                        raise ValueError(f"Labels differ from conversion manifest; rebuild: {record['id']}")
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Dataset validation failed while checking {current_path}: {exc}") from exc
        if split in {"train", "val"} and not positive:
            raise ValueError(f"No positive labels in {split}")
        result[split] = {"images": len(images), "positive": positive, "negative": len(images) - positive}
        print(f"[validate] {split} passed: {result[split]}", flush=True)
    print(f"Dataset validation passed: {result}", flush=True)
    return result


def run(args) -> int:
    validate_dataset(args.data)
    return 0
