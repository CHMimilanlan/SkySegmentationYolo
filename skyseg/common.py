"""Image, mask and manifest I/O shared by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path) -> Image.Image:
    # Materialize and remove EXIF orientation so training pixels and masks agree.
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError(f"Multi-frame images must first be exported to individual frames: {path}")
        if im.mode.startswith("I") or im.mode == "F":
            raise ValueError(f"Convert scientific/high-bit-depth data to an explicit 8-bit stretch first: {path}")
        return ImageOps.exif_transpose(im).convert("RGB")


def sample_id(relative_path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(relative_path).stem)[:50] or "image"
    return f"{stem}_{hashlib.sha256(relative_path.encode()).hexdigest()[:16]}"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image)
    if arr.ndim != 2 or arr.shape != (size[1], size[0]):
        raise ValueError(f"Mask must be a single-channel image with size {size}: {path}")
    if not set(np.unique(arr)).issubset({0, 1, 255}):
        raise ValueError(f"Mask values must be 0/1 or 0/255 (no antialiased edges): {path}")
    return arr > 0


def save_overlay(path: Path, image: Image.Image, mask: np.ndarray, max_side: int = 1280) -> None:
    arr = np.asarray(image).copy()
    arr[mask] = (arr[mask].astype(np.float32) * 0.55 + np.array([30, 200, 255]) * 0.45).astype(np.uint8)
    overlay = Image.fromarray(arr)
    overlay.thumbnail((max_side, max_side))
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path, quality=90)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(intersection / union) if union else 1.0


def require_separate_output(source: Path, output: Path) -> None:
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must not be inside the input directory (would be read as input on reruns).")
