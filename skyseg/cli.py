"""Command-line entry point: python -m skyseg <stage> --help."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SAM3 sky masks → reviewed YOLOv8s-seg training dataset → model")
    commands = root.add_subparsers(dest="command", required=True)
    annotate = commands.add_parser("annotate", help="Generate SAM3 sky masks and review previews")
    annotate.add_argument("--images", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--prompt", default="sky")
    annotate.add_argument("--checkpoint", type=Path, help="Local official sam3.pt; omit for HF download")
    annotate.add_argument("--bpe-path", type=Path, help="Optional explicit tokenizer vocabulary path")
    annotate.add_argument("--device", default="cuda:0")
    annotate.add_argument("--confidence", type=float, default=0.5)
    annotate.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    annotate.add_argument("--min-sky-fraction", type=float, default=0.005)
    annotate.add_argument("--max-sky-fraction", type=float, default=0.995)
    annotate.add_argument("--limit", type=int, help="Try the first N images before a full run")
    annotate.add_argument("--resume", action="store_true")
    annotate.set_defaults(handler="annotate:run")

    build = commands.add_parser("build", help="Convert masks to YOLO polygons and split data")
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--reviews", type=Path, help="Edited review.csv; paths inside it are relative to this CSV")
    build.add_argument("--approved-only", action="store_true", help="Only explicit accept/negative reviews")
    build.add_argument("--group-by", choices=["image", "parent"], default="image")
    build.add_argument("--val-ratio", type=float, default=0.1)
    build.add_argument("--test-ratio", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--min-area", type=int, default=64, help="Minimum contour area in original-image pixels")
    build.add_argument("--epsilon", type=float, default=1.0, help="Polygon simplification tolerance in pixels")
    build.add_argument("--min-polygon-iou", type=float, default=0.98)
    build.add_argument("--max-instances", type=int, default=100)
    build.set_defaults(handler="dataset:run")

    validate = commands.add_parser("validate", help="Validate labels, masks and split isolation")
    validate.add_argument("--data", type=Path, required=True)
    validate.set_defaults(handler="validate:run")

    clean_png = commands.add_parser("clean-png", help="Inspect/remove unused PNG ICC profiles without changing pixel data")
    png_source = clean_png.add_mutually_exclusive_group(required=True)
    png_source.add_argument("--images", type=Path, help="Image file or recursively scanned directory")
    png_source.add_argument("--data", type=Path, help="Resolve actual train/val/test image paths from a built data.yaml")
    clean_png.add_argument("--in-place", action="store_true", help="Remove all iCCP chunks atomically; default only inspects")
    clean_png.set_defaults(handler="png_profiles:run")

    train = commands.add_parser("train", help="Train the requested YOLOv8s-seg model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--config", type=Path, default=Path("configs/train_yolov8s_seg.yaml"))
    train.add_argument("--model", help="Pretrained segmentation weights or --resume last.pt")
    train.add_argument("--epochs", type=int)
    train.add_argument("--imgsz", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--device")
    train.add_argument("--workers", type=int)
    train.add_argument("--project", type=Path)
    train.add_argument("--name")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-tensorboard", action="store_true", help="Disable this pipeline's TensorBoard writer")
    train.add_argument("--val-save-limit", type=int, help="Save the first N validation images each epoch; 0 saves all (default)")
    train.add_argument("--val-mask-conf", type=float, help="Confidence filter for saved masks/previews (default .25); validation metrics unchanged")
    train.add_argument("--dry-run", action="store_true", help="Validate inputs and print config without torch/GPU")
    train.set_defaults(handler="train:run")

    for command in ("predict", "evaluate"):
        sub = commands.add_parser(command, help="Predict binary sky masks" if command == "predict" else "Evaluate pixel-level sky IoU")
        sub.add_argument("--model", required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--device", default="0", help="Ultralytics device string: 0, 0,1, or cpu")
        sub.add_argument("--confidence", type=float, default=0.25)
        sub.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
        sub.add_argument("--max-det", type=int, default=100)
        if command == "predict":
            sub.add_argument("--images", type=Path, required=True)
            sub.add_argument("--save-ground", action="store_true", help="Also save binary complement; assumes sky/non-sky closed set")
        else:
            sub.add_argument("--data", type=Path, required=True)
            sub.add_argument("--split", choices=["val", "test"], default="test")
        sub.set_defaults(handler=f"inference:run_{command}")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    module, handler = args.handler.split(":")
    try:
        return getattr(importlib.import_module(f"skyseg.{module}"), handler)(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
