"""SAM3 text-prompt annotation with resumable records and review artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .common import (
    discover_images, file_hash, pixel_hash, read_rgb, require_separate_output,
    sample_id, save_mask, save_overlay, write_json, write_jsonl,
)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no native torch.bfloat16 dtype (scores may use BF16 under
        # autocast); keep boolean masks boolean instead of copying them to FP32.
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def merge_predictions(masks, scores, shape: tuple[int, int], confidence: float):
    """Accept official SAM3 [N,1,H,W] or [N,H,W] outputs, including N=0."""
    scores = _numpy(scores).reshape(-1)
    masks = _numpy(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] != len(scores) or masks.shape[1:] != shape:
        raise ValueError(f"Unexpected SAM3 mask/score shapes: {masks.shape}, {scores.shape}; expected [N,1,{shape}]")
    if not np.isfinite(scores).all() or not np.isfinite(masks).all():
        raise ValueError("SAM3 produced non-finite masks/scores")
    kept = scores >= confidence
    union = np.any(masks[kept] > 0.5, axis=0)
    return union, scores[kept].tolist()


class Sam3Backend:
    def __init__(self, args):
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError("Install official facebookresearch/sam3 and CUDA PyTorch; see README section 1.") from exc
        self.torch = torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the PyTorch build and GPU driver.")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("SAM3 supports --device cpu or cuda[:index] in this pipeline")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        # Upstream _setup_device_and_mode only handles literal 'cuda'. Build on
        # CPU and move explicitly so cuda:1, etc. actually select the right GPU.
        kwargs = {"device": "cpu", "eval_mode": True}
        if args.checkpoint:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            kwargs.update(checkpoint_path=str(args.checkpoint.resolve()), load_from_HF=False)
        if args.bpe_path:
            kwargs["bpe_path"] = str(args.bpe_path.resolve())
        self.model = build_sam3_image_model(**kwargs).to(self.device).eval()
        self.processor = Sam3Processor(self.model, device=str(self.device), confidence_threshold=args.confidence)
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.bf16 = use_bf16 and args.precision == "bf16"
        if args.precision == "bf16" and not use_bf16:
            print("BF16 unavailable on this device; using FP32.", flush=True)

    def predict(self, image, prompt):
        torch = self.torch
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else nullcontext()
        with torch.inference_mode(), context:
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            # No GPU tensors/states escape this call or accumulate across images.
            return _numpy(output["masks"]), _numpy(output["scores"])


def run(args, backend=None) -> int:
    root, output = args.images.resolve(), args.output.resolve()
    if not root.is_dir():
        raise ValueError("--images must be a directory")
    require_separate_output(root, output)
    if not 0 <= args.confidence <= 1 or not args.prompt.strip():
        raise ValueError("Use a non-empty prompt and confidence in [0,1]")
    if not 0 <= args.min_sky_fraction <= args.max_sky_fraction <= 1:
        raise ValueError("Invalid sky fraction thresholds")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    images = discover_images(root)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError(f"No supported images found in {root}")
    config = {
        "schema_version": 1, "images": str(root), "prompt": args.prompt,
        "confidence": args.confidence, "precision": args.precision, "device": args.device,
        "min_sky_fraction": args.min_sky_fraction, "max_sky_fraction": args.max_sky_fraction,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "facebook/sam3",
        "checkpoint_stat": [args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns] if args.checkpoint else None,
        "bpe_path": str(args.bpe_path.resolve()) if args.bpe_path else None,
    }
    if output.exists() and any(output.iterdir()):
        config_file = output / "run.json"
        if not args.resume or not config_file.exists():
            raise ValueError("Output is non-empty; use a new directory, or --resume with the same annotation settings")
        if json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Annotation settings changed. Use a new output directory to preserve provenance.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", config)
    records = []
    invalidated_reviews = set()
    for index, path in enumerate(images, 1):
        relative = path.relative_to(root).as_posix()
        sid = sample_id(relative)
        record_path = output / "records" / f"{sid}.json"
        source_sha = file_hash(path)
        if args.resume and record_path.exists():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            complete = all((output / previous[key]).is_file() for key in ("image", "mask", "overlay") if key in previous)
            if previous.get("source_sha256") == source_sha and previous["status"] != "error" and complete:
                records.append(previous)
                print(f"[{index}/{len(images)}] resume {relative}", flush=True)
                continue
            if previous.get("source_sha256") != source_sha:
                invalidated_reviews.add(sid)
        record = {"id": sid, "source": relative, "source_sha256": source_sha,
                  "group": path.relative_to(root).parent.as_posix(), "status": "error"}
        try:
            image = read_rgb(path)
            # Initialize lazily: a fully cached resume needs no GPU/model download.
            if backend is None:
                backend = Sam3Backend(args)
            masks, scores = backend.predict(image, args.prompt)
            mask, kept_scores = merge_predictions(masks, scores, (image.height, image.width), args.confidence)
            fraction = float(mask.mean())
            warnings = []
            if not mask.any():
                warnings.append("empty_prediction_not_confirmed_negative")
            elif not args.min_sky_fraction <= fraction <= args.max_sky_fraction:
                warnings.append("sky_fraction_outlier")
            record.update(
                status="ok" if mask.any() else "empty", width=image.width, height=image.height,
                image=f"images/{sid}.png", mask=f"masks/{sid}.png", overlay=f"overlays/{sid}.jpg",
                pixel_sha256=pixel_hash(image), scores=kept_scores, sky_fraction=fraction,
                warnings=warnings,
            )
            (output / "images").mkdir(exist_ok=True)
            image.save(output / record["image"])
            save_mask(output / record["mask"], mask)
            save_overlay(output / record["overlay"], image, mask)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            # A failed dependency/checkpoint initialization affects every image.
            if backend is None:
                write_json(record_path, record)
                raise
        write_json(record_path, record)
        records.append(record)
        print(f"[{index}/{len(images)}] {record['status']}: {relative}", flush=True)
    write_jsonl(output / "manifest.jsonl", records)
    # Never overwrite human decisions on resume. This file is an editable template.
    review = output / "review.csv"
    existing = {}
    if review.exists():
        with review.open(encoding="utf-8-sig", newline="") as f:
            existing = {row["id"]: row for row in csv.DictReader(f)}
    with review.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "decision", "mask_path", "comment"])
        writer.writeheader()
        for record in records:
            row = existing.get(record["id"], {"id": record["id"], "decision": "", "mask_path": "", "comment": ""})
            if record["id"] in invalidated_reviews:
                row = {"id": record["id"], "decision": "", "mask_path": "", "comment": "Source changed; review again"}
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    counts = dict(Counter(record["status"] for record in records))
    write_json(output / "summary.json", counts)
    print(f"Annotation summary: {counts}; inspect overlays/ and review.csv", flush=True)
    return 1 if counts.get("error", 0) else 0
