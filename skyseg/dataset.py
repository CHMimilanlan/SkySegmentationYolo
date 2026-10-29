"""Review selection, polygon conversion and reproducible dataset splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .common import (file_hash, pixel_hash, read_jsonl, read_mask, read_rgb,
                     require_separate_output, save_mask, save_overlay, write_json, write_jsonl)
from .polygons import labels_text, mask_to_polygons, polygons_to_mask


def load_reviews(path: Path | None, valid_ids: set[str]) -> dict:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "decision"}.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV requires id,decision columns; optional mask_path,comment")
        for row in reader:
            sid = row["id"].strip()
            decision = row["decision"].strip().lower()
            if sid not in valid_ids or sid in result:
                raise ValueError(f"Unknown or duplicate review id: {sid}")
            if decision not in {"", "accept", "reject", "negative"}:
                raise ValueError(f"Invalid decision {decision!r} for {sid}")
            mask_path = row.get("mask_path", "").strip()
            if mask_path and decision != "accept":
                raise ValueError(f"A replacement mask requires decision=accept: {sid}")
            result[sid] = {"decision": decision, "comment": row.get("comment", ""),
                           "mask_path": (path.parent / mask_path).resolve() if mask_path else None}
    return result


def split_groups(items: list[dict], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Assign whole groups to one split; ratios apply to group counts, not images."""
    if not 0 < val_ratio < 1 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Require 0 < val_ratio, 0 <= test_ratio and val_ratio + test_ratio < 1")
    groups = sorted({item["split_group"] for item in items})
    required = 3 if test_ratio > 0 else 2
    if len(groups) < required:
        raise ValueError(f"Need at least {required} independent groups/images; found {len(groups)}. "
                         "Use --test-ratio 0 for two groups, or supply more independent scenes.")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_ratio))
    n_test = max(1, round(len(groups) * test_ratio)) if test_ratio else 0
    # Reserve at least one training group even for very small datasets.
    while n_val + n_test >= len(groups):
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Unable to allocate nonempty training split")
    assignment = {g: "train" for g in groups}
    assignment.update({g: "val" for g in groups[:n_val]})
    assignment.update({g: "test" for g in groups[n_val:n_val + n_test]})
    return assignment


def run(args) -> int:
    source, output = args.annotations.resolve(), args.output.resolve()
    require_separate_output(source, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset output must be empty. Use a new directory to avoid stale images/labels.")
    if not 0 <= args.min_polygon_iou <= 1 or args.min_area < 0 or args.epsilon < 0 or args.max_instances < 1:
        raise ValueError("Invalid polygon conversion thresholds")
    records = read_jsonl(source / "manifest.jsonl")
    valid_ids = {record["id"] for record in records}
    if len(valid_ids) != len(records):
        raise ValueError("Duplicate ids in annotation manifest")
    reviews = load_reviews(args.reviews.resolve() if args.reviews else None, valid_ids)
    candidates, reports, seen = [], [], {}
    for record in records:
        sid = record["id"]
        review = reviews.get(sid, {})
        decision = review.get("decision", "")
        report = {"id": sid, "source": record["source"], "decision": decision, "included": False}
        reports.append(report)
        if decision == "reject":
            report["reason"] = "human_rejected"
            continue
        if record["status"] == "error":
            report["reason"] = "annotation_error_rerun_or_repair_first"
            continue
        if args.approved_only and decision not in {"accept", "negative"}:
            report["reason"] = "not_reviewed"
            continue
        if not decision and (record["status"] != "ok" or record.get("warnings")):
            report["reason"] = "needs_review"
            continue
        image_path = source / record["image"]
        image = read_rgb(image_path)
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"Image size changed after annotation: {sid}")
        digest = pixel_hash(image)
        if digest != record["pixel_sha256"]:
            raise ValueError(f"Image changed after annotation; regenerate annotations: {sid}")
        if decision == "negative":
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask_path = None
        else:
            mask_path = review.get("mask_path") or source / record["mask"]
            mask = read_mask(mask_path, image.size)
            if not mask.any():
                report["reason"] = "empty_mask_requires_explicit_negative_decision"
                continue
        polygons, stats = mask_to_polygons(mask, args.min_area, args.epsilon)
        report.update(stats)
        if mask.any() and not polygons:
            report["reason"] = "all_components_filtered"
            continue
        if stats["polygon_iou"] < args.min_polygon_iou:
            report["reason"] = "polygon_roundtrip_loss"
            continue
        if len(polygons) > args.max_instances:
            report["reason"] = "too_many_components"
            continue
        mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
        if digest in seen:
            previous_id, previous_mask = seen[digest]
            if previous_mask != mask_digest:
                raise ValueError(f"Duplicate image has conflicting masks: {previous_id}, {sid}; fix review decisions")
            report.update(reason="duplicate_pixels", duplicate_of=previous_id)
            continue
        seen[digest] = (sid, mask_digest)
        group = record["group"] if args.group_by == "parent" else sid
        candidates.append({"record": record, "image_path": image_path, "mask_path": mask_path,
                           "polygons": polygons, "split_group": group, "report": report,
                           "mask_sha256": mask_digest})
    # Write diagnostics even when filtering leaves too little data to split.
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "conversion_report.jsonl", reports)
    assignment = split_groups(candidates, args.val_ratio, args.test_ratio, args.seed)
    for split in {"train", "val"}:
        if not any(item["polygons"] and assignment[item["split_group"]] == split for item in candidates):
            raise ValueError(f"{split} has no positive sky labels; adjust independent groups/seed/reviews")
    manifest = []
    for split in sorted(set(assignment.values())):
        for folder in ("images", "labels", "masks", "label_overlays"):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
    for item in candidates:
        record = item["record"]
        sid, split = record["id"], assignment[item["split_group"]]
        image = read_rgb(item["image_path"])
        mask = (read_mask(item["mask_path"], image.size) if item["mask_path"] is not None
                else np.zeros((image.height, image.width), dtype=bool))
        image_rel, label_rel, mask_rel = f"images/{split}/{sid}.png", f"labels/{split}/{sid}.txt", f"masks/{split}/{sid}.png"
        shutil.copy2(item["image_path"], output / image_rel)
        (output / label_rel).write_text(labels_text(item["polygons"]), encoding="utf-8")
        save_mask(output / mask_rel, mask)
        save_overlay(output / "label_overlays" / split / f"{sid}.jpg", image,
                     polygons_to_mask(item["polygons"], mask.shape))
        item["report"].update(included=True, split=split, split_group=item["split_group"])
        manifest.append({
            "id": sid, "source": record["source"], "split": split, "split_group": item["split_group"],
            "image": image_rel, "label": label_rel, "mask": mask_rel,
            "pixel_sha256": record["pixel_sha256"], "mask_sha256": item["mask_sha256"],
            "decision": item["report"]["decision"], "instances": len(item["polygons"]),
            "polygon_iou": item["report"]["polygon_iou"],
        })
    config = {"path": str(output), "train": "images/train", "val": "images/val", "names": {0: "sky"}, "nc": 1}
    if args.test_ratio:
        config["test"] = "images/test"
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_jsonl(output / "dataset_manifest.jsonl", manifest)
    write_jsonl(output / "conversion_report.jsonl", reports)
    write_json(output / "build_config.json", {
        "annotations": str(source), "annotation_run": json.loads((source / "run.json").read_text(encoding="utf-8")),
        "reviews": str(args.reviews.resolve()) if args.reviews else None,
        "reviews_sha256": file_hash(args.reviews) if args.reviews else None,
        "approved_only": args.approved_only, "group_by": args.group_by,
        "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "seed": args.seed,
        "min_area": args.min_area, "epsilon": args.epsilon,
        "min_polygon_iou": args.min_polygon_iou, "max_instances": args.max_instances,
    })
    counts = dict(Counter(item["split"] for item in manifest))
    summary = {"included": len(manifest), "excluded": len(records) - len(manifest), "splits": counts,
               "exclusion_reasons": dict(Counter(r["reason"] for r in reports if not r["included"]))}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.group_by == "image":
        print("Image-level split: adjacent frames/same-scene images can leak across splits. Use --group-by parent for sequences.")
    from .validate import validate_dataset
    validate_dataset(output / "data.yaml")
    return 0
