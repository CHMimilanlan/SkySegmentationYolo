"""CPU regression tests. Model APIs are faked; no checkpoints/downloads are required."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from skyseg import annotate, dataset, inference, train
from skyseg.cli import main, parser
from skyseg.common import read_jsonl, read_rgb, sample_id, save_mask
from skyseg.polygons import labels_text, mask_to_polygons, polygons_to_mask
from skyseg.validate import read_labels, validate_dataset


class FakeSam3:
    def __init__(self):
        self.calls = 0

    def predict(self, image, prompt):
        assert prompt == "sky"
        self.calls += 1
        mask = np.zeros((1, 1, image.height, image.width), dtype=bool)
        if np.asarray(image)[0, 0, 0] != 250:
            mask[:, :, :image.height // 2] = True
        return mask, np.array([0.95], dtype=np.float32)


def args(command, *values):
    return parser().parse_args([command, *map(str, values)])


def create_annotations(tmp_path, extra_empty=False, duplicate=False):
    raw, output = tmp_path / "raw", tmp_path / "annotations"
    for i in range(8):
        folder = raw / f"scene_{i // 2}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (96, 64), (i * 20 + 1, 20, 30)).save(folder / f"星空_{i}.png")
    if extra_empty:
        Image.new("RGB", (96, 64), (250, 20, 30)).save(raw / "empty.png")
    if duplicate:
        import shutil
        shutil.copy2(raw / "scene_0/星空_0.png", raw / "duplicate.png")
    backend = FakeSam3()
    arguments = args("annotate", "--images", raw, "--output", output)
    assert annotate.run(arguments, backend=backend) == 0
    return raw, output, backend


def write_reviews(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "decision", "mask_path", "comment"])
        writer.writeheader()
        writer.writerows(rows)


def test_sam3_masks_union_shapes_empty_and_scores():
    masks = np.zeros((2, 1, 30, 40), dtype=bool)
    masks[0, 0, :10] = True
    masks[1, 0, 20:] = True
    merged, scores = annotate.merge_predictions(masks, [0.9, 0.8], (30, 40), 0.5)
    assert merged.sum() == 800 and len(scores) == 2
    merged, _ = annotate.merge_predictions(masks, [0.9, 0.2], (30, 40), 0.5)
    assert merged.sum() == 400
    empty, _ = annotate.merge_predictions(np.empty((0, 1, 30, 40)), [], (30, 40), 0.5)
    assert empty.shape == (30, 40) and not empty.any()
    with pytest.raises(ValueError):
        annotate.merge_predictions(np.zeros((1, 20, 20)), [0.8], (30, 40), 0.5)
    with pytest.raises(ValueError):
        annotate.merge_predictions(masks, [float("nan"), 0.8], (30, 40), 0.5)
    # Torch BF16 cannot call .numpy() directly; emulate its required conversion.
    class BfloatTensor:
        dtype = "torch.bfloat16"

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return ArrayTensor([0.75])

        def numpy(self):
            raise AssertionError("BF16 must be converted before NumPy")
    assert annotate._numpy(BfloatTensor()).tolist() == [0.75]


def test_polygon_disconnected_regions_holes_and_degenerate(tmp_path):
    mask = np.zeros((100, 160), dtype=bool)
    mask[:35, :] = True  # touches three image edges
    mask[70:90, 10:35] = True
    polygons, stats = mask_to_polygons(mask, min_area=0, epsilon=0)
    assert len(polygons) == 2 and stats["polygon_iou"] == 1.0
    assert np.array_equal(mask, polygons_to_mask(polygons, mask.shape))
    label = tmp_path / "label.txt"
    label.write_text(labels_text(polygons), encoding="utf-8")
    assert len(read_labels(label)) == 2
    mask[5:30, 30:80] = False
    _, stats = mask_to_polygons(mask, min_area=0, epsilon=0)
    assert stats["holes"] == 1 and stats["polygon_iou"] < 0.9
    tiny = np.zeros((20, 20), dtype=bool)
    tiny[3, 3] = True
    polygons, stats = mask_to_polygons(tiny, min_area=0)
    assert not polygons and stats["polygon_iou"] == 0


def test_exif_orientation_and_ids(tmp_path):
    image = Image.new("RGB", (80, 40), "navy")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "旋转.jpg"
    image.save(path, exif=exif)
    normalized = read_rgb(path)
    assert normalized.size == (40, 80)
    assert normalized.getexif().get(274) is None
    assert sample_id("a/image.jpg") != sample_id("b/image.jpg")
    assert sample_id("a/image.jpg") != sample_id("a/image.png")


def test_end_to_end_build_review_split_validate_dry_run(tmp_path):
    _, annotations, backend = create_annotations(tmp_path, extra_empty=True, duplicate=True)
    rows = read_jsonl(annotations / "manifest.jsonl")
    empty = next(row for row in rows if row["status"] == "empty")
    output = tmp_path / "dataset"
    assert dataset.run(args("build", "--annotations", annotations, "--output", output)) == 0
    included = read_jsonl(output / "dataset_manifest.jsonl")
    assert len(included) == 8
    assert empty["id"] not in {row["id"] for row in included}
    reports = read_jsonl(output / "conversion_report.jsonl")
    assert sum(row.get("reason") == "duplicate_pixels" for row in reports) == 1
    result = validate_dataset(output / "data.yaml")
    assert sum(split["images"] for split in result.values()) == 8
    assert train.run(args("train", "--data", output / "data.yaml", "--dry-run")) == 0
    # No implicit overwrite; no stale files survive a changed split.
    with pytest.raises(ValueError, match="must be empty"):
        dataset.run(args("build", "--annotations", annotations, "--output", output))
    # Resume doesn't call the backend again or reset human decisions.
    review = annotations / "review.csv"
    write_reviews(review, [{"id": empty["id"], "decision": "negative"}])
    calls = backend.calls
    raw = tmp_path / "raw"
    assert annotate.run(args("annotate", "--images", raw, "--output", annotations, "--resume"), backend=backend) == 0
    assert backend.calls == calls
    with review.open(newline="") as f:
        assert next(r for r in csv.DictReader(f) if r["id"] == empty["id"])["decision"] == "negative"


def test_parent_groups_deterministic_and_no_leakage(tmp_path):
    _, annotations, _ = create_annotations(tmp_path)
    manifests = []
    for name in ("first", "second"):
        output = tmp_path / name
        dataset.run(args("build", "--annotations", annotations, "--output", output, "--group-by", "parent"))
        manifests.append(read_jsonl(output / "dataset_manifest.jsonl"))
    assert manifests[0] == manifests[1]
    groups = {}
    for row in manifests[0]:
        groups.setdefault(row["split_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    with pytest.raises(ValueError, match="independent"):
        dataset.split_groups([{"split_group": "same_scene"}] * 3, 0.1, 0.1, 42)


def test_review_replacement_negative_and_approved_only(tmp_path):
    _, annotations, _ = create_annotations(tmp_path, extra_empty=True)
    rows = read_jsonl(annotations / "manifest.jsonl")
    empty = next(row for row in rows if row["status"] == "empty")
    positives = [row for row in rows if row["status"] == "ok"]
    corrected = np.zeros((64, 96), dtype=bool)
    corrected[:40] = True
    save_mask(annotations / "corrected.png", corrected)
    review = annotations / "review.csv"
    write_reviews(review, [
        {"id": empty["id"], "decision": "negative"},
        {"id": positives[0]["id"], "decision": "accept", "mask_path": "corrected.png"},
        *[{"id": row["id"], "decision": "accept"} for row in positives[1:]],
    ])
    output = tmp_path / "reviewed"
    dataset.run(args("build", "--annotations", annotations, "--output", output,
                     "--reviews", review, "--approved-only"))
    manifest = read_jsonl(output / "dataset_manifest.jsonl")
    negative = next(row for row in manifest if row["id"] == empty["id"])
    assert (output / negative["label"]).read_text() == ""
    assert np.asarray(Image.open(output / next(r["mask"] for r in manifest if r["id"] == positives[0]["id"]))).sum() == corrected.sum() * 255
    validate_dataset(output / "data.yaml")


def test_lossy_holes_excluded_and_corrupt_label_rejected(tmp_path):
    _, annotations, _ = create_annotations(tmp_path)
    rows = read_jsonl(annotations / "manifest.jsonl")
    first = rows[0]
    mask = np.zeros((64, 96), dtype=bool)
    mask[:32] = True
    mask[5:25, 10:80] = False
    save_mask(annotations / first["mask"], mask)
    output = tmp_path / "dataset"
    dataset.run(args("build", "--annotations", annotations, "--output", output))
    report = next(r for r in read_jsonl(output / "conversion_report.jsonl") if r["id"] == first["id"])
    assert report["reason"] == "polygon_roundtrip_loss"
    manifest = read_jsonl(output / "dataset_manifest.jsonl")
    (output / manifest[0]["label"]).write_text("0 1.5 0 0 1 0 0\n")
    with pytest.raises(ValueError, match="outside"):
        validate_dataset(output / "data.yaml")


def test_duplicate_pixels_conflicting_labels_fail(tmp_path):
    _, annotations, _ = create_annotations(tmp_path, duplicate=True)
    rows = read_jsonl(annotations / "manifest.jsonl")
    duplicate = next(row for row in rows if row["source"] == "duplicate.png")
    mask = np.zeros((64, 96), dtype=bool)
    mask[:40] = True
    save_mask(annotations / duplicate["mask"], mask)
    with pytest.raises(ValueError, match="conflicting"):
        dataset.run(args("build", "--annotations", annotations, "--output", tmp_path / "dataset"))


def test_resume_source_change_invalidates_review(tmp_path):
    raw, annotations, backend = create_annotations(tmp_path)
    first = read_jsonl(annotations / "manifest.jsonl")[0]
    write_reviews(annotations / "review.csv", [{"id": first["id"], "decision": "accept"}])
    Image.new("RGB", (96, 64), (199, 33, 44)).save(raw / first["source"])
    annotate.run(args("annotate", "--images", raw, "--output", annotations, "--resume"), backend=backend)
    with (annotations / "review.csv").open(newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["id"] == first["id"])
    assert row["decision"] == "" and "changed" in row["comment"]


def test_annotation_errors_are_not_negatives(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (64, 64)).save(raw / "valid.png")
    (raw / "broken.jpg").write_bytes(b"not an image")
    output = tmp_path / "annotations"
    assert annotate.run(args("annotate", "--images", raw, "--output", output), backend=FakeSam3()) == 1
    rows = read_jsonl(output / "manifest.jsonl")
    assert {row["status"] for row in rows} == {"error", "ok"}
    assert not any("mask" in row for row in rows if row["status"] == "error")


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeYolo:
    def predict(self, **kwargs):
        assert kwargs["retina_masks"] is True and isinstance(kwargs["source"], Image.Image)
        image = kwargs["source"]
        masks = np.zeros((2, image.height, image.width))
        masks[0, :10] = 1
        masks[1, 20:30] = 1
        return [SimpleNamespace(masks=SimpleNamespace(data=ArrayTensor(masks)),
                                boxes=SimpleNamespace(cls=ArrayTensor([0, 0]), conf=ArrayTensor([0.9, 0.8])),
                                speed={"inference": 1.0})]


def test_yolo_union_inference_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "load_model", lambda _: FakeYolo())
    image = Image.new("RGB", (48, 40), "navy")
    raw = tmp_path / "raw"
    raw.mkdir()
    image.save(raw / "sky.png")
    output = tmp_path / "predictions"
    options = args("predict", "--model", "fake.pt", "--images", raw, "--output", output, "--save-ground")
    assert inference.run_predict(options) == 0
    mask = np.asarray(Image.open(next((output / "masks").glob("*.png")))) > 0
    ground = np.asarray(Image.open(next((output / "ground_masks").glob("*.png")))) > 0
    assert mask.sum() == 20 * 48 and np.array_equal(ground, ~mask)
    counts = inference.confusion_counts(np.array([[True, True], [False, False]]),
                                         np.array([[True, False], [True, False]]))
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    metrics = inference.semantic_metrics(counts)
    assert metrics["sky_iou"] == pytest.approx(1 / 3) and metrics["sky_dice"] == 0.5
    assert inference.semantic_metrics({k: 0 for k in counts})["sky_iou"] == 1


def test_semantic_evaluate_integration(tmp_path, monkeypatch):
    _, annotations, _ = create_annotations(tmp_path)
    output = tmp_path / "dataset"
    dataset.run(args("build", "--annotations", annotations, "--output", output))
    monkeypatch.setattr(inference, "load_model", lambda _: FakeYolo())
    metrics_out = tmp_path / "metrics"
    assert inference.run_evaluate(args("evaluate", "--model", "fake.pt", "--data", output / "data.yaml",
                                       "--split", "test", "--output", metrics_out)) == 0
    metrics = json.loads((metrics_out / "semantic_metrics.json").read_text())
    assert metrics["images"] == 1 and 0 < metrics["sky_iou"] < 1


@pytest.mark.parametrize("text", ["0 0.5 0.5 0.1 0.1", "1 0 0 1 0 1 1", "0 nan 0 1 0 1 1", "0 0 0 0 0 0 0"])
def test_invalid_labels(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_labels(path)


def test_train_overrides_and_cli_errors():
    options = args("train", "--data", "data.yaml", "--imgsz", "960", "--batch", "4", "--device", "cpu")
    model, config = train.training_settings(options)
    assert model == "yolov8s-seg.pt"
    assert config["imgsz"] == 960 and config["batch"] == 4 and config["device"] == "cpu"
    assert config["mask_ratio"] == 1 and config["overlap_mask"] is False
    assert main(["validate", "--data", "/no/such/data.yaml"]) == 1
