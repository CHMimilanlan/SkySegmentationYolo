"""Pre-training validation progress, decode count and integrity regressions."""

from __future__ import annotations

import io
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest
from PIL import Image

from skyseg import annotate, dataset, validate as validation
from skyseg.cli import parser
from skyseg.common import read_jsonl, save_mask, write_jsonl
from skyseg.polygons import labels_text


@pytest.fixture
def built_dataset(tmp_path):
    raw, annotations, output = tmp_path / "raw", tmp_path / "annotations", tmp_path / "dataset"
    raw.mkdir()
    for index, size in enumerate(((64, 48), (96, 64), (80, 120), (128, 72))):
        Image.new("RGB", size, (20 + index * 30, 40, 60)).save(raw / f"{index}.png")

    class Backend:
        def predict(self, image, prompt):
            masks = np.zeros((1, 1, image.height, image.width), dtype=bool)
            masks[:, :, :image.height // 2] = True
            return masks, np.array([0.9])

    args = parser().parse_args(["annotate", "--images", str(raw), "--output", str(annotations)])
    assert annotate.run(args, backend=Backend()) == 0
    args = parser().parse_args(["build", "--annotations", str(annotations), "--output", str(output)])
    assert dataset.run(args) == 0
    return output


def test_progress_precedes_decode_and_each_file_is_read_once(built_dataset, monkeypatch):
    stdout, stderr = io.StringIO(), io.StringIO()
    reads = {name: Counter() for name in ("read_rgb", "read_mask", "read_labels")}
    for name in reads:
        original = getattr(validation, name)

        def observed(path, *args, _name=name, _original=original, **kwargs):
            reads[_name][str(path)] += 1
            if _name == "read_rgb":
                assert "[validate] Checking" in stdout.getvalue()
                assert f"file={path.name}" in stderr.getvalue()
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(validation, name, observed)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = validation.validate_dataset(built_dataset / "data.yaml")
    assert sum(split["images"] for split in result.values()) == 4
    for counts in reads.values():
        assert len(counts) == 4 and set(counts.values()) == {1}
    assert "[validate] Loading dataset configuration" in stdout.getvalue()
    assert "Dataset validation passed" in stdout.getvalue()
    for split in ("train", "val", "test"):
        assert f"Validate {split}" in stderr.getvalue()


def test_decode_error_identifies_file(built_dataset, capsys):
    record = read_jsonl(built_dataset / "dataset_manifest.jsonl")[0]
    path = built_dataset / record["image"]
    path.write_bytes(b"not a valid image")
    with pytest.raises(ValueError, match="Dataset validation failed while checking") as error:
        validation.validate_dataset(built_dataset / "data.yaml")
    assert str(path) in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    output = capsys.readouterr()
    assert path.name in output.err


def test_profile_cleanup_keeps_dataset_manifest_valid(built_dataset):
    from skyseg.png_profiles import strip_png_icc

    records = read_jsonl(built_dataset / "dataset_manifest.jsonl")
    for record in records:
        path = built_dataset / record["image"]
        with Image.open(path) as image:
            copy = image.copy()
        copy.save(path, icc_profile=b"fixture ICC profile")
        assert strip_png_icc(path, in_place=True)
    result = validation.validate_dataset(built_dataset / "data.yaml")
    assert sum(split["images"] for split in result.values()) == len(records)


@pytest.mark.parametrize("kind,expected", [
    ("image", "Image pixels differ from manifest"),
    ("mask", "Reference mask changed"),
    ("labels", "Labels differ from conversion manifest"),
    ("groups", "Scene group leaks between splits"),
])
def test_single_pass_keeps_integrity_checks(built_dataset, kind, expected):
    manifest_path = built_dataset / "dataset_manifest.jsonl"
    records = read_jsonl(manifest_path)
    record = records[0]
    if kind == "image":
        path = built_dataset / record["image"]
        with Image.open(path) as image:
            size = image.size
        Image.new("RGB", size, (199, 33, 44)).save(path)
    elif kind == "mask":
        path = built_dataset / record["mask"]
        with Image.open(path) as image:
            mask = np.asarray(image) > 0
        mask[0, 0] = ~mask[0, 0]
        save_mask(path, mask)
    elif kind == "labels":
        path = built_dataset / record["label"]
        polygons = validation.read_labels(path)
        polygons[0][0, 0] = 0.1
        path.write_text(labels_text(polygons), encoding="utf-8")
    else:
        for row in records:
            row["split_group"] = "same_scene"
        write_jsonl(manifest_path, records)
    with pytest.raises(ValueError, match=expected):
        validation.validate_dataset(built_dataset / "data.yaml")
