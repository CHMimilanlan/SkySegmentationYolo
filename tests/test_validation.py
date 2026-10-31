"""Pre-training validation progress, decode count and integrity regressions."""

from __future__ import annotations

import io
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from skyseg import annotate, dataset, train, validate as validation
from skyseg.cli import parser
from skyseg.common import read_jsonl, save_mask, write_jsonl
from skyseg.polygons import labels_text
from skyseg.png_profiles import strip_png_icc


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
    records = read_jsonl(built_dataset / "dataset_manifest.jsonl")
    for record in records:
        path = built_dataset / record["image"]
        with Image.open(path) as image:
            copy = image.copy()
        copy.save(path, icc_profile=b"fixture ICC profile")
        assert strip_png_icc(path, in_place=True)
    result = validation.validate_dataset(built_dataset / "data.yaml")
    assert sum(split["images"] for split in result.values()) == len(records)


def add_dataset_profiles(root):
    paths = [root / record["image"] for record in read_jsonl(root / "dataset_manifest.jsonl")]
    for path in paths:
        with Image.open(path) as image:
            copy = image.copy()
        copy.save(path, icc_profile=b"fixture ICC profile")
    return paths


def test_builder_cleans_old_annotations_without_modifying_sources(built_dataset, tmp_path):
    annotations = built_dataset.parent / "annotations"
    sources = []
    for record in read_jsonl(annotations / "manifest.jsonl"):
        path = annotations / record["image"]
        with Image.open(path) as image:
            copy = image.copy()
        copy.save(path, icc_profile=b"fixture ICC profile")
        sources.append(path)
    source_bytes = {path: path.read_bytes() for path in sources}
    output = tmp_path / "rebuilt"
    args = parser().parse_args(["build", "--annotations", str(annotations), "--output", str(output)])
    assert dataset.run(args) == 0
    assert all(not strip_png_icc(output / row["image"]) for row in read_jsonl(output / "dataset_manifest.jsonl"))
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    validation.validate_dataset(output / "data.yaml")


def test_clean_command_uses_yaml_root_from_another_working_directory(built_dataset, tmp_path, monkeypatch, capsys):
    from skyseg.png_profiles import run

    paths = add_dataset_profiles(built_dataset)
    data = built_dataset / "data.yaml"
    config = yaml.safe_load(data.read_text())
    config["path"] = "."  # relative to data.yaml, never the process's cwd
    data.write_text(yaml.safe_dump(config))
    unrelated = tmp_path / "other_working_directory"
    unrelated.mkdir()
    decoy = unrelated / "decoy.png"
    Image.new("RGB", (20, 16)).save(decoy, icc_profile=b"leave this unrelated file alone")
    decoy_bytes = decoy.read_bytes()
    monkeypatch.chdir(unrelated)
    args = parser().parse_args(["clean-png", "--data", str(data), "--in-place"])
    assert run(args) == 0
    assert all(not strip_png_icc(path) for path in paths)
    assert decoy.read_bytes() == decoy_bytes
    output = capsys.readouterr().out
    assert f"Resolved dataset root: {built_dataset}" in output
    assert all(str(built_dataset / "images" / split) in output for split in ("train", "val", "test"))
    validation.validate_dataset(built_dataset / "data.yaml")


@pytest.mark.parametrize("resume", [False, True])
def test_training_cleans_existing_dataset_before_yolo_reads_it(built_dataset, tmp_path, monkeypatch, capfd, resume):
    paths = add_dataset_profiles(built_dataset)
    trained = []
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"fake checkpoint")

    class FakeYOLO:
        def __init__(self, model, task):
            assert task == "segment"
            assert all(not strip_png_icc(path) for path in paths)
            self.ckpt = {"train_args": {"data": str(built_dataset / "data.yaml")}}
            self.trainer = SimpleNamespace(save_dir=tmp_path / "runs")

        def train(self, **kwargs):
            # Mimic independent repeated training/validation OpenCV reads.
            for _ in range(2):
                for path in paths:
                    assert cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) is not None
            trained.append(kwargs)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    # Keep this regression focused on cleanup before YOLO reads the images;
    # the real trainer/validator adapter has its own monitoring tests.
    monkeypatch.setattr(train, "run_training", lambda model, monitoring, resume, config:
                        model.train(**({"resume": True} if resume else {}), **config))
    monkeypatch.setattr(train.importlib.metadata, "version", lambda _: "test")
    arguments = ["train", "--data", str(built_dataset / "data.yaml")]
    if resume:
        arguments.extend(["--resume", "--model", str(checkpoint)])
    assert train.run(parser().parse_args(arguments)) == 0
    assert len(trained) == 1
    assert bool(trained[0].get("resume")) == resume
    assert "libpng warning" not in capfd.readouterr().err
    validation.validate_dataset(built_dataset / "data.yaml")


def test_dry_run_does_not_modify_profiled_dataset(built_dataset):
    paths = add_dataset_profiles(built_dataset)
    original = {path: path.read_bytes() for path in paths}
    args = parser().parse_args(["train", "--data", str(built_dataset / "data.yaml"), "--dry-run"])
    assert train.run(args) == 0
    assert all(path.read_bytes() == content for path, content in original.items())


def test_training_rejects_ambiguous_relative_root_before_cleaning(built_dataset):
    paths = add_dataset_profiles(built_dataset)
    original = {path: path.read_bytes() for path in paths}
    data = built_dataset / "data.yaml"
    config = yaml.safe_load(data.read_text())
    config["path"] = "."
    data.write_text(yaml.safe_dump(config))
    args = parser().parse_args(["train", "--data", str(data)])
    with pytest.raises(ValueError, match="skyseg and Ultralytics read the same images") as error:
        train.run(args)
    assert f"set path: {built_dataset}" in str(error.value)
    assert all(path.read_bytes() == content for path, content in original.items())


def test_training_stops_on_cleanup_failure_before_loading_yolo(built_dataset, monkeypatch):
    from skyseg import png_profiles

    paths = add_dataset_profiles(built_dataset)
    original = {path: path.read_bytes() for path in paths}

    def fail_copy(*args):
        raise PermissionError("simulated read-only dataset")

    monkeypatch.setattr(png_profiles, "_copy_bytes", fail_copy)
    args = parser().parse_args(["train", "--data", str(built_dataset / "data.yaml")])
    with pytest.raises(ValueError, match="PNG ICC preparation failed; training was not started"):
        train.run(args)
    assert all(path.read_bytes() == content for path, content in original.items())


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
