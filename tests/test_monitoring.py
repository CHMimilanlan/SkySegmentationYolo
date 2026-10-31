"""Epoch logging, mask geometry, confidence filtering and resume regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from skyseg.cli import parser
from skyseg.monitoring import TrainingMonitor, monitoring_settings, restore_mask
from skyseg.train import training_settings


class Writer:
    def __init__(self, **kwargs):
        self.options, self.scalars, self.images = kwargs, [], []
        self.flushed, self.closed = 0, False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_image(self, tag, image, step, **kwargs):
        self.images.append((tag, image.copy(), step))

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


def trainer(root, epoch=0, resume=False):
    return SimpleNamespace(save_dir=root, epoch=epoch, start_epoch=epoch, args=SimpleNamespace(resume=resume),
                           tloss=[1, 2, 3, 4], metrics={"val/seg_loss": 0.7, "metrics/mAP50(M)": 0.8},
                           lr={"lr/pg0": 0.001},
                           label_loss_items=lambda values, prefix: {f"{prefix}/{name}_loss": value
                               for name, value in zip(("box", "seg", "cls", "dfl"), values)})


def test_monitoring_settings_and_yolo_arguments_are_separate(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("model: yolov8s-seg.pt\nval: false\nplots: false\nskyseg:\n  val_save_limit: 5\n")
    args = parser().parse_args(["train", "--data", "data.yaml", "--config", str(config),
                                "--val-save-limit", "2", "--no-tensorboard"])
    _, values = training_settings(args)
    assert values["val"] is True and values["plots"] is True and "skyseg" not in values
    settings = monitoring_settings(args, {"skyseg": {"val_save_limit": 5}})
    assert settings["val_save_limit"] == 2 and settings["tensorboard"] is False


@pytest.mark.parametrize("settings", [{"val_save_limit": -1}, {"val_mask_conf": 1.1},
                                     {"tensorboard_images": -1}, {"typo": True}])
def test_invalid_monitoring_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        monitoring_settings(SimpleNamespace(), {"skyseg": settings})


@pytest.mark.parametrize("original,input_shape,gains,pad,mask_shape", [
    ((80, 40), (64, 64), (0.8, 0.8), (0, 16), (16, 16)),
    ((40, 80), (64, 64), (0.8, 0.8), (16, 0), (64, 64)),
    ((77, 51), (64, 96), (64 / 51, 96 / 77), (0, 0), (16, 24)),
])
def test_masks_return_to_original_shape_after_letterbox(original, input_shape, gains, pad, mask_shape):
    mask = np.ones(mask_shape, dtype=bool)
    restored = restore_mask(mask, original, input_shape, (gains, pad))
    assert restored.shape == (original[1], original[0]) and restored.all()


def test_letterbox_padding_does_not_become_sky():
    mask = np.zeros((64, 64), dtype=bool)
    mask[:16] = True
    restored = restore_mask(mask, (80, 40), (64, 64), ((0.8, 0.8), (0, 16)))
    assert not restored.any()


def test_every_epoch_keeps_masks_and_logs_current_train_and_validation_losses(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (80, 40), "navy").save(source)
    monitor = TrainingMonitor(writer_factory=Writer)
    state = trainer(tmp_path / "run")
    monitor.on_train_start(state)
    writer = monitor.writer
    for epoch in range(2):
        state.epoch = epoch
        monitor.on_train_epoch_start(state)
        monitor.start_validation(state)
        masks = np.zeros((3, 64, 64), dtype=bool)
        masks[0, 16:32] = True
        masks[1, 32:48] = True
        masks[2] = True
        monitor.save_prediction(source, masks, [0, 0, 1], [0.9, 0.1, 0.99],
                                (64, 64), ((0.8, 0.8), (0, 16)))
        monitor.finish_validation()
        monitor.on_fit_epoch_end(state)
        folder = Path(state.save_dir) / "validation" / f"epoch_{epoch + 1:04d}"
        with Image.open(folder / "masks/image.png") as image:
            mask = np.asarray(image)
        assert mask.shape == (40, 80) and mask[:18].all() and not mask[22:].any()
        assert (folder / "overlays/image.jpg").exists()
        records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        assert records[0]["instances"] == 1
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["epoch"] == epoch + 1 and metrics["saved_images"] == 1
    assert ("train/total_loss", 10.0, 1) in writer.scalars
    assert ("val/seg_loss", 0.7, 2) in writer.scalars
    assert len(writer.images) == 2
    count = len(writer.scalars)
    state.epoch += 1  # final best.pt evaluation is not a third training epoch
    monitor.on_fit_epoch_end(state)
    assert len(writer.scalars) == count
    monitor.close()
    assert writer.closed and writer.flushed >= 3


def test_empty_prediction_limit_resume_step_and_interrupted_validation(tmp_path):
    state = trainer(tmp_path / "run", epoch=4, resume=True)
    monitor = TrainingMonitor({"val_save_limit": 1}, writer_factory=Writer)
    monitor.on_train_start(state)
    writer = monitor.writer
    assert writer.options["purge_step"] == 5
    monitor.on_train_epoch_start(state)
    monitor.start_validation(state)
    for index in range(2):
        source = tmp_path / f"{index}.png"
        Image.new("RGB", (20, 10)).save(source)
        monitor.save_prediction(source, None, [], [], (10, 20), ((1, 1), (0, 0)))
    assert monitor.saved == 1
    with Image.open(monitor.epoch_dir / "masks/0.png") as image:
        assert not np.asarray(image).any()
    monitor.close()
    assert (monitor.epoch_dir / "predictions.jsonl.tmp").exists()
    assert not (monitor.epoch_dir / "predictions.jsonl").exists()
    assert writer.closed


def test_non_main_process_does_not_create_writer_or_files(tmp_path):
    monitor = TrainingMonitor(main_process=False, writer_factory=lambda **_: pytest.fail("non-main writer"))
    state = trainer(tmp_path / "run")
    monitor.on_train_start(state)
    monitor.on_train_epoch_start(state)
    monitor.start_validation(state)
    monitor.on_fit_epoch_end(state)
    monitor.close()
    assert not Path(state.save_dir).exists()


def test_distributed_artifacts_include_all_shards_and_deduplicate_sampler_padding(tmp_path):
    state = trainer(tmp_path / "run")
    sources = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (20, 10)).save(path)
        sources.append(path)
    monitors = [TrainingMonitor({"tensorboard": False}, rank=rank, world_size=2, main_process=rank == 0)
                for rank in range(2)]
    for rank, monitor in enumerate(monitors):
        monitor.on_train_epoch_start(state)
        monitor.start_validation(state)
        for index in ((0, 2) if rank == 0 else (1, 0)):
            monitor.save_prediction(sources[index], None, [], [], (10, 20), ((1, 1), (0, 0)))
        monitor.finish_validation()
    main = monitors[0]
    main.merge_distributed_predictions()
    main.on_fit_epoch_end(state)
    records = [json.loads(line) for line in (main.epoch_root / "predictions.jsonl").read_text().splitlines()]
    assert {row["id"] for row in records} == {"0", "1", "2"} and main.saved == 3
    assert all((main.epoch_root / row["mask"]).is_file() for row in records)
