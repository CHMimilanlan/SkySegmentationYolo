"""Opt-in real CPU training: SKYSEG_RUN_TRAINING_SMOKE=1 pytest -q -s tests/test_training_smoke.py."""

import json
import os

import numpy as np
import pytest
import yaml
from PIL import Image

from skyseg import annotate, dataset, train
from skyseg.cli import parser


@pytest.mark.skipif(os.environ.get("SKYSEG_RUN_TRAINING_SMOKE") != "1", reason="opt-in real CPU training")
@pytest.mark.parametrize("interrupt_and_resume", [False, True])
def test_real_training_validates_and_saves_every_epoch(tmp_path, monkeypatch, interrupt_and_resume):
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    torch.set_num_threads(2)
    # This test needs no downloaded assets, including an optional Arial font.
    # Use the locally installed plotting font for the upstream font check.
    import matplotlib.font_manager
    import ultralytics.data.utils as data_utils
    import ultralytics.utils.plotting as plotting
    local_font = lambda *_, **__: matplotlib.font_manager.findfont("DejaVu Sans")
    monkeypatch.setattr(data_utils, "check_font", local_font)
    monkeypatch.setattr(plotting, "check_font", local_font)
    # Disable optional upstream integrations in this isolated test only.
    for key in ("tensorboard", "wandb", "mlflow", "comet", "clearml"):
        if key in ultralytics.settings:
            monkeypatch.setitem(ultralytics.settings, key, False)
    raw, annotations, data = tmp_path / "raw", tmp_path / "annotations", tmp_path / "data"
    raw.mkdir()
    for i, size in enumerate(((64, 48), (96, 64), (80, 120), (128, 72))):
        Image.new("RGB", size, (20 + i * 30, 40, 60)).save(raw / f"{i}.png")

    class Backend:
        def predict(self, image, prompt):
            masks = np.zeros((1, 1, image.height, image.width), dtype=bool)
            masks[:, :, :image.height // 2] = True
            return masks, np.array([0.9])

    assert annotate.run(parser().parse_args(["annotate", "--images", str(raw), "--output", str(annotations)]),
                        backend=Backend()) == 0
    assert dataset.run(parser().parse_args(["build", "--annotations", str(annotations), "--output", str(data)])) == 0
    from pathlib import Path
    config = yaml.safe_load(Path("configs/train_yolov8s_seg.yaml").read_text())
    epochs = 3 if interrupt_and_resume else 2
    config.update(model="yolov8s-seg.yaml", epochs=epochs, imgsz=64, batch=2, device="cpu", workers=0,
                  amp=False, pretrained=False, project=str(tmp_path / "runs"), name="smoke", exist_ok=True,
                  conf=0.000001, max_det=4, val=False, plots=False)
    config["skyseg"]["val_mask_conf"] = 0.0
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run = tmp_path / "runs/smoke"
    arguments = ["train", "--data", str(data / "data.yaml"), "--config", str(config_path)]
    if interrupt_and_resume:
        original = train.run_training

        def stop_at_second_epoch(state):
            if state.epoch == 1:
                raise InterruptedError("simulate an interrupted training process")

        def first_run(model, monitoring, *, resume, config):
            model.add_callback("on_train_epoch_start", stop_at_second_epoch)
            return original(model, monitoring, resume=resume, config=config)

        monkeypatch.setattr(train, "run_training", first_run)
        with pytest.raises(InterruptedError):
            train.run(parser().parse_args(arguments))
        monkeypatch.setattr(train, "run_training", original)
        arguments += ["--resume", "--model", str(run / "weights/last.pt")]
    assert train.run(parser().parse_args(arguments)) == 0
    val_images = list((data / "images/val").glob("*.png"))
    for step in range(1, epochs + 1):
        folder = run / "validation" / f"epoch_{step:04d}"
        records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        assert len(records) == len(val_images)
        for record in records:
            with Image.open(folder / record["mask"]) as mask, Image.open(record["source"]) as image:
                assert mask.size == image.size
            assert (folder / record["overlay"]).is_file()
        assert json.loads((folder / "metrics.json").read_text())["epoch"] == step
    assert not (run / "validation" / f"epoch_{epochs + 1:04d}").exists()
    assert (run / "results.png").is_file() and (run / "results.csv").is_file()
    events = EventAccumulator(str(run / "tensorboard")).Reload()
    assert [event.step for event in events.Scalars("train/seg_loss")] == list(range(1, epochs + 1))
    assert [event.step for event in events.Scalars("val/seg_loss")] == list(range(1, epochs + 1))
    assert events.Tags()["images"]
