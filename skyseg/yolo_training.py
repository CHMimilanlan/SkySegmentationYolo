"""Importable Ultralytics adapters; loaded only for actual YOLO training."""

from __future__ import annotations

import inspect
import os
from copy import copy

from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import RANK

from .monitoring import TrainingMonitor, settings_from_environment


class SkySegmentationValidator(SegmentationValidator):
    def __call__(self, *args, **kwargs):
        trainer = kwargs.get("trainer", args[0] if args else None)
        self.skyseg_monitor = getattr(trainer, "skyseg_monitor", None)
        if self.skyseg_monitor is not None:
            self.skyseg_monitor.start_validation(trainer)
        result = super().__call__(*args, **kwargs)
        if self.skyseg_monitor is not None:
            self.skyseg_monitor.finish_validation()
            import torch.distributed as distributed
            if (self.skyseg_monitor.world_size > 1 and distributed.is_available()
                    and distributed.is_initialized()):
                distributed.barrier()
            self.skyseg_monitor.merge_distributed_predictions()
        return result

    def _prepare_batch(self, si, batch):
        prepared = super()._prepare_batch(si, batch)
        prepared["_skyseg_index"] = si
        return prepared

    def _prepare_pred(self, *args, **kwargs):
        result = super()._prepare_pred(*args, **kwargs)
        # Ultralytics 8.3's tensor API computes instance masks here. Reuse
        # those masks instead of decoding prototypes or running inference again.
        if self.skyseg_monitor is not None and len(args) == 3:
            pred, prepared, _ = args
            index = prepared["_skyseg_index"]
            self._save(index, result[1], pred[:, 5], pred[:, 4])
        return result

    def _save(self, index, masks, classes, scores):
        batch = self.skyseg_batch
        self.skyseg_monitor.save_prediction(batch["im_file"][index], masks, classes, scores,
                                            batch["img"].shape[2:], batch["ratio_pad"][index])

    def update_metrics(self, preds, batch):
        self.skyseg_batch = batch
        try:
            result = super().update_metrics(preds, batch)
            if self.skyseg_monitor is not None:
                if isinstance(preds, list) and all(isinstance(pred, dict) for pred in preds):
                    # Newer Ultralytics returns masks directly after postprocess.
                    for index, pred in enumerate(preds):
                        self._save(index, pred["masks"], pred["cls"], pred["conf"])
                elif isinstance(preds, tuple) and len(preds) == 2:
                    for index, pred in enumerate(preds[0]):
                        if len(pred) == 0:
                            self._save(index, None, [], [])
                else:
                    raise RuntimeError("Unsupported Ultralytics validation prediction format; "
                                       "cannot save per-epoch masks safely")
            return result
        finally:
            self.skyseg_batch = None  # do not retain a GPU batch between iterations


class SkySegmentationTrainer(SegmentationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Resume may restore val=False from old checkpoints; enforce the
        # requested validation and plots for both new and resumed training.
        self.args.val = True
        self.args.plots = True
        # Older Ultralytics validates the entire val set on rank zero only.
        # Newer validators gather stats across all ranks and split val images.
        validation_world_size = int(os.environ.get("WORLD_SIZE", "1")) if hasattr(
            SegmentationValidator, "gather_stats") else 1
        self.skyseg_monitor = TrainingMonitor(settings_from_environment(), main_process=RANK in {-1, 0},
                                              rank=max(0, RANK), world_size=validation_world_size)
        events = {"on_train_start": self.skyseg_monitor.on_train_start,
                  "on_train_epoch_start": self.skyseg_monitor.on_train_epoch_start,
                  "on_fit_epoch_end": self.skyseg_monitor.on_fit_epoch_end,
                  "on_train_end": self.skyseg_monitor.close, "teardown": self.skyseg_monitor.close}
        for event, callback in events.items():
            # DDP can inherit parent callbacks. Each process gets its own
            # monitor/writer, with only rank zero producing artifacts.
            self.callbacks[event] = [item for item in self.callbacks[event]
                                     if not isinstance(getattr(item, "__self__", None), TrainingMonitor)]
            self.add_callback(event, callback)

    def get_validator(self):
        if "proto" in inspect.signature(SegmentationValidator._prepare_pred).parameters:
            self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return SkySegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                        args=copy(self.args), _callbacks=self.callbacks)
