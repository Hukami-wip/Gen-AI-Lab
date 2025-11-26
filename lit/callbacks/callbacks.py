import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class MetricsCallback(Callback):
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        pl_module.logger.experiment.add_scalar(
            "model_total_parameters", float(total_params), 0
        )
        pl_module.logger.experiment.add_scalar(
            "model_trainable_parameters", float(trainable_params), 0
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_start_time = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.monotonic() - self.train_batch_start_time
        batch_size = batch[0].size(0)
        throughput = batch_size / batch_time
        pl_module.log("train_throughput_samples_per_sec", throughput)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.val_batch_start_time = time.monotonic()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        latency = time.monotonic() - self.val_batch_start_time
        pl_module.log("validation_latency_sec", latency)

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.test_batch_start_time = time.monotonic()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        latency = time.monotonic() - self.test_batch_start_time
        pl_module.log("test_latency_sec", latency)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.is_global_zero:
            final_metrics = {
                k: v.item() for k, v in trainer.logged_metrics.items() if "test" in k
            }
            pl_module.logger.log_hyperparams(pl_module.hparams, final_metrics)
