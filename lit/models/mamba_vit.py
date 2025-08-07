import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import Accuracy

from models.mamba.mamba_vit import MambaViT


class LitMambaViT(LightningModule):
    def __init__(
        self,
        model_dim: int,
        n_layers: int,
        n_classes: int,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # We will move the model implementation here later
        self.model = MambaViT(
            model_dim=model_dim,
            n_layers=n_layers,
            n_classes=n_classes,
            **kwargs,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
