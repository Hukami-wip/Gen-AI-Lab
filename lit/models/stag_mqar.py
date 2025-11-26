import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from models.stag import Stag


class StagMQAR(LightningModule):
    """
    STAG model for Multi-Query Associative Recall task.

    Architecture:
        Embedding -> STAG backbone -> Linear head

    The model processes the full sequence and predicts at each position.
    Loss is computed only on query positions (using ignore_index=-100).
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        state_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, model_dim)

        # STAG backbone
        self.backbone = Stag(
            model_dim=model_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Output head
        self.head = nn.Linear(model_dim, vocab_size)

        # Loss with ignore_index for masking KV portion
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(x)  # (batch, seq_len, model_dim)
        x = self.backbone(x)  # (batch, seq_len, model_dim)
        return self.head(x)  # (batch, seq_len, vocab_size)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.criterion(
            logits.view(-1, self.hparams.vocab_size), targets.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.criterion(
            logits.view(-1, self.hparams.vocab_size), targets.view(-1)
        )

        # Compute accuracy only on query positions (where target != -100)
        preds = logits.argmax(dim=-1)
        mask = targets != -100
        correct = (preds[mask] == targets[mask]).float()
        acc = correct.mean() if mask.sum() > 0 else torch.tensor(0.0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)

        loss = self.criterion(
            logits.view(-1, self.hparams.vocab_size), targets.view(-1)
        )

        preds = logits.argmax(dim=-1)
        mask = targets != -100
        correct = (preds[mask] == targets[mask]).float()
        acc = correct.mean() if mask.sum() > 0 else torch.tensor(0.0)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
