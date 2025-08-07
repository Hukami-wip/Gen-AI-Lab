import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from lit.nn.positional_encoding import PositionalEncoding


class TransformerLitModel(LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(src)
        return self.fc_out(output)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )

        preds = torch.argmax(outputs, dim=-1)
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.output_dim), targets.view(-1)
        )

        preds = torch.argmax(outputs, dim=-1)
        acc = (preds == targets).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
