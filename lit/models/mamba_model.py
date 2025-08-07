import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class MambaLitModel(LightningModule):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = MambaConfig(
            d_model=self.hparams.d_model,
            n_layer=self.hparams.n_layer,
            vocab_size=self.hparams.vocab_size,
        )
        self.model = MambaLMHeadModel(config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # Mamba expects inputs of type long
        inputs = inputs.squeeze(-1).long()
        targets = targets.squeeze(-1).long()

        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.vocab_size), targets.view(-1)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.squeeze(-1).long()
        targets = targets.squeeze(-1).long()

        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.vocab_size), targets.view(-1)
        )

        preds = torch.argmax(outputs, dim=-1)
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.squeeze(-1).long()
        targets = targets.squeeze(-1).long()

        outputs = self.forward(inputs)
        loss = self.criterion(
            outputs.view(-1, self.hparams.vocab_size), targets.view(-1)
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
